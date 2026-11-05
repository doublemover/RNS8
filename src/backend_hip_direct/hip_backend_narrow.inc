#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_fused_narrow_rns(const void*, const void*, int8_t*, int, int, int,
                                                int, int, int, bool, bool);
extern "C" int rns8_hip_native_i64_abs_max(const int64_t*, int, int, int, void*);
extern "C" int rns8_hip_native_domain(const void*, int, int, int, void*, bool);
extern "C" int rns8_hip_native_i64_narrow_gemm(const int64_t*, const int64_t*, int64_t*, int, int,
                                               int, int, int, int);
extern "C" int rns8_hip_native_byte_wmma(const void*, const void*, void*, int, int, int, int, int,
                                         int, bool);
extern "C" int rns8_hip_native_narrow_split_partials(const int64_t*, const int64_t*, int64_t*, int,
                                                     int, int, int, int);
extern "C" int rns8_hip_native_narrow_split_encode(const int64_t*, int8_t*, int, int, int, int);
extern "C" int rns8_hip_native_narrow_group_stage(const NativeNarrowTask*, int, int, int, int, int,
                                                  uint32_t, bool, bool);
#endif

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
namespace {
// Arithmetic admission belongs to the checked scan or immutable owner proof.
// Share the actual launch/timing boundary between those two authorities.
rns8_status launch_admitted_narrow_product(const void* a, const void* b, void* output, int64_t m,
                                           int64_t n, int64_t k, int64_t lda, int64_t ldb,
                                           int64_t ldc, uint32_t prefix, bool byte_wmma,
                                           bool unsigned_inputs, bool* write_started) {
  if (write_started)
    *write_started = true;
  const char* label =
      byte_wmma ? (prefix ? "native_byte_wmma_fused_rns" : "native_byte_wmma_gemm")
                : (prefix ? "narrow_native_i64_fused_rns_kernel" : "narrow_native_i64_gemm_kernel");
  const auto error = timed_hip_operation(label, [&] {
    const auto launched = static_cast<hipError_t>(
        prefix ? rns8_hip_native_fused_narrow_rns(a, b, static_cast<int8_t*>(output), int(m),
                                                  int(n), int(k), int(lda), int(ldb), int(prefix),
                                                  byte_wmma, unsigned_inputs)
        : byte_wmma ? rns8_hip_native_byte_wmma(a, b, output, int(m), int(n), int(k), int(lda),
                                                int(ldb), int(ldc), unsigned_inputs)
                    : rns8_hip_native_i64_narrow_gemm(static_cast<const int64_t*>(a),
                                                      static_cast<const int64_t*>(b),
                                                      static_cast<int64_t*>(output), int(m), int(n),
                                                      int(k), int(lda), int(ldb), int(ldc)));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status encode_narrow_product(const void* product, void* output, int64_t m, int64_t n,
                                  uint32_t prefix, bool unsigned_inputs, bool& commit_started) {
  commit_started = true;
  const auto error = timed_hip_operation("narrow_native_encode_rns", [&] {
    const auto code = unsigned_inputs
                          ? rns8_hip_direct_pack_u64_device(static_cast<const uint64_t*>(product),
                                                            static_cast<int8_t*>(output), int(m),
                                                            int(n), int(n), int(prefix))
                          : rns8_hip_direct_pack_i64_device(static_cast<const int64_t*>(product),
                                                            static_cast<int8_t*>(output), int(m),
                                                            int(n), int(n), int(prefix));
    return code == int(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}
} // namespace
#endif

rns8_status hip_direct_scan_native_domain(int device_id, const void* source, int64_t rows,
                                          int64_t cols, int64_t ld, bool unsigned_inputs,
                                          hip_direct_device_buffer& scratch,
                                          NativeDomainRange& out) {
  constexpr uint64_t bytes = 2 * sizeof(uint64_t);
  if (!source || !scratch || scratch.device_id() != device_id || scratch.bytes() < bytes ||
      !qualified_i32_matrix_address(rows, cols, ld, 1, 8))
    return RNS8_INVALID_ARGUMENT;
  if (!nonoverlapping_byte_ranges(source, uint64_t(rows) * uint64_t(ld) * 8, scratch.get(), bytes))
    return RNS8_INVALID_ARGUMENT;
  if (scratch.pending())
    return RNS8_BUSY;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  scratch.mark_pending();
  const auto error = timed_hip_operation("narrow_native_domain_scan", [&] {
    auto* words = static_cast<uint64_t*>(scratch.get());
    auto issued = hipMemset(words, 0xff, sizeof(uint64_t));
    if (issued == hipSuccess)
      issued = hipMemset(words + 1, 0, sizeof(uint64_t));
    if (issued == hipSuccess)
      issued = static_cast<hipError_t>(
          rns8_hip_native_domain(source, int(rows), int(cols), int(ld), words, unsigned_inputs));
    return issued == hipSuccess ? hipDeviceSynchronize() : issued;
  });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  uint64_t extrema[2]{};
  status = hip_direct_copy_device_to_host(device_id, extrema, scratch.get(), sizeof(extrema));
  if (status != RNS8_SUCCESS)
    return status;
  scratch.mark_completed();
  if (extrema[0] > extrema[1])
    return RNS8_BACKEND_FAILURE;
  NativeDomainRange result;
  if (unsigned_inputs)
    result = NativeDomainRange::unsigned_bounds(extrema[0], extrema[1]);
  else {
    const uint64_t low_bits = extrema[0] ^ (UINT64_C(1) << 63);
    const uint64_t high_bits = extrema[1] ^ (UINT64_C(1) << 63);
    int64_t minimum = 0, maximum = 0;
    std::memcpy(&minimum, &low_bits, sizeof(minimum));
    std::memcpy(&maximum, &high_bits, sizeof(maximum));
    result = NativeDomainRange::signed_bounds(minimum, maximum);
  }
  if (!result.well_formed())
    return RNS8_BACKEND_FAILURE;
  out = result;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)unsigned_inputs;
  (void)out;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_narrow_checked_device(int device_id, const int64_t* a,
                                                      const int64_t* b, int64_t* c, int64_t m,
                                                      int64_t n, int64_t k, int64_t lda,
                                                      int64_t ldb, int64_t ldc,
                                                      uint64_t* out_abs_sum_bound,
                                                      uint32_t rns_prefix, bool* write_started) {
  if (write_started)
    *write_started = false;
  if (rns_prefix > 20 || (rns_prefix && ldc != n))
    return RNS8_INVALID_ARGUMENT;
  if (!a || !b || !c || !out_abs_sum_bound || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n ||
      ldc < n)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc) ||
      (rns_prefix && !qualified_i32_matrix_address(m, n, ldc, rns_prefix)))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes =
      static_cast<uint64_t>(m) * ldc * (rns_prefix ? rns_prefix : sizeof(int64_t));
  if (!nonoverlapping_byte_ranges(c, bytes, a, static_cast<uint64_t>(m) * lda * sizeof(int64_t)) ||
      !nonoverlapping_byte_ranges(c, bytes, b, static_cast<uint64_t>(k) * ldb * sizeof(int64_t)))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer maxima;
  status = maxima.allocate(device_id, 2 * sizeof(uint64_t));
  if (status != RNS8_SUCCESS)
    return status;
  maxima.mark_pending();
  if (hipMemset(maxima.get(), 0, 2 * sizeof(uint64_t)) != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  auto* device_maxima = static_cast<uint64_t*>(maxima.get());
  const auto scan = timed_hip_operation("narrow_native_bound_scan", [&]() {
    auto error = static_cast<hipError_t>(rns8_hip_native_i64_abs_max(
        a, static_cast<int>(m), static_cast<int>(k), static_cast<int>(lda), device_maxima));
    if (error == hipSuccess)
      error = static_cast<hipError_t>(rns8_hip_native_i64_abs_max(
          b, static_cast<int>(k), static_cast<int>(n), static_cast<int>(ldb), device_maxima + 1));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (scan != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  uint64_t bounds[2]{};
  status = hip_direct_copy_device_to_host(device_id, bounds, device_maxima, sizeof(bounds));
  if (status != RNS8_SUCCESS)
    return status;
  maxima.mark_completed();
  constexpr uint64_t limit = INT64_MAX;
  uint64_t sum_bound = 0;
  if (!native_partial_sum_bound(bounds[0], bounds[1], static_cast<uint64_t>(k), limit, sum_bound))
    return RNS8_RANGE_ERROR;
  const auto product_status = launch_admitted_narrow_product(
      a, b, c, m, n, k, lda, ldb, ldc, rns_prefix, false, false, write_started);
  if (product_status != RNS8_SUCCESS)
    return product_status;
  *out_abs_sum_bound = sum_bound;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_narrow_rns_certified_device(
    int device_id, const NativeDomainCertificate& a, const NativeDomainCertificate& b, void* output,
    uint32_t prefix, bool byte_wmma, bool fused, const NativeWmmaDeviceCapability& capability,
    void* native_product, uint64_t product_bytes, uint64_t budget, bool& commit_started,
    bool split_k) {
  commit_started = false;
  const auto& lhs = a.identity;
  const auto& rhs = b.identity;
  if (!output || !a.matches(lhs) || !b.matches(rhs) || lhs.device != device_id ||
      rhs.device != device_id || lhs.unsigned_inputs != rhs.unsigned_inputs ||
      lhs.cols != rhs.rows || lhs.method != NativeBoundMethod::tile_max ||
      rhs.method != NativeBoundMethod::tile_max || lhs.panel_k || rhs.panel_k || lhs.row_tiles ||
      rhs.row_tiles)
    return RNS8_INVALID_ARGUMENT;
  const bool unsigned_inputs = lhs.unsigned_inputs;
  if (unsigned_inputs && !byte_wmma)
    return RNS8_UNSUPPORTED_BACKEND;
  const auto m = lhs.rows, n = rhs.cols, k = lhs.cols;
  const auto support =
      query_narrow_rns_support(m, n, k, lhs.ld, rhs.ld, prefix, byte_wmma, budget, fused, split_k);
  if (support.status != RNS8_SUCCESS)
    return support.status;
  const uint64_t output_bytes = uint64_t(m) * uint64_t(n) * prefix;
  const uint64_t a_bytes = uint64_t(m) * uint64_t(lhs.ld) * 8;
  const uint64_t b_bytes = uint64_t(k) * uint64_t(rhs.ld) * 8;
  if (!nonoverlapping_byte_ranges(output, output_bytes, lhs.native, a_bytes) ||
      !nonoverlapping_byte_ranges(output, output_bytes, rhs.native, b_bytes))
    return RNS8_INVALID_ARGUMENT;
  if (support.payload_bytes &&
      (!native_product || product_bytes < support.payload_bytes ||
       reinterpret_cast<uintptr_t>(native_product) % alignof(uint64_t) ||
       !nonoverlapping_byte_ranges(native_product, support.payload_bytes, output, output_bytes) ||
       !nonoverlapping_byte_ranges(native_product, support.payload_bytes, lhs.native, a_bytes) ||
       !nonoverlapping_byte_ranges(native_product, support.payload_bytes, rhs.native, b_bytes)))
    return RNS8_INVALID_ARGUMENT;
  if (byte_wmma) {
    if (capability.provider != &hip_direct_query_native_wmma_capability ||
        capability.device_id != device_id || !capability.target[0])
      return RNS8_INVALID_ARGUMENT;
    if (!capability.supported)
      return RNS8_UNSUPPORTED_BACKEND;
  }
  uint64_t bound = 0;
  const auto admission =
      native_narrow_domain_admission(a.range, b.range, uint64_t(k), byte_wmma, bound);
  if (admission != RNS8_SUCCESS)
    return admission;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS)
    return selected;
  if (split_k) {
    // The full-K magnitude envelope also bounds each lane, block and merge
    // partial. Output currentness changes only when the merge is submitted.
    const auto partials = timed_hip_operation("narrow_native_split_k_partials", [&] {
      const auto error = static_cast<hipError_t>(rns8_hip_native_narrow_split_partials(
          static_cast<const int64_t*>(lhs.native), static_cast<const int64_t*>(rhs.native),
          static_cast<int64_t*>(native_product), int(m), int(n), int(k), int(lhs.ld), int(rhs.ld)));
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
    if (partials != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    commit_started = true;
    const auto merged = timed_hip_operation("narrow_native_split_k_encode", [&] {
      const auto error = static_cast<hipError_t>(rns8_hip_native_narrow_split_encode(
          static_cast<const int64_t*>(native_product), static_cast<int8_t*>(output), int(m), int(n),
          int(k), int(prefix)));
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
    return merged == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }
  const auto product = launch_admitted_narrow_product(
      lhs.native, rhs.native, fused ? output : native_product, m, n, k, lhs.ld, rhs.ld, n,
      fused ? prefix : 0, byte_wmma, unsigned_inputs, fused ? &commit_started : nullptr);
  if (product != RNS8_SUCCESS || fused)
    return product;
  return encode_narrow_product(native_product, output, m, n, prefix, unsigned_inputs,
                               commit_started);
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_launch_admitted_narrow_group(int device_id, const NativeNarrowTask* tasks,
                                                    uint32_t count, int64_t m, int64_t n, int64_t k,
                                                    uint32_t prefix, uint32_t method,
                                                    bool unsigned_inputs, bool& commit_started) {
  commit_started = false;
  const auto support = query_narrow_group_support(m, n, k, prefix, method, count, UINT64_MAX);
  if (support.status != RNS8_SUCCESS)
    return support.status;
  if (!tasks)
    return RNS8_INVALID_ARGUMENT;
  if (unsigned_inputs && !native_narrow_byte_wmma(method))
    return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS)
    return selected;
  const bool split = native_narrow_split(method);
  if (!split)
    commit_started = true;
  const auto first = timed_hip_operation(
      split                             ? "narrow_group_split_k_partials"
      : native_narrow_byte_wmma(method) ? "narrow_group_byte_wmma_fused_rns"
                                        : "narrow_group_scalar_fused_rns",
      [&] {
        const auto error = static_cast<hipError_t>(
            rns8_hip_native_narrow_group_stage(tasks, int(count), int(m), int(n), int(k),
                                               int(prefix), method, unsigned_inputs, false));
        return error == hipSuccess ? hipDeviceSynchronize() : error;
      });
  if (first != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  if (!split)
    return RNS8_SUCCESS;
  commit_started = true;
  const auto merged = timed_hip_operation("narrow_group_split_k_encode", [&] {
    const auto error = static_cast<hipError_t>(rns8_hip_native_narrow_group_stage(
        tasks, int(count), int(m), int(n), int(k), int(prefix), method, unsigned_inputs, true));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  return merged == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_narrow_rns_device(int device_id, const void* a, const void* b, void* c,
                                              int64_t m, int64_t n, int64_t k, int64_t lda,
                                              int64_t ldb, uint32_t prefix, bool unsigned_inputs,
                                              bool byte_wmma, uint64_t budget, bool& commit_started,
                                              bool fused) {
  commit_started = false;
  if (!a || !b || !c)
    return RNS8_INVALID_ARGUMENT;
  if (unsigned_inputs && !byte_wmma)
    return RNS8_UNSUPPORTED_BACKEND;
  const auto support =
      query_narrow_rns_support(m, n, k, lda, ldb, prefix, byte_wmma, budget, fused);
  if (support.status != RNS8_SUCCESS)
    return support.status;
  const uint64_t output_bytes = uint64_t(m) * uint64_t(n) * prefix;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, uint64_t(m) * uint64_t(lda) * 8) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, uint64_t(k) * uint64_t(ldb) * 8))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (fused) {
    uint64_t bound = 0;
    return byte_wmma
               ? hip_direct_gemm_native_byte_wmma_checked_device(device_id, a, b, c, m, n, k, lda,
                                                                 ldb, n, unsigned_inputs, &bound,
                                                                 prefix, &commit_started)
               : hip_direct_gemm_i64_narrow_checked_device(
                     device_id, static_cast<const int64_t*>(a), static_cast<const int64_t*>(b),
                     static_cast<int64_t*>(c), m, n, k, lda, ldb, n, &bound, prefix,
                     &commit_started);
  }
  hip_direct_device_buffer product;
  auto status = product.allocate(device_id, static_cast<size_t>(support.payload_bytes));
  if (status != RNS8_SUCCESS)
    return status;
  uint64_t bound = 0;
  product.mark_pending();
  if (byte_wmma)
    status = hip_direct_gemm_native_byte_wmma_checked_device(
        device_id, a, b, product.get(), m, n, k, lda, ldb, n, unsigned_inputs, &bound);
  else
    status = hip_direct_gemm_i64_narrow_checked_device(
        device_id, static_cast<const int64_t*>(a), static_cast<const int64_t*>(b),
        static_cast<int64_t*>(product.get()), m, n, k, lda, ldb, n, &bound);
  if (status != RNS8_SUCCESS) {
    // Admission/range rejection has completed the synchronous scan and never
    // submitted product work. Device/host failures retain uncertain storage.
    if (status != RNS8_BACKEND_FAILURE && status != RNS8_INTERNAL_ERROR)
      product.mark_completed();
    return status;
  }
  // The CRT-capacity proof remains independent of native accumulation safety.
  const auto encoded =
      encode_narrow_product(product.get(), c, m, n, prefix, unsigned_inputs, commit_started);
  if (encoded == RNS8_SUCCESS)
    product.mark_completed();
  return encoded;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
