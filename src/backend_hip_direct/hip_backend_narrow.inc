#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_i64_abs_max(const int64_t*, int, int, int, void*);
extern "C" int rns8_hip_native_i64_narrow_gemm(const int64_t*, const int64_t*, int64_t*, int, int, int, int, int, int);
#endif

rns8_status hip_direct_gemm_i64_narrow_checked_device(
    int device_id, const int64_t* a, const int64_t* b, int64_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint64_t* out_abs_sum_bound) {
  if (!a || !b || !c || !out_abs_sum_bound || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes = static_cast<uint64_t>(m) * ldc * sizeof(int64_t);
  if (!nonoverlapping_byte_ranges(c, bytes, a, static_cast<uint64_t>(m) * lda * sizeof(int64_t)) ||
      !nonoverlapping_byte_ranges(c, bytes, b, static_cast<uint64_t>(k) * ldb * sizeof(int64_t)))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer maxima;
  status = maxima.allocate(device_id, 2 * sizeof(uint64_t));
  if (status != RNS8_SUCCESS) return status;
  if (hipMemset(maxima.get(), 0, 2 * sizeof(uint64_t)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  auto* device_maxima = static_cast<uint64_t*>(maxima.get());
  const auto scan = timed_hip_operation("narrow_native_bound_scan", [&]() {
    auto error = static_cast<hipError_t>(rns8_hip_native_i64_abs_max(a, static_cast<int>(m), static_cast<int>(k),
                                                                  static_cast<int>(lda), device_maxima));
    if (error == hipSuccess)
      error = static_cast<hipError_t>(rns8_hip_native_i64_abs_max(b, static_cast<int>(k), static_cast<int>(n),
                                                               static_cast<int>(ldb), device_maxima + 1));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (scan != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint64_t bounds[2]{};
  status = hip_direct_copy_device_to_host(device_id, bounds, device_maxima, sizeof(bounds));
  if (status != RNS8_SUCCESS) return status;
  constexpr uint64_t limit = INT64_MAX;
  uint64_t sum_bound = 0;
  if (!native_partial_sum_bound(bounds[0], bounds[1], static_cast<uint64_t>(k), limit, sum_bound))
    return RNS8_RANGE_ERROR;
  const auto error = timed_hip_operation("narrow_native_i64_gemm_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_native_i64_narrow_gemm(
        a, b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc)));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  *out_abs_sum_bound = sum_bound;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_narrow_rns_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    uint32_t prefix, bool unsigned_inputs, bool byte_wmma,
    uint64_t budget, bool& commit_started) {
  commit_started=false;
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  if (unsigned_inputs && !byte_wmma) return RNS8_UNSUPPORTED_BACKEND;
  const auto support=query_narrow_rns_support(m,n,k,lda,ldb,prefix,byte_wmma,budget);
  if (support.status!=RNS8_SUCCESS) return support.status;
  const uint64_t output_bytes=uint64_t(m)*uint64_t(n)*prefix;
  if (!nonoverlapping_byte_ranges(c,output_bytes,a,uint64_t(m)*uint64_t(lda)*8) ||
      !nonoverlapping_byte_ranges(c,output_bytes,b,uint64_t(k)*uint64_t(ldb)*8))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_direct_device_buffer product;
  auto status=product.allocate(device_id,static_cast<size_t>(support.payload_bytes));
  if (status!=RNS8_SUCCESS) return status;
  uint64_t bound=0;
  if (byte_wmma)
    status=hip_direct_gemm_native_byte_wmma_checked_device(device_id,a,b,product.get(),
        m,n,k,lda,ldb,n,unsigned_inputs,&bound);
  else
    status=hip_direct_gemm_i64_narrow_checked_device(device_id,
        static_cast<const int64_t*>(a),static_cast<const int64_t*>(b),
        static_cast<int64_t*>(product.get()),m,n,k,lda,ldb,n,&bound);
  if (status!=RNS8_SUCCESS) return status;
  // The owner's independent complete-expression proof admits the CRT capacity.
  // The device scan above separately admits every native accumulation partial.
  commit_started=true;
  const auto error=timed_hip_operation("narrow_native_encode_rns",[&]() {
    const int code=unsigned_inputs
        ? rns8_hip_direct_pack_u64_device(static_cast<const uint64_t*>(product.get()),
            static_cast<int8_t*>(c),int(m),int(n),int(n),int(prefix))
        : rns8_hip_direct_pack_i64_device(static_cast<const int64_t*>(product.get()),
            static_cast<int8_t*>(c),int(m),int(n),int(n),int(prefix));
    return code==int(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return error==hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
