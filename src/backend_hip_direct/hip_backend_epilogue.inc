#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_gemm_rns_axpby(const int8_t*, const int8_t*, const int8_t*, int8_t*,
                                     int, int, int, int, int, int, int, int, int64_t, int64_t, void*);
#endif

rns8_status hip_direct_gemm_rns_axpby_device(
    int device_id, const void* a, const void* b, const void* d, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldd, int64_t ldc,
    uint32_t prefix, int64_t alpha, int64_t beta) {
  if (!a || !b || !c || !d || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldd < n || ldc < n ||
      !prefix || prefix > RNS8_MAX_SUPPORTED_PREFIX) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc, prefix) ||
      !qualified_i32_matrix_address(m, n, ldd, prefix)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * prefix;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * prefix) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * prefix) ||
      !nonoverlapping_byte_ranges(c, output_bytes, d, static_cast<uint64_t>(m) * ldd * prefix))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("rns_gemm_axpby_fused_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_gemm_rns_axpby(
        static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<const int8_t*>(d),
        static_cast<int8_t*>(c), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldd), static_cast<int>(ldc), prefix, alpha, beta, nullptr));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)alpha; (void)beta;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

// Internal enqueue-only route: caller owns validated storage, range admission,
// device/stream selection, reservations and completion publication.
rns8_status hip_direct_enqueue_axpby_no_publish(
    const rns8_plan* plan, const rns8_matrix* a, const rns8_matrix* b,
    const rns8_matrix* d, rns8_matrix* c, int64_t alpha, int64_t beta, void* stream) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto error = static_cast<hipError_t>(rns8_hip_gemm_rns_axpby(
      static_cast<const int8_t*>(a->hip_residues), static_cast<const int8_t*>(b->hip_residues),
      static_cast<const int8_t*>(d->hip_residues), static_cast<int8_t*>(c->hip_residues),
      static_cast<int>(plan->desc.m), static_cast<int>(plan->desc.n), static_cast<int>(plan->desc.k),
      static_cast<int>(a->desc.cols), static_cast<int>(b->desc.cols), static_cast<int>(d->desc.cols),
      static_cast<int>(c->desc.cols), plan->prefix, alpha, beta, stream));
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)plan; (void)a; (void)b; (void)d; (void)c; (void)alpha; (void)beta; (void)stream;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
