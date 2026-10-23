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
