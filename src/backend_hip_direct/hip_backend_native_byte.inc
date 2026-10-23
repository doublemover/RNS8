#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_byte_domain(const void *, int, int, int,
                                           uint64_t *, uint64_t *, bool);
extern "C" int rns8_hip_native_byte_wmma(const void *, const void *, void *,
                                         int, int, int, int, int, int, bool);
#endif
rns8_status hip_direct_gemm_native_byte_wmma_checked_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, bool unsigned_inputs,
    uint64_t *out_abs_sum_bound) {
  if (!a || !b || !c || !out_abs_sum_bound || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, ldc, 1, 8))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes = static_cast<uint64_t>(m) * ldc * 8;
  if (!nonoverlapping_byte_ranges(c, bytes, a,
                                  static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, bytes, b,
                                  static_cast<uint64_t>(k) * ldb * 8))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, 4 * sizeof(uint64_t));
  if (status != RNS8_SUCCESS)
    return status;
  if (hipMemset(scratch.get(), 0, 4 * sizeof(uint64_t)) != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  auto *words = static_cast<uint64_t *>(scratch.get());
  const auto scan = timed_hip_operation("native_byte_wmma_admission", [&]() {
    auto error = static_cast<hipError_t>(
        rns8_hip_native_wmma_capability(reinterpret_cast<int *>(words + 3)));
    if (error == hipSuccess)
      error = static_cast<hipError_t>(rns8_hip_native_byte_domain(
          a, static_cast<int>(m), static_cast<int>(k), static_cast<int>(lda),
          words, words + 2, unsigned_inputs));
    if (error == hipSuccess)
      error = static_cast<hipError_t>(rns8_hip_native_byte_domain(
          b, static_cast<int>(k), static_cast<int>(n), static_cast<int>(ldb),
          words + 1, words + 2, unsigned_inputs));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (scan != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  uint64_t proof[4]{};
  status =
      hip_direct_copy_device_to_host(device_id, proof, words, sizeof(proof));
  if (status != RNS8_SUCCESS)
    return status;
  if (!proof[3])
    return RNS8_UNSUPPORTED_BACKEND;
  uint64_t bound = 0;
  if (proof[2] ||
      !native_partial_sum_bound(proof[0], proof[1], static_cast<uint64_t>(k),
                                INT32_MAX, bound))
    return RNS8_RANGE_ERROR;
  const auto error = timed_hip_operation("native_byte_wmma_gemm", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_native_byte_wmma(
        a, b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc),
        unsigned_inputs));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  *out_abs_sum_bound = bound;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
