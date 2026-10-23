#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_strassen_transform(const int8_t *, int8_t *, int, int,
                                           int, int, int, int, int, bool, bool);
extern "C" int rns8_hip_strassen_combine(const int8_t *, int8_t *, int, int,
                                         int, int, int, int, bool);
extern "C" int rns8_hip_strassen_commit(const int8_t *, int8_t *, int, int, int,
                                        int);
#endif
static rns8_status hip_direct_gemm_bilinear_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started, bool rank23) {
  commit_started = false;
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n ||
      ldc < n || !prefix || prefix > 20)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc, prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  if (!nonoverlapping_byte_ranges(c, uint64_t(m) * ldc * prefix, a,
                                  uint64_t(m) * lda * prefix) ||
      !nonoverlapping_byte_ranges(c, uint64_t(m) * ldc * prefix, b,
                                  uint64_t(k) * ldb * prefix))
    return RNS8_INVALID_ARGUMENT;
  const uint64_t blocks = rank23 ? 3 : 2;
  const uint64_t hm = (uint64_t(m) + blocks - 1) / blocks,
                 hn = (uint64_t(n) + blocks - 1) / blocks,
                 hk = (uint64_t(k) + blocks - 1) / blocks;
  const uint64_t ab = hm * hk * prefix, bb = hk * hn * prefix,
                 pb = hm * hn * prefix, cb = uint64_t(m) * n * prefix;
  const uint64_t scratch_bytes = ab + bb + pb + cb;
  if (scratch_bytes > SIZE_MAX || scratch_bytes > max_scratch_bytes)
    return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, scratch_bytes);
  if (status != RNS8_SUCCESS)
    return status;
  auto *ta = static_cast<int8_t *>(scratch.get());
  auto *tb = ta + ab;
  auto *product = tb + bb;
  auto *stage = product + pb;
  status = hip_direct_zero(device_id, stage, cb);
  if (status != RNS8_SUCCESS)
    return status;
  for (int term = 0; term < (rank23 ? 23 : 7); ++term) {
    auto error = timed_hip_operation(
        rank23 ? "rank23_transform_pair" : "strassen_transform_pair", [&]() {
          auto launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
              static_cast<const int8_t *>(a), ta, static_cast<int>(m),
              static_cast<int>(k), static_cast<int>(lda), static_cast<int>(hm),
              static_cast<int>(hk), prefix, term, true, rank23));
          if (launch != hipSuccess)
            return launch;
          launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
              static_cast<const int8_t *>(b), tb, static_cast<int>(k),
              static_cast<int>(n), static_cast<int>(ldb), static_cast<int>(hk),
              static_cast<int>(hn), prefix, term, false, rank23));
          return launch == hipSuccess ? hipDeviceSynchronize() : launch;
        });
    if (error != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    status = hip_direct_gemm_rns_device(device_id, ta, tb, product, hm, hn, hk,
                                        hk, hn, hn, prefix);
    if (status != RNS8_SUCCESS)
      return status;
    error = timed_hip_operation(
        rank23 ? "rank23_recombine" : "strassen_recombine", [&]() {
          const auto launch = static_cast<hipError_t>(rns8_hip_strassen_combine(
              product, stage, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(hm), static_cast<int>(hn), prefix, term,
              rank23));
          return launch == hipSuccess ? hipDeviceSynchronize() : launch;
        });
    if (error != hipSuccess)
      return RNS8_BACKEND_FAILURE;
  }
  commit_started = true;
  const auto error =
      timed_hip_operation(rank23 ? "rank23_commit" : "strassen_commit", [&]() {
        const auto launch = static_cast<hipError_t>(rns8_hip_strassen_commit(
            stage, static_cast<int8_t *>(c), static_cast<int>(m),
            static_cast<int>(n), static_cast<int>(ldc), prefix));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gemm_strassen_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb,
                                         ldc, prefix, max_scratch_bytes,
                                         commit_started, false);
}
rns8_status hip_direct_gemm_rank23_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb,
                                         ldc, prefix, max_scratch_bytes,
                                         commit_started, true);
}
