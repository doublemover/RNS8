#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_split_k_produce(const int8_t*, const int8_t*, uint8_t*, int, int, int, int, int, int, int);
extern "C" int rns8_hip_split_k_merge(const uint8_t*, uint8_t*, int64_t, int, int);
extern "C" int rns8_hip_split_k_store(const uint8_t*, int8_t*, int, int, int, int);
#endif

rns8_status hip_direct_gemm_rns_split_k_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t splits, uint64_t max_scratch_bytes) {
  if (!a || !b || !c || a == c || b == c || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || !prefix || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      !splits || splits > 4096 || splits > static_cast<uint64_t>(k)) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc, prefix)) return RNS8_UNSUPPORTED_BACKEND;
  if ((static_cast<uint64_t>(k) + splits - 1) / splits > 65536) return RNS8_RANGE_ERROR;
  const uint64_t plane_cells = static_cast<uint64_t>(m) * n * prefix;
  const uint64_t merge_slots = (splits + 127) / 128;
  if (plane_cells > SIZE_MAX / (splits + merge_slots)) return RNS8_RANGE_ERROR;
  const uint64_t bytes = plane_cells * (splits + merge_slots);
  if (bytes > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* first = static_cast<uint8_t*>(scratch.get());
  auto* second = first + plane_cells * splits;
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (run("split_k_producer_kernel", [&]() {
        return rns8_hip_split_k_produce(static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), first,
                                      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                      static_cast<int>(lda), static_cast<int>(ldb), prefix, splits);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t fragments = splits;
  while (fragments > 1) {
    if (run("split_k_merge_kernel", [&]() {
          return rns8_hip_split_k_merge(first, second, m * n, prefix, fragments);
        }) != hipSuccess) return RNS8_BACKEND_FAILURE;
    std::swap(first, second);
    fragments = (fragments + 127) / 128;
  }
  return run("split_k_store_kernel", [&]() {
    return rns8_hip_split_k_store(first, static_cast<int8_t*>(c), static_cast<int>(m), static_cast<int>(n),
                                static_cast<int>(ldc), prefix);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
