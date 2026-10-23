#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_split_k_canonicalize(uint8_t*, int64_t, int, int);
#endif

rns8_status hip_direct_gemm_native_wmma_split_k_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t group, bool unsigned_inputs, uint32_t splits,
    uint32_t workers, uint64_t max_scratch_bytes, bool& commit_started) try {
  commit_started = false;
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      !prefix || prefix > 20 || (group != 1 && group != 2 && group != 4) ||
      !splits || splits > 4096 || splits > static_cast<uint64_t>(k) || !workers || workers > 1024)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, ldc, prefix)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * prefix;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * 8)) return RNS8_INVALID_ARGUMENT;
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
  std::vector<NativeWmmaTask> tasks;
  tasks.reserve(splits);
  for (uint32_t fragment = 0; fragment < splits; ++fragment) {
    const int64_t begin = k * fragment / splits, end = k * (fragment + 1) / splits;
    tasks.push_back({static_cast<const uint64_t*>(a) + begin, static_cast<const uint64_t*>(b) + begin * ldb,
                     reinterpret_cast<int8_t*>(first + plane_cells * fragment), m, n, end - begin,
                     lda, ldb, n, prefix, group, static_cast<uint32_t>(unsigned_inputs)});
  }
  bool scratch_may_be_written = false;
  status = hip_direct_gemm_native_wmma_queue(device_id, tasks, workers, max_scratch_bytes - bytes, scratch_may_be_written);
  if (status != RNS8_SUCCESS) return status;
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (run("native_split_k_canonicalize", [&]() {
        return rns8_hip_split_k_canonicalize(first, m * n, prefix, splits);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t fragments = splits;
  while (fragments > 1) {
    if (run("native_split_k_merge", [&]() {
          return rns8_hip_split_k_merge(first, second, m * n, prefix, fragments);
        }) != hipSuccess) return RNS8_BACKEND_FAILURE;
    std::swap(first, second);
    fragments = (fragments + 127) / 128;
  }
  commit_started = true;
  return run("native_split_k_store", [&]() {
    return rns8_hip_split_k_store(first, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(ldc), prefix);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
  return RNS8_INTERNAL_ERROR;
}
