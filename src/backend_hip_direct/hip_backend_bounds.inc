#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_axis_bounds(const void*, const void*, uint64_t*, int, int, int, int, int, bool);
extern "C" int rns8_hip_native_tile_bounds(
    const uint64_t*, NativeTileBound*, uint32_t*, int, int, int, int, bool);
extern "C" int rns8_hip_native_bound_compact(
    const NativeTileBound*, const uint32_t*, uint32_t*, uint32_t*, uint32_t*, int);
#endif

rns8_status hip_direct_discover_native_tile_bounds(
    int device_id, const void* a, const void* b, int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, bool unsigned_inputs, uint32_t capacity,
    NativeTileBound* device_bounds, uint32_t* device_indices, uint32_t* device_offsets,
    uint64_t max_scratch_bytes) {
  if (!a || !b || !device_bounds || !device_indices || !device_offsets ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || !capacity || capacity > 20)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t rows = (static_cast<uint64_t>(m) + 15) / 16;
  const uint64_t cols = (static_cast<uint64_t>(n) + 15) / 16;
  if (rows > INT32_MAX / cols) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t tiles = rows * cols;
  const void* outputs[] = {device_bounds, device_indices, device_offsets};
  const uint64_t sizes[] = {tiles * sizeof(NativeTileBound), tiles * sizeof(uint32_t), 22 * sizeof(uint32_t)};
  for (int i = 0; i < 3; ++i) {
    if (!nonoverlapping_byte_ranges(outputs[i], sizes[i], a, static_cast<uint64_t>(m) * lda * 8) ||
        !nonoverlapping_byte_ranges(outputs[i], sizes[i], b, static_cast<uint64_t>(k) * ldb * 8))
      return RNS8_INVALID_ARGUMENT;
    for (int j = 0; j < i; ++j)
      if (!nonoverlapping_byte_ranges(outputs[i], sizes[i], outputs[j], sizes[j])) return RNS8_INVALID_ARGUMENT;
  }
  const uint64_t maxima_bytes = (rows + cols) * sizeof(uint64_t);
  const uint64_t scratch_bytes = maxima_bytes + (22 + 21) * sizeof(uint32_t);
  if (scratch_bytes > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, static_cast<std::size_t>(scratch_bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* maxima = static_cast<uint64_t*>(scratch.get());
  auto* counts = reinterpret_cast<uint32_t*>(maxima + rows + cols);
  auto* cursors = counts + 22;
  if (hipMemset(counts, 0, 22 * sizeof(uint32_t)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (run("native_bound_axis_scan", [&]() {
        return rns8_hip_native_axis_bounds(a, b, maxima, m, n, k, lda, ldb, unsigned_inputs);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  if (run("native_bound_prefix_selection", [&]() {
        return rns8_hip_native_tile_bounds(maxima, device_bounds, counts, rows, cols, k, capacity, !unsigned_inputs);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t rejected = 0;
  status = hip_direct_copy_device_to_host(device_id, &rejected, counts + 21, sizeof(rejected));
  if (status != RNS8_SUCCESS) return status;
  if (rejected) return RNS8_RANGE_ERROR;
  return run("native_bound_bucket_compaction", [&]() {
    return rns8_hip_native_bound_compact(device_bounds, counts, cursors, device_offsets, device_indices, tiles);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
