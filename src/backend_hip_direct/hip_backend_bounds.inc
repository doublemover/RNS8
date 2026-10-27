#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_single_axis(const void*, int, int, int, bool, bool, void*, bool, int, int);
extern "C" int rns8_hip_native_axis_bounds(const void*, const void*, uint64_t*, int, int, int, int, int, bool);
extern "C" int rns8_hip_native_axis_l1_bounds(const void*, const void*, NativeAxisSummary*, int, int, int, int, int, bool, int, int);
extern "C" int rns8_hip_native_l1_tile_bounds(const NativeAxisSummary*, NativeTileBound*, uint32_t*, int, int, int, int, bool, int, int);
extern "C" int rns8_hip_native_tile_bounds(
    const uint64_t*, NativeTileBound*, uint32_t*, int, int, int, int, bool);
extern "C" int rns8_hip_native_bound_compact(
    const NativeTileBound*, const uint32_t*, uint32_t*, uint32_t*, uint32_t*, int);
#endif

rns8_status hip_direct_scan_native_summary(int device_id, const void* source,
    int64_t rows, int64_t cols, int64_t ld, bool unsigned_inputs, bool row_tiles,
    NativeBoundMethod method, uint32_t panel_k, void* summaries, uint64_t summary_bytes) {
  if (!source || !summaries || rows <= 0 || cols <= 0 || ld < cols ||
      (method != NativeBoundMethod::tile_max && method != NativeBoundMethod::axis_l1 && method != NativeBoundMethod::panel_l1) ||
      (method == NativeBoundMethod::panel_l1 && (!panel_k || panel_k > 65536))) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(rows,cols,ld,1,8)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t k = uint64_t(row_tiles ? cols : rows);
  const uint64_t extent = method == NativeBoundMethod::panel_l1 ? panel_k : k;
  const uint64_t panels = (k + extent - 1) / extent;
  if (panels > 65535) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes = ((uint64_t(row_tiles ? rows : cols)+15)/16)*panels*
      (method == NativeBoundMethod::tile_max ? sizeof(uint64_t) : sizeof(NativeAxisSummary));
  if (bytes > summary_bytes) return RNS8_WORKSPACE_TOO_SMALL;
  if (!nonoverlapping_byte_ranges(source,uint64_t(rows)*ld*8,summaries,bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  return timed_hip_operation(row_tiles ? "native_bound_cached_a_scan" : "native_bound_cached_b_scan", [&]() {
    const auto code = static_cast<hipError_t>(rns8_hip_native_single_axis(source,int(rows),int(cols),int(ld),
        unsigned_inputs,row_tiles,summaries,method != NativeBoundMethod::tile_max,int(extent),int(panels)));
    return code == hipSuccess ? hipDeviceSynchronize() : code;
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs; return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_discover_native_tile_bounds(
    int device_id, const void* a, const void* b, int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, bool unsigned_inputs, uint32_t capacity,
    NativeTileBound* device_bounds, uint32_t* device_indices, uint32_t* device_offsets,
    uint64_t max_scratch_bytes, NativeBoundMethod method, uint32_t panel_k, const void* retained_summaries) {
  if (!a || !b || !device_bounds || !device_indices || !device_offsets ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || !capacity || capacity > 20 ||
      (method != NativeBoundMethod::tile_max && method != NativeBoundMethod::axis_l1 && method != NativeBoundMethod::panel_l1) ||
      (method == NativeBoundMethod::panel_l1 && (!panel_k || panel_k > 65536)))
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
  const bool panel = method == NativeBoundMethod::panel_l1;
  const bool l1 = method != NativeBoundMethod::tile_max;
  const uint64_t panel_extent = panel ? panel_k : static_cast<uint64_t>(k);
  const uint64_t panels = (static_cast<uint64_t>(k) + panel_extent - 1) / panel_extent;
  if (panels > 65535) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t maxima_bytes = (rows + cols) * panels * (l1 ? sizeof(NativeAxisSummary) : sizeof(uint64_t));
  const uint64_t scratch_bytes = maxima_bytes + (22 + 21) * sizeof(uint32_t);
  if (scratch_bytes > SIZE_MAX) return RNS8_RANGE_ERROR;
  if (scratch_bytes > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
  if (retained_summaries) {
    for (int i=0; i<3; ++i)
      if (!nonoverlapping_byte_ranges(retained_summaries,maxima_bytes,outputs[i],sizes[i])) return RNS8_INVALID_ARGUMENT;
  }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, static_cast<std::size_t>(scratch_bytes - (retained_summaries ? maxima_bytes : 0)));
  if (status != RNS8_SUCCESS) return status;
  const void* axis = retained_summaries ? retained_summaries : scratch.get();
  const auto* maxima = static_cast<const uint64_t*>(axis);
  const auto* summaries = static_cast<const NativeAxisSummary*>(axis);
  auto* counts = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(scratch.get()) + (retained_summaries ? 0 : maxima_bytes));
  auto* cursors = counts + 22;
  if (hipMemset(counts, 0, 22 * sizeof(uint32_t)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (!retained_summaries && run(panel ? "native_bound_panel_l1_scan" : l1 ? "native_bound_axis_l1_scan" : "native_bound_axis_scan", [&]() {
        if (l1) return rns8_hip_native_axis_l1_bounds(a, b, static_cast<NativeAxisSummary*>(scratch.get()), static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), static_cast<int>(lda), static_cast<int>(ldb), unsigned_inputs,
            static_cast<int>(panel_extent), static_cast<int>(panels));
        return rns8_hip_native_axis_bounds(a, b, static_cast<uint64_t*>(scratch.get()), static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), static_cast<int>(lda), static_cast<int>(ldb), unsigned_inputs);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  if (run("native_bound_prefix_selection", [&]() {
        if (l1) return rns8_hip_native_l1_tile_bounds(summaries, device_bounds, counts, static_cast<int>(rows),
            static_cast<int>(cols), static_cast<int>(k), static_cast<int>(capacity), !unsigned_inputs,
            static_cast<int>(panel_extent), static_cast<int>(panels));
        return rns8_hip_native_tile_bounds(maxima, device_bounds, counts, static_cast<int>(rows),
            static_cast<int>(cols), static_cast<int>(k), static_cast<int>(capacity), !unsigned_inputs);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t rejected = 0;
  status = hip_direct_copy_device_to_host(device_id, &rejected, counts + 21, sizeof(rejected));
  if (status != RNS8_SUCCESS) return status;
  if (rejected) return RNS8_RANGE_ERROR;
  return run("native_bound_bucket_compaction", [&]() {
    return rns8_hip_native_bound_compact(device_bounds, counts, cursors, device_offsets, device_indices, static_cast<int>(tiles));
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
