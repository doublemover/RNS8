template <typename NativeT>
int rns8_native_axis_l1_launch(const void* a, const void* b, rns8::detail::NativeAxisSummary* summaries,
                               int m, int n, int k, int lda, int ldb, int panel_k, int panels) {
  const uint32_t tile_rows = (static_cast<uint64_t>(m) + 15) / 16;
  const uint32_t tile_cols = (static_cast<uint64_t>(n) + 15) / 16;
  hipLaunchKernelGGL((rns8_native_axis_l1_kernel<NativeT, true>), dim3(tile_rows, panels), dim3(256), 0, nullptr,
                    static_cast<const NativeT*>(a), m, k, lda, summaries, panel_k, panels);
  const auto first = hipGetLastError();
  if (first != hipSuccess) return static_cast<int>(first);
  hipLaunchKernelGGL((rns8_native_axis_l1_kernel<NativeT, false>), dim3(tile_cols, panels), dim3(256), 0, nullptr,
                    static_cast<const NativeT*>(b), k, n, ldb,
                    summaries + static_cast<uint64_t>(tile_rows) * panels, panel_k, panels);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_axis_l1_bounds(
    const void* a, const void* b, rns8::detail::NativeAxisSummary* summaries,
    int m, int n, int k, int lda, int ldb, bool unsigned_inputs, int panel_k, int panels) {
  return unsigned_inputs ? rns8_native_axis_l1_launch<uint64_t>(a, b, summaries, m, n, k, lda, ldb, panel_k, panels)
                         : rns8_native_axis_l1_launch<int64_t>(a, b, summaries, m, n, k, lda, ldb, panel_k, panels);
}
extern "C" int rns8_hip_native_l1_tile_bounds(
    const rns8::detail::NativeAxisSummary* summaries, rns8::detail::NativeTileBound* bounds, uint32_t* counts,
    int tile_rows, int tile_cols, int k, int capacity, bool signed_result, int panel_k, int panels) {
  const int tile_count = tile_rows * tile_cols;
  hipLaunchKernelGGL(rns8_native_l1_tile_bounds_kernel, dim3((static_cast<uint64_t>(tile_count) + 255) / 256),
                    dim3(256), 0, nullptr, summaries, summaries + static_cast<uint64_t>(tile_rows) * panels, bounds, counts,
                    tile_count, tile_cols, k, capacity, signed_result, panel_k, panels);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_axis_bounds(
    const void* a, const void* b, uint64_t* maxima, int m, int n, int k, int lda, int ldb, bool unsigned_inputs) {
  const uint32_t tile_rows = (static_cast<uint64_t>(m) + 15) / 16;
  const uint32_t tile_cols = (static_cast<uint64_t>(n) + 15) / 16;
  if (unsigned_inputs) {
    hipLaunchKernelGGL((rns8_native_axis_max_kernel<uint64_t, true>), dim3(tile_rows), dim3(256), 0, nullptr,
        static_cast<const uint64_t*>(a), m, k, lda, maxima, true);
    const auto first = hipGetLastError();
    if (first != hipSuccess) return static_cast<int>(first);
    hipLaunchKernelGGL((rns8_native_axis_max_kernel<uint64_t, false>), dim3(tile_cols), dim3(256), 0, nullptr,
        static_cast<const uint64_t*>(b), k, n, ldb, maxima + tile_rows, true);
  } else {
    hipLaunchKernelGGL((rns8_native_axis_max_kernel<int64_t, true>), dim3(tile_rows), dim3(256), 0, nullptr,
        static_cast<const int64_t*>(a), m, k, lda, maxima, false);
    const auto first = hipGetLastError();
    if (first != hipSuccess) return static_cast<int>(first);
    hipLaunchKernelGGL((rns8_native_axis_max_kernel<int64_t, false>), dim3(tile_cols), dim3(256), 0, nullptr,
        static_cast<const int64_t*>(b), k, n, ldb, maxima + tile_rows, false);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_tile_bounds(
    const uint64_t* maxima, rns8::detail::NativeTileBound* bounds, uint32_t* counts,
    int tile_rows, int tile_cols, int k, int capacity, bool signed_result) {
  const int tile_count = tile_rows * tile_cols;
  hipLaunchKernelGGL(rns8_native_tile_bounds_kernel, dim3((static_cast<uint64_t>(tile_count) + 255) / 256),
      dim3(256), 0, nullptr, maxima, maxima + tile_rows, bounds, counts, tile_count, tile_cols, k, capacity, signed_result);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_bound_compact(
    const rns8::detail::NativeTileBound* bounds, const uint32_t* counts,
    uint32_t* cursors, uint32_t* offsets, uint32_t* indices, int tile_count) {
  hipLaunchKernelGGL(rns8_native_bound_offsets_kernel, dim3(1), dim3(1), 0, nullptr, counts, offsets, cursors);
  const auto first = hipGetLastError();
  if (first != hipSuccess) return static_cast<int>(first);
  hipLaunchKernelGGL(rns8_native_bound_scatter_kernel, dim3((static_cast<uint64_t>(tile_count) + 255) / 256),
      dim3(256), 0, nullptr, bounds, cursors, indices, tile_count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_native_global_bound(const rns8::detail::NativeTileBound* bounds,
    uint64_t count,rns8::detail::NativeTileBound* output,uint32_t capacity,bool signed_result) {
  hipLaunchKernelGGL(rns8_native_global_bound_kernel,dim3(1),dim3(256),0,nullptr,bounds,count,output,capacity,signed_result);
  return int(hipGetLastError());
}
