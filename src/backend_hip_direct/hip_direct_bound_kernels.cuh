template <typename NativeT, bool RowTiles>
__global__ void rns8_native_axis_max_kernel(
    const NativeT* source, int rows, int cols, int ld, uint64_t* maxima, bool unsigned_inputs) {
  __shared__ uint64_t partial[256];
  const int lane = threadIdx.x;
  const int64_t origin = static_cast<int64_t>(blockIdx.x) * 16;
  const int64_t axis = RowTiles ? rows : cols;
  const int64_t extent = axis - origin < 16 ? axis - origin : 16;
  const int64_t elements = extent * (RowTiles ? cols : rows);
  uint64_t maximum = 0;
  for (int64_t i = lane; i < elements; i += 256) {
    const int64_t row = RowTiles ? origin + i / cols : i / extent;
    const int64_t col = RowTiles ? i % cols : origin + i % extent;
    const NativeT value = source[row * ld + col];
    const uint64_t bits = static_cast<uint64_t>(value);
    const uint64_t magnitude = !unsigned_inputs && value < 0 ? uint64_t{0} - bits : bits;
    if (magnitude > maximum) maximum = magnitude;
  }
  partial[lane] = maximum;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (lane < step && partial[lane + step] > partial[lane]) partial[lane] = partial[lane + step];
    __syncthreads();
  }
  if (!lane) maxima[blockIdx.x] = partial[0];
}

__global__ void rns8_native_tile_bounds_kernel(
    const uint64_t* a, const uint64_t* b, rns8::detail::NativeTileBound* bounds,
    uint32_t* counts, int tile_count, int tile_cols, int k, int capacity, bool signed_result) {
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile >= tile_count) return;
  auto bound = rns8::detail::native_tile_max_bound(a[tile / tile_cols], b[tile % tile_cols], k);
  rns8::detail::select_native_bound_prefix(bound, signed_result, capacity, rns8_direct_crt_table.products);
  bounds[tile] = bound;
  atomicAdd(counts + bound.required_prefix, 1u);
}
__global__ void rns8_native_bound_offsets_kernel(const uint32_t* counts, uint32_t* offsets, uint32_t* cursors) {
  uint32_t sum = 0;
  for (int p = 0; p <= 20; ++p) {
    offsets[p] = sum;
    cursors[p] = sum;
    sum += counts[p];
  }
  offsets[21] = sum;
}
__global__ void rns8_native_bound_scatter_kernel(
    const rns8::detail::NativeTileBound* bounds, uint32_t* cursors, uint32_t* indices, int tile_count) {
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile >= tile_count || bounds[tile].required_prefix > 20) return;
  const uint32_t slot = atomicAdd(cursors + bounds[tile].required_prefix, 1u);
  indices[slot] = static_cast<uint32_t>(tile);
}
