template <typename NativeT, bool RowTiles>
__global__ void rns8_native_axis_l1_kernel(
    const NativeT* source, int rows, int cols, int ld, rns8::detail::NativeAxisSummary* summaries, int panel_k, int panels) {
  __shared__ uint64_t maxima[256], low[256], high[256];
  const int lane = threadIdx.x;
  const int axis_lane = RowTiles ? lane / 16 : lane % 16;
  const int k_lane = RowTiles ? lane % 16 : lane / 16;
  const int slot = axis_lane * 16 + k_lane;
  const int64_t axis = static_cast<int64_t>(blockIdx.x) * 16 + axis_lane;
  const int axis_size = RowTiles ? rows : cols;
  const int k_size = RowTiles ? cols : rows;
  const int64_t begin = int64_t{blockIdx.y} * panel_k;
  const int64_t end = begin + panel_k < k_size ? begin + panel_k : k_size;
  uint64_t maximum = 0, sum_low = 0, sum_high = 0;
  if (axis < axis_size)
    for (int64_t q = begin + k_lane; q < end; q += 16) {
      const NativeT value = source[RowTiles ? axis * ld + q : q * ld + axis];
      const uint64_t bits = static_cast<uint64_t>(value);
      const uint64_t magnitude = value < 0 ? uint64_t{0} - bits : bits;
      maximum = magnitude > maximum ? magnitude : maximum;
      const uint64_t old = sum_low;
      sum_low += magnitude;
      sum_high += sum_low < old;
    }
  maxima[slot] = maximum; low[slot] = sum_low; high[slot] = sum_high;
  __syncthreads();
  for (int step = 8; step; step /= 2) {
    if (k_lane < step) {
      const uint64_t old = low[slot];
      low[slot] += low[slot + step];
      high[slot] += high[slot + step] + (low[slot] < old);
      maxima[slot] = maxima[slot] > maxima[slot + step] ? maxima[slot] : maxima[slot + step];
    }
    __syncthreads();
  }
  if (!lane) {
    rns8::detail::NativeAxisSummary summary{};
    for (int row = 0; row < 16; ++row) {
      const int index = row * 16;
      summary.maximum = summary.maximum > maxima[index] ? summary.maximum : maxima[index];
      if (high[index] > summary.l1_high || (high[index] == summary.l1_high && low[index] > summary.l1_low)) {
        summary.l1_low = low[index]; summary.l1_high = high[index];
      }
    }
    summaries[static_cast<uint64_t>(blockIdx.x) * panels + blockIdx.y] = summary;
  }
}

__global__ void rns8_native_l1_tile_bounds_kernel(
    const rns8::detail::NativeAxisSummary* a, const rns8::detail::NativeAxisSummary* b,
    rns8::detail::NativeTileBound* bounds, uint32_t* counts,
    int tile_count, int tile_cols, int k, int capacity, bool signed_result, int panel_k, int panels) {
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile >= tile_count) return;
  rns8::detail::NativeTileBound bound{};
  for (int panel = 0; panel < panels; ++panel) {
    const int64_t remaining = k - int64_t{panel} * panel_k;
    const auto terms = static_cast<uint32_t>(remaining < panel_k ? remaining : panel_k);
    const auto part = rns8::detail::native_axis_l1_bound(
        a[(tile / tile_cols) * panels + panel], b[(tile % tile_cols) * panels + panel], terms);
    rns8::detail::add_native_panel_bound(bound, part);
  }
  rns8::detail::select_native_bound_prefix(bound, signed_result, capacity, rns8_direct_crt_table.products);
  bounds[tile] = bound;
  atomicAdd(counts + bound.required_prefix, 1u);
}

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

// Reduce the SAME tile certificates consumed by adaptive execution.
__global__ void rns8_native_global_bound_kernel(const rns8::detail::NativeTileBound* bounds,
    uint64_t count,rns8::detail::NativeTileBound* output,uint32_t capacity,bool signed_result) {
  using namespace rns8::detail;
  __shared__ uint64_t magnitudes[256][3];
  const uint32_t lane=threadIdx.x;NativeTileBound maximum{};
  for(uint64_t i=lane;i<count;i+=256)if(native_bound_less(maximum,bounds[i]))maximum=bounds[i];
  for(int limb=0;limb<3;++limb)magnitudes[lane][limb]=maximum.magnitude[limb];
  __syncthreads();
  for(uint32_t step=128;step;step/=2) {
    if(lane<step) {
      NativeTileBound other{};
      for(int limb=0;limb<3;++limb)other.magnitude[limb]=magnitudes[lane+step][limb];
      if(native_bound_less(maximum,other))maximum=other;
      for(int limb=0;limb<3;++limb)magnitudes[lane][limb]=maximum.magnitude[limb];
    }
    __syncthreads();
  }
  if(!lane) {
    select_native_bound_prefix(maximum,signed_result,capacity,rns8_direct_crt_table.products);
    *output=maximum;
  }
}
