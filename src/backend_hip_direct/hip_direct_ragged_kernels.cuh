__global__ void rns8_materialize_ragged_kernel(
    const int8_t* source, const rns8::detail::RaggedResidueTile* tiles,
    int tile_rows, int tile_cols, int grid_cols, bool signed_value,
    int8_t* destination, rns8::detail::ResidueLayout target, int* status) {
  const uint64_t cell = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= target.rows * target.cols) return;
  const uint64_t row = cell / target.cols, col = cell % target.cols;
  const auto& tile = tiles[(row / tile_rows) * grid_cols + col / tile_cols];
  const int local = (row % tile_rows) * tile_cols + col % tile_cols;
  const int area = tile_rows * tile_cols;
  const auto* input = source + tile.byte_offset;
  for (uint32_t p = 0; p < tile.prefix; ++p) {
    const int value = input[p * area + local], modulus = rns8_default_moduli_device[p];
    if (value < -(modulus / 2) || value > (modulus - 1) / 2) {
      atomicMax(status, 1); return;
    }
  }
  rns8_u192_device value{}, product{};
  rns8_reconstruct_canonical_wide_device(input, local, area, tile.prefix, &value, &product);
  const bool negative = signed_value && rns8_u192_centered_is_negative_device(value, product);
  const auto magnitude = negative ? rns8_u192_sub_device(product, value) : value;
  const rns8_u192_device bound{tile.magnitude_bound[0],tile.magnitude_bound[1],tile.magnitude_bound[2]};
  if (rns8_u192_compare_device(magnitude, bound) > 0) { atomicMax(status, 2); return; }
  for (uint32_t p = 0; p < target.prefix; ++p) {
    int8_t stored = 0;
    if (p < tile.prefix) stored = input[p * area + local];
    else {
      const uint32_t modulus = rns8_default_moduli_device[p];
      uint32_t residue = rns8_u192_mod_u32_device(magnitude, modulus);
      if (negative && residue) residue = modulus - residue;
      const int centered = residue >= (modulus + 1) / 2 ? static_cast<int>(residue) - static_cast<int>(modulus)
                                                       : static_cast<int>(residue);
      stored = static_cast<int8_t>(centered);
    }
    destination[target.offset(p,row,col)] = stored;
  }
}

// One block owns each directory tile. Only its maximum magnitude leaves the GPU.
__global__ void rns8_discover_ragged_bounds_kernel(const int8_t* source, int rows, int cols,
    int prefix, int tile_rows, int tile_cols, int grid_cols, bool signed_value,
    rns8_u192_device source_bound, rns8::detail::RaggedResidueTile* directory, int* status) {
  __shared__ rns8_u192_device maxima[128];
  const int tr = blockIdx.x / grid_cols, tc = blockIdx.x % grid_cols;
  const int area = tile_rows * tile_cols, cells = rows * cols;
  rns8_u192_device maximum{};
  for (int local = threadIdx.x; local < area; local += blockDim.x) {
    const int row = tr * tile_rows + local / tile_cols;
    const int col = tc * tile_cols + local % tile_cols;
    if (row >= rows || col >= cols) continue;
    const int cell = row * cols + col;
    bool valid = true;
    for (int p = 0; p < prefix; ++p) {
      const int value = source[p * cells + cell], modulus = rns8_default_moduli_device[p];
      if (value < -(modulus / 2) || value > (modulus - 1) / 2) valid = false;
    }
    if (!valid) { atomicMax(status, 1); continue; }
    rns8_u192_device value{}, product{};
    rns8_reconstruct_canonical_wide_device(source, cell, cells, prefix, &value, &product);
    const auto magnitude = signed_value && rns8_u192_centered_is_negative_device(value, product)
        ? rns8_u192_sub_device(product, value) : value;
    if (rns8_u192_compare_device(magnitude, source_bound) > 0) atomicMax(status, 2);
    if (rns8_u192_compare_device(magnitude, maximum) > 0) maximum = magnitude;
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride &&
        rns8_u192_compare_device(maxima[threadIdx.x + stride], maxima[threadIdx.x]) > 0)
      maxima[threadIdx.x] = maxima[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) {
    auto& tile = directory[blockIdx.x];
    tile.byte_offset = 0; tile.prefix = 0; tile.reserved = 0;
    tile.magnitude_bound[0] = maxima[0].limb0;
    tile.magnitude_bound[1] = maxima[0].limb1;
    tile.magnitude_bound[2] = maxima[0].limb2;
  }
}

__global__ void rns8_compact_ragged_kernel(const int8_t* source, int rows, int cols,
    const rns8::detail::RaggedResidueTile* directory, int tile_rows, int tile_cols,
    int grid_cols, int8_t* destination) {
  const auto& tile = directory[blockIdx.x];
  const int tr = blockIdx.x / grid_cols, tc = blockIdx.x % grid_cols;
  const int area = tile_rows * tile_cols;
  for (uint32_t index = threadIdx.x; index < tile.prefix * area; index += blockDim.x) {
    const int local = index % area, row = tr * tile_rows + local / tile_cols;
    const int col = tc * tile_cols + local % tile_cols;
    destination[tile.byte_offset + index] = row < rows && col < cols
        ? source[(index / area) * rows * cols + row * cols + col] : int8_t{0};
  }
}
