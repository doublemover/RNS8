__global__ void rns8_export_u8_modulus_kernel(
    const int8_t* residues,
    uint8_t* dst,
    int rows,
    int cols,
    int ld,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  dst[static_cast<int64_t>(row) * ld + col] =
      static_cast<uint8_t>(rns8_canonical_from_centered_device(residues[idx], modulus));
}

template <int Modulus>
__global__ void rns8_export_u8_fixed_modulus_kernel(
    const int8_t* residues,
    uint8_t* dst,
    int rows,
    int cols,
    int ld) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  dst[static_cast<int64_t>(row) * ld + col] =
      static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx]));
}

__device__ void rns8_export_i64_cell_device(
    int64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    uint64_t bound,
    int* status) {
  if (rns8_u192_centered_is_negative_device(x, product)) {
    const rns8_u192_device magnitude = rns8_u192_sub_device(product, x);
    constexpr uint64_t int64_min_magnitude = 0x8000000000000000ULL;
    if (rns8_u192_gt_u64_device(magnitude, bound) || rns8_u192_gt_u64_device(magnitude, int64_min_magnitude)) {
      atomicCAS(status, 0, 5);
      return;
    }
    if (magnitude.limb0 == int64_min_magnitude) {
      dst[cell] = (-9223372036854775807LL - 1LL);
      return;
    }
    dst[cell] = -static_cast<int64_t>(magnitude.limb0);
    return;
  }

  if (rns8_u192_gt_u64_device(x, bound) || rns8_u192_gt_u64_device(x, 0x7fffffffffffffffULL)) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = static_cast<int64_t>(x.limb0);
}

__device__ void rns8_export_u64_cell_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    uint64_t bound,
    int* status) {
  if (rns8_u192_gt_u64_device(x, bound)) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = x.limb0;
}

__device__ void rns8_export_i64_cell_device(
    const int8_t* residues,
    int64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_i64_cell_device(dst, cell, x, product, bound, status);
}

template <int Prefix>
__device__ void rns8_export_i64_fixed_prefix_cell_device(
    const int8_t* residues,
    int64_t* dst,
    int cell,
    int elements,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_i64_cell_device(dst, cell, x, product, bound, status);
}

__device__ void rns8_export_u64_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_u64_cell_device(dst, cell, x, bound, status);
}

template <int Prefix>
__device__ void rns8_export_u64_fixed_prefix_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_u64_cell_device(dst, cell, x, bound, status);
}

__device__ void rns8_export_i64_bounded_prefix_cell_device(
    const int8_t* residues,
    int64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  switch (prefix) {
    case 1:
      rns8_export_i64_fixed_prefix_cell_device<1>(residues, dst, cell, elements, bound, status);
      return;
    case 2:
      rns8_export_i64_fixed_prefix_cell_device<2>(residues, dst, cell, elements, bound, status);
      return;
    case 3:
      rns8_export_i64_fixed_prefix_cell_device<3>(residues, dst, cell, elements, bound, status);
      return;
    case 4:
      rns8_export_i64_fixed_prefix_cell_device<4>(residues, dst, cell, elements, bound, status);
      return;
    case 5:
      rns8_export_i64_fixed_prefix_cell_device<5>(residues, dst, cell, elements, bound, status);
      return;
    case 6:
      rns8_export_i64_fixed_prefix_cell_device<6>(residues, dst, cell, elements, bound, status);
      return;
    case 7:
      rns8_export_i64_fixed_prefix_cell_device<7>(residues, dst, cell, elements, bound, status);
      return;
    case 8:
      rns8_export_i64_fixed_prefix_cell_device<8>(residues, dst, cell, elements, bound, status);
      return;
    case kRns8DefaultBoundedPrefix:
      rns8_export_i64_fixed_prefix_cell_device<kRns8DefaultBoundedPrefix>(
          residues, dst, cell, elements, bound, status);
      return;
    default:
      rns8_export_i64_cell_device(residues, dst, cell, elements, prefix, bound, status);
      return;
  }
}

__device__ void rns8_export_u64_bounded_prefix_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  switch (prefix) {
    case 1:
      rns8_export_u64_fixed_prefix_cell_device<1>(residues, dst, cell, elements, bound, status);
      return;
    case 2:
      rns8_export_u64_fixed_prefix_cell_device<2>(residues, dst, cell, elements, bound, status);
      return;
    case 3:
      rns8_export_u64_fixed_prefix_cell_device<3>(residues, dst, cell, elements, bound, status);
      return;
    case 4:
      rns8_export_u64_fixed_prefix_cell_device<4>(residues, dst, cell, elements, bound, status);
      return;
    case 5:
      rns8_export_u64_fixed_prefix_cell_device<5>(residues, dst, cell, elements, bound, status);
      return;
    case 6:
      rns8_export_u64_fixed_prefix_cell_device<6>(residues, dst, cell, elements, bound, status);
      return;
    case 7:
      rns8_export_u64_fixed_prefix_cell_device<7>(residues, dst, cell, elements, bound, status);
      return;
    case 8:
      rns8_export_u64_fixed_prefix_cell_device<8>(residues, dst, cell, elements, bound, status);
      return;
    case kRns8DefaultBoundedPrefix:
      rns8_export_u64_fixed_prefix_cell_device<kRns8DefaultBoundedPrefix>(
          residues, dst, cell, elements, bound, status);
      return;
    default:
      rns8_export_u64_cell_device(residues, dst, cell, elements, prefix, bound, status);
      return;
  }
}

__global__ void rns8_export_i64_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_i64_cell_device(residues, dst, cell, elements, prefix, bound, status);
}

template <int Prefix>
__global__ void rns8_export_i64_fixed_prefix_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
}

__global__ void rns8_export_u64_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_u64_cell_device(residues, dst, cell, elements, prefix, bound, status);
}

template <int Prefix>
__global__ void rns8_export_u64_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
}

__global__ void rns8_export_i64_grouped_kernel(
    const int8_t* const* residue_ptrs,
    int64_t* dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  int64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_i64_cell_device(residues, task_dst, cell, elements, prefix, bound, status);
}

template <int Prefix>
__global__ void rns8_export_i64_grouped_fixed_prefix_kernel(
    const int8_t* const* residue_ptrs,
    int64_t* dst,
    int task_count,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  int64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell, elements, bound, status);
}

__global__ void rns8_export_u64_grouped_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_u64_cell_device(residues, task_dst, cell, elements, prefix, bound, status);
}

template <int Prefix>
__global__ void rns8_export_u64_grouped_fixed_prefix_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell, elements, bound, status);
}

__global__ void rns8_export_i64_scheduled_kernel(
    const int8_t* residues,
    int64_t* dst,
    const rns8_plan_tile_schedule_entry* schedule,
    const uint64_t* bounds,
    const uint8_t* zero_a_rows,
    const uint8_t* zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* status) {
  const int blocks_per_tile = (max_tile_elements + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x);
  if (blocks_per_tile <= 0) {
    return;
  }
  const int entry_index = static_cast<int>(blockIdx.x) / blocks_per_tile;
  if (entry_index >= entry_count) {
    return;
  }
  const int tile_block = static_cast<int>(blockIdx.x) - entry_index * blocks_per_tile;
  const int tile_cell = tile_block * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (tile_cell >= max_tile_elements) {
    return;
  }
  const rns8_plan_tile_schedule_entry entry = schedule[entry_index];
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_row = static_cast<int>(entry.row_offset) + local_row;
  const int global_col = static_cast<int>(entry.col_offset) + local_col;
  const int global_cell = global_row * cols + global_col;
  if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
    dst[global_cell] = 0;
    return;
  }
  if ((entry.flags & kRns8TileScheduleZeroRowColProduct) != 0 &&
      zero_a_rows != nullptr && zero_b_cols != nullptr &&
      (zero_a_rows[global_row] != 0 || zero_b_cols[global_col] != 0)) {
    dst[global_cell] = 0;
    return;
  }
  rns8_export_i64_bounded_prefix_cell_device(
      residues, dst, global_cell, rows * cols, static_cast<int>(entry.selected_prefix), bounds[entry_index], status);
}

__global__ void rns8_export_u64_scheduled_kernel(
    const int8_t* residues,
    uint64_t* dst,
    const rns8_plan_tile_schedule_entry* schedule,
    const uint64_t* bounds,
    const uint8_t* zero_a_rows,
    const uint8_t* zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* status) {
  const int blocks_per_tile = (max_tile_elements + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x);
  if (blocks_per_tile <= 0) {
    return;
  }
  const int entry_index = static_cast<int>(blockIdx.x) / blocks_per_tile;
  if (entry_index >= entry_count) {
    return;
  }
  const int tile_block = static_cast<int>(blockIdx.x) - entry_index * blocks_per_tile;
  const int tile_cell = tile_block * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (tile_cell >= max_tile_elements) {
    return;
  }
  const rns8_plan_tile_schedule_entry entry = schedule[entry_index];
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_row = static_cast<int>(entry.row_offset) + local_row;
  const int global_col = static_cast<int>(entry.col_offset) + local_col;
  const int global_cell = global_row * cols + global_col;
  if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
    dst[global_cell] = 0;
    return;
  }
  if ((entry.flags & kRns8TileScheduleZeroRowColProduct) != 0 &&
      zero_a_rows != nullptr && zero_b_cols != nullptr &&
      (zero_a_rows[global_row] != 0 || zero_b_cols[global_col] != 0)) {
    dst[global_cell] = 0;
    return;
  }
  rns8_export_u64_bounded_prefix_cell_device(
      residues, dst, global_cell, rows * cols, static_cast<int>(entry.selected_prefix), bounds[entry_index], status);
}

__device__ void rns8_export_exact_wide_unsigned_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    int limb_count,
    int* status) {
  if (!rns8_u192_unsigned_fits_limbs_device(x, limb_count)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_limbs_device(dst + static_cast<int64_t>(cell) * limb_count, x, limb_count);
}

template <int LimbCount>
__device__ void rns8_export_exact_wide_unsigned_fixed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    int* status) {
  if (!rns8_u192_unsigned_fits_fixed_limbs_device<LimbCount>(x)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_fixed_limbs_device<LimbCount>(dst + static_cast<int64_t>(cell) * LimbCount, x);
}

__device__ void rns8_export_exact_wide_signed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    int limb_count,
    int* status);

__global__ void rns8_export_exact_wide_signed_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_exact_wide_signed_limbs_device(dst, cell, x, product, limb_count, status);
}

__global__ void rns8_export_exact_wide_unsigned_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_exact_wide_unsigned_limbs_device(dst, cell, x, limb_count, status);
}

template <int Prefix>
__global__ void rns8_export_exact_wide_unsigned_limbs_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_limbs_device(dst, cell, x, limb_count, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_fixed_limbs_device<LimbCount>(dst, cell, x, status);
}

__device__ void rns8_export_exact_wide_signed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    int limb_count,
    int* status) {
  uint64_t* cell_dst = dst + static_cast<int64_t>(cell) * limb_count;
  if (rns8_u192_centered_is_negative_device(x, product)) {
    const rns8_u192_device magnitude = rns8_u192_sub_device(product, x);
    if (!rns8_u192_signed_negative_magnitude_fits_limbs_device(magnitude, limb_count)) {
      atomicCAS(status, 0, 5);
      return;
    }
    rns8_store_u192_negative_twos_complement_limbs_device(cell_dst, magnitude, limb_count);
    return;
  }

  if (!rns8_u192_signed_positive_fits_limbs_device(x, limb_count)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_limbs_device(cell_dst, x, limb_count);
}

template <int LimbCount>
__device__ void rns8_export_exact_wide_signed_fixed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    int* status) {
  uint64_t* cell_dst = dst + static_cast<int64_t>(cell) * LimbCount;
  if (rns8_u192_centered_is_negative_device(x, product)) {
    const rns8_u192_device magnitude = rns8_u192_sub_device(product, x);
    if (!rns8_u192_signed_negative_magnitude_fits_fixed_limbs_device<LimbCount>(magnitude)) {
      atomicCAS(status, 0, 5);
      return;
    }
    rns8_store_u192_negative_twos_complement_fixed_limbs_device<LimbCount>(cell_dst, magnitude);
    return;
  }

  if (!rns8_u192_signed_positive_fits_fixed_limbs_device<LimbCount>(x)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_fixed_limbs_device<LimbCount>(cell_dst, x);
}

template <int Prefix>
__global__ void rns8_export_exact_wide_signed_limbs_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_limbs_device(dst, cell, x, product, limb_count, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_fixed_limbs_device<LimbCount>(dst, cell, x, product, status);
}

__global__ void rns8_export_exact_wide_unsigned_grouped_limbs_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * limb_count;
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_exact_wide_unsigned_limbs_device(task_dst, cell, x, limb_count, nullptr);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_grouped_fixed_prefix_fixed_limbs_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * LimbCount;
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_fixed_limbs_device<LimbCount>(task_dst, cell, x, nullptr);
}

__global__ void rns8_export_exact_wide_signed_grouped_limbs_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * limb_count;
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  rns8_export_exact_wide_signed_limbs_device(task_dst, cell, x, product, limb_count, nullptr);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_grouped_fixed_prefix_fixed_limbs_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * LimbCount;
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_fixed_limbs_device<LimbCount>(task_dst, cell, x, product, nullptr);
}

