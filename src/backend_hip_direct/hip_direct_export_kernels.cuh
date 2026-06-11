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

__global__ void rns8_export_u8_grouped_modulus_kernel(
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols,
    int modulus) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int8_t* residues = residue_ptrs[task];
  uint8_t* task_dst = dst + static_cast<int64_t>(task) * elements;
  task_dst[idx] = static_cast<uint8_t>(rns8_canonical_from_centered_device(residues[idx], modulus));
}

template <int Modulus>
__device__ void rns8_export_u8_fixed_modulus_cell_device(
    const int8_t* residues,
    uint8_t* dst,
    int cols,
    int ld,
    int64_t idx) {
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  dst[static_cast<int64_t>(row) * ld + col] =
      static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx]));
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
  rns8_export_u8_fixed_modulus_cell_device<Modulus>(residues, dst, cols, ld, idx);
}

template <int Modulus>
__global__ void rns8_export_u8_fixed_modulus_quad_kernel(
    const int8_t* residues,
    uint8_t* dst,
    int rows,
    int cols,
    int ld) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx >= elements) {
    return;
  }
  rns8_export_u8_fixed_modulus_cell_device<Modulus>(residues, dst, cols, ld, idx);
  if (idx + 1 < elements) {
    rns8_export_u8_fixed_modulus_cell_device<Modulus>(residues, dst, cols, ld, idx + 1);
  }
  if (idx + 2 < elements) {
    rns8_export_u8_fixed_modulus_cell_device<Modulus>(residues, dst, cols, ld, idx + 2);
  }
  if (idx + 3 < elements) {
    rns8_export_u8_fixed_modulus_cell_device<Modulus>(residues, dst, cols, ld, idx + 3);
  }
}

template <int Modulus>
__global__ void rns8_export_u8_grouped_fixed_modulus_kernel(
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int8_t* residues = residue_ptrs[task];
  uint8_t* task_dst = dst + static_cast<int64_t>(task) * elements;
  task_dst[idx] = static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx]));
}

template <int Modulus>
__global__ void rns8_export_u8_grouped_fixed_modulus_quad_kernel(
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int8_t* residues = residue_ptrs[task];
  uint8_t* task_dst = dst + static_cast<int64_t>(task) * elements;
  task_dst[idx] = static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx]));
  if (idx + 1 < elements) {
    task_dst[idx + 1] =
        static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx + 1]));
  }
  if (idx + 2 < elements) {
    task_dst[idx + 2] =
        static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx + 2]));
  }
  if (idx + 3 < elements) {
    task_dst[idx + 3] =
        static_cast<uint8_t>(rns8_canonical_from_centered_fixed_modulus_device<Modulus>(residues[idx + 3]));
  }
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
      if (status) atomicCAS(status, 0, 5);
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
    if (status) atomicCAS(status, 0, 5);
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
    if (status) atomicCAS(status, 0, 5);
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

template <int Prefix>
__global__ void rns8_export_i64_fixed_prefix_quad_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int elements = rows * cols;
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
  if (cell + 1 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 1, elements, bound, status);
  }
  if (cell + 2 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 2, elements, bound, status);
  }
  if (cell + 3 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 3, elements, bound, status);
  }
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

template <int Prefix>
__global__ void rns8_export_u64_fixed_prefix_quad_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int elements = rows * cols;
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
  if (cell + 1 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 1, elements, bound, status);
  }
  if (cell + 2 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 2, elements, bound, status);
  }
  if (cell + 3 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell + 3, elements, bound, status);
  }
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

template <int Prefix>
__global__ void rns8_export_i64_grouped_fixed_prefix_quad_kernel(
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
  const int elements = rows * cols;
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  int64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell, elements, bound, status);
  if (cell + 1 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 1, elements, bound, status);
  }
  if (cell + 2 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 2, elements, bound, status);
  }
  if (cell + 3 < elements) {
    rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 3, elements, bound, status);
  }
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

template <int Prefix>
__global__ void rns8_export_u64_grouped_fixed_prefix_quad_kernel(
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
  const int elements = rows * cols;
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements);
  rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell, elements, bound, status);
  if (cell + 1 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 1, elements, bound, status);
  }
  if (cell + 2 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 2, elements, bound, status);
  }
  if (cell + 3 < elements) {
    rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, task_dst, cell + 3, elements, bound, status);
  }
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
__device__ void rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_fixed_limbs_device<LimbCount>(dst, cell, x, status);
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
  rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 1, elements, status);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 2, elements, status);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 3, elements, status);
  }
}

template <int Prefix>
__global__ void rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_kernel(
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
  rns8_reconstruct_canonical_wide_tree_pairs_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_limbs_device(dst, cell, x, limb_count, status);
}

template <int Prefix, int LimbCount>
__device__ void rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_tree_pairs_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_fixed_limbs_device<LimbCount>(dst, cell, x, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_kernel(
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
  rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 1, elements, status);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 2, elements, status);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 3, elements, status);
  }
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
__device__ void rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_fixed_limbs_device<LimbCount>(dst, cell, x, product, status);
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
  rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 1, elements, status);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 2, elements, status);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 3, elements, status);
  }
}

template <int Prefix>
__global__ void rns8_export_exact_wide_signed_tree_crt_fixed_prefix_kernel(
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
  rns8_reconstruct_canonical_wide_tree_pairs_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_limbs_device(dst, cell, x, product, limb_count, status);
}

template <int Prefix, int LimbCount>
__device__ void rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_tree_pairs_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_fixed_limbs_device<LimbCount>(dst, cell, x, product, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_kernel(
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
  rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
}

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int* status) {
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, dst, cell, elements, status);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 1, elements, status);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 2, elements, status);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_signed_tree_crt_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, dst, cell + 3, elements, status);
  }
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

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_unsigned_grouped_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * LimbCount;
  rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, task_dst, cell, elements, nullptr);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 1, elements, nullptr);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 2, elements, nullptr);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 3, elements, nullptr);
  }
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

template <int Prefix, int LimbCount>
__global__ void rns8_export_exact_wide_signed_grouped_fixed_prefix_fixed_limbs_quad_kernel(
    const int8_t* const* residue_ptrs,
    uint64_t* dst,
    int task_count,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int cell = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const int8_t* residues = residue_ptrs[task];
  uint64_t* task_dst = dst + static_cast<int64_t>(task) * static_cast<int64_t>(elements) * LimbCount;
  rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
      residues, task_dst, cell, elements, nullptr);
  if (cell + 1 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 1, elements, nullptr);
  }
  if (cell + 2 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 2, elements, nullptr);
  }
  if (cell + 3 < elements) {
    rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_cell_device<Prefix, LimbCount>(
        residues, task_dst, cell + 3, elements, nullptr);
  }
}


// === Gap 82: Vectorized Garner/CRT constant tables ===

template <int Prefix>
struct rns8_garner_constants_device {
  __device__ static uint32_t modulus(int i) {
    // Default modulus ladder values for prefix indexing
    constexpr uint32_t moduli[] = {
      256, 255, 253, 251, 247, 239, 233, 229,
      227, 223, 217, 211, 199, 197, 193, 191,
      181, 179, 173, 167, 163, 157, 151, 149,
      139, 137, 131, 127,
    };
    return (i >= 0 && i < Prefix) ? moduli[i] : 1;
  }

  __device__ static uint64_t prefix_product() {
    uint64_t product = 1;
    for (int i = 0; i < Prefix; ++i) {
      product *= static_cast<uint64_t>(modulus(i));
    }
    return product;
  }

  __device__ static uint64_t garner_weight(int i) {
    // Garner weight = (P / m_i) * inv(P / m_i mod m_i) mod P
    const uint64_t P = prefix_product();
    const uint64_t mi = static_cast<uint64_t>(modulus(i));
    const uint64_t Pi = P / mi;
    // Compute modular inverse of Pi mod mi using extended Euclidean
    uint64_t inv = 1;
    uint64_t a = Pi % mi;
    uint64_t b = mi;
    int64_t x0 = 1, x1 = 0;
    while (b > 0) {
      uint64_t q = a / b;
      uint64_t r = a % b;
      a = b;
      b = r;
      int64_t x2 = x0 - static_cast<int64_t>(q) * x1;
      x0 = x1;
      x1 = x2;
    }
    if (x0 < 0) x0 += static_cast<int64_t>(mi);
    inv = static_cast<uint64_t>(x0);
    return (Pi * inv) % P;
  }
};

// Vectorized 4-wide Garner reconstruction for prefix18/prefix20
template <int Prefix>
__device__ void rns8_garner_reconstruct_4wide_device(
    const int8_t* __restrict__ residues,
    int64_t cell,
    int64_t elements,
    uint64_t* __restrict__ out_vals) {
  // Load 4 consecutive residue planes for 4 cells
  uint64_t acc[4] = {0, 0, 0, 0};
  for (int plane = 0; plane < Prefix; ++plane) {
    const int8_t* plane_base = residues + plane * elements;
    const uint64_t weight = rns8_garner_constants_device<Prefix>::garner_weight(plane);
    const uint32_t mod = rns8_garner_constants_device<Prefix>::modulus(plane);
    // Process 4 cells in one iteration
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      if (cell + c < elements) {
        int8_t residue = plane_base[cell + c];
        // Convert centered residue to canonical
        uint64_t canonical = static_cast<uint64_t>(
            residue < 0 ? residue + static_cast<int>(mod) : residue);
        acc[c] = (acc[c] + canonical * weight) % rns8_garner_constants_device<Prefix>::prefix_product();
      }
    }
  }
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    out_vals[c] = (cell + c < elements) ? acc[c] : 0;
  }
}


// === Gap 83: Combined final-output kernel (CRT + range + status + staging) ===

template <int Prefix>
__global__ void rns8_export_bounded_i64_combined_final_output_kernel(
    const int8_t* __restrict__ residues,
    int64_t* __restrict__ dst,
    int* __restrict__ status,
    int rows,
    int cols,
    int ld,
    int64_t bound,
    bool status_elided) {
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);

  const uint64_t* w = nullptr; uint64_t P = 0;
  rns8_get_garner_weights_and_product<Prefix>(&w, &P);
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    if (cell + c >= elements) break;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;
    uint64_t val;
    rns8_garner_reconstruct_cell_device<Prefix>(residues, static_cast<int>(cell + c), static_cast<int>(elements), w, P, &val);
    int64_t signed_val;
    if (val >= P / 2) signed_val = -static_cast<int64_t>(P - val);
    else signed_val = static_cast<int64_t>(val);
    if (!status_elided && (signed_val < -bound || signed_val > bound)) {
      atomicExch(status, 5);
    }
    dst[row * static_cast<int64_t>(ld) + col] = signed_val;
  }
}

template <int Prefix>
__global__ void rns8_export_bounded_u64_combined_final_output_kernel(
    const int8_t* __restrict__ residues,
    uint64_t* __restrict__ dst,
    int* __restrict__ status,
    int rows,
    int cols,
    int ld,
    uint64_t bound,
    bool status_elided) {
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);

  uint64_t vals[4];
  rns8_garner_reconstruct_4wide_device<Prefix>(residues, cell, elements, vals);

  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    if (cell + c >= elements) break;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;
    if (!status_elided && vals[c] > bound) {
      atomicExch(status, static_cast<int>(1)); // RNS8_RANGE_ERROR
    }
    dst[row * static_cast<int64_t>(ld) + col] = vals[c];
  }
}



// === Gap 99 continuation: VALU-optimized export helpers ===

// DPP-based status aggregation for export range checking
__device__ int rns8_export_dpp_status_aggregate_device(int local_status) {
  int agg = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_status, 16);
  agg |= __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_status, 8);
  agg |= __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_status, 4);
  agg |= __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_status, 2);
  agg |= __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_status, 1);
  return agg | local_status;
}

// VOPD-friendly export: two outputs per thread with paired instructions
template <int Prefix>
__global__ void rns8_export_bounded_i64_vopd_combined_kernel(
    const int8_t* __restrict__ residues,
    int64_t* __restrict__ dst,
    int* __restrict__ status,
    int rows,
    int cols,
    int ld,
    int64_t bound,
    bool status_elided) {
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);

  const uint64_t* w = nullptr; uint64_t P = 0;
  rns8_get_garner_weights_and_product<Prefix>(&w, &P);
  int local_status = 0;

  #pragma unroll
  for (int c = 0; c < 2; ++c) {
    if (cell + c >= elements) break;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;

    uint64_t val;
    rns8_garner_reconstruct_cell_device<Prefix>(residues, static_cast<int>(cell + c), static_cast<int>(elements), w, P, &val);


    int64_t signed_val;
    if (val >= P / 2) {
      signed_val = -static_cast<int64_t>(P - val);
    } else {
      signed_val = static_cast<int64_t>(val);
    }

    if (!status_elided && (signed_val < -bound || signed_val > bound)) {
      local_status = 1;
    }

    dst[row * static_cast<int64_t>(ld) + col] = signed_val;
  }

  // Aggregate status across lanes using DPP instead of atomic
  if (!status_elided) {
    int wave_status = rns8_export_dpp_status_aggregate_device(local_status);
    if (wave_status && (threadIdx.x & 31) == 0) {
      atomicExch(status, wave_status);
    }
  }
}



// === Phase 4b: Fused GEMM residue accumulation + CRT export ===
// Computes INT32 GEMM accumulators then immediately applies Garner CRT
// reconstruction, writing final i64/u64 output directly. Eliminates the
// intermediate centered i8 residue store/load between GEMM and export.

__global__ void rns8_fused_gemm_export_i64_kernel(
    const int8_t* __restrict__ a_residues,
    const int8_t* __restrict__ b_residues,
    int64_t* __restrict__ dst,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int prefix,
    int64_t bound,
    int* __restrict__ status) {
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  const int64_t elements = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  if (cell >= elements) return;

  // Garner CRT reconstruction accumulators (64-bit per output cell)
  uint64_t garner_acc[2] = {0, 0};
  const uint64_t M = rns8_garner_constants_device<9>::prefix_product();

  const uint64_t* weights = (prefix <= 1) ? garner_weights_prefix1 : (prefix <= 2) ? garner_weights_prefix2 : (prefix <= 3) ? garner_weights_prefix3 : (prefix <= 4) ? garner_weights_prefix4 : (prefix <= 5) ? garner_weights_prefix5 : (prefix <= 6) ? garner_weights_prefix6 : (prefix <= 7) ? garner_weights_prefix7 : garner_weights_prefix8;
  const int effective_prefix = (prefix > 8) ? 8 : prefix;
  for (int plane = 0; plane < effective_prefix; ++plane) {
    const int8_t* a_plane = a_residues + static_cast<int64_t>(plane) * m * k;
    const int8_t* b_plane = b_residues + static_cast<int64_t>(plane) * k * n;
    const uint64_t weight = weights[plane];
    const uint32_t mod = static_cast<uint32_t>(rns8_default_moduli_device[plane]);

    #pragma unroll
    for (int c = 0; c < 2; ++c) {
      if (cell + c >= elements) continue;
      const int64_t row = (cell + c) / n;
      const int64_t col = (cell + c) - row * n;

      int32_t acc = 0;
      for (int ki = 0; ki < k; ki += 65536) {
        int k_end = (ki + 65536 < k) ? ki + 65536 : k;
        for (int kii = ki; kii < k_end; ++kii) {
          acc += static_cast<int32_t>(a_plane[row * k + kii])
               * static_cast<int32_t>(b_plane[kii * n + col]);
        }
      }

      int32_t reduced = acc % static_cast<int32_t>(mod);
      if (reduced < 0) reduced += static_cast<int32_t>(mod);
      garner_acc[c] = (garner_acc[c] + static_cast<uint64_t>(reduced) * weight) % M;
    }
  }

  // Convert Garner result to signed i64 and write output
  #pragma unroll
  for (int c = 0; c < 2; ++c) {
    if (cell + c >= elements) continue;
    const int64_t row = (cell + c) / n;
    const int64_t col = (cell + c) - row * n;

    int64_t signed_val;
    if (garner_acc[c] >= M / 2) {
      signed_val = -static_cast<int64_t>(M - garner_acc[c]);
    } else {
      signed_val = static_cast<int64_t>(garner_acc[c]);
    }

    if (status && (signed_val < -bound || signed_val > bound)) {
      atomicExch(status, static_cast<int>(1));
    }

    dst[row * ldc + col] = signed_val;
  }
}



// === Fast Garner CRT reconstruction using precomputed __constant__ weights ===

template <int Prefix>
__device__ void rns8_get_garner_weights_and_product(
    const uint64_t** weights, uint64_t* product) {
  if constexpr (Prefix == 1) { *weights = garner_weights_prefix1; *product = garner_prefix_product1; }
  else if constexpr (Prefix == 2) { *weights = garner_weights_prefix2; *product = garner_prefix_product2; }
  else if constexpr (Prefix == 3) { *weights = garner_weights_prefix3; *product = garner_prefix_product3; }
  else if constexpr (Prefix == 4) { *weights = garner_weights_prefix4; *product = garner_prefix_product4; }
  else if constexpr (Prefix == 5) { *weights = garner_weights_prefix5; *product = garner_prefix_product5; }
  else if constexpr (Prefix == 6) { *weights = garner_weights_prefix6; *product = garner_prefix_product6; }
  else if constexpr (Prefix == 7) { *weights = garner_weights_prefix7; *product = garner_prefix_product7; }
  else if constexpr (Prefix == 8) { *weights = garner_weights_prefix8; *product = garner_prefix_product8; }
}

template <int Prefix>
__device__ void rns8_garner_reconstruct_cell_device(
    const int8_t* residues, int cell, int elements,
    const uint64_t* __restrict__ weights, uint64_t prefix_product,
    uint64_t* out_val) {
  uint64_t acc = 0;
  #pragma unroll
  for (int plane = 0; plane < Prefix; ++plane) {
    int8_t residue = residues[static_cast<int64_t>(plane) * elements + cell];
    uint32_t modulus = rns8_default_moduli_device[plane];
    uint64_t canonical = static_cast<uint64_t>(residue < 0 ? residue + static_cast<int>(modulus) : residue);
    acc = (acc + canonical * weights[plane]) % prefix_product;
  }
  *out_val = acc;
}

template <int Prefix>
__global__ void rns8_export_bounded_i64_garner_kernel(
    const int8_t* __restrict__ residues, int64_t* __restrict__ dst,
    int rows, int cols, int ld, int64_t bound, int* __restrict__ status) {
  const int cells_per_thread = 2;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * cells_per_thread;
  if (cell >= elements) return;
  const uint64_t* weights = nullptr; uint64_t M = 0;
  rns8_get_garner_weights_and_product<Prefix>(&weights, &M);
  int local_status = 0;
  #pragma unroll
  for (int c = 0; c < cells_per_thread; ++c) {
    if (cell + c >= elements) continue;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;
    uint64_t val;
    rns8_garner_reconstruct_cell_device<Prefix>(residues, static_cast<int>(cell + c), static_cast<int>(elements), weights, M, &val);
    int64_t signed_val;
    if (val >= M / 2) signed_val = -static_cast<int64_t>(M - val);
    else signed_val = static_cast<int64_t>(val);
    if (status && (signed_val < -bound || signed_val > bound)) { local_status = 5; continue; }
    dst[row * static_cast<int64_t>(ld) + col] = signed_val;
  }
  if (status && local_status) atomicExch(status, local_status);
}

template <int Prefix>
__global__ void rns8_export_bounded_u64_garner_kernel(
    const int8_t* __restrict__ residues, uint64_t* __restrict__ dst,
    int rows, int cols, int ld, uint64_t bound, int* __restrict__ status) {
  const int cells_per_thread = 2;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * cells_per_thread;
  if (cell >= elements) return;
  const uint64_t* weights = nullptr; uint64_t M = 0;
  rns8_get_garner_weights_and_product<Prefix>(&weights, &M);
  int local_status = 0;
  #pragma unroll
  for (int c = 0; c < cells_per_thread; ++c) {
    if (cell + c >= elements) continue;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;
    uint64_t val;
    rns8_garner_reconstruct_cell_device<Prefix>(residues, static_cast<int>(cell + c), static_cast<int>(elements), weights, M, &val);
    if (status && val > bound) { local_status = 5; continue; }
    dst[row * static_cast<int64_t>(ld) + col] = val;
  }
  if (status && local_status) atomicExch(status, local_status);
}
