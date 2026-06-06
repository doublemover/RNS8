__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_uniform_small_i8_ab_i32_grouped_prefix9_kernel(
        const int8_t* A,
        const int8_t* B,
        int8_t* C_base,
        int m,
        int n,
        int k,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int accumulate) {
  const int modulus_index = static_cast<int>(blockIdx.z);
  if (modulus_index >= kRns8DefaultBoundedPrefix) {
    return;
  }
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  int8_t* C = C_base + c_plane_offset;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = tile_row + thread_row;
  const int col = tile_col + thread_col;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output_active = row < m && col < n;
  int32_t acc = 0;

  for (int tile_k = 0; tile_k < k_block; tile_k += kRns8HipTileK) {
    const int tile_extent =
        k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    for (int index = lane; index < kRns8HipTileM * kRns8HipTileK; index += block_threads) {
      const int local_row = index / kRns8HipTileK;
      const int local_k = index - local_row * kRns8HipTileK;
      const int global_row = tile_row + local_row;
      const int source_k = k_offset + tile_k + local_k;
      a_tile[local_row][local_k] =
          global_row < m && local_k < tile_extent ? A[global_row * lda + source_k] : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n ? B[source_k * ldb + global_col] : 0;
    }

    __syncthreads();

    if (output_active) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        acc += static_cast<int32_t>(a_tile[thread_row][kk]) * static_cast<int32_t>(b_tile[kk][thread_col]);
      }
    }

    __syncthreads();
  }

  if (output_active && accumulate) {
    acc += static_cast<int32_t>(C[row * ldc + col]);
  }
  if (output_active) {
    C[row * ldc + col] =
        rns8_reduce_to_centered_default_modulus_fixed_device(acc, modulus_index, modulus, modulus_reciprocal);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * (kRns8HipTileN / 2))
    rns8_ring_gemm_uniform_small_i8_ab_i32_colpair_grouped_prefix9_kernel(
        const int8_t* A,
        const int8_t* B,
        int8_t* C_base,
        int m,
        int n,
        int k,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int accumulate) {
  const int modulus_index = static_cast<int>(blockIdx.z);
  if (modulus_index >= kRns8DefaultBoundedPrefix) {
    return;
  }
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  int8_t* C = C_base + c_plane_offset;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col_pair = static_cast<int>(threadIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = tile_row + thread_row;
  const int col0 = tile_col + thread_col_pair * 2;
  const int col1 = col0 + 1;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col_pair;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output0_active = row < m && col0 < n;
  const bool output1_active = row < m && col1 < n;
  int32_t acc0 = 0;
  int32_t acc1 = 0;

  for (int tile_k = 0; tile_k < k_block; tile_k += kRns8HipTileK) {
    const int tile_extent =
        k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    for (int index = lane; index < kRns8HipTileM * kRns8HipTileK; index += block_threads) {
      const int local_row = index / kRns8HipTileK;
      const int local_k = index - local_row * kRns8HipTileK;
      const int global_row = tile_row + local_row;
      const int source_k = k_offset + tile_k + local_k;
      a_tile[local_row][local_k] =
          global_row < m && local_k < tile_extent ? A[global_row * lda + source_k] : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n ? B[source_k * ldb + global_col] : 0;
    }

    __syncthreads();

    if (output0_active || output1_active) {
      const int local_col0 = thread_col_pair * 2;
      const int local_col1 = local_col0 + 1;
      for (int kk = 0; kk < tile_extent; ++kk) {
        const int32_t a = static_cast<int32_t>(a_tile[thread_row][kk]);
        acc0 += a * static_cast<int32_t>(b_tile[kk][local_col0]);
        acc1 += a * static_cast<int32_t>(b_tile[kk][local_col1]);
      }
    }

    __syncthreads();
  }

  if (output0_active && accumulate) {
    acc0 += static_cast<int32_t>(C[row * ldc + col0]);
  }
  if (output1_active && accumulate) {
    acc1 += static_cast<int32_t>(C[row * ldc + col1]);
  }
  if (output0_active) {
    C[row * ldc + col0] =
        rns8_reduce_to_centered_default_modulus_fixed_device(acc0, modulus_index, modulus, modulus_reciprocal);
  }
  if (output1_active) {
    C[row * ldc + col1] =
        rns8_reduce_to_centered_default_modulus_fixed_device(acc1, modulus_index, modulus, modulus_reciprocal);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN) rns8_ring_gemm_i8_i32_scheduled_kernel(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    const rns8_plan_tile_schedule_entry* schedule,
    const uint8_t* zero_a_rows,
    const uint8_t* zero_b_cols,
    int entry_count,
    int k_offset,
    int k_block,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int accumulate) {
  const int entry_index = static_cast<int>(blockIdx.z);
  if (entry_index >= entry_count) {
    return;
  }
  const rns8_plan_tile_schedule_entry entry = schedule[entry_index];
  if (modulus_index >= static_cast<int>(entry.selected_prefix)) {
    return;
  }

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_local_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_local_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = static_cast<int>(entry.row_offset) + tile_local_row + thread_row;
  const int col = static_cast<int>(entry.col_offset) + tile_local_col + thread_col;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output_active =
      tile_local_row + thread_row < static_cast<int>(entry.row_extent) &&
      tile_local_col + thread_col < static_cast<int>(entry.col_extent);
  if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
    if (output_active) {
      C[row * ldc + col] = 0;
    }
    return;
  }
  const bool zero_row_col_product =
      (entry.flags & kRns8TileScheduleZeroRowColProduct) != 0 && zero_a_rows != nullptr && zero_b_cols != nullptr;
  if (zero_row_col_product) {
    bool block_zero_rows = true;
    for (int local_row = 0; local_row < kRns8HipTileM; ++local_row) {
      const int output_row = tile_local_row + local_row;
      if (output_row >= static_cast<int>(entry.row_extent)) {
        continue;
      }
      const int global_row = static_cast<int>(entry.row_offset) + output_row;
      if (zero_a_rows[global_row] == 0) {
        block_zero_rows = false;
        break;
      }
    }
    bool block_zero_cols = true;
    for (int local_col = 0; local_col < kRns8HipTileN; ++local_col) {
      const int output_col = tile_local_col + local_col;
      if (output_col >= static_cast<int>(entry.col_extent)) {
        continue;
      }
      const int global_col = static_cast<int>(entry.col_offset) + output_col;
      if (zero_b_cols[global_col] == 0) {
        block_zero_cols = false;
        break;
      }
    }
    if (block_zero_rows || block_zero_cols) {
      if (output_active) {
        C[row * ldc + col] = 0;
      }
      return;
    }
  }
  const bool output_zero_by_row_col =
      output_active && zero_row_col_product && (zero_a_rows[row] != 0 || zero_b_cols[col] != 0);
  int32_t acc = 0;

  for (int tile_k = 0; tile_k < k_block; tile_k += kRns8HipTileK) {
    const int tile_extent =
        k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    for (int index = lane; index < kRns8HipTileM * kRns8HipTileK; index += block_threads) {
      const int local_row = index / kRns8HipTileK;
      const int local_k = index - local_row * kRns8HipTileK;
      const int local_output_row = tile_local_row + local_row;
      const int global_row = static_cast<int>(entry.row_offset) + local_output_row;
      const int source_k = k_offset + tile_k + local_k;
      a_tile[local_row][local_k] =
          local_output_row < static_cast<int>(entry.row_extent) && local_k < tile_extent
              ? A[global_row * lda + source_k]
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int local_output_col = tile_local_col + local_col;
      const int global_col = static_cast<int>(entry.col_offset) + local_output_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && local_output_col < static_cast<int>(entry.col_extent)
              ? B[source_k * ldb + global_col]
              : 0;
    }

    __syncthreads();

    if (output_active && !output_zero_by_row_col) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        acc += static_cast<int32_t>(a_tile[thread_row][kk]) * static_cast<int32_t>(b_tile[kk][thread_col]);
      }
    }

    __syncthreads();
  }

  if (output_active && !output_zero_by_row_col && accumulate) {
    acc += static_cast<int32_t>(C[row * ldc + col]);
  }
  if (output_zero_by_row_col) {
    C[row * ldc + col] = 0;
  } else if (output_active) {
    C[row * ldc + col] =
        rns8_reduce_to_centered_default_modulus_fixed_device(acc, modulus_index, modulus, modulus_reciprocal);
  }
}

