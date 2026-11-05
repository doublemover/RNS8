__global__ void __launch_bounds__(kRns8HipTileM* kRns8HipTileN)
    rns8_ring_gemm_uniform_small_i8_ab_i32_grouped_prefix9_kernel(const int8_t* A, const int8_t* B,
                                                                  int8_t* C_base, int m, int n,
                                                                  int k, int k_offset, int k_block,
                                                                  int lda, int ldb, int ldc,
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile, b_tile,
      {m, n, tile_row, tile_col, k_offset, k_block, ldc, row, col, output_active, accumulate},
      rns8::detail::CenteredResidueTileInput{A, lda},
      rns8::detail::CenteredResidueTileInput{B, ldb}, C,
      Rns8DefaultLadderTileReducer{modulus_index, modulus, modulus_reciprocal});
}

__global__ void __launch_bounds__(kRns8HipTileM*(kRns8HipTileN / 2))
    rns8_ring_gemm_uniform_small_i8_ab_i32_colpair_grouped_prefix9_kernel(
        const int8_t* A, const int8_t* B, int8_t* C_base, int m, int n, int k, int k_offset,
        int k_block, int lda, int ldb, int ldc, int accumulate) {
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
    const int tile_extent = k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    rns8::detail::load_residue_tile<kRns8HipTileM, kRns8HipTileN, kRns8HipTileK,
                                    kRns8HipTileNPadded>(
        a_tile, b_tile, {m, n, tile_row, tile_col, k_offset}, tile_k, tile_extent, lane,
        block_threads, rns8::detail::CenteredResidueTileInput{A, lda},
        rns8::detail::CenteredResidueTileInput{B, ldb});

    __syncthreads();

    if (output0_active || output1_active) {
      rns8::detail::accumulate_residue_tile_pair(&a_tile[thread_row][0],
                                                 &b_tile[0][thread_col_pair * 2],
                                                 kRns8HipTileNPadded, tile_extent, acc0, acc1);
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
    C[row * ldc + col0] = rns8_reduce_to_centered_default_modulus_fixed_device(
        acc0, modulus_index, modulus, modulus_reciprocal);
  }
  if (output1_active) {
    C[row * ldc + col1] = rns8_reduce_to_centered_default_modulus_fixed_device(
        acc1, modulus_index, modulus, modulus_reciprocal);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM* kRns8HipTileN)
    rns8_ring_gemm_i8_i32_scheduled_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                                           const rns8_plan_tile_schedule_entry* schedule,
                                           const uint8_t* zero_a_rows, const uint8_t* zero_b_cols,
                                           int entry_count, int k_offset, int k_block, int lda,
                                           int ldb, int ldc, int modulus,
                                           uint32_t modulus_reciprocal, int modulus_index,
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
  const bool output_active = tile_local_row + thread_row < static_cast<int>(entry.row_extent) &&
                             tile_local_col + thread_col < static_cast<int>(entry.col_extent);
  if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
    if (output_active) {
      C[row * ldc + col] = 0;
    }
    return;
  }
  const bool zero_row_col_product = (entry.flags & kRns8TileScheduleZeroRowColProduct) != 0 &&
                                    zero_a_rows != nullptr && zero_b_cols != nullptr;
  if (zero_row_col_product &&
      rns8::detail::residue_tile_zero_product<kRns8HipTileM, kRns8HipTileN>(
          zero_a_rows, zero_b_cols,
          {static_cast<int>(entry.row_extent), static_cast<int>(entry.col_extent), tile_local_row,
           tile_local_col, static_cast<int>(entry.row_offset),
           static_cast<int>(entry.col_offset)})) {
    if (output_active)
      C[row * ldc + col] = 0;
    return;
  }
  const bool output_zero_by_row_col =
      output_active && zero_row_col_product && (zero_a_rows[row] != 0 || zero_b_cols[col] != 0);
  // Local bounds stop loads at this schedule entry, while C keeps global indices.
  const int8_t* tile_A = A + static_cast<int>(entry.row_offset) * lda;
  const int8_t* tile_B = B + static_cast<int>(entry.col_offset);
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile, b_tile,
      {static_cast<int>(entry.row_extent), static_cast<int>(entry.col_extent), tile_local_row,
       tile_local_col, k_offset, k_block, ldc, row, col, output_active && !output_zero_by_row_col,
       accumulate},
      rns8::detail::CenteredResidueTileInput{tile_A, lda},
      rns8::detail::CenteredResidueTileInput{tile_B, ldb}, C,
      Rns8DefaultLadderTileReducer{modulus_index, modulus, modulus_reciprocal});
  if (output_zero_by_row_col)
    C[row * ldc + col] = 0;
}

__global__ void __launch_bounds__(kRns8HipTileM* kRns8HipTileN)
    rns8_ring_gemm_i8_i32_grouped_active_schedule_kernel(
        const int8_t* A_base, const int8_t* B_base, int8_t* C_base,
        const rns8_plan_tile_schedule_entry* active_schedule, const uint8_t* zero_a_rows,
        const uint8_t* zero_b_cols, int active_entry_count, int m, int k_total, int k_offset,
        int k_block, int lda, int ldb, int ldc, int accumulate) {
  const int entry_index = static_cast<int>(blockIdx.z);
  if (entry_index >= active_entry_count) {
    return;
  }
  const rns8_plan_tile_schedule_entry entry = active_schedule[entry_index];
  const int modulus_index = static_cast<int>(entry.group_index);
  if (modulus_index < 0 || modulus_index >= static_cast<int>(entry.selected_prefix) ||
      modulus_index >= kRns8DefaultModulusCount ||
      (entry.flags & kRns8TileScheduleZeroOutput) != 0) {
    return;
  }

  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t b_plane_offset = static_cast<int64_t>(modulus_index) *
                                 static_cast<int64_t>(k_total) * static_cast<int64_t>(ldb);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* A = A_base + a_plane_offset;
  const int8_t* B = B_base + b_plane_offset;
  int8_t* C = C_base + c_plane_offset;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_local_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_local_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = static_cast<int>(entry.row_offset) + tile_local_row + thread_row;
  const int col = static_cast<int>(entry.col_offset) + tile_local_col + thread_col;
  const bool output_active = tile_local_row + thread_row < static_cast<int>(entry.row_extent) &&
                             tile_local_col + thread_col < static_cast<int>(entry.col_extent);
  const bool zero_row_col_product = (entry.flags & kRns8TileScheduleZeroRowColProduct) != 0 &&
                                    zero_a_rows != nullptr && zero_b_cols != nullptr;
  if (zero_row_col_product &&
      rns8::detail::residue_tile_zero_product<kRns8HipTileM, kRns8HipTileN>(
          zero_a_rows, zero_b_cols,
          {static_cast<int>(entry.row_extent), static_cast<int>(entry.col_extent), tile_local_row,
           tile_local_col, static_cast<int>(entry.row_offset),
           static_cast<int>(entry.col_offset)})) {
    if (output_active)
      C[row * ldc + col] = 0;
    return;
  }
  const bool output_zero_by_row_col =
      output_active && zero_row_col_product && (zero_a_rows[row] != 0 || zero_b_cols[col] != 0);
  // Local bounds stop loads at this schedule entry, while C keeps global indices.
  const int8_t* tile_A = A + static_cast<int>(entry.row_offset) * lda;
  const int8_t* tile_B = B + static_cast<int>(entry.col_offset);
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile, b_tile,
      {static_cast<int>(entry.row_extent), static_cast<int>(entry.col_extent), tile_local_row,
       tile_local_col, k_offset, k_block, ldc, row, col, output_active && !output_zero_by_row_col,
       accumulate},
      rns8::detail::CenteredResidueTileInput{tile_A, lda},
      rns8::detail::CenteredResidueTileInput{tile_B, ldb}, C,
      Rns8DefaultLadderTileReducer{modulus_index, modulus, modulus_reciprocal});
  if (output_zero_by_row_col)
    C[row * ldc + col] = 0;
}
