extern "C" int rns8_hip_direct_ring_gemm_i8_scheduled_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || !d_schedule || entry_count <= 0 || max_tile_row_blocks <= 0 ||
      max_tile_col_blocks <= 0 || k <= 0 || lda < k || ldb <= 0 || ldc <= 0 || modulus <= 0 ||
      modulus_index < 0 || selected_prefix <= 0 || modulus_index >= selected_prefix ||
      selected_prefix > kRns8DefaultModulusCount ||
      modulus != rns8_default_moduli_host[modulus_index] ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid(max_tile_col_blocks, max_tile_row_blocks, entry_count);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_scheduled_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        d_schedule,
        d_zero_a_rows,
        d_zero_b_cols,
        entry_count,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        modulus_index,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_active_schedule_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_active_schedule,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int active_entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int m,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || !d_active_schedule || active_entry_count <= 0 ||
      max_tile_row_blocks <= 0 || max_tile_col_blocks <= 0 || m <= 0 || k <= 0 ||
      lda < k || ldb <= 0 || ldc <= 0 || safe_k_block <= 0 ||
      safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid(max_tile_col_blocks, max_tile_row_blocks, active_entry_count);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_grouped_active_schedule_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        d_active_schedule,
        d_zero_a_rows,
        d_zero_b_cols,
        active_entry_count,
        m,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      nullptr);
}

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    void* stream) {
  return rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      reinterpret_cast<hipStream_t>(stream));
}

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_device(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_on_stream(
      d_a_ptrs,
      d_b_ptrs,
      d_c_ptrs,
      task_count,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      nullptr);
}

__global__ void rns8_zero_scheduled_residue_tiles_kernel(
    int8_t* C,
    const rns8_plan_tile_schedule_entry* schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc) {
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
  if ((entry.flags & kRns8TileScheduleZeroOutput) == 0) {
    return;
  }
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int64_t cell_offset =
      static_cast<int64_t>(static_cast<int>(entry.row_offset) + local_row) * ldc +
      static_cast<int64_t>(static_cast<int>(entry.col_offset) + local_col);
  const int64_t plane_stride = static_cast<int64_t>(rows) * ldc;
  for (int prefix = 0; prefix < static_cast<int>(entry.selected_prefix); ++prefix) {
    C[static_cast<int64_t>(prefix) * plane_stride + cell_offset] = 0;
  }
}

extern "C" int rns8_hip_direct_zero_scheduled_residue_tiles_device(
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc) {
  if (!d_c || !d_schedule || entry_count <= 0 || max_tile_elements <= 0 || rows <= 0 || ldc <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t block_count =
      static_cast<int64_t>(entry_count) *
      ((static_cast<int64_t>(max_tile_elements) + threads - 1) / threads);
  if (block_count <= 0 || block_count > 0xffffffffLL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      rns8_zero_scheduled_residue_tiles_kernel,
      dim3(static_cast<unsigned int>(block_count)),
      dim3(threads),
      0,
      nullptr,
      d_c,
      d_schedule,
      entry_count,
      max_tile_elements,
      rows,
      ldc);
  return static_cast<int>(hipGetLastError());
}

