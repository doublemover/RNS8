__global__ void pack_a_rowmajor_kernel(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= padded_rows || kk >= gemm_k_block) {
    return;
  }
  packed[row * gemm_k_block + kk] =
      row < rows && kk < valid_k_block ? a[row * lda + k_offset + kk] : static_cast<int8_t>(0);
}

__global__ void pack_b_colmajor_kernel(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= padded_n || kk >= gemm_k_block) {
    return;
  }
  packed[col * gemm_k_block + kk] =
      col < n && kk < valid_k_block ? b[(k_offset + kk) * ldb + col] : static_cast<int8_t>(0);
}

__global__ void pack_b_colmajor_prefix_kernel(
    const int8_t* b,
    int8_t* packed,
    uint64_t packed_plane_bytes,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const uint32_t modulus_index = static_cast<uint32_t>(blockIdx.z);
  if (col >= padded_n || kk >= gemm_k_block) {
    return;
  }
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(valid_k_block) * static_cast<int64_t>(ldb);
  const uint64_t packed_plane_offset = static_cast<uint64_t>(modulus_index) * packed_plane_bytes;
  const uint64_t packed_index =
      packed_plane_offset + static_cast<uint64_t>(col) * static_cast<uint64_t>(gemm_k_block) + static_cast<uint64_t>(kk);
  const int64_t b_index = b_plane_offset + static_cast<int64_t>(kk) * static_cast<int64_t>(ldb) + col;
  packed[packed_index] = col < n && kk < valid_k_block ? b[b_index] : static_cast<int8_t>(0);
}

__global__ void __launch_bounds__(kRns8WmmaWaveThreads) rocwmma_i8_residue_gemm_kernel(
    const int8_t* a_pack,
    const int8_t* b_pack,
    int8_t* c,
    int rows,
    int cols,
    int padded_rows,
    int padded_cols,
    int gemm_k_block,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, row_major>;
  using FragB = fragment<matrix_b, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, col_major>;
  using FragAcc = fragment<accumulator, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int32_t>;

  __shared__ int32_t tile_acc[kRns8WmmaTileM * kRns8WmmaTileN];

  const int tile_col = static_cast<int>(blockIdx.x) * kRns8WmmaTileN;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8WmmaTileM;
  if (tile_row >= padded_rows || tile_col >= padded_cols) {
    return;
  }

  FragA frag_a;
  FragB frag_b;
  FragAcc acc;
  fill_fragment(acc, 0);

  for (int kk = 0; kk < gemm_k_block; kk += kRns8WmmaTileK) {
    load_matrix_sync(frag_a, a_pack + tile_row * gemm_k_block + kk, gemm_k_block);
    load_matrix_sync(frag_b, b_pack + tile_col * gemm_k_block + kk, gemm_k_block);
    mma_sync(acc, frag_a, frag_b, acc);
  }

  store_matrix_sync(tile_acc, acc, kRns8WmmaTileN, mem_row_major);
  __syncthreads();

  const int lane = static_cast<int>(threadIdx.x);
  for (int index = lane; index < kRns8WmmaTileM * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    const int local_row = index / kRns8WmmaTileN;
    const int local_col = index - local_row * kRns8WmmaTileN;
    const int row = tile_row + local_row;
    const int col = tile_col + local_col;
    if (row < rows && col < cols) {
      int32_t value = tile_acc[index];
      if (accumulate) {
        value += static_cast<int32_t>(c[row * ldc + col]);
      }
      c[row * ldc + col] = reduce_to_centered_device(value, modulus, modulus_reciprocal);
    }
  }
}

__device__ int64_t wrap64_compact_limb_index(int64_t row, int64_t col, int64_t cols, int limb) {
  return ((row * cols + col) * 8) + limb;
}

__device__ int8_t wrap64_signed_byte_lane(uint8_t value) {
  return value < 128u ? static_cast<int8_t>(value) : static_cast<int8_t>(static_cast<int>(value) - 256);
}

__device__ int8_t wrap64_lane_component(uint8_t value, int high_bit_component) {
  return high_bit_component != 0 ? static_cast<int8_t>(value >> 7u) : wrap64_signed_byte_lane(value);
}

__device__ void rocwmma_wrap64_accumulate_component(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t k_base,
    int valid_k,
    int tile_row,
    int tile_col,
    int a_limb,
    int b_limb,
    int a_high_bit_component,
    int b_high_bit_component,
    int diagonal,
    int64_t coefficient,
    int8_t* a_pack,
    int8_t* b_pack,
    int32_t* tile_acc,
    int64_t* diagonal_acc) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, row_major>;
  using FragB = fragment<matrix_b, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, col_major>;
  using FragAcc = fragment<accumulator, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int32_t>;

  const int lane = static_cast<int>(threadIdx.x);
  for (int index = lane; index < kRns8WmmaTileM * kRns8WmmaTileK; index += kRns8WmmaWaveThreads) {
    const int local_k = index % kRns8WmmaTileK;
    const int local_row = index / kRns8WmmaTileK;
    const int64_t row = static_cast<int64_t>(tile_row + local_row);
    const int64_t kk = k_base + local_k;
    int8_t value = 0;
    if (row < m && local_k < valid_k) {
      const uint8_t byte = a_limbs[wrap64_compact_limb_index(row, kk, k, a_limb)];
      value = wrap64_lane_component(byte, a_high_bit_component);
    }
    a_pack[index] = value;
  }
  for (int index = lane; index < kRns8WmmaTileK * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    const int local_k = index % kRns8WmmaTileK;
    const int local_col = index / kRns8WmmaTileK;
    const int64_t col = static_cast<int64_t>(tile_col + local_col);
    const int64_t kk = k_base + local_k;
    int8_t value = 0;
    if (col < n && local_k < valid_k) {
      const uint8_t byte = b_limbs[wrap64_compact_limb_index(kk, col, n, b_limb)];
      value = wrap64_lane_component(byte, b_high_bit_component);
    }
    b_pack[index] = value;
  }
  __syncthreads();

  FragA frag_a;
  FragB frag_b;
  FragAcc acc;
  fill_fragment(acc, 0);
  load_matrix_sync(frag_a, a_pack, kRns8WmmaTileK);
  load_matrix_sync(frag_b, b_pack, kRns8WmmaTileK);
  mma_sync(acc, frag_a, frag_b, acc);
  store_matrix_sync(tile_acc, acc, kRns8WmmaTileN, mem_row_major);
  __syncthreads();

  for (int index = lane; index < kRns8WmmaTileM * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    const int local_row = index / kRns8WmmaTileN;
    const int local_col = index - local_row * kRns8WmmaTileN;
    const int64_t row = static_cast<int64_t>(tile_row + local_row);
    const int64_t col = static_cast<int64_t>(tile_col + local_col);
    if (row < m && col < n) {
      diagonal_acc[diagonal * kRns8WmmaTileM * kRns8WmmaTileN + index] +=
          static_cast<int64_t>(tile_acc[index]) * coefficient;
    }
  }
  __syncthreads();
}

