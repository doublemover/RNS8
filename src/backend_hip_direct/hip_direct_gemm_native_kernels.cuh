__device__ int8_t rns8_center_native_operand_device(int64_t value, int modulus_index, int modulus) {
  return rns8_center_i64_default_modulus_fixed_device(value, modulus_index, modulus);
}

__device__ int8_t rns8_center_native_operand_device(uint64_t value, int modulus_index, int modulus) {
  return rns8_center_u64_default_modulus_fixed_device(value, modulus_index, modulus);
}

template <typename NativeT>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_native_i64_i32_grouped_prefix9_kernel(
        const NativeT* A,
        const NativeT* B,
        int8_t* C_base,
        int m,
        int n,
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
          global_row < m && local_k < tile_extent
              ? rns8_center_native_operand_device(A[global_row * lda + source_k], modulus_index, modulus)
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n
              ? rns8_center_native_operand_device(B[source_k * ldb + global_col], modulus_index, modulus)
              : 0;
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

template <typename NativeT>
__global__ void __launch_bounds__(kRns8HipTileM * (kRns8HipTileN / 2))
    rns8_ring_gemm_native_i64_i32_colpair_grouped_prefix9_kernel(
        const NativeT* A,
        const NativeT* B,
        int8_t* C_base,
        int m,
        int n,
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
          global_row < m && local_k < tile_extent
              ? rns8_center_native_operand_device(A[global_row * lda + source_k], modulus_index, modulus)
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n
              ? rns8_center_native_operand_device(B[source_k * ldb + global_col], modulus_index, modulus)
              : 0;
    }

    __syncthreads();

    if (output0_active || output1_active) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        const int32_t a = static_cast<int32_t>(a_tile[thread_row][kk]);
        if (output0_active) {
          acc0 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2]);
        }
        if (output1_active) {
          acc1 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2 + 1]);
        }
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

template <typename NativeT>
__device__ int8_t rns8_uniform_small_native_operand_device(NativeT value) {
  return static_cast<int8_t>(value);
}

template <typename NativeT, bool UniformSmallA>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_native_a_i8_b_i32_grouped_prefix9_kernel(
        const NativeT* A,
        const int8_t* B_base,
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
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * static_cast<int64_t>(ldb);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* B = B_base + b_plane_offset;
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
          global_row < m && local_k < tile_extent
              ? (UniformSmallA
                     ? rns8_uniform_small_native_operand_device(A[global_row * lda + source_k])
                     : rns8_center_native_operand_device(A[global_row * lda + source_k], modulus_index, modulus))
              : 0;
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

template <typename NativeT, bool UniformSmallA>
__global__ void __launch_bounds__(kRns8HipTileM * (kRns8HipTileN / 2))
    rns8_ring_gemm_native_a_i8_b_i32_colpair_grouped_prefix9_kernel(
        const NativeT* A,
        const int8_t* B_base,
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
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * static_cast<int64_t>(ldb);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* B = B_base + b_plane_offset;
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
          global_row < m && local_k < tile_extent
              ? (UniformSmallA
                     ? rns8_uniform_small_native_operand_device(A[global_row * lda + source_k])
                     : rns8_center_native_operand_device(A[global_row * lda + source_k], modulus_index, modulus))
              : 0;
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
      for (int kk = 0; kk < tile_extent; ++kk) {
        const int32_t a = static_cast<int32_t>(a_tile[thread_row][kk]);
        if (output0_active) {
          acc0 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2]);
        }
        if (output1_active) {
          acc1 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2 + 1]);
        }
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

template <typename NativeT, bool UniformSmallB>
__global__ void __launch_bounds__(kRns8HipTileM * (kRns8HipTileN / 2))
    rns8_ring_gemm_i8_a_native_b_i32_colpair_grouped_prefix9_kernel(
        const int8_t* A_base,
        const NativeT* B,
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
  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* A = A_base + a_plane_offset;
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
          local_k < tile_extent && global_col < n
              ? (UniformSmallB
                     ? rns8_uniform_small_native_operand_device(B[source_k * ldb + global_col])
                     : rns8_center_native_operand_device(B[source_k * ldb + global_col], modulus_index, modulus))
              : 0;
    }

    __syncthreads();

    if (output0_active || output1_active) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        const int32_t a = static_cast<int32_t>(a_tile[thread_row][kk]);
        if (output0_active) {
          acc0 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2]);
        }
        if (output1_active) {
          acc1 += a * static_cast<int32_t>(b_tile[kk][thread_col_pair * 2 + 1]);
        }
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

