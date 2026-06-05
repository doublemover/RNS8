__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN) rns8_ring_gemm_i8_i32_tiled_kernel(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int m,
    int n,
    int k_offset,
    int k_block,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate) {
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
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus, modulus_reciprocal);
  }
}

template <int Modulus>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_i8_i32_fixed_modulus_kernel(
        const int8_t* A,
        const int8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int accumulate) {
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
    C[row * ldc + col] = rns8_reduce_to_centered_fixed_modulus_device<Modulus>(acc);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_u8_native_i32_kernel(
        const uint8_t* A,
        const uint8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int modulus,
        uint32_t modulus_reciprocal,
        int accumulate) {
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
              ? rns8_center_u8_device(A[global_row * lda + source_k], modulus)
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n
              ? rns8_center_u8_device(B[source_k * ldb + global_col], modulus)
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
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus, modulus_reciprocal);
  }
}

template <int Modulus>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_u8_native_i32_fixed_modulus_kernel(
        const uint8_t* A,
        const uint8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int accumulate) {
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
              ? rns8_center_u8_fixed_modulus_device<Modulus>(A[global_row * lda + source_k])
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n
              ? rns8_center_u8_fixed_modulus_device<Modulus>(B[source_k * ldb + global_col])
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
    C[row * ldc + col] = rns8_reduce_to_centered_fixed_modulus_device<Modulus>(acc);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_u8_native_a_i8_b_i32_kernel(
        const uint8_t* A,
        const int8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int modulus,
        uint32_t modulus_reciprocal,
        int accumulate) {
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
              ? rns8_center_u8_device(A[global_row * lda + source_k], modulus)
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
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus, modulus_reciprocal);
  }
}

template <int Modulus>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_u8_native_a_i8_b_i32_fixed_modulus_kernel(
        const uint8_t* A,
        const int8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int accumulate) {
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
              ? rns8_center_u8_fixed_modulus_device<Modulus>(A[global_row * lda + source_k])
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
    C[row * ldc + col] = rns8_reduce_to_centered_fixed_modulus_device<Modulus>(acc);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_i8_i32_grouped_prefix_kernel(
        const int8_t* A_base,
        const int8_t* B_base,
        int8_t* C_base,
        int m,
        int n,
        int k_total,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int grouped_prefix,
        int accumulate) {
  const int modulus_index = static_cast<int>(blockIdx.z);
  if (modulus_index >= grouped_prefix) {
    return;
  }
  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k_total) * static_cast<int64_t>(ldb);
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

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_i8_i32_grouped_task_prefix_kernel(
        const int8_t* const* A_ptrs,
        const int8_t* const* B_ptrs,
        int8_t* const* C_ptrs,
        int task_count,
        int m,
        int n,
        int k_total,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int grouped_prefix,
        int accumulate) {
  const int combined_index = static_cast<int>(blockIdx.z);
  const int task_index = combined_index / grouped_prefix;
  const int modulus_index = combined_index - task_index * grouped_prefix;
  if (task_index >= task_count || modulus_index >= grouped_prefix) {
    return;
  }
  const int8_t* A_base = A_ptrs[task_index];
  const int8_t* B_base = B_ptrs[task_index];
  int8_t* C_base = C_ptrs[task_index];
  if (!A_base || !B_base || !C_base) {
    return;
  }
  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k_total) * static_cast<int64_t>(ldb);
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

