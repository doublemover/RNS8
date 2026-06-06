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

