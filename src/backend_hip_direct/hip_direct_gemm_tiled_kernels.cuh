#include "hip_direct_tiled_residue_core.cuh"

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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DynamicTileReducer{modulus,modulus_reciprocal});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8FixedTileReducer<Modulus>{});
}
// === Packed integer-dot fixed-modulus GEMM (not a matrix instruction) ===
// Uses v_dot4_i32_iu8 neg_lo:[1,1,0] (hardware i8 dot4, signed).
// Same tile layout as scalar: a_tile[M][K], b_tile[K][Npadded].
// A: contiguous 32-bit LDS load. B: manual 4-byte gather.
// neg_lo:[1,1,0] works around ROCm 7.1 assembler bug where the
// v_dot4_i32_i8 alias dropped sign-extension neg_lo bits.
template <int Modulus>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_i8_i32_dp4a_fixed_modulus_kernel(
        const int8_t* A, const int8_t* B, int8_t* C,
        int m, int n, int k_offset, int k_block,
        int lda, int ldb, int ldc, int accumulate) {
  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = tile_row + thread_row;
  const int col = tile_col + thread_col;
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::packed>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8FixedTileReducer<Modulus>{});
}


__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_i8_i32_grouped_modulus_kernel(
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
        int modulus,
        uint32_t modulus_reciprocal,
        int accumulate) {
  const int task_index = static_cast<int>(blockIdx.z);
  if (task_index >= task_count) {
    return;
  }
  const int8_t* A = A_ptrs[task_index];
  const int8_t* B = B_ptrs[task_index];
  int8_t* C = C_ptrs[task_index];
  if (!A || !B || !C) {
    return;
  }
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
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DynamicTileReducer{modulus,modulus_reciprocal});
}

constexpr int kRns8HipGemvN1Threads = 256;
constexpr int kRns8HipGemvSmallNMaxN = 8;

__global__ void __launch_bounds__(kRns8HipGemvN1Threads)
    rns8_ring_gemv_n1_i8_i32_grouped_prefix_kernel(
        const int8_t* A_base,
        const int8_t* B_base,
        int8_t* C_base,
        int m,
        int k,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int grouped_prefix,
        int accumulate) {
  const int row = static_cast<int>(blockIdx.x);
  const int modulus_index = static_cast<int>(blockIdx.y);
  if (row >= m || modulus_index >= grouped_prefix) {
    return;
  }

  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * static_cast<int64_t>(ldb);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* A = A_base + a_plane_offset;
  const int8_t* B = B_base + b_plane_offset;
  int8_t* C = C_base + c_plane_offset;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));

  int32_t thread_acc = 0;
  for (int kk = static_cast<int>(threadIdx.x); kk < k_block; kk += static_cast<int>(blockDim.x)) {
    const int source_k = k_offset + kk;
    thread_acc +=
        static_cast<int32_t>(A[row * lda + source_k]) * static_cast<int32_t>(B[source_k * ldb]);
  }

  __shared__ int32_t partials[kRns8HipGemvN1Threads];
  partials[threadIdx.x] = thread_acc;
  __syncthreads();

  for (int stride = kRns8HipGemvN1Threads / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int32_t acc = partials[0];
    if (accumulate) {
      acc += static_cast<int32_t>(C[row * ldc]);
    }
    C[row * ldc] =
        rns8_reduce_to_centered_default_modulus_fixed_device(acc, modulus_index, modulus, modulus_reciprocal);
  }
}

__global__ void __launch_bounds__(kRns8HipGemvN1Threads)
    rns8_ring_gemv_small_n_i8_i32_grouped_prefix_kernel(
        const int8_t* A_base,
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
        int grouped_prefix,
        int accumulate) {
  const int row = static_cast<int>(blockIdx.x);
  const int modulus_index = static_cast<int>(blockIdx.y);
  if (row >= m || modulus_index >= grouped_prefix || n <= 0 || n > kRns8HipGemvSmallNMaxN) {
    return;
  }

  const int64_t a_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(lda);
  const int64_t b_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * static_cast<int64_t>(ldb);
  const int64_t c_plane_offset =
      static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * static_cast<int64_t>(ldc);
  const int8_t* A = A_base + a_plane_offset;
  const int8_t* B = B_base + b_plane_offset;
  int8_t* C = C_base + c_plane_offset;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));

  int32_t thread_acc[kRns8HipGemvSmallNMaxN] = {};
  for (int kk = static_cast<int>(threadIdx.x); kk < k_block; kk += static_cast<int>(blockDim.x)) {
    const int source_k = k_offset + kk;
    const int32_t a_value = static_cast<int32_t>(A[row * lda + source_k]);
    for (int col = 0; col < kRns8HipGemvSmallNMaxN; ++col) {
      if (col < n) {
        thread_acc[col] += a_value * static_cast<int32_t>(B[source_k * ldb + col]);
      }
    }
  }

  __shared__ int32_t partials[kRns8HipGemvN1Threads][kRns8HipGemvSmallNMaxN];
  for (int col = 0; col < kRns8HipGemvSmallNMaxN; ++col) {
    partials[threadIdx.x][col] = thread_acc[col];
  }
  __syncthreads();

  for (int stride = kRns8HipGemvN1Threads / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      for (int col = 0; col < kRns8HipGemvSmallNMaxN; ++col) {
        partials[threadIdx.x][col] += partials[threadIdx.x + stride][col];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int col = 0; col < kRns8HipGemvSmallNMaxN; ++col) {
      if (col < n) {
        int32_t acc = partials[0][col];
        if (accumulate) {
          acc += static_cast<int32_t>(C[row * ldc + col]);
        }
        C[row * ldc + col] =
            rns8_reduce_to_centered_default_modulus_fixed_device(acc, modulus_index, modulus, modulus_reciprocal);
      }
    }
  }
}

template <int Modulus>
__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_finite_ring_gemm_i8_i32_grouped_fixed_modulus_kernel(
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
        int accumulate) {
  const int task_index = static_cast<int>(blockIdx.z);
  if (task_index >= task_count) {
    return;
  }
  const int8_t* A = A_ptrs[task_index];
  const int8_t* B = B_ptrs[task_index];
  int8_t* C = C_ptrs[task_index];
  if (!A || !B || !C) {
    return;
  }
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
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8FixedTileReducer<Modulus>{});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::NativeByteResidueTileInput<0>{A,lda,uint32_t(modulus)},rns8::detail::NativeByteResidueTileInput<0>{B,ldb,uint32_t(modulus)},C,Rns8DynamicTileReducer{modulus,modulus_reciprocal});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::NativeByteResidueTileInput<Modulus>{A,lda,uint32_t(Modulus)},rns8::detail::NativeByteResidueTileInput<Modulus>{B,ldb,uint32_t(Modulus)},C,Rns8FixedTileReducer<Modulus>{});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::NativeByteResidueTileInput<0>{A,lda,uint32_t(modulus)},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DynamicTileReducer{modulus,modulus_reciprocal});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::NativeByteResidueTileInput<Modulus>{A,lda,uint32_t(Modulus)},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8FixedTileReducer<Modulus>{});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::selected_default_moduli>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DefaultLadderTileReducer{modulus_index,modulus,modulus_reciprocal});
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_i8_i32_region_kernel(
        const int8_t* A,
        const int8_t* B,
        int8_t* C,
        int m,
        int n,
        int k_total,
        int k_offset,
        int k_block,
        int lda,
        int ldb,
        int ldc,
        int region_row_offset,
        int region_col_offset,
        int region_rows,
        int region_cols,
        int modulus,
        uint32_t modulus_reciprocal,
        int accumulate) {
  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int region_tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int region_tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = region_row_offset + region_tile_row + thread_row;
  const int col = region_col_offset + region_tile_col + thread_col;
  const bool output_active =
      row < m && col < n && region_tile_row + thread_row < region_rows && region_tile_col + thread_col < region_cols;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,region_row_offset+region_tile_row,region_col_offset+region_tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DynamicTileReducer{modulus,modulus_reciprocal});
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN)
    rns8_ring_gemm_i8_i32_grouped_prefix_region_kernel(
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
        int region_row_offset,
        int region_col_offset,
        int region_rows,
        int region_cols,
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
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint32_t modulus_reciprocal =
      static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));
  const int8_t* A = A_base + a_plane_offset;
  const int8_t* B = B_base + b_plane_offset;
  int8_t* C = C_base + c_plane_offset;

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int region_tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int region_tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = region_row_offset + region_tile_row + thread_row;
  const int col = region_col_offset + region_tile_col + thread_col;
  const bool output_active =
      row < m && col < n && region_tile_row + thread_row < region_rows && region_tile_col + thread_col < region_cols;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,region_row_offset+region_tile_row,region_col_offset+region_tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DefaultLadderTileReducer{modulus_index,modulus,modulus_reciprocal});
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
  const bool output_active = row < m && col < n;
  rns8_tiled_residue_body<rns8::detail::ResidueTileDot::configured>(
      a_tile,b_tile,{m,n,tile_row,tile_col,k_offset,k_block,ldc,row,col,output_active,accumulate},
      rns8::detail::CenteredResidueTileInput{A,lda},rns8::detail::CenteredResidueTileInput{B,ldb},C,Rns8DefaultLadderTileReducer{modulus_index,modulus,modulus_reciprocal});
}

