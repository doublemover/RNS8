#pragma once

// Included in the existing kernel namespace after device reducers/constants.
// Entrypoints retain explicit byte-LDS arrays and uniform admission guards.
struct Rns8TiledResidueProblem {
  int m, n, tile_row, tile_col, k_offset, k_block, ldc, row, col;
  bool output_active;
  int accumulate;
};
struct Rns8DynamicTileReducer {
  int modulus;
  uint32_t reciprocal;
  __device__ __forceinline__ int8_t operator()(int32_t value) const {
    return rns8_reduce_to_centered_device(value, modulus, reciprocal);
  }
};
template <int Modulus> struct Rns8FixedTileReducer {
  static constexpr int modulus = Modulus;
  __device__ __forceinline__ int8_t operator()(int32_t value) const {
    return rns8_reduce_to_centered_fixed_modulus_device<Modulus>(value);
  }
};
struct Rns8DefaultLadderTileReducer {
  int modulus_index, modulus;
  uint32_t reciprocal;
  __device__ __forceinline__ int8_t operator()(int32_t value) const {
    return rns8_reduce_to_centered_default_modulus_fixed_device(value, modulus_index, modulus,
                                                                reciprocal);
  }
};

template <rns8::detail::ResidueTileDot DotPolicy, class InputA, class InputB, class Reducer>
__device__ __forceinline__ void
rns8_tiled_residue_body(int8_t (&a_tile)[kRns8HipTileM][kRns8HipTileK],
                        int8_t (&b_tile)[kRns8HipTileK][kRns8HipTileNPadded],
                        Rns8TiledResidueProblem problem, InputA a, InputB b, int8_t* output,
                        Reducer reduce) {
  const int thread_row = int(threadIdx.y), thread_col = int(threadIdx.x);
  const int lane = thread_row * int(blockDim.x) + thread_col;
  const int threads = int(blockDim.x * blockDim.y);
  int32_t acc = 0;
  for (int tile_k = 0; tile_k < problem.k_block; tile_k += kRns8HipTileK) {
    const int remaining = problem.k_block - tile_k;
    const int extent = remaining < kRns8HipTileK ? remaining : kRns8HipTileK;
    rns8::detail::load_residue_tile<kRns8HipTileM, kRns8HipTileN, kRns8HipTileK,
                                    kRns8HipTileNPadded>(
        a_tile, b_tile,
        {problem.m, problem.n, problem.tile_row, problem.tile_col, problem.k_offset}, tile_k,
        extent, lane, threads, a, b);
    // All admitted CTA threads participate, including inactive output lanes.
    __syncthreads();
    if (problem.output_active)
      acc = rns8::detail::accumulate_residue_tile(
          &a_tile[thread_row][0], &b_tile[0][thread_col], kRns8HipTileNPadded, extent, acc,
          rns8::detail::residue_tile_packed<DotPolicy>(reduce.modulus));
    __syncthreads();
  }
  if (problem.output_active) {
    if (problem.accumulate)
      acc += int32_t(output[problem.row * problem.ldc + problem.col]);
    output[problem.row * problem.ldc + problem.col] = reduce(acc);
  }
}
