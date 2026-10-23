// One wave owns a complete output tile and modulus. Only local LDS holds
// packed byte fragments and INT32 fragment stores; no global AB temporary.
__global__ void __launch_bounds__(kRns8WmmaWaveThreads) rocwmma_exact_axpby_kernel(
    const int8_t* a, const int8_t* b, const int8_t* d, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldd, int ldc, int64_t alpha, int64_t beta) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, 16, 16, 16, int8_t, row_major>;
  using FragB = fragment<matrix_b, 16, 16, 16, int8_t, col_major>;
  using FragAcc = fragment<accumulator, 16, 16, 16, int32_t>;
  __shared__ __align__(16) int8_t ap[256], bp[256];
  __shared__ int32_t partial[256];
  constexpr int owned = 256 / kRns8WmmaWaveThreads;
  int32_t residues[owned]{};
  const int lane = threadIdx.x, plane = blockIdx.z;
  const int64_t tile_row = int64_t{blockIdx.y} * 16, tile_col = int64_t{blockIdx.x} * 16;
  const int modulus = rocwmma_default_modulus_device(plane);
  const uint32_t reciprocal = rns8::detail::finite_u8::modulus_reciprocal_u32(modulus);
  for (int64_t begin = 0; begin < k; begin += kRns8SignedSafeInt32KBlock) {
    const int64_t end = begin + kRns8SignedSafeInt32KBlock < k ? begin + kRns8SignedSafeInt32KBlock : k;
    FragAcc acc;
    fill_fragment(acc, 0);
    for (int64_t kk = begin; kk < end; kk += 16) {
      for (int index = lane; index < 256; index += kRns8WmmaWaveThreads) {
        const int64_t row = tile_row + index / 16, col = tile_col + index / 16, q = kk + index % 16;
        ap[index] = row < m && q < end ? a[(int64_t{plane} * m + row) * lda + q] : int8_t{0};
        bp[index] = col < n && q < end ? b[(int64_t{plane} * k + q) * ldb + col] : int8_t{0};
      }
      __syncthreads();
      FragA fa; FragB fb;
      load_matrix_sync(fa, ap, 16);
      load_matrix_sync(fb, bp, 16);
      mma_sync(acc, fa, fb, acc);
      __syncthreads(); // All fragment loads finish before the next LDS write.
    }
    store_matrix_sync(partial, acc, 16, mem_row_major);
    __syncthreads();
    for (int slot = 0; slot < owned; ++slot) {
      const int index = lane + slot * kRns8WmmaWaveThreads;
      // Full signed-byte domain: 65536*128*128 + 128 fits INT32.
      residues[slot] = rns8::detail::finite_u8::reduce_to_centered_i32(
          partial[index] + residues[slot], modulus, reciprocal);
    }
    __syncthreads();
  }
  for (int slot = 0; slot < owned; ++slot) {
    const int index = lane + slot * kRns8WmmaWaveThreads;
    const int64_t row = tile_row + index / 16, col = tile_col + index % 16;
    if (row < m && col < n)
      c[(int64_t{plane} * m + row) * ldc + col] = rns8::detail::centered_residue_axpby(
          static_cast<int8_t>(residues[slot]), d[(int64_t{plane} * m + row) * ldd + col],
          alpha, beta, modulus);
  }
}
