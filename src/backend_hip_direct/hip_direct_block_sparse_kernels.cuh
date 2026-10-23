__global__ void rns8_block_sparse_gemm_kernel(
    const int8_t *a, const uint32_t *ao, const uint32_t *ai, const int8_t *b,
    const uint32_t *bo, const uint32_t *bi, int8_t *c, int m, int n, int k,
    int ldc, int prefix, int tile_cols) {
  __shared__ int8_t at[1024], bt[1024];
  const int lane = threadIdx.x, lr = lane / 16, lc = lane % 16;
  const int64_t rt = blockIdx.x / tile_cols, ct = blockIdx.x % tile_cols;
  const int64_t row = rt * 16 + lr, col = ct * 16 + lc;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  uint32_t x = ao[rt], y = bo[ct], xe = ao[rt + 1], ye = bo[ct + 1];
  int sum = 0;
  while (x < xe && y < ye) {
    if (ai[x] < bi[y]) {
      ++x;
      continue;
    }
    if (bi[y] < ai[x]) {
      ++y;
      continue;
    }
    const uint64_t ap = (uint64_t{x} * prefix + plane) * 1024;
    const uint64_t bp = (uint64_t{y} * prefix + plane) * 1024;
    for (int cell = lane; cell < 1024; cell += 256) {
      at[cell] = a[ap + cell];
      bt[cell] = b[bp + cell];
    }
    __syncthreads();
    const int64_t start = uint64_t{ai[x]} * 64;
    const int extent = k - start < 64 ? static_cast<int>(k - start) : 64;
    for (int q = 0; q < extent; ++q)
      sum += static_cast<int>(at[lr * 64 + q]) * bt[q * 16 + lc];
    // At most 64*128*128 plus a carried residue: no cancellation assumption.
    sum %= modulus;
    __syncthreads();
    ++x;
    ++y;
  }
  if (row < m && col < n) {
    if (sum < 0)
      sum += modulus;
    if (sum >= (modulus + 1) / 2)
      sum -= modulus;
    c[(static_cast<int64_t>(plane) * m + row) * ldc + col] =
        static_cast<int8_t>(sum);
  }
}
