// Feature detection runs in the device compilation, where target-specific
// builtin availability is meaningful. Unsupported code objects never write C.
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32)
#define RNS8_DEVICE_RDNA3_INT4 1
#else
#define RNS8_DEVICE_RDNA3_INT4 0
#endif
__global__ void rns8_int4_capability_kernel(int* supported) {
  *supported = RNS8_DEVICE_RDNA3_INT4;
}

__global__ __launch_bounds__(32) void rns8_finite_int4_wmma_kernel(
    const int8_t* a, const int8_t* b, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldc, int modulus) {
#if RNS8_DEVICE_RDNA3_INT4
  using packed2 = int __attribute__((ext_vector_type(2)));
  using accum8 = int __attribute__((ext_vector_type(8)));
  const int lane = threadIdx.x;
  const int64_t tile_row = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t tile_col = static_cast<int64_t>(blockIdx.x) * 16;
  const int64_t ar = tile_row + (lane & 15), bc = tile_col + (lane & 15);
  accum8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int64_t kk = 0; kk < k; kk += 16) {
    packed2 av = {0, 0}, bv = {0, 0};
    for (int word = 0; word < 2; ++word) {
      uint32_t ap = 0, bp = 0;
      for (int nibble = 0; nibble < 8; ++nibble) {
        const int64_t q = kk + word * 8 + nibble;
        if (q < k && ar < m)
          ap |= rns8::detail::finite_int4_nibble(a[ar * lda + q], modulus) << (4 * nibble);
        if (q < k && bc < n)
          bp |= rns8::detail::finite_int4_nibble(b[q * ldb + bc], modulus) << (4 * nibble);
      }
      av[word] = static_cast<int>(ap);
      bv[word] = static_cast<int>(bp);
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, av, true, bv, acc, false);
    // Every partial is bounded by 65536*64+15, including carried residue.
    // This bound is independent of cancellation and the total K dimension.
    if (((kk + 16) % 65536) == 0)
      for (int reg = 0; reg < 8; ++reg) acc[reg] %= modulus;
  }
  const int64_t col = tile_col + (lane & 15);
  for (int reg = 0; reg < 8; ++reg) {
    const int64_t row = tile_row + 2 * reg + (lane >> 4);
    if (row < m && col < n) {
      int value = acc[reg] % modulus;
      if (value < 0) value += modulus;
      if (value >= (modulus + 1) / 2) value -= modulus;
      c[row * ldc + col] = static_cast<int8_t>(value);
    }
  }
#else
  (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
  (void)lda; (void)ldb; (void)ldc; (void)modulus;
#endif
}
#undef RNS8_DEVICE_RDNA3_INT4
