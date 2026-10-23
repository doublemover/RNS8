template <typename NativeT>
__global__ void rns8_native_byte_domain_kernel(const NativeT *source, int rows,
                                               int cols, int ld,
                                               unsigned long long *maximum,
                                               unsigned long long *invalid) {
  __shared__ unsigned long long magnitudes[256];
  __shared__ unsigned flags[256];
  const unsigned lane = threadIdx.x;
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * 256 + lane;
  uint64_t magnitude = 0;
  unsigned bad = 0;
  if (cell < static_cast<int64_t>(rows) * cols) {
    const NativeT value = source[(cell / cols) * ld + cell % cols];
    if constexpr (std::is_signed_v<NativeT>) {
      magnitude = value < 0 ? uint64_t{0} - static_cast<uint64_t>(value)
                            : static_cast<uint64_t>(value);
      bad = value < -128 || value > 127;
    } else {
      magnitude = value;
      bad = value > 255;
    }
  }
  magnitudes[lane] = magnitude;
  flags[lane] = bad;
  __syncthreads();
  for (unsigned step = 128; step; step /= 2) {
    if (lane < step) {
      if (magnitudes[lane + step] > magnitudes[lane])
        magnitudes[lane] = magnitudes[lane + step];
      flags[lane] |= flags[lane + step];
    }
    __syncthreads();
  }
  if (!lane) {
    atomicMax(maximum, magnitudes[0]);
    if (flags[0])
      atomicExch(invalid, 1ull);
  }
}

template <typename NativeT>
__global__ __launch_bounds__(32) void rns8_native_byte_wmma_kernel(
    const NativeT *a, const NativeT *b, NativeT *c, int m, int n, int k,
    int lda, int ldb, int ldc) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) &&  \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  using fragment = int __attribute__((ext_vector_type(4)));
  using accumulator = int __attribute__((ext_vector_type(8)));
  const int lane = threadIdx.x;
  const int64_t tile_row = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t tile_col = static_cast<int64_t>(blockIdx.x) * 16;
  const int64_t ar = tile_row + (lane & 15), bc = tile_col + (lane & 15);
  accumulator acc{};
  for (int64_t kk = 0; kk < k; kk += 16) {
    fragment av{}, bv{};
    for (int word = 0; word < 4; ++word) {
      uint32_t ap = 0, bp = 0;
      for (int byte = 0; byte < 4; ++byte) {
        const int64_t q = kk + word * 4 + byte;
        const auto aa = ar < m && q < k ? a[ar * lda + q] : NativeT{};
        const auto bb = bc < n && q < k ? b[q * ldb + bc] : NativeT{};
        ap |= static_cast<uint32_t>(static_cast<uint8_t>(aa)) << (8 * byte);
        bp |= static_cast<uint32_t>(static_cast<uint8_t>(bb)) << (8 * byte);
      }
      av[word] = static_cast<int>(ap);
      bv[word] = static_cast<int>(bp);
    }
    // Host admission proves the entire absolute product sum <= INT32_MAX.
    // K-tail zero padding does not enlarge that bound. Saturation stays off.
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        std::is_signed_v<NativeT>, av, std::is_signed_v<NativeT>, bv, acc,
        false);
  }
  for (int reg = 0; reg < 8; ++reg) {
    const int64_t row = tile_row + 2 * reg + (lane >> 4);
    if (row < m && bc < n)
      c[row * ldc + bc] = static_cast<NativeT>(acc[reg]);
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
#endif
}
