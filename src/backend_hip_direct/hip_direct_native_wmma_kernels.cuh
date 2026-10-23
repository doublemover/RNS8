#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define RNS8_DEVICE_NATIVE_WMMA 1
#else
#define RNS8_DEVICE_NATIVE_WMMA 0
#endif
__global__ void rns8_native_wmma_capability_kernel(int* supported) {
  *supported = RNS8_DEVICE_NATIVE_WMMA;
}

// Each wave owns one output tile and Group consecutive default-basis planes.
// Native loads are outside the channel loop; channels have separate encoded
// fragments and accumulators. No INT32 or packed-input global workspace exists.
template <typename NativeT, int Group>
__global__ __launch_bounds__(32) void rns8_native_grouped_wmma_kernel(
    const NativeT* a, const NativeT* b, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldc, int prefix) {
#if RNS8_DEVICE_NATIVE_WMMA
  using fragment = int __attribute__((ext_vector_type(4)));
  using accumulator = int __attribute__((ext_vector_type(8)));
  const int lane = threadIdx.x, first_plane = blockIdx.z * Group;
  const int64_t tile_row = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t tile_col = static_cast<int64_t>(blockIdx.x) * 16;
  const int64_t ar = tile_row + (lane & 15), bc = tile_col + (lane & 15);
  accumulator acc[Group]{};
  int moduli[Group]{};
  for (int channel = 0; channel < Group; ++channel)
    if (first_plane + channel < prefix)
      moduli[channel] = rns8_default_moduli_device[first_plane + channel];
  for (int64_t kk = 0; kk < k; kk += 16) {
    fragment av[Group]{}, bv[Group]{};
    for (int word = 0; word < 4; ++word) {
      uint32_t ap[Group]{}, bp[Group]{};
      for (int byte = 0; byte < 4; ++byte) {
        const int64_t q = kk + word * 4 + byte;
        const NativeT raw_a = ar < m && q < k ? a[ar * lda + q] : NativeT{};
        const NativeT raw_b = bc < n && q < k ? b[q * ldb + bc] : NativeT{};
        for (int channel = 0; channel < Group; ++channel) {
          const int plane = first_plane + channel;
          if (plane >= prefix) continue;
          ap[channel] |= static_cast<uint32_t>(static_cast<uint8_t>(
              rns8_center_native_operand_device(raw_a, plane, moduli[channel]))) << (8 * byte);
          bp[channel] |= static_cast<uint32_t>(static_cast<uint8_t>(
              rns8_center_native_operand_device(raw_b, plane, moduli[channel]))) << (8 * byte);
        }
      }
      for (int channel = 0; channel < Group; ++channel) {
        av[channel][word] = static_cast<int>(ap[channel]);
        bv[channel][word] = static_cast<int>(bp[channel]);
      }
    }
    for (int channel = 0; channel < Group; ++channel) {
      if (first_plane + channel >= prefix) continue;
      acc[channel] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
          true, av[channel], true, bv[channel], acc[channel], false);
      // |every partial| <= 65536*128*128+255 < INT32_MAX.
      // Signed remainder carries at most 255 into the next safe slab.
      if (((kk + 16) % 65536) == 0)
        for (int reg = 0; reg < 8; ++reg) acc[channel][reg] %= moduli[channel];
    }
  }
  const int64_t col = tile_col + (lane & 15);
  for (int channel = 0; channel < Group; ++channel) {
    const int plane = first_plane + channel;
    if (plane >= prefix) continue;
    const int modulus = moduli[channel];
    for (int reg = 0; reg < 8; ++reg) {
      const int64_t row = tile_row + 2 * reg + (lane >> 4);
      if (row < m && col < n) {
        int value = acc[channel][reg] % modulus;
        if (value < 0) value += modulus;
        if (value >= (modulus + 1) / 2) value -= modulus;
        c[(static_cast<int64_t>(plane) * m + row) * ldc + col] = static_cast<int8_t>(value);
      }
    }
  }
#else
  (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
  (void)lda; (void)ldb; (void)ldc; (void)prefix;
#endif
}
#undef RNS8_DEVICE_NATIVE_WMMA
