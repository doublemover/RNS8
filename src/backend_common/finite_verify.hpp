#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_VERIFY_INLINE __host__ __device__ inline
#else
#define RNS8_VERIFY_INLINE inline
#endif
namespace rns8::detail::finite_verify {
// Keep this implementation and its constants stable: public sampler version 1.
inline uint8_t next_bit(uint64_t& state) {
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return static_cast<uint8_t>((z ^ (z >> 31)) >> 63);
}
RNS8_VERIFY_INLINE uint32_t decode(int8_t value, uint32_t modulus, uint32_t& flags) {
  const int lo = -static_cast<int>(modulus / 2);
  const int hi = static_cast<int>((modulus - 1) / 2);
  if (value < lo || value > hi) flags |= 2;
  // Even malformed bytes map safely; the entire trial reports invalid storage.
  return static_cast<uint32_t>((int(value) + int(modulus) * 64) % int(modulus));
}
// Reduce after EVERY term: accumulator <= 255 + 255*255 = 65280.
// Address products have been checked by the public finite storage admission.
inline uint32_t cpu_trial(const int8_t* a, const int8_t* b, const int8_t* c,
    uint64_t m, uint64_t n, uint64_t k, uint32_t q, const uint8_t* r, uint32_t* br) {
  uint32_t flags = 0;
  for (uint64_t j = 0; j < k; ++j) {
    uint32_t value = 0;
    for (uint64_t col = 0; col < n; ++col)
      value = (value + decode(b[j*n+col], q, flags) * r[col]) % q;
    br[j] = value;
  }
  for (uint64_t row = 0; row < m; ++row) {
    uint32_t abr = 0, cr = 0;
    for (uint64_t j = 0; j < k; ++j)
      abr = (abr + decode(a[row*k+j], q, flags) * br[j]) % q;
    for (uint64_t col = 0; col < n; ++col)
      cr = (cr + decode(c[row*n+col], q, flags) * r[col]) % q;
    if (abr != cr) flags |= 1;
  }
  return flags;
}
}
#undef RNS8_VERIFY_INLINE
