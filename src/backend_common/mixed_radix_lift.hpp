#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_LIFT_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_LIFT_INLINE inline
#endif
namespace rns8::detail {
// Valid canonical mixed-radix digits, byte bases/target >=2, count >=1.
// X=sum(d_i*product_{j<i} m_j). Signed negative values represent X-P,
// so subtract P modulo the new modulus; reducing X alone is not a signed lift.
template <typename Modulus>
RNS8_LIFT_INLINE uint32_t mixed_radix_lift_residue(const uint32_t *digits,
                                                   const Modulus *moduli,
                                                   uint32_t count,
                                                   uint32_t target,
                                                   bool negative) {
  uint32_t residue = 0;
  for (uint32_t i = count; i-- > 0;)
    residue = (residue * moduli[i] + digits[i]) % target;
  if (negative) {
    uint32_t product = 1;
    for (uint32_t i = 0; i < count; ++i)
      product = product * moduli[i] % target;
    residue = (residue + target - product) % target;
  }
  return residue;
}
} // namespace rns8::detail
#undef RNS8_LIFT_INLINE
