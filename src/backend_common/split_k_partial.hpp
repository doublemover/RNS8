#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_PARTIAL_HD __host__ __device__
#else
#define RNS8_PARTIAL_HD
#endif
namespace rns8::detail {
// A centered byte is a signed value, not its uint8 bit pattern modulo p.
// In particular 0xff denotes -1 -> p-1, not 255 for odd p.
RNS8_PARTIAL_HD constexpr uint32_t split_k_partial_canonical(uint8_t bits,uint32_t modulus,bool centered) {
  if(!centered)return bits;
  const int32_t value=bits>=128?static_cast<int32_t>(bits)-256:bits;
  return static_cast<uint32_t>(value<0?value+static_cast<int32_t>(modulus):value);
}
RNS8_PARTIAL_HD constexpr int8_t split_k_partial_centered(uint8_t bits,uint32_t modulus,bool centered) {
  const int32_t canonical=static_cast<int32_t>(split_k_partial_canonical(bits,modulus,centered));
  return static_cast<int8_t>(canonical>=static_cast<int32_t>((modulus+1)/2)?canonical-static_cast<int32_t>(modulus):canonical);
}
} // namespace rns8::detail
#undef RNS8_PARTIAL_HD
