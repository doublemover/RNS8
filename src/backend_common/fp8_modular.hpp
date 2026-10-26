#pragma once
#include "sha256.hpp"
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_FP8_HD __host__ __device__
#else
#define RNS8_FP8_HD
#endif
namespace rns8::detail::fp8_modular {
inline constexpr uint32_t format_version=1,safe_k=65536;
RNS8_FP8_HD inline bool integer_partial(float value,int& result) {
  if(!(value>=-4194304.0f && value<=4194304.0f))return false;
  const int integer=int(value);if(float(integer)!=value)return false;result=integer;return true;
}
RNS8_FP8_HD constexpr int low(int value){return ((value+136)&15)-8;}
RNS8_FP8_HD constexpr int high(int value){return (value-low(value))/16;}
// CDNA3 E4M3 FNUZ: bias8, unique positive zero. Only the exact integer
// alphabet [-8,8] is admitted; no host floating casts or scale-byte guesses.
RNS8_FP8_HD constexpr uint8_t encode(int value) {
  if(!value)return 0;
  const unsigned magnitude=value<0?unsigned(-value):unsigned(value);
  const unsigned exponent=magnitude>=8?3:magnitude>=4?2:magnitude>=2?1:0;
  return uint8_t((value<0?128:0)|((exponent+8)<<3)|((magnitude<<(3-exponent))-8));
}
constexpr int decode(uint8_t code) {
  if(!code)return 0;
  const int exponent=int((code>>3)&15)-8;
  const int significand=8+(code&7);
  const int magnitude=exponent>=3?significand<<(exponent-3):significand>>(3-exponent);
  return code&128?-magnitude:magnitude;
}
constexpr bool algebra() {
  for(int value=-128;value<=127;++value) {
    if(low(value)<-8 || low(value)>7 || high(value)<-8 || high(value)>8 || low(value)+16*high(value)!=value)return false;
    if(decode(encode(low(value)))!=low(value) || decode(encode(high(value)))!=high(value))return false;
  }
  // Every individual floating product and all intermediate partial sums are
  // integers with |sum| <=2^22, below the 24-bit binary32 integer limit.
  return uint64_t(safe_k)*8*8<(uint64_t{1}<<24) && safe_k%32==0;
}
static_assert(algebra());
inline std::string identity(const std::string& target,const std::string& build) {
  return sha256("fp8-modular-v1;gfx942;E4M3FNUZ.bias8;radix16-balanced-two-digit;four-products;"
      "wave64.16x16x32;fp32-exact-integer-accumulation-required;slab65536;"
      "reduce-each-product-before-weighted-composition;no-default-dispatch;"+target+";"+build);
}
} // namespace rns8::detail::fp8_modular
#undef RNS8_FP8_HD
