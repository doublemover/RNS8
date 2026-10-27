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
struct Coordinate{unsigned row,col;};
struct Rdna4Layout {
  // The official FP8 policy splits K into contiguous eight-byte lane groups.
  // Its two packed registers differ from FP16's four-register interleave.
  RNS8_FP8_HD static constexpr Coordinate a(unsigned lane,unsigned byte){return {lane&15,(lane>>4)*8+byte};}
  RNS8_FP8_HD static constexpr Coordinate b(unsigned lane,unsigned byte){const auto v=a(lane,byte);return {v.col,v.row};}
  RNS8_FP8_HD static constexpr Coordinate c(unsigned lane,unsigned reg){return {(lane>>4)*8+reg,lane&15};}
  static constexpr bool bijective(){
    bool aa[256]{},bb[256]{},cc[256]{};
    for(unsigned lane=0;lane<32;++lane)for(unsigned element=0;element<8;++element){
      const auto x=a(lane,element),y=b(lane,element),z=c(lane,element);
      if(x.row>=16 || x.col>=16 || y.row>=16 || y.col>=16 || z.row>=16 || z.col>=16 || aa[x.row*16+x.col] || bb[y.row*16+y.col] || cc[z.row*16+z.col])return false;
      aa[x.row*16+x.col]=bb[y.row*16+y.col]=cc[z.row*16+z.col]=true;
    }
    return true;
  }
};
static_assert(Rdna4Layout::bijective());
RNS8_FP8_HD inline bool integer_partial(float value,int& result,int bound=4194304) {
  if(!(value>=-float(bound) && value<=float(bound)))return false;
  const int integer=int(value);if(float(integer)!=value)return false;result=integer;return true;
}
RNS8_FP8_HD constexpr int low(int value){return ((value+136)&15)-8;}
RNS8_FP8_HD constexpr int high(int value){return (value-low(value))/16;}
// Exact E4M3 integers: FNUZ uses bias8, OCP bias7. Four-product digits
// use [-8,8]; the three-product alphabets extend to [-15,15].
// No host floating casts or scale-byte guesses.
RNS8_FP8_HD constexpr uint8_t encode(int value,bool ocp=false) {
  if(!value)return 0;
  const unsigned magnitude=value<0?unsigned(-value):unsigned(value);
  const unsigned exponent=magnitude>=16?4:magnitude>=8?3:magnitude>=4?2:magnitude>=2?1:0;
  const unsigned significand=exponent<=3?magnitude<<(3-exponent):magnitude>>(exponent-3);
  return uint8_t((value<0?128:0)|((exponent+(ocp?7:8))<<3)|(significand-8));
}
constexpr int decode(uint8_t code,bool ocp=false) {
  if(!code)return 0;
  const int exponent=int((code>>3)&15)-(ocp?7:8);
  const int significand=8+(code&7);
  const int magnitude=exponent>=3?significand<<(exponent-3):significand>>(3-exponent);
  return code&128?-magnitude:magnitude;
}
// Ozaki-II three-product identity: x=16*h+l, s=h+l. Each component is
// an exactly representable integer in either E4M3 alphabet, including -15.
RNS8_FP8_HD constexpr int three_high(int x){return x<0?-((-x+15)/16):(x+15)/16;}
RNS8_FP8_HD constexpr int component(int x,unsigned term,bool three){
  return three?(term==0?three_high(x):term==1?x-16*three_high(x):x-15*three_high(x)):
      (term==0?low(x):high(x));
}
RNS8_FP8_HD constexpr int weight(unsigned term,bool three){return three?(term==0?240:term==1?-15:16):(term==0?1:term==3?256:16);}
RNS8_FP8_HD constexpr int partial_bound(unsigned term,bool three){return int(safe_k)*(three?(term==0?64:term==1?225:196):64);}
constexpr bool three_algebra(){
  for(int x=-128;x<=127;++x) {
    if(16*component(x,0,true)+component(x,1,true)!=x)return false;
    for(unsigned term=0;term<3;++term)for(bool ocp:{false,true})
      if(decode(encode(component(x,term,true),ocp),ocp)!=component(x,term,true))return false;
  }
  return uint64_t(safe_k)*225<(uint64_t{1}<<24);
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
static_assert(three_algebra());
inline std::string identity(const std::string& target,const std::string& build,uint32_t format=1,uint32_t method=0) {
  return sha256(std::string("fp8-modular-v2;")+(format==1?"gfx942;E4M3FNUZ.bias8;wave64.16x16x32;":"gfx1201;E4M3OCP.bias7;wave32.16x16x16;")+
      (method>=2?"ceil-radix16-three-product;240,-15,16;":"balanced-radix16-four-product;1,16,16,256;")+
      "fp32-exact-integer-accumulation-required;slab65536;"
      "reduce-each-product-before-weighted-composition;no-default-dispatch;"+target+";"+build);
}
} // namespace rns8::detail::fp8_modular
#undef RNS8_FP8_HD
