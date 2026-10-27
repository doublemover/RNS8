#pragma once
#include "fp8_modular.hpp"
#include <array>
#if defined(__HIPCC__)
#define RNS8_MX_HD __host__ __device__
#else
#define RNS8_MX_HD
#endif
namespace rns8::detail::mxfp_modular {
// CDNA4 f8f6f4 selectors: E4M3=0, E2M3=2, E3M2=3, E2M1=4.
// Each digit alphabet is an integer subset of its named floating format.
struct Format {unsigned bits,mantissa,bias,selector,digits,radix_bits;int min_scale,max_scale;};
RNS8_MX_HD constexpr Format format(unsigned selector) {
  return selector==4?Format{4,1,1,4,4,2,-1,1}:
      selector==2?Format{6,3,1,2,3,3,0,3}:
      selector==3?Format{6,2,3,3,3,3,-2,4}:Format{8,3,7,0,2,4,-5,9};
}
struct Pair {unsigned a,b;};
RNS8_MX_HD constexpr Pair pair(unsigned id) {
  return id==3?Pair{4,4}:id==4?Pair{2,2}:id==5?Pair{3,3}:id==6?Pair{0,0}:
      id==7?Pair{0,4}:id==8?Pair{4,0}:id==9?Pair{2,4}:Pair{4,2};
}
RNS8_MX_HD constexpr unsigned terms(unsigned id){const auto p=pair(id);return format(p.a).digits*format(p.b).digits;}
RNS8_MX_HD constexpr int digit(int value,unsigned index,unsigned selector) {
  const auto f=format(selector);const int radix=1<<f.radix_bits;
  for(unsigned i=0;i<index;++i){const int low=((value+128*radix+radix/2)&(radix-1))-radix/2;value=(value-low)/radix;}
  return index+1==f.digits?value:((value+128*radix+radix/2)&(radix-1))-radix/2;
}
RNS8_MX_HD constexpr int weight(unsigned term,unsigned id) {
  const auto p=pair(id);const auto a=format(p.a),b=format(p.b);
  return 1<<((term/b.digits)*a.radix_bits+(term%b.digits)*b.radix_bits);
}
RNS8_MX_HD constexpr uint8_t encode(int value,unsigned selector,int scale) {
  if(!value)return 0;
  const auto f=format(selector);const unsigned mag=value<0?unsigned(-value):unsigned(value);
  const unsigned log=mag>=8?3:mag>=4?2:mag>=2?1:0;
  const int exponent=int(log)-scale;
  const unsigned payload=exponent>=1-int(f.bias)?
      (unsigned(exponent+int(f.bias))<<f.mantissa)|((mag<<(f.mantissa-log))-(1u<<f.mantissa)):
      mag<<unsigned(int(f.bias+f.mantissa)-1-scale);
  return uint8_t((value<0?1u<<(f.bits-1):0)|payload);
}
// Dyadic reference decoder; value=numerator*2^exponent. No floating conversion.
struct Dyadic {int numerator,exponent;};
RNS8_MX_HD constexpr Dyadic decode(uint8_t code,unsigned selector,int scale) {
  const auto f=format(selector);const unsigned fraction=code&((1u<<f.mantissa)-1);
  const unsigned field=(code>>f.mantissa)&((1u<<(f.bits-f.mantissa-1))-1);
  const int magnitude=field?int((1u<<f.mantissa)+fraction):int(fraction);
  return {(code>>(f.bits-1))?-magnitude:magnitude,
      int(field?field:1)-int(f.bias)-int(f.mantissa)+scale};
}
struct Layout {
  RNS8_MX_HD static constexpr fp8_modular::Coordinate a(unsigned lane,unsigned item,unsigned selector=4){return {lane&15,selector==0?(lane>>4)*16+(item/16)*64+item%16:(lane>>4)*32+item};}
  RNS8_MX_HD static constexpr fp8_modular::Coordinate b(unsigned lane,unsigned item,unsigned selector=4){const auto v=a(lane,item,selector);return {v.col,v.row};}
  RNS8_MX_HD static constexpr fp8_modular::Coordinate c(unsigned lane,unsigned reg){return {(lane>>4)*4+reg,lane&15};}
  static constexpr bool bijective(unsigned selector=4){
    bool a_seen[2048]{},b_seen[2048]{},c_seen[256]{};
    for(unsigned lane=0;lane<64;++lane) {
      for(unsigned item=0;item<32;++item){const auto x=a(lane,item,selector),y=b(lane,item,selector);
        if(a_seen[x.row*128+x.col] || b_seen[y.row*16+y.col])return false;
        a_seen[x.row*128+x.col]=b_seen[y.row*16+y.col]=true;}
      for(unsigned reg=0;reg<4;++reg){const auto x=c(lane,reg);if(c_seen[x.row*16+x.col])return false;c_seen[x.row*16+x.col]=true;}
    }
    return true;
  }
};
static_assert(Layout::bijective(0) && Layout::bijective(2) && Layout::bijective(3) && Layout::bijective(4));
constexpr bool algebra() {
  for(unsigned selector:{0u,2u,3u,4u}) {
    const auto f=format(selector);
    for(int value=-128;value<=127;++value) {
      int reconstructed=0;
      for(unsigned d=0;d<f.digits;++d) {
        const int v=digit(value,d,selector);reconstructed+=v*(1<<(d*f.radix_bits));
        for(int scale=f.min_scale;scale<=f.max_scale;++scale) {
          const auto x=decode(encode(v,selector,scale),selector,scale);
          if(x.exponent>=0){if(x.numerator*(1<<x.exponent)!=v)return false;}
          else if(x.numerator!=v*(1<<-x.exponent))return false;
        }
      }
      if(reconstructed!=value)return false;
    }
  }
  return true;
}
inline std::string identity(unsigned id,const std::string& build,const std::string& scales) {
  return sha256("mxfp-modular-v1;gfx950;16x16x128;independent-A-B-selectors;balanced-integer-digits;"
      "E8M0.bias127;inverse-physical-encoding;reset-every128;integer-reduce-before-weight;"+
      std::to_string(id)+";"+build+";"+scales);
}
}
#undef RNS8_MX_HD
