#pragma once
#include "mfma_i8_layout.hpp"
#if defined(__HIPCC__)
#define RNS8_F64_LAYOUT_HD __host__ __device__
#else
#define RNS8_F64_LAYOUT_HD
#endif
namespace rns8::detail {
// FP64 output uses interleaved rows, unlike the four-consecutive-row INT32 map.
struct MfmaF64Layout {
  RNS8_F64_LAYOUT_HD static constexpr MfmaCoordinate a(unsigned lane){return {lane%16,lane/16};}
  RNS8_F64_LAYOUT_HD static constexpr MfmaCoordinate b(unsigned lane){return {lane/16,lane%16};}
  RNS8_F64_LAYOUT_HD static constexpr MfmaCoordinate c(unsigned lane,unsigned reg){return {4*reg+lane/16,lane%16};}
  static constexpr bool bijective(){bool aa[64]{},bb[64]{},cc[256]{};
    for(unsigned lane=0;lane<64;++lane){const auto x=a(lane),y=b(lane);if(aa[x.row*4+x.col] || bb[y.row*16+y.col])return false;aa[x.row*4+x.col]=bb[y.row*16+y.col]=true;
      for(unsigned reg=0;reg<4;++reg){const auto z=c(lane,reg);if(cc[z.row*16+z.col])return false;cc[z.row*16+z.col]=true;}}
    for(bool x:aa)if(!x)return false;for(bool x:bb)if(!x)return false;for(bool x:cc)if(!x)return false;return true;
  }
};
static_assert(MfmaF64Layout::bijective());
}
#undef RNS8_F64_LAYOUT_HD
