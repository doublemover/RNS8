#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_MFMA_LAYOUT_HD __host__ __device__
#else
#define RNS8_MFMA_LAYOUT_HD
#endif
namespace rns8::detail {
// Single-block wave64 16x16 MFMA: CDNA2 K16, CDNA3 K32, CDNA4 K64.
// These are distinct physical input widths, never RDNA wave32 replicas.
struct MfmaCoordinate {uint32_t row,col;};
template<unsigned K> struct MfmaI8Layout {
  static_assert(K==16 || K==32 || K==64);
  static constexpr unsigned wave=64,rows=16,cols=16,k=K,input_bytes_per_lane=K/4,accumulators_per_lane=4;
  RNS8_MFMA_LAYOUT_HD static constexpr MfmaCoordinate a(unsigned lane,unsigned byte){return {lane&15,(lane>>4)*input_bytes_per_lane+byte};}
  RNS8_MFMA_LAYOUT_HD static constexpr MfmaCoordinate b(unsigned lane,unsigned byte){return {(lane>>4)*input_bytes_per_lane+byte,lane&15};}
  RNS8_MFMA_LAYOUT_HD static constexpr MfmaCoordinate c(unsigned lane,unsigned reg){return {(lane>>4)*4+reg,lane&15};}
  static constexpr bool bijective(){
    bool aa[16*K]{},bb[K*16]{},cc[256]{};
    for(unsigned lane=0;lane<wave;++lane){for(unsigned byte=0;byte<input_bytes_per_lane;++byte){const auto x=a(lane,byte),y=b(lane,byte);if(x.row>=16 || x.col>=K || y.row>=K || y.col>=16 || aa[x.row*K+x.col] || bb[y.row*16+y.col])return false;aa[x.row*K+x.col]=bb[y.row*16+y.col]=true;}
      for(unsigned reg=0;reg<4;++reg){const auto z=c(lane,reg);if(z.row>=16 || z.col>=16 || cc[z.row*16+z.col])return false;cc[z.row*16+z.col]=true;}}
    for(bool x:aa)if(!x)return false;for(bool x:bb)if(!x)return false;for(bool x:cc)if(!x)return false;return true;
  }
};
static_assert(MfmaI8Layout<16>::bijective());static_assert(MfmaI8Layout<32>::bijective());static_assert(MfmaI8Layout<64>::bijective());
}

#undef RNS8_MFMA_LAYOUT_HD
