#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_TDM_HD __host__ __device__
#else
#define RNS8_TDM_HD
#endif
namespace rns8::detail {
struct NativeTdmProblem {
  uint64_t m=0,n=0,k=0,lda=0,ldb=0,ldc=0;
  uint32_t prefix=0,group=1,mode=0,cluster=1,compute=0,cluster_axis=0;
  bool signed_input=true;
};
struct NativeTdmResources {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t native_a_bytes=0,native_b_bytes=0,output_bytes=0,grid_blocks=0;
  uint32_t lds_bytes=32768,status_bytes=4;
};
inline NativeTdmResources query_native_tdm(const NativeTdmProblem& p) {
  NativeTdmResources r;AddressDomain a,b,c;
  if(!p.m || !p.n || !p.k || !p.prefix || p.prefix>20 || p.mode>2 || p.compute>1 || p.cluster_axis>1 ||
      (p.cluster_axis && p.mode!=2) ||
      (p.group!=1 && p.group!=2 && p.group!=4) ||
      (p.cluster!=1 && p.cluster!=2 && p.cluster!=4) || (p.mode==2)!=(p.cluster>1) ||
      !checked_host_plane_address(p.m,p.k,p.lda,1,8,a) ||
      !checked_host_plane_address(p.k,p.n,p.ldb,1,8,b) ||
      !checked_host_plane_address(p.m,p.n,p.ldc,p.prefix,1,c))return r;
  const uint64_t rows=(p.m+15)/16,cols=(p.n+15)/16,groups=(p.prefix+p.group-1)/p.group;
  if(rows>65535 || cols>UINT32_MAX/p.cluster || p.k>UINT32_MAX || p.lda>UINT32_MAX || p.ldb>UINT32_MAX)return r;
  r.native_a_bytes=a.byte_extent;r.native_b_bytes=b.byte_extent;r.output_bytes=c.byte_extent;
  r.grid_blocks=p.cluster_axis?rows*((cols+p.cluster-1)/p.cluster)*p.cluster*groups:
      rows*cols*p.cluster*((groups+p.cluster-1)/p.cluster);r.status=RNS8_SUCCESS;return r;
}
// Explicit gfx1250 wave32 IU8 16x16x64 fragment model. CK's gfx12
// k_per_blk=32, subgroup=16 and contiguous eight-row accumulator mapping;
// model coverage is independent of actual emitted-ISA/device qualification.
struct TdmWmmaLayout {
  struct Cell{unsigned row,col;};
  RNS8_TDM_HD static constexpr Cell a(unsigned lane,unsigned byte){return {lane%16,(lane/16)*32+byte};}
  RNS8_TDM_HD static constexpr Cell b(unsigned lane,unsigned byte){return {(lane/16)*32+byte,lane%16};}
  RNS8_TDM_HD static constexpr Cell c(unsigned lane,unsigned reg){return {(lane/16)*8+reg,lane%16};}
};
struct TdmDescriptor2D {uint32_t group0[4]{},group1[8]{};};
// CDNA5 ISA, tensor D# groups0/1. 64-bit elements, ordinary unpadded 2D,
// no descriptor iteration or atomic-barrier side effect. Stride is ELEMENTS.
RNS8_TDM_HD constexpr bool make_native_tdm_descriptor(uint64_t global,uint32_t local,
    uint32_t tensor_rows,uint32_t tensor_cols,uint32_t tile_rows,uint32_t tile_cols,
    uint64_t stride,uint32_t multicast,TdmDescriptor2D& d) {
  if(global>>57 || (global&7) || (local&7) || local>=65536 || !tensor_rows || !tensor_cols ||
      !tile_rows || tile_rows>65535 || !tile_cols || tile_cols>65535 || stride<tensor_cols || stride>>48 ||
      multicast>15 || uint64_t(tile_rows)*tile_cols*8>65536-local)return false;
  d={};d.group0[0]=1;d.group0[1]=local;d.group0[2]=uint32_t(global);d.group0[3]=uint32_t(global>>32)|(2u<<30);
  d.group1[0]=(3u<<16)|multicast;
  d.group1[1]=tensor_cols<<16;
  d.group1[2]=(tensor_cols>>16)|(tensor_rows<<16);
  d.group1[3]=(tensor_rows>>16)|(tile_cols<<16);d.group1[4]=tile_rows;
  d.group1[5]=uint32_t(stride);d.group1[6]=uint32_t(stride>>32);return true;
}
// Host logical address oracle, independent of packed descriptor bit fields.
inline bool native_tdm_source(uint32_t row,uint32_t col,uint32_t valid_rows,uint32_t valid_cols,
    uint64_t stride,uint64_t& index) {
  if(row>=valid_rows || col>=valid_cols)return false;
  if(row && stride>(UINT64_MAX-col)/row)return false;
  index=uint64_t(row)*stride+col;
  return true;
}
} // namespace rns8::detail
#undef RNS8_TDM_HD
