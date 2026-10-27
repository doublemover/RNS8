#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
namespace rns8::detail {
struct EncodedBLayout {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t k_words=0,payload_bytes=0,total_bytes=0;
};
inline EncodedBLayout encoded_b_layout(int64_t k,int64_t n,int64_t ldb,uint32_t prefix,uint64_t budget) {
  EncodedBLayout out;
  if(k<=0 || n<=0 || ldb<n || !prefix || prefix>20) return out;
  if(!qualified_i32_matrix_address(k,n,ldb,1,8)) { out.status=RNS8_UNSUPPORTED_BACKEND; return out; }
  out.k_words=((uint64_t(k)+15)/16)*4;
  AddressDomain address;
  if(!checked_host_plane_address(n,out.k_words,out.k_words,prefix,4,address) || address.byte_extent>SIZE_MAX-4) {
    out.status=RNS8_RANGE_ERROR; return out;
  }
  out.payload_bytes=address.byte_extent;
  out.total_bytes=out.payload_bytes+4; // retained capability word, naturally aligned
  out.status=out.total_bytes>budget?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;
  return out;
}
} // namespace rns8::detail
