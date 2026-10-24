#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
namespace rns8::detail {
struct WideGemmSupport {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  AddressDomain a{},b{},c{};
  uint64_t logical_outputs=0;
};
inline WideGemmSupport query_wide_gemm_support(int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix) {
  WideGemmSupport out;
  if (m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldc<n || !prefix || prefix>20) return out;
  if (!checked_host_plane_address(m,k,lda,prefix,1,out.a) ||
      !checked_host_plane_address(k,n,ldb,prefix,1,out.b) ||
      !checked_host_plane_address(m,n,ldc,prefix,1,out.c)) {
    out.status=RNS8_RANGE_ERROR; return out;
  }
  // Dense logical cells are no larger than the proved padded destination span.
  out.logical_outputs=uint64_t(m)*uint64_t(n)*prefix;
  out.status=RNS8_SUCCESS;
  return out;
}
} // namespace rns8::detail
