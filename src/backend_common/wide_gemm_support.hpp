#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
namespace rns8::detail {
struct WideGemmSupport {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  AddressDomain a{},b{},c{};
  uint64_t logical_outputs=0;
};
enum class FiniteGemmAddressWidth : uint32_t { signed32=0,unsigned64=1 };
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
inline WideGemmSupport query_finite_indexed_window_support(int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint16_t modulus,int64_t first_row,int64_t rows,
    FiniteGemmAddressWidth width) {
  WideGemmSupport out;
  if(modulus<2 || modulus>256 || m<=0 || first_row<0 || first_row>=m || rows<=0 || rows>m-first_row ||
      (width!=FiniteGemmAddressWidth::signed32 && width!=FiniteGemmAddressWidth::unsigned64))return out;
  out=query_wide_gemm_support(m,n,k,lda,ldb,ldc,1);
  if(out.status!=RNS8_SUCCESS)return out;
  if(width==FiniteGemmAddressWidth::signed32 && !qualified_i32_gemm_address(m,n,k,lda,ldb,ldc)) {
    out.status=RNS8_UNSUPPORTED_BACKEND;return out;
  }
  // The complete C span proves this sub-window multiplication fits too.
  out.logical_outputs=uint64_t(rows)*uint64_t(n);
  return out;
}
} // namespace rns8::detail
