#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
namespace rns8::detail {
inline rns8_status query_split_k_resources(int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t fragments,
    uint64_t budget,rns8_split_k_resources& out) {
  if(m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldc<n || !prefix || prefix>20 ||
     !fragments || fragments>4096 || fragments>uint64_t(k)) return RNS8_INVALID_ARGUMENT;
  if(!qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,prefix)) return RNS8_UNSUPPORTED_BACKEND;
  if((uint64_t(k)+fragments-1)/fragments>65536) return RNS8_RANGE_ERROR;
  const uint64_t cells=uint64_t(m)*n*prefix;
  const uint32_t merge_slots=(fragments+127)/128;
  const uint64_t slots=uint64_t(fragments)+merge_slots;
  if(cells>uint64_t(PTRDIFF_MAX)/slots) return RNS8_RANGE_ERROR;
  const uint64_t bytes=cells*slots;
  if(bytes>budget) return RNS8_WORKSPACE_TOO_SMALL;
  rns8_split_k_resources result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
  result.fragments=fragments;result.partial_bytes=cells*fragments;
  result.merge_bytes=cells*merge_slots;result.scratch_bytes=bytes;
  for(auto count=fragments;count>1;count=(count+127)/128) ++result.merge_levels;
  result.fragment_window=fragments;result.window_count=1;
  out=result;return RNS8_SUCCESS;
}
inline rns8_status query_split_k_window_resources(int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t fragments,uint32_t window,
    uint64_t budget,rns8_split_k_resources& out) {
  if(!window || window>fragments) return RNS8_INVALID_ARGUMENT;
  rns8_split_k_resources result{};
  const auto status=query_split_k_resources(m,n,k,lda,ldb,ldc,prefix,fragments,UINT64_MAX,result);
  if(status!=RNS8_SUCCESS) return status;
  if(window<fragments) {
    const uint64_t cells=uint64_t(m)*n*prefix;
    result.partial_bytes=cells*window;result.merge_bytes=cells*((window+127)/128);
    result.running_bytes=cells;result.scratch_bytes=result.partial_bytes+result.merge_bytes+cells;
    result.fragment_window=window;result.window_count=(fragments+window-1)/window;
    result.merge_levels=0;
    for(auto count=window;count>1;count=(count+127)/128) ++result.merge_levels;
  }
  if(result.scratch_bytes>budget) return RNS8_WORKSPACE_TOO_SMALL;
  out=result;return RNS8_SUCCESS;
}

}
