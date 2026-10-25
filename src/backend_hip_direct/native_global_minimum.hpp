#pragma once
#include "backend_common/adaptive_native_support.hpp"
#include "backend_common/operation_support.hpp"
#include <algorithm>
namespace rns8::detail {
static_assert(sizeof(NativeTileBound)==32,"global proof readback ABI");
struct NativeGlobalMinimumSupport {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  AdaptiveNativeSupport discovery;
  uint64_t suffix_bytes=0,peak_scratch_bytes=0;
};
inline NativeGlobalMinimumSupport query_native_global_minimum_support(
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,
    uint32_t capacity,uint32_t group,uint64_t budget) {
  NativeGlobalMinimumSupport out;
  out.discovery=query_adaptive_native_support(m,n,k,lda,ldb,ldc,capacity,group,
      NativeBoundMethod::axis_l1,256,0,UINT64_MAX);
  out.status=out.discovery.status;if(out.status!=RNS8_SUCCESS)return out;
  if(capacity>1) {
    const auto lift=query_exact_lift_support(m*n,1,capacity,UINT64_MAX);
    out.status=lift.status;if(out.status!=RNS8_SUCCESS)return out;out.suffix_bytes=lift.scratch_bytes;
  }
  // Discovery records retire before production. Suffix storage is allocated
  // only after the selected prefix is known, and after the capability probe.
  // Bound the maximum phase, rather than summing disjoint scratch lifetimes.
  const auto production_extra=std::max<uint64_t>(out.suffix_bytes,sizeof(int));
  if(out.discovery.peak_scratch_bytes>SIZE_MAX-sizeof(NativeTileBound) ||
      out.discovery.stage_bytes>SIZE_MAX-production_extra){out.status=RNS8_RANGE_ERROR;return out;}
  out.peak_scratch_bytes=std::max(out.discovery.peak_scratch_bytes+sizeof(NativeTileBound),out.discovery.stage_bytes+production_extra);
  out.status=out.peak_scratch_bytes>budget?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;return out;
}
struct NativeGlobalMinimumReceipt {
  uint32_t required_prefix=0,compute_prefix=0;
  NativeTileBound bound;
  uint64_t proof_readback_bytes=0;
  double discovery_ns=0,conversion_product_ns=0,completion_ns=0,commit_ns=0,cleanup_ns=0;
};
// Private synchronous research control. Fresh device axis-L1 proofs on every
// call; minimum UNIFORM prefix for those proofs, then exact completed output.
// Caller retains immutable native allocations for the call. No cached proof or
// caller generation assertion is reused. Failed preflight preserves c.
rns8_status hip_direct_gemm_global_minimum_native_device(int device_id,const void* a,const void* b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,bool unsigned_inputs,
    uint32_t capacity,uint32_t group,uint64_t budget,bool& commit_started,NativeGlobalMinimumReceipt&);
} // namespace rns8::detail
