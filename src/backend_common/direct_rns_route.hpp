#pragma once
#include <cstdint>
namespace rns8::detail {
enum class DirectRnsRoute:uint32_t {grouped=0,persistent_small=1,gemv_n1=2,gemv_small_n=3};
// One pure decision for public metadata and actual uniform dispatch. Parallel
// skinny execution is explicit research policy, not an unmeasured AUTO win.
inline DirectRnsRoute direct_rns_route(int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,bool parallel) {
  if(parallel && n==1)return DirectRnsRoute::gemv_n1;
  if(parallel && n>1 && n<=8)return DirectRnsRoute::gemv_small_n;
  if(m>0 && n>0 && m<=64/n && prefix && lda==k && ldb==n && ldc==n)return DirectRnsRoute::persistent_small;
  return DirectRnsRoute::grouped;
}
inline const char* direct_rns_route_name(DirectRnsRoute route) {
  switch(route){case DirectRnsRoute::persistent_small:return "direct_hip_persistent_small_rns_gemm_v1";
    case DirectRnsRoute::gemv_n1:return "direct_hip_selected_prefix_rns_gemv_n1_v1";
    case DirectRnsRoute::gemv_small_n:return "direct_hip_selected_prefix_rns_gemv_small_n_v1";
    default:return "direct_hip_selected_prefix_grouped_rns_gemm_v1";}
}
inline const char* direct_rns_route_event(DirectRnsRoute route) {
  switch(route){case DirectRnsRoute::gemv_n1:return "rns_gemv_n1_kernel_group";
    case DirectRnsRoute::gemv_small_n:return "rns_gemv_small_n_kernel_group";
    default:return "rns_gemm_kernel_group";}
}
}
