#pragma once
#include "rank23_slp_resources.hpp"
namespace rns8::detail {
struct BilinearSelection {
  bool valid=false,matrix=false,slp=false,recompute=false;
  uint32_t algorithm=0,blocks=0;
};
inline BilinearSelection bilinear_selection(uint32_t method) {
  if(method>15)return {};
  if(method>=12)return {true,bool(method&1),true,method>=14,6,4};
  if(method>=10)return {true,method==11,false,false,5,4};
  const auto algorithm=method%5;
  return {true,method>=5,algorithm>=2,algorithm==4,algorithm,algorithm?3u:2u};
}
inline BilinearResources query_bilinear_method_resources(uint32_t method,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget) {
  const auto selected=bilinear_selection(method);if(!selected.valid)return {};
  if(selected.algorithm==6)return query_bilinear_dag_resources(m,n,k,lda,ldb,ldc,prefix,budget,4,48,selected.recompute);
  return selected.slp?query_rank23_slp_resources(m,n,k,lda,ldb,ldc,prefix,budget,selected.recompute):
      query_bilinear_resources(m,n,k,lda,ldb,ldc,prefix,selected.blocks,budget);
}
}
