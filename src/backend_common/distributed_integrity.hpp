#pragma once
#include "distributed_inputs.hpp"
namespace rns8::detail::distributed {
inline constexpr uint32_t integrity_moduli[2]={257,263};
inline rns8_status distributed_commit_status(uint32_t flags) {
  if(flags&1)return RNS8_RANGE_ERROR;
  return flags?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS;
}
// Independent native modular dots. The candidate is a complete reconstructed
// integer; neither check channel is derived from its computational residues.
inline uint32_t reference_integrity_cell(const Plan& plan,const Inputs& inputs,
    uint64_t row,uint64_t col,const uint64_t* candidate) {
  const auto& r=plan.request();
  Integer value=0;
  for(unsigned i=0;i<3;++i)value+=Integer(candidate[i])<<(64*i);
  if(plan.signed_value() && (candidate[2]>>63))value-=Integer(1)<<192;
  uint32_t mask=0;
  for(unsigned p=0;p<2;++p) {
    const auto q=integrity_moduli[p];uint32_t sum=0;
    for(uint64_t k=0;k<r.k;++k)
      sum=(sum+native_residue(inputs.lhs()[row*r.lda+k],plan.signed_value(),q)*
          native_residue(inputs.rhs()[k*r.ldb+col],plan.signed_value(),q))%q;
    Integer observed=value%q;if(observed<0)observed+=q;
    if(observed!=sum)mask|=4u<<p;
  }
  return mask;
}
} // namespace rns8::detail::distributed
