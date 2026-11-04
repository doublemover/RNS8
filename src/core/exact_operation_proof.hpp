#pragma once
#include "exact_range.hpp"
#include <memory>

namespace rns8::detail {
// An immutable mathematical certificate, separate from per-call storage,
// reservations, transfer envelopes, pending producers and stream admission.
class ExactOperationProof {
 public:
  static rns8_status create(const rns8_plan& plan,const rns8_matrix& a,const rns8_matrix& b,
      const rns8_matrix* d,int64_t alpha,int64_t beta,std::shared_ptr<const ExactOperationProof>& out) {
    if(plan.desc.semantics!=RNS8_BOUNDED_I64 && plan.desc.semantics!=RNS8_BOUNDED_U64 &&
        !exact_wide_semantics(plan.desc.semantics))return RNS8_UNSUPPORTED_BACKEND;
    auto staged=std::shared_ptr<ExactOperationProof>(new ExactOperationProof(plan,a,b,d,alpha,beta));
    const auto status=d?exact_axpby_range(plan,a,b,*d,alpha,beta,staged->bound_):
      exact_gemm_range(plan,a,b,staged->bound_,&staged->ranges_);
    if(status!=RNS8_SUCCESS)return status;
    out=std::move(staged);return RNS8_SUCCESS;
  }
  bool matches(const rns8_plan& plan,const rns8_matrix& a,const rns8_matrix& b,
      const rns8_matrix* d=nullptr,int64_t alpha=0,int64_t beta=0) const {
    const auto& x=plan.desc;
    return plan_==&plan && contract_==plan.validated_contract && prefix_==plan.prefix && product_==plan.modulus_product &&
      desc_.m==x.m && desc_.n==x.n && desc_.k==x.k && desc_.semantics==x.semantics && desc_.bound_kind==x.bound_kind &&
      desc_.bound==x.bound && desc_.max_prefix==x.max_prefix && desc_.flags==x.flags &&
      desc_.tile_m==x.tile_m && desc_.tile_n==x.tile_n && desc_.finite_modulus==x.finite_modulus &&
      alpha_==alpha && beta_==beta && inputs_[0].matches(a) && inputs_[1].matches(b) &&
      (d?inputs_[2].matches(*d):!inputs_[2].matrix);
  }
  void stage(cpp_int& bound,rns8_exact_matrix_ranges* ranges=nullptr) const {
    bound=bound_;if(ranges)*ranges=ranges_;
  }
  uint64_t retained_object_bytes() const noexcept {
    return sizeof(*this)+(ranges_.rows.capacity()+ranges_.cols.capacity())*sizeof(rns8_exact_axis_range);
  }
  uint64_t logical_limb_bytes() const noexcept {
    const auto bytes=[](const cpp_int& value){return uint64_t(value.backend().size())*sizeof(boost::multiprecision::limb_type);};
    uint64_t result=bytes(product_)+bytes(bound_);
    for(const auto* axes:{&ranges_.rows,&ranges_.cols})for(const auto& axis:*axes)
      result+=bytes(axis.max_magnitude)+bytes(axis.absolute_sum)+bytes(axis.sum)+bytes(axis.minimum)+
        bytes(axis.maximum)+bytes(axis.deviation_sum)+bytes(axis.squared_sum);
    return result;
  }
 private:
  struct Input {
    const rns8_matrix* matrix=nullptr;uint64_t instance=0,generation=0;
    bool matches(const rns8_matrix& value) const noexcept {
      return matrix==&value && instance==value.matrix_instance_id && generation==value.content_generation && !value.generation_exhausted;
    }
  };
  ExactOperationProof(const rns8_plan& plan,const rns8_matrix& a,const rns8_matrix& b,const rns8_matrix* d,int64_t alpha,int64_t beta)
      :plan_(&plan),contract_(plan.validated_contract),desc_(plan.desc),prefix_(plan.prefix),product_(plan.modulus_product),alpha_(alpha),beta_(beta) {
    inputs_[0]={&a,a.matrix_instance_id,a.content_generation};inputs_[1]={&b,b.matrix_instance_id,b.content_generation};
    if(d)inputs_[2]={d,d->matrix_instance_id,d->content_generation};
  }
  const rns8_plan* plan_;
  std::shared_ptr<const rns8_validated_plan_contract> contract_;
  rns8_gemm_desc desc_;
  uint32_t prefix_;
  cpp_int product_,bound_;
  rns8_exact_matrix_ranges ranges_;
  int64_t alpha_,beta_;
  Input inputs_[3];
};
}
