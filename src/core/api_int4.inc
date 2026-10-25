#include "backend_common/int4_encoding.hpp"
struct rns8_int4_recipe {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;const rns8_matrix* b=nullptr;
  uint64_t instance=0,generation=0;uint32_t mode=0;
  rns8::detail::hip_direct_device_buffer packed;
  std::shared_ptr<const rns8::detail::arithmetic::Guard> guard;
  const rns8_matrix* pending_a=nullptr;rns8_matrix* pending_c=nullptr;rns8_workspace* pending_workspace=nullptr;
  bool pending=false,usable=false;
};
namespace {
void release_int4_pending(rns8_int4_recipe& owner) {
  if(!owner.pending)return;
  --owner.b->in_flight_readers;
  if(owner.pending_a && owner.pending_a!=owner.b)--owner.pending_a->in_flight_readers;
  if(owner.pending_c)owner.pending_c->in_flight_writer=false;
  if(owner.pending_workspace)owner.pending_workspace->in_flight=false;
  owner.pending=false;owner.pending_a=nullptr;owner.pending_c=nullptr;owner.pending_workspace=nullptr;
}
std::shared_ptr<const rns8::detail::arithmetic::Guard> int4_guard(const rns8_plan& plan,const rns8_matrix& b,uint32_t mode) {
  using namespace rns8::detail::arithmetic;
  Certificate c;c.representation=Representation::signed_finite_i4;c.semantics=plan.desc.semantics;
  auto& i=c.instruction;i.target=plan.backend_target_id;i.opcode="wmma_i32_16x16x16_iu4_w32.signed";
  i.compiler_build=plan.arithmetic_guard->certificate.instruction.compiler_build;i.compiler_flags=i.compiler_build;
  i.operand_bits=4;i.accumulator_bits=32;i.safe_k=65536;i.lhs_abs=i.rhs_abs=8;i.prior_abs=15;
  i.lowering_identity="finite-signed-int4-v1;B=[ceilK16][N][2]low-K-low-nibble;zero-tail;duplicated-wave-half;mode="+std::to_string(mode)+
    ";modulus="+std::to_string(plan.desc.finite_modulus)+";instance="+std::to_string(b.matrix_instance_id)+";generation="+std::to_string(b.content_generation)+
    ";plan="+plan.arithmetic_guard->checked.sha256;
  const uint32_t modulus=plan.desc.finite_modulus;
  Node a;a.bits=4;a.input={-Integer(modulus/2),Integer((modulus-1)/2),1,modulus};a.equality=Equality::modular_value;c.dag.push_back(a);c.dag.push_back(a);
  Node prior;prior.input={-Integer(modulus-1),Integer(modulus-1),1,modulus};prior.equality=Equality::modular_value;c.dag.push_back(prior);
  Node dot;dot.operation=Operation::dot;dot.equality=Equality::modular_value;dot.left=0;dot.right=1;dot.prior=2;dot.count=65536;c.dag.push_back(dot);
  Node reduce;reduce.operation=Operation::centered_reduce;reduce.equality=Equality::modular_value;reduce.left=3;reduce.modulus=modulus;reduce.bits=4;c.dag.push_back(reduce);
  return std::make_shared<const Guard>(std::move(c));
}
}
rns8_status rns8_prepare_int4_recipe(rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* b,const rns8_int4_desc* desc,rns8_int4_recipe** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !plan || !b || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) || desc->mode>1)return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(b->in_flight_writer)return RNS8_BUSY;
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.problem=plan->desc;q.operation=RNS8_OP_INT4;
    q.execution_modes=RNS8_EXEC_SYNC;q.algorithm=desc->mode;q.ldb=b->desc.cols;q.scratch_budget=desc->max_preparation_bytes;
    const auto provider=operation_support(*ctx,q,plan->backend);if(provider.status!=RNS8_SUCCESS)return provider.status;
    if(!context_accepts_backend(*ctx,plan->backend) || !plan_schedule_contract_matches(*plan) || !plan->arithmetic_guard ||
       !matrix_backend_compatible_with_plan(*ctx,*b,plan->backend) || b->generation_exhausted || b->hip_device_id!=ctx->device_id ||
       b->desc.semantics!=plan->desc.semantics || b->finite_modulus!=plan->desc.finite_modulus || b->desc.rows!=plan->desc.k || b->desc.cols!=plan->desc.n ||
       !finite_matrix_storage_matches(*b,plan->backend,plan->desc.k,plan->desc.n) || !rns_residue_state_current_for_backend(*b,plan->backend))return RNS8_INVALID_ARGUMENT;
    if(ctx->prepared_references==UINT64_MAX || plan->prepared_references==UINT64_MAX || b->prepared_references==UINT64_MAX || b->in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto owner=std::make_unique<rns8_int4_recipe>();owner->guard=int4_guard(*plan,*b,desc->mode);if(!owner->guard->checked.admitted())return RNS8_RANGE_ERROR;
    auto status=hip_direct_prepare_int4_capability(ctx->device_id);
    if(status!=RNS8_SUCCESS){if(status==RNS8_BACKEND_FAILURE && hip_direct_synchronize(ctx->device_id)!=RNS8_SUCCESS)ctx->hip_failure_poisoned=true;return status;}
    if(desc->mode){status=owner->packed.allocate(ctx->device_id,static_cast<size_t>(provider.required_device_bytes));if(status!=RNS8_SUCCESS)return status;}
    owner->ctx=ctx;owner->plan=plan;owner->b=b;owner->instance=b->matrix_instance_id;owner->generation=b->content_generation;owner->mode=desc->mode;
    ++ctx->prepared_references;++plan->prepared_references;++b->prepared_references;
    if(desc->mode){++b->in_flight_readers;owner->pending=true;
      status=hip_direct_pack_int4_b(ctx->device_id,static_cast<const int8_t*>(b->hip_residues),owner->packed.get(),plan->desc.k,plan->desc.n,b->desc.cols,plan->desc.finite_modulus);
      if(status!=RNS8_SUCCESS && hip_direct_synchronize(ctx->device_id)!=RNS8_SUCCESS){ctx->hip_failure_poisoned=true;*out=owner.release();return status;}
      release_int4_pending(*owner);
      if(status!=RNS8_SUCCESS){*out=owner.release();const auto cleanup=rns8_destroy_int4_recipe(*out);if(cleanup==RNS8_SUCCESS)*out=nullptr;return status;}}
    owner->usable=true;*out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_int4_info(const rns8_int4_recipe* owner,rns8_int4_info* info) {
  if(!owner || !info || !rns8::detail::valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
  *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;info->mode=owner->mode;info->layout_version=1;
  info->modulus=owner->plan->desc.finite_modulus;info->signed_operands=1;info->source_instance=owner->instance;info->source_generation=owner->generation;
  info->retained_device_bytes=owner->packed.bytes();info->safe_k=65536;info->pending_cleanup=owner->pending;
  info->source_is_current=owner->usable && !owner->b->generation_exhausted && owner->b->matrix_instance_id==owner->instance && owner->b->content_generation==owner->generation &&
    rns_residue_state_current_for_backend(*owner->b,owner->plan->backend);
  info->arithmetic_proven=owner->guard->checked.admitted();set_text(info->certificate_sha256,sizeof(info->certificate_sha256),owner->guard->checked.sha256);
  set_text(info->target_id,sizeof(info->target_id),owner->ctx->device_info.gcn_arch);return RNS8_SUCCESS;
}
rns8_status rns8_gemm_int4(rns8_int4_recipe* owner,const rns8_matrix* a,rns8_matrix* c,rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!owner || !a || !c || !workspace || !owner->usable)return RNS8_INVALID_ARGUMENT;
    if(owner->pending)return RNS8_BUSY;
    auto& ctx=*owner->ctx;const auto& plan=*owner->plan;const auto& b=*owner->b;
    auto status=validate_plan_context_workspace(ctx,plan,*workspace);if(status!=RNS8_SUCCESS)return status;
    status=validate_finite_gemm_operands(ctx,plan,uint16_t(plan.desc.finite_modulus),*a,b,*c);if(status!=RNS8_SUCCESS)return status;
    if(b.generation_exhausted || b.matrix_instance_id!=owner->instance || b.content_generation!=owner->generation || !owner->guard->checked.admitted())return RNS8_INVALID_ARGUMENT;
    if(c->generation_exhausted || a->in_flight_readers==UINT64_MAX || b.in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(workspace->in_flight)return RNS8_BUSY;
    // Reserve all borrowed storage before invalidation/launch. An undrainable
    // failure retains these leases in the cleanup-only owner, never publishes C.
    invalidate_output_currentness(*c);
    ++b.in_flight_readers;if(a!=&b)++a->in_flight_readers;c->in_flight_writer=true;workspace->in_flight=true;
    owner->pending=true;owner->pending_a=a;owner->pending_c=c;owner->pending_workspace=workspace;
    status=hip_direct_gemm_int4_admitted(ctx.device_id,static_cast<const int8_t*>(a->hip_residues),static_cast<const int8_t*>(b.hip_residues),owner->packed.get(),
      static_cast<int8_t*>(c->hip_residues),plan.desc.m,plan.desc.n,plan.desc.k,a->desc.cols,b.desc.cols,c->desc.cols,plan.desc.finite_modulus);
    if(status!=RNS8_SUCCESS){if(hip_direct_synchronize(ctx.device_id)!=RNS8_SUCCESS){ctx.hip_failure_poisoned=true;owner->usable=false;return status;}
      release_int4_pending(*owner);return status;}
    release_int4_pending(*owner);mark_output_device_residues_current(*c,&plan);c->finite_modulus=uint16_t(plan.desc.finite_modulus);
    c->exact_axis_ranges.rows.clear();c->exact_axis_ranges.cols.clear();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_int4_recipe(rns8_int4_recipe* owner) {
  if(!owner)return RNS8_SUCCESS;
  if(owner->pending){const auto status=rns8::detail::hip_direct_synchronize(owner->ctx->device_id);if(status!=RNS8_SUCCESS)return status;release_int4_pending(*owner);}
  const auto status=owner->packed.reset();if(status!=RNS8_SUCCESS)return status;
  --owner->b->prepared_references;--owner->plan->prepared_references;--owner->ctx->prepared_references;delete owner;return RNS8_SUCCESS;
}
