#include "core/api_internal.hpp"
#include "backend_common/fp8_modular.hpp"
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_fp8_recipe {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;
  hip_direct_device_buffer probe,products;uint32_t method=0;
  std::string certificate;
  const rns8_matrix *a=nullptr,*b=nullptr;rns8_matrix* c=nullptr;rns8_workspace* workspace=nullptr;
  bool pending=false,usable=false;
};
namespace {
void fp8_release(rns8_fp8_recipe& owner) {
  if(!owner.pending)return;
  if(owner.a)--owner.a->in_flight_readers;
  if(owner.b && owner.b!=owner.a)--owner.b->in_flight_readers;
  if(owner.c)owner.c->in_flight_writer=false;
  if(owner.workspace)owner.workspace->in_flight=false;
  owner.a=owner.b=nullptr;owner.c=nullptr;owner.workspace=nullptr;owner.pending=false;
  owner.probe.mark_completed();
  owner.products.mark_completed();
}
}
rns8_status rns8_prepare_fp8_recipe(rns8_context* ctx,const rns8_plan* plan,const rns8_fp8_desc* desc,rns8_fp8_recipe** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !plan || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) || desc->method>1 ||
        (desc->flags&~RNS8_FP8_ASSUME_NAMED_INSTRUCTION_MODEL))return RNS8_INVALID_ARGUMENT;
    if(desc->format!=RNS8_FP8_GFX942_E4M3_FNUZ)return RNS8_UNSUPPORTED_BACKEND;
    if(!(desc->flags&RNS8_FP8_ASSUME_NAMED_INSTRUCTION_MODEL))return RNS8_UNSUPPORTED_BACKEND;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.problem=plan->desc;
    q.operation=RNS8_OP_FP8;q.algorithm=desc->method;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=desc->max_preparation_bytes;
    const auto support=operation_support(*ctx,q,plan->backend);if(support.status!=RNS8_SUCCESS)return support.status;
    if(!context_accepts_backend(*ctx,plan->backend) || !plan_schedule_contract_matches(*plan) || !plan->arithmetic_guard ||
        !plan->arithmetic_guard->checked.admitted())return RNS8_INVALID_ARGUMENT;
    if(desc->max_preparation_bytes<support.required_device_bytes)return RNS8_WORKSPACE_TOO_SMALL;
    if(ctx->prepared_references==UINT64_MAX || plan->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto owner=std::make_unique<rns8_fp8_recipe>();
    char build[65]{};const auto build_status=rns8_get_pipeline_build_identity(build,sizeof(build));if(build_status!=RNS8_SUCCESS)return build_status;
    owner->certificate=fp8_modular::identity(plan->backend_target_id,std::string(build)+";plan="+plan->arithmetic_guard->checked.sha256+
        ";modulus="+std::to_string(plan->desc.finite_modulus)+";method="+std::to_string(desc->method));
    owner->method=desc->method;
    auto status=owner->probe.allocate(ctx->device_id,4);if(status!=RNS8_SUCCESS)return status;
    if(desc->method){status=owner->products.allocate(ctx->device_id,support.required_device_bytes-4);if(status!=RNS8_SUCCESS)return status;}
    owner->ctx=ctx;owner->plan=plan;++ctx->prepared_references;++plan->prepared_references;
    owner->pending=true;owner->probe.mark_pending();
    status=hip_direct_fp8_modular_prepare(ctx->device_id,owner->probe.get());
    if(status!=RNS8_SUCCESS && hip_direct_synchronize(ctx->device_id)!=RNS8_SUCCESS) {
      ctx->hip_failure_poisoned=true;*out=owner.release();return status;
    }
    fp8_release(*owner);
    if(status!=RNS8_SUCCESS) {
      *out=owner.release();const auto closed=rns8_destroy_fp8_recipe(*out);if(closed==RNS8_SUCCESS)*out=nullptr;return status;
    }
    owner->usable=true;*out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_fp8_info(const rns8_fp8_recipe* owner,rns8_fp8_info* info) {
  if(!owner || !info || !valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
  *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;info->format=1;
  info->modulus=owner->plan->desc.finite_modulus;info->model_proven=fp8_modular::algebra();
  info->pending_cleanup=owner->pending;info->safe_k=fp8_modular::safe_k;info->retained_device_bytes=owner->probe.bytes()+owner->products.bytes();info->method=owner->method;
  set_text(info->certificate_sha256,sizeof(info->certificate_sha256),owner->certificate);
  set_text(info->target_id,sizeof(info->target_id),owner->plan->backend_target_id);return RNS8_SUCCESS;
}
rns8_status rns8_gemm_fp8_research(rns8_fp8_recipe* owner,const rns8_matrix* a,const rns8_matrix* b,rns8_matrix* c,rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !owner->usable || !a || !b || !c || !workspace)return RNS8_INVALID_ARGUMENT;
    if(owner->pending || workspace->in_flight)return RNS8_BUSY;
    auto& ctx=*owner->ctx;const auto& plan=*owner->plan;
    auto status=validate_plan_context_workspace(ctx,plan,*workspace);if(status!=RNS8_SUCCESS)return status;
    status=validate_finite_gemm_operands(ctx,plan,uint16_t(plan.desc.finite_modulus),*a,*b,*c);if(status!=RNS8_SUCCESS)return status;
    if(c->generation_exhausted || a->in_flight_readers==UINT64_MAX || b->in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
    invalidate_output_currentness(*c);++a->in_flight_readers;if(a!=b)++b->in_flight_readers;
    c->in_flight_writer=true;workspace->in_flight=true;
    owner->a=a;owner->b=b;owner->c=c;owner->workspace=workspace;owner->pending=true;owner->probe.mark_pending();if(owner->method)owner->products.mark_pending();
    status=hip_direct_fp8_modular_gemm(ctx.device_id,static_cast<const int8_t*>(a->hip_residues),static_cast<const int8_t*>(b->hip_residues),
        static_cast<int8_t*>(c->hip_residues),plan.desc.m,plan.desc.n,plan.desc.k,a->desc.cols,b->desc.cols,c->desc.cols,plan.desc.finite_modulus,owner->probe.get(),owner->products.get());
    if(status!=RNS8_SUCCESS && hip_direct_synchronize(ctx.device_id)!=RNS8_SUCCESS){ctx.hip_failure_poisoned=true;owner->usable=false;return status;}
    fp8_release(*owner);if(status==RNS8_VERIFICATION_FAILED)owner->usable=false;if(status!=RNS8_SUCCESS)return status;
    mark_output_device_residues_current(*c,&plan);c->finite_modulus=uint16_t(plan.desc.finite_modulus);
    c->exact_axis_ranges.rows.clear();c->exact_axis_ranges.cols.clear();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_fp8_recipe(rns8_fp8_recipe* owner) {
  if(!owner)return RNS8_SUCCESS;
  if(owner->pending){const auto status=hip_direct_synchronize(owner->ctx->device_id);if(status!=RNS8_SUCCESS)return status;fp8_release(*owner);}
  auto status=owner->products.reset();if(status!=RNS8_SUCCESS)return status;
  status=owner->probe.reset();if(status!=RNS8_SUCCESS)return status;
  --owner->ctx->prepared_references;--owner->plan->prepared_references;delete owner;return RNS8_SUCCESS;
}
