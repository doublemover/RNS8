#include "backend_hip_direct/encoded_b_backend.hpp"
struct rns8_encoded_b {
  rns8_context* ctx=nullptr;
  const rns8_matrix* source=nullptr;
  uint64_t instance=0,generation=0;
  rns8::detail::hip_direct_device_buffer storage;
};
rns8_status rns8_prepare_encoded_b(rns8_context* ctx,const rns8_matrix* b,uint64_t budget,rns8_encoded_b** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !b || !out) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || b->backend!=ctx->backend || !exact_wide_semantics(b->desc.semantics))
      return RNS8_UNSUPPORTED_BACKEND;
    if(b->in_flight_writer) return RNS8_BUSY;
    if(b->prepared_references==UINT64_MAX || ctx->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    if(b->hip_device_id!=ctx->device_id || b->generation_exhausted || !b->device_native_current ||
        !(b->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !rns_residue_state_current_for_backend(*b,ctx->backend) ||
        b->exact_range_prefix<b->prefix) return RNS8_INVALID_ARGUMENT;
    const auto layout=encoded_b_layout(b->desc.rows,b->desc.cols,b->desc.cols,b->prefix,budget);
    if(layout.status!=RNS8_SUCCESS) return layout.status;
    const bool u=b->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    const void* native=u?b->hip_native_u64:b->hip_native_i64;
    const auto native_bytes=u?b->hip_native_u64_bytes:b->hip_native_i64_bytes;
    if(!native || native_bytes!=uint64_t(b->desc.rows)*uint64_t(b->desc.cols)*8) return RNS8_INVALID_ARGUMENT;
    auto cache=std::make_unique<rns8_encoded_b>();
    auto status=cache->storage.allocate(ctx->device_id,static_cast<size_t>(layout.total_bytes));
    if(status!=RNS8_SUCCESS) return status;
    status=hip_prepare_encoded_b(ctx->device_id,native,cache->storage.get(),b->desc.rows,b->desc.cols,
        b->desc.cols,b->prefix,u,layout.total_bytes);
    if(status!=RNS8_SUCCESS) return status;
    cache->ctx=ctx; cache->source=b; cache->instance=b->matrix_instance_id; cache->generation=b->content_generation;
    ++ctx->prepared_references; ++b->prepared_references;
    *out=cache.release(); return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_encoded_b(rns8_encoded_b* cache) {
  if(!cache) return RNS8_SUCCESS;
  const auto status=cache->storage.reset();
  --cache->source->prepared_references; --cache->ctx->prepared_references;
  delete cache; return status;
}
rns8_status rns8_gemm_encoded_b_exact(rns8_context* ctx,const rns8_plan* plan,
    const rns8_matrix* a,const rns8_encoded_b* cache,rns8_matrix* c,uint32_t group,rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !plan || !a || !cache || !c || !workspace || cache->ctx!=ctx ||
        (group!=1 && group!=2 && group!=4)) return RNS8_INVALID_ARGUMENT;
    if(plan->backend!=RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(plan->desc.semantics))
      return RNS8_UNSUPPORTED_BACKEND;
    auto status=validate_plan_context_workspace(*ctx,*plan,*workspace);
    if(status!=RNS8_SUCCESS) return status;
    const auto& b=*cache->source;
    status=validate_rns_gemm_operands(*ctx,*plan,*a,b,*c,true);
    if(status!=RNS8_SUCCESS) return status;
    if(c->generation_exhausted) return RNS8_RANGE_ERROR;
    if(b.generation_exhausted || b.matrix_instance_id!=cache->instance || b.content_generation!=cache->generation ||
        b.prefix!=plan->prefix || !a->device_native_current || !(a->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT))
      return RNS8_INVALID_ARGUMENT;
    const bool u=plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    const void* native=u?a->hip_native_u64:a->hip_native_i64;
    const auto native_bytes=u?a->hip_native_u64_bytes:a->hip_native_i64_bytes;
    if(!native || native_bytes!=uint64_t(a->desc.rows)*uint64_t(a->desc.cols)*8) return RNS8_INVALID_ARGUMENT;
    cpp_int bound; rns8_exact_matrix_ranges ranges;
    status=exact_gemm_range(*plan,*a,b,bound,&ranges); if(status!=RNS8_SUCCESS) return status;
    bool started=false;
    struct CommitGuard { rns8_matrix& c; bool& started; bool published=false;
      ~CommitGuard() { if(started && !published) invalidate_output_currentness(c); }
    } write{*c,started};
    status=hip_gemm_encoded_b(ctx->device_id,native,cache->storage.get(),c->hip_residues,
        plan->desc.m,plan->desc.n,plan->desc.k,a->desc.cols,c->desc.cols,plan->prefix,group,u,started);
    if(status!=RNS8_SUCCESS) return status;
    mark_output_device_residues_current(*c); commit_exact_range(*c,*plan,bound,&ranges);
    write.published=true; return RNS8_SUCCESS;
  });
}
