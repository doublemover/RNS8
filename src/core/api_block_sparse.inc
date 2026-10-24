#include "core/hip_resources.hpp"
#include <memory>
#include <type_traits>

struct rns8_block_sparse_operand {
  rns8_context* ctx=nullptr;
  const rns8_matrix* source=nullptr;
  rns8::detail::BlockSparseOperand layout;
  rns8::detail::hip_direct_device_buffer payload;
};

rns8_status rns8_prepare_block_sparse_operand(
    rns8_context* ctx, const rns8_matrix* source, rns8_block_sparse_role role,
    uint64_t max_preparation_bytes, rns8_block_sparse_operand** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !source || !out ||
        (role!=RNS8_BLOCK_SPARSE_LHS && role!=RNS8_BLOCK_SPARSE_RHS)) return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if (ctx->backend!=RNS8_BACKEND_HIP_DIRECT || source->backend!=ctx->backend ||
        !exact_wide_semantics(source->desc.semantics)) return RNS8_UNSUPPORTED_BACKEND;
    if (source->in_flight_writer) return RNS8_BUSY;
    if (source->prepared_references==UINT64_MAX || ctx->prepared_references==UINT64_MAX)
      return RNS8_RANGE_ERROR;
    if (source->hip_device_id!=ctx->device_id || source->generation_exhausted ||
        !(source->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !source->device_native_current ||
        !rns_residue_state_current_for_backend(*source,ctx->backend) ||
        source->exact_range_prefix<source->prefix) return RNS8_INVALID_ARGUMENT;
    AddressDomain address;
    if (!checked_host_plane_address(source->desc.rows,source->desc.cols,source->desc.cols,1,8,address))
      return RNS8_RANGE_ERROR;
    const bool u=source->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    const void* native=u?source->hip_native_u64:source->hip_native_i64;
    const auto native_bytes=u?source->hip_native_u64_bytes:source->hip_native_i64_bytes;
    if (!native || native_bytes!=address.byte_extent) return RNS8_INVALID_ARGUMENT;
    if (address.byte_extent>max_preparation_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    const uint64_t pack_budget=(max_preparation_bytes-address.byte_extent)/2;
    auto result=std::make_unique<rns8_block_sparse_operand>();
    std::vector<int8_t> packed;
    const auto packed_role=role==RNS8_BLOCK_SPARSE_LHS?BlockSparseRole::lhs:BlockSparseRole::rhs;
    const auto prepare=[&](auto tag) -> rns8_status {
      using T=decltype(tag);
      std::vector<T> host(static_cast<size_t>(address.byte_extent/8));
      auto status=hip_direct_copy_device_to_host(ctx->device_id,host.data(),native,static_cast<size_t>(address.byte_extent));
      if (status!=RNS8_SUCCESS) return status;
      if constexpr(std::is_signed_v<T>)
        return pack_block_sparse_i64(host.data(),source->desc.rows,source->desc.cols,source->desc.cols,
            source->prefix,packed_role,source->matrix_instance_id,source->content_generation,
            pack_budget,result->layout,packed);
      else
        return pack_block_sparse_u64(host.data(),source->desc.rows,source->desc.cols,source->desc.cols,
            source->prefix,packed_role,source->matrix_instance_id,source->content_generation,
            pack_budget,result->layout,packed);
    };
    auto status=u?prepare(uint64_t{}):prepare(int64_t{});
    if (status!=RNS8_SUCCESS) return status;
    if (!packed.empty()) {
      status=result->payload.allocate(ctx->device_id,packed.size());
      if (status!=RNS8_SUCCESS) return status;
      status=hip_direct_copy_host_to_device(ctx->device_id,result->payload.get(),packed.data(),packed.size());
      if (status!=RNS8_SUCCESS) return status;
    }
    // No fallible work after taking lifetime references. Repacking is allowed;
    // execute compares the stored identity and generation before using the cache.
    result->ctx=ctx; result->source=source;
    ++ctx->prepared_references; ++source->prepared_references;
    *out=result.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_block_sparse_operand(rns8_block_sparse_operand* operand) {
  if (!operand) return RNS8_SUCCESS;
  // Synchronous handle use/destruction requires external thread exclusion.
  const auto status=operand->payload.reset();
  --operand->source->prepared_references;
  --operand->ctx->prepared_references;
  delete operand;
  return status;
}
rns8_status rns8_gemm_block_sparse_exact_with_method(
    rns8_context* ctx, const rns8_plan* plan,
    const rns8_block_sparse_operand* a, const rns8_block_sparse_operand* b,
    rns8_matrix* c,rns8_block_sparse_method method,uint32_t scalar_intersection_limit,
    uint64_t max_scratch_bytes, rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (method<RNS8_BLOCK_SPARSE_SCALAR || method>RNS8_BLOCK_SPARSE_HYBRID ||
        (method!=RNS8_BLOCK_SPARSE_HYBRID && scalar_intersection_limit)) return RNS8_INVALID_ARGUMENT;
    if (!ctx || !plan || !a || !b || !c || !workspace || a->ctx!=ctx || b->ctx!=ctx ||
        a->layout.role!=BlockSparseRole::lhs || b->layout.role!=BlockSparseRole::rhs)
      return RNS8_INVALID_ARGUMENT;
    if (plan->backend!=RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(plan->desc.semantics))
      return RNS8_UNSUPPORTED_BACKEND;
    auto status=validate_plan_context_workspace(*ctx,*plan,*workspace);
    if (status!=RNS8_SUCCESS) return status;
    status=validate_rns_gemm_operands(*ctx,*plan,*a->source,*b->source,*c,true);
    if (status!=RNS8_SUCCESS) return status;
    if (c->generation_exhausted) return RNS8_RANGE_ERROR;
    for (const auto* operand : {a,b}) {
      const auto& source=*operand->source;
      if (source.generation_exhausted || operand->layout.prefix!=plan->prefix ||
          !valid_block_sparse_operand(operand->layout,source.matrix_instance_id,source.content_generation))
        return RNS8_INVALID_ARGUMENT;
    }
    cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    status=exact_gemm_range(*plan,*a->source,*b->source,bound,&ranges);
    if (status!=RNS8_SUCCESS) return status;
    bool commit_started=false;
    struct CommitGuard {
      rns8_matrix& output; bool& started; bool published=false;
      ~CommitGuard() { if (started && !published) invalidate_output_currentness(output); }
    } write{*c,commit_started};
    status=hip_direct_gemm_block_sparse_device(ctx->device_id,a->payload.get(),a->layout,
        a->source->matrix_instance_id,a->source->content_generation,b->payload.get(),b->layout,
        b->source->matrix_instance_id,b->source->content_generation,c->hip_residues,c->desc.cols,
        max_scratch_bytes,&commit_started,static_cast<uint32_t>(method),scalar_intersection_limit);
    if (status!=RNS8_SUCCESS) return status;
    mark_output_device_residues_current(*c);
    commit_exact_range(*c,*plan,bound,&ranges);
    write.published=true;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_block_sparse_exact(rns8_context* ctx,const rns8_plan* plan,
    const rns8_block_sparse_operand* a,const rns8_block_sparse_operand* b,rns8_matrix* c,
    uint64_t budget,rns8_workspace* workspace) {
  return rns8_gemm_block_sparse_exact_with_method(ctx,plan,a,b,c,RNS8_BLOCK_SPARSE_SCALAR,0,budget,workspace);
}
