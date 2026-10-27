struct rns8_quarter_crt_export {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;const rns8_matrix* source=nullptr;
  const rns8::detail::quarter_crt::Certificate certificate;
  std::vector<uint64_t> staged;
  rns8::detail::hip_direct_device_buffer device;
  bool pending=false;
  explicit rns8_quarter_crt_export(rns8::detail::quarter_crt::Certificate c):certificate(std::move(c)){}
};
rns8_status rns8_prepare_quarter_crt_export(rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* source,
    const rns8_quarter_crt_desc* desc,rns8_quarter_crt_export** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !plan || !source || !desc || !out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!quarter_crt_backend_supported(plan->backend))return RNS8_UNSUPPORTED_BACKEND;
    if(!valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) || desc->mode>2 || !desc->limb_count || desc->limb_count>32)return RNS8_INVALID_ARGUMENT;
    if(source->in_flight_writer)return RNS8_BUSY;
    if(!exact_wide_semantics(plan->desc.semantics) || (desc->mode==0 && plan->desc.semantics!=RNS8_EXACT_WIDE_SIGNED) ||
        (desc->mode==1 && plan->desc.semantics!=RNS8_EXACT_WIDE_UNSIGNED))return RNS8_INVALID_ARGUMENT;
    const auto valid=validate_export_matrix(*ctx,*plan,*source,plan->desc.semantics,RNS8_BOUND_NONE,plan->prefix);
    if(valid!=RNS8_SUCCESS)return valid;
    if(source->generation_exhausted || source->exact_range_prefix<plan->prefix || !source->content_generation)return RNS8_RANGE_ERROR;
    if(ctx->prepared_references==UINT64_MAX || plan->prepared_references==UINT64_MAX || source->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(!plan->arithmetic_guard)return RNS8_INVALID_ARGUMENT;
    quarter_crt::Certificate c;c.prefix=plan->prefix;c.limbs=desc->limb_count;c.mode=desc->mode;c.bound=source->exact_max_magnitude;
    c.source_instance=source->matrix_instance_id;c.source_generation=source->content_generation;c.target=plan->backend_target_id;
    c.build=plan->arithmetic_guard->certificate.instruction.compiler_build;c.plan_hash=plan->arithmetic_guard->checked.sha256;
    c=quarter_crt::certify(std::move(c));if(!c.admitted)return RNS8_RANGE_ERROR;
    const uint64_t cells=uint64_t(plan->desc.m)*uint64_t(plan->desc.n);
    if(!valid_limb_export_access(plan->desc.m,plan->desc.n,plan->desc.n,c.limbs) ||
        (hip_resident_rns_backend(plan->backend) && cells>uint64_t(INT32_MAX-255)))return RNS8_RANGE_ERROR;
    const uint64_t bytes=cells*c.limbs*8,owners=hip_resident_rns_backend(plan->backend)?2:1;
    if(bytes>SIZE_MAX || bytes>desc->max_scratch_bytes/owners)return RNS8_WORKSPACE_TOO_SMALL;
    auto owner=std::make_unique<rns8_quarter_crt_export>(std::move(c));owner->staged.resize(static_cast<size_t>(cells*desc->limb_count));
    if(hip_resident_rns_backend(plan->backend)){const auto status=owner->device.allocate(ctx->device_id,static_cast<size_t>(bytes));if(status!=RNS8_SUCCESS)return status;}
    owner->ctx=ctx;owner->plan=plan;owner->source=source;++ctx->prepared_references;++plan->prepared_references;++source->prepared_references;
    *out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_quarter_crt_info(const rns8_quarter_crt_export* owner,rns8_quarter_crt_info* info) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !info || !rns8::detail::valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
    const auto& c=owner->certificate;*info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;
    info->prefix=c.prefix;info->mode=c.mode;info->limb_count=c.limbs;info->delayed_digits=c.digits;
    info->source_instance=c.source_instance;info->source_generation=c.source_generation;
    info->retained_payload_bytes=owner->staged.capacity()*8+owner->device.bytes();
    info->algebraic_verdict=RNS8_PROOF_PROVEN;info->target_execution_verdict=RNS8_PROOF_NOT_RUN;
    for(uint32_t l=0;l<3;++l)info->magnitude_bound[l]=static_cast<uint64_t>((c.bound>>(64*l))&UINT64_MAX);
    std::copy(c.sha256.begin(),c.sha256.end(),info->certificate_sha256);return RNS8_SUCCESS;
  });
}
rns8_status rns8_export_quarter_crt_limbs(rns8_quarter_crt_export* owner,uint64_t* dst,int64_t ld) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!owner || !dst)return RNS8_INVALID_ARGUMENT;const auto& c=owner->certificate;const auto& plan=*owner->plan;const auto& source=*owner->source;
    if(owner->pending || source.in_flight_writer)return RNS8_BUSY;
    if(!valid_limb_export_access(plan.desc.m,plan.desc.n,ld,c.limbs) || source.generation_exhausted ||
        source.matrix_instance_id!=c.source_instance || source.content_generation!=c.source_generation)return RNS8_INVALID_ARGUMENT;
    const auto valid=validate_export_matrix(*owner->ctx,plan,source,plan.desc.semantics,RNS8_BOUND_NONE,plan.prefix);if(valid!=RNS8_SUCCESS)return valid;
    if(source.in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
    PreparedExportReadLease read(source,owner->pending);
    const int64_t cells=plan.desc.m*plan.desc.n;
    if(owner->device.get()) {
      owner->pending=true;
      auto status=hip_direct_export_quarter_crt_device(owner->ctx->device_id,source.hip_residues,owner->device.get(),cells,c.prefix,c.digits,c.limbs,c.mode);
      if(status==RNS8_SUCCESS)status=hip_direct_copy_device_to_host(owner->ctx->device_id,owner->staged.data(),owner->device.get(),owner->staged.size()*8);
      if(status!=RNS8_SUCCESS) {
        if(hip_direct_synchronize(owner->ctx->device_id)!=RNS8_SUCCESS){owner->ctx->hip_failure_poisoned=true;return status;}
        owner->pending=false;return status;
      }
      owner->pending=false;
    } else {
      uint32_t residues[20]{};
      for(int64_t cell=0;cell<cells;++cell) {
        for(uint32_t p=0;p<c.prefix;++p){const int value=source.residues[size_t(p)*cells+cell];residues[p]=value<0?uint32_t(value+garner::kConstants.moduli[p]):uint32_t(value);}
        if(!quarter_crt::reconstruct_host(c,residues,owner->staged.data()+size_t(cell)*c.limbs)){return RNS8_RANGE_ERROR;}
      }
    }
    // Caller publication is last, after the complete staged result and drain.
    for(int64_t row=0;row<plan.desc.m;++row)std::copy_n(owner->staged.data()+size_t(row*plan.desc.n)*c.limbs,
        size_t(plan.desc.n)*c.limbs,dst+size_t(row*ld)*c.limbs);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_quarter_crt_export(rns8_quarter_crt_export* owner) {
  if(!owner)return RNS8_SUCCESS;
  if(owner->pending){const auto status=rns8::detail::hip_direct_synchronize(owner->ctx->device_id);if(status!=RNS8_SUCCESS)return status;
    owner->pending=false;--owner->source->in_flight_readers;}
  const auto status=owner->device.reset();if(status!=RNS8_SUCCESS)return status;
  --owner->ctx->prepared_references;--owner->plan->prepared_references;--owner->source->prepared_references;
  delete owner;return RNS8_SUCCESS;
}
