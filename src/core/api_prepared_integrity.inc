// Integrity is a separate opt-in commit condition, never an arithmetic guard.
static const void* integrity_native_pointer(const rns8_matrix& matrix,bool u) {
  return u?matrix.hip_native_u64:matrix.hip_native_i64;
}
static rns8_status admit_prepared_integrity(const rns8_prepared_execution& e,
    const boost::multiprecision::cpp_int& bound) {
  if(e.d || e.plan->backend!=RNS8_BACKEND_HIP_DIRECT ||
      !rns8::detail::exact_wide_semantics(e.plan->desc.semantics) || !e.plan->tile_schedule.empty())
    return RNS8_UNSUPPORTED_BACKEND;
  const auto provider=plan_operation_preflight(*e.ctx,*e.plan,RNS8_OP_NATIVE_INTEGRITY,RNS8_EXEC_SYNC,
      RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_CURRENT_RANGE_PROOF);
  if(provider!=RNS8_SUCCESS)return provider;
  if(bound<0 || bound>=(boost::multiprecision::cpp_int(1)<<160))return RNS8_RANGE_ERROR;
  const bool u=e.plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
  for(const auto* matrix:{e.a,e.b}) {
    if(!rns8::detail::qualified_i32_matrix_address(matrix->desc.rows,matrix->desc.cols,matrix->desc.cols,1,8))
      return RNS8_UNSUPPORTED_BACKEND;
    const auto bytes=u?matrix->hip_native_u64_bytes:matrix->hip_native_i64_bytes;
    if(!(matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !matrix->device_native_current ||
        !integrity_native_pointer(*matrix,u) || bytes!=uint64_t(matrix->desc.rows)*uint64_t(matrix->desc.cols)*8)
      return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}
static rns8_status close_prepared_integrity(rns8_prepared_execution& e) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(!e.integrity_flag && !e.integrity_host_flag.get() && !e.integrity_stream)return RNS8_SUCCESS;
  rns8::detail::hip_scoped_device device(e.ctx->device_id);
  if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(e.integrity_stream) {
    if(hipStreamDestroy(e.integrity_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    e.integrity_stream=nullptr;
  }
  if(e.integrity_host_flag.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
  return e.integrity_flag.reset();
}
rns8_status rns8_set_prepared_native_integrity(rns8_prepared_execution* e,
    const rns8_native_integrity_options* options) {
  return guard_api([&]() -> rns8_status {
    if(!e || !options || !rns8::detail::valid_abi(options->struct_size,options->abi_version,sizeof(*options)) ||
        options->mode>1 || options->flags || options->reserved)return RNS8_INVALID_ARGUMENT;
    if(e->running || e->group_references || e->dependent_references || e->producer_count)return RNS8_BUSY;
    if(!options->mode) {
      // On a failed free retain the owner and retryable allocation identity.
      e->native_integrity_enabled=false;
      return close_prepared_integrity(*e);
    }
    if(options->max_scratch_bytes<8)return RNS8_WORKSPACE_TOO_SMALL;
    auto status=validate_prepared(*e);if(status!=RNS8_SUCCESS)return status;
    if(!inputs_match(*e))return RNS8_INVALID_ARGUMENT;
    boost::multiprecision::cpp_int bound;rns8_exact_matrix_ranges ranges;
    status=prepared_range(*e,bound,ranges);if(status!=RNS8_SUCCESS)return status;
    status=admit_prepared_integrity(*e,bound);if(status!=RNS8_SUCCESS)return status;
    status=rns8_prepare_hip_submission(e);if(status!=RNS8_SUCCESS)return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device(e->ctx->device_id);
    if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(!e->integrity_flag){status=e->integrity_flag.allocate(e->ctx->device_id,4);if(status!=RNS8_SUCCESS)return status;}
    if(e->integrity_host_flag.ensure(e->ctx->device_id,4)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(!e->integrity_stream && hipStreamCreateWithFlags(&e->integrity_stream,hipStreamNonBlocking)!=hipSuccess)
      return RNS8_BACKEND_FAILURE;
    e->native_integrity_enabled=true;return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
static void begin_prepared_integrity(rns8_prepared_execution& e,const uint64_t bound[3]) {
  auto& report=e.native_integrity_report;report={};
  report.struct_size=sizeof(report);report.abi_version=RNS8_ABI_VERSION;
  report.enabled=e.native_integrity_enabled;report.operation_id=e.id;report.epoch=e.epoch;
  report.a_instance=e.a_instance;report.a_generation=e.a_generation;
  report.b_instance=e.b_instance;report.b_generation=e.b_generation;
  report.c_instance=e.c->matrix_instance_id;report.c_generation=e.c->content_generation;
  if(!e.native_integrity_enabled)return;
  report.computational_basis_version=1;report.computational_prefix=e.plan->prefix;
  report.unsigned_inputs=e.plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
  report.check_moduli[0]=257;report.check_moduli[1]=263;
  for(unsigned i=0;i<3;++i)report.absolute_bound[i]=bound[i];
  report.retained_device_bytes=4;report.retained_pinned_host_bytes=4;
}
static rns8_status enqueue_prepared_integrity(rns8_prepared_execution& e,void* stream) {
  if(!e.native_integrity_enabled)return RNS8_SUCCESS;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto& report=e.native_integrity_report;
  rns8::detail::IntegrityIdentity identity{e.id,e.epoch,e.a_instance,e.a_generation,e.b_instance,e.b_generation,1,e.plan->prefix};
  return rns8::detail::hip_direct_enqueue_native_product_integrity(e.ctx->device_id,
      integrity_native_pointer(*e.a,report.unsigned_inputs),integrity_native_pointer(*e.b,report.unsigned_inputs),
      e.c->hip_residues,e.plan->desc.m,e.plan->desc.n,e.plan->desc.k,e.a->desc.cols,e.b->desc.cols,
      report.unsigned_inputs!=0,report.absolute_bound,identity,identity,e.integrity_flag.get(),e.integrity_host_flag.get(),stream);
#else
  (void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
static rns8_status finish_prepared_integrity(rns8_prepared_execution& e,rns8_status operation) {
  auto& report=e.native_integrity_report;
  if(!e.native_integrity_enabled)return operation;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(operation==RNS8_SUCCESS) {
    const auto failures=*static_cast<const uint32_t*>(e.integrity_host_flag.get());
    if(failures&~15u)operation=RNS8_BACKEND_FAILURE; // copy not completed / malformed checker result
    else {report.failure_mask=failures;report.outcome=failures?2:1;
      if(failures)operation=RNS8_VERIFICATION_FAILED;}
  }
#endif
  if(operation==RNS8_CANCELED)report.outcome=4;
  else if(operation!=RNS8_SUCCESS && report.outcome!=2)report.outcome=3;
  report.operation_status=operation;
  return operation;
}
rns8_status rns8_get_prepared_native_integrity_report(const rns8_prepared_execution* e,
    rns8_completion_token token,rns8_native_integrity_report* report) {
  if(!e || !report || !rns8::detail::valid_abi(report->struct_size,report->abi_version,sizeof(*report)) ||
      token.prepared_id!=e->id || !token.epoch || token.epoch!=e->epoch ||
      e->native_integrity_report.operation_id!=token.prepared_id || e->native_integrity_report.epoch!=token.epoch)
    return RNS8_INVALID_ARGUMENT;
  if(e->running)return RNS8_BUSY;
  *report=e->native_integrity_report;return RNS8_SUCCESS;
}
