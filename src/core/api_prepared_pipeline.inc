static uint32_t prepared_output_prefix(const rns8_prepared_execution& e) {
  return e.transfer?e.transfer->output_prefix():e.plan->prefix;
}
static rns8_semantics prepared_semantics(const rns8_prepared_execution& e) {
  return e.transfer?e.c->desc.semantics:e.plan->desc.semantics;
}
static rns8_status prove_transfer_pipeline(rns8_prepared_execution& e,
    boost::multiprecision::cpp_int& bound,rns8_exact_matrix_ranges& ranges) {
  if(!e.transfer->reads_matrix() || e.producer_count!=1)return RNS8_INVALID_ARGUMENT;
  const auto& binding=e.producers[0];auto& p=*binding.producer;
  if(p.ctx!=e.ctx || p.c!=e.a || binding.token.prepared_id!=p.id || binding.token.epoch!=p.epoch || !p.epoch ||
      (p.transfer && !p.transfer->produces_resident()) || prepared_semantics(p)!=e.a->desc.semantics)
    return RNS8_INVALID_ARGUMENT;
  if(p.event_dependency_depth>=64)return RNS8_RANGE_ERROR;
  ranges={};
  if(p.running) {
    if(!p.hip_submission_prepared || !p.pending_stream || !p.completion_event.get() || !p.c->in_flight_writer)return RNS8_INVALID_ARGUMENT;
    if(p.async_failure.stage!=RNS8_ASYNC_FAILURE_NONE)return RNS8_BACKEND_FAILURE;
    if(p.cancel_requested)return RNS8_CANCELED;
    return e.transfer->validate(&p.pending_bound,prepared_output_prefix(p),bound);
  }
  rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  const auto status=rns8_get_prepared_completion(&p,binding.token,&info);
  if(status!=RNS8_SUCCESS)return status;
  if(info.operation_status!=RNS8_SUCCESS)return info.operation_status;
  if(!info.output_is_current)return RNS8_INVALID_ARGUMENT;
  return e.transfer->validate(nullptr,0,bound);
}
// Token-qualified pending inputs are the only exception to ordinary writer
// exclusion. This preflight never changes matrix currentness or reservations.
rns8_status prove_pipeline_inputs(rns8_prepared_execution& e,
    boost::multiprecision::cpp_int& bound,rns8_exact_matrix_ranges& ranges) {
  if(e.transfer)return prove_transfer_pipeline(e,bound,ranges);
  if(!e.producer_count || (!rns8::detail::exact_wide_semantics(e.plan->desc.semantics) &&
      e.plan->desc.semantics!=RNS8_BOUNDED_I64 && e.plan->desc.semantics!=RNS8_BOUNDED_U64))
    return RNS8_UNSUPPORTED_BACKEND;
  auto status=validate_plan_context_workspace(*e.ctx,*e.plan,*e.workspace);
  if(status!=RNS8_SUCCESS) return status;
  for(uint32_t i=0;i<e.producer_count;++i) {
    const auto& binding=e.producers[i];const auto& p=*binding.producer;
    if(p.ctx!=e.ctx || binding.token.prepared_id!=p.id || binding.token.epoch!=p.epoch || !p.epoch)
      return RNS8_INVALID_ARGUMENT;
    if(prepared_semantics(p)!=e.plan->desc.semantics || prepared_output_prefix(p)<e.plan->prefix) return RNS8_INVALID_ARGUMENT;
    if(p.event_dependency_depth>=64) return RNS8_RANGE_ERROR;
    if(p.running) {
      if(!p.hip_submission_prepared || !p.pending_stream || !p.completion_event.get()) return RNS8_UNSUPPORTED_BACKEND;
      if(p.async_failure.stage!=RNS8_ASYNC_FAILURE_NONE) return RNS8_BACKEND_FAILURE;
      if(p.cancel_requested) return RNS8_CANCELED;
      if(!p.c->in_flight_writer) return RNS8_INVALID_ARGUMENT;
    } else {
      rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      status=rns8_get_prepared_completion(&p,binding.token,&info);
      if(status!=RNS8_SUCCESS) return status;
      if(info.operation_status!=RNS8_SUCCESS) return info.operation_status;
      if(!info.completed || !info.output_is_current) return RNS8_INVALID_ARGUMENT;
    }
  }
  std::array<rns8_matrix,3> proof_inputs;
  const std::array<const rns8_matrix*,3> inputs{e.a,e.b,e.d};
  const std::array<uint64_t,3> instances{e.a_instance,e.b_instance,e.d_instance};
  const std::array<uint64_t,3> generations{e.a_generation,e.b_generation,e.d_generation};
  for(size_t slot=0;slot<inputs.size();++slot) {
    const auto* input=inputs[slot];if(!input) continue;
    if(input->generation_exhausted) return RNS8_RANGE_ERROR;
    const rns8_prepared_execution* producer=nullptr;
    for(uint32_t i=0;i<e.producer_count;++i)
      if(e.producers[i].producer->c==input) producer=e.producers[i].producer;
    if(!producer && (input->matrix_instance_id!=instances[slot] || input->content_generation!=generations[slot]))
      return RNS8_INVALID_ARGUMENT;
    if(!producer || !producer->running) {
      if(input->in_flight_writer) return RNS8_BUSY;
      if(!rns_residue_state_current_for_backend(*input,e.plan->backend)) return RNS8_INVALID_ARGUMENT;
      if(!materialized_rns_prefix_covers(*input,e.plan->prefix)) return RNS8_RANGE_ERROR;
      proof_inputs[slot].exact_range_prefix=input->exact_range_prefix;
      proof_inputs[slot].exact_max_magnitude=input->exact_max_magnitude;
      proof_inputs[slot].exact_axis_ranges=input->exact_axis_ranges;
    } else {
      proof_inputs[slot].exact_range_prefix=prepared_output_prefix(*producer);
      proof_inputs[slot].exact_max_magnitude=producer->pending_bound;
      proof_inputs[slot].exact_axis_ranges=producer->pending_ranges;
    }
  }
  if(e.c->generation_exhausted) return RNS8_RANGE_ERROR;
  // Writer exceptions were tied to live, same-context producer tokens above.
  status=e.d?validate_exact_axpby_operands(*e.ctx,*e.plan,*e.a,*e.b,*e.d,*e.c,e.alpha,e.beta,false,true):
      validate_rns_gemm_operands(*e.ctx,*e.plan,*e.a,*e.b,*e.c,false,true);
  if(status!=RNS8_SUCCESS) return status;
  return e.d?rns8::detail::exact_axpby_range(*e.plan,proof_inputs[0],proof_inputs[1],proof_inputs[2],e.alpha,e.beta,bound):
      rns8::detail::exact_gemm_range(*e.plan,proof_inputs[0],proof_inputs[1],bound,&ranges);
}
// Observation may publish producers, but submission never does. Keep producer
// handles and reader reservations until all dependent device work is drained.
rns8_status observe_pipeline_inputs(rns8_prepared_execution& e,bool wait,rns8_status& operation) {
  operation=RNS8_SUCCESS;
  if(!e.event_dependencies) return RNS8_SUCCESS;
  for(uint32_t i=0;i<e.producer_count;++i) {
    const auto& binding=e.producers[i];auto* producer=binding.producer;
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    const auto status=producer->running
        ?(wait?rns8_wait_prepared_hip(producer,binding.token,&info):rns8_query_prepared_hip(producer,binding.token,&info))
        :rns8_get_prepared_completion(producer,binding.token,&info);
    if(status!=RNS8_SUCCESS) return status;
    if(info.operation_status!=RNS8_SUCCESS && operation==RNS8_SUCCESS) operation=info.operation_status;
    else if(info.operation_status==RNS8_SUCCESS && (!info.completed || !info.output_is_current) && operation==RNS8_SUCCESS)
      operation=RNS8_INVALID_ARGUMENT;
  }
  // Readers have prevented mutation since submission. Use the actual published
  // generations, never a prediction of the producer's next identity.
  bind_inputs(e);
  return RNS8_SUCCESS;
}
