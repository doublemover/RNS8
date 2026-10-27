// Public ownership for the private native queue. The arithmetic family remains
// direct-HIP exact-wide WMMA; no CPU implementation is labelled matrix-engine.
namespace {
bool native_group_busy(const rns8_prepared_group& group) {
  if(group.running)return true;
  for(const auto* e:group.executions)if(e->running || e->producer_count || e->dependent_references)return true;
  return false;
}
rns8_status close_native_group_graph(rns8_prepared_group& group) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(group.native_executable) {
    if(hipGraphExecDestroy(group.native_executable)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    group.native_executable=nullptr;
  }
  if(group.native_graph) {
    if(hipGraphDestroy(group.native_graph)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    group.native_graph=nullptr;
  }
#else
  (void)group;
#endif
  return RNS8_SUCCESS;
}
rns8_status close_native_group(rns8_prepared_group& group) {
  if(!group.native_queue
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      && !group.native_stream && !group.native_graph && !group.native_executable
#endif
  )return RNS8_SUCCESS;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_scoped_device device(group.executions.front()->ctx->device_id);
  if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
  auto status=close_native_group_graph(group);if(status!=RNS8_SUCCESS)return status;
  if(group.native_queue) {status=group.native_queue->close();if(status!=RNS8_SUCCESS)return status;}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(group.native_stream) {
    if(hipStreamDestroy(group.native_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    group.native_stream=nullptr;
  }
#endif
  return RNS8_SUCCESS;
}
rns8_status admit_native_group_child(rns8_prepared_execution& e,
    const rns8_native_group_binding& binding,rns8_prepared_group::NativeBinding& value) {
  using namespace rns8::detail;
  if(!valid_abi(binding.struct_size,binding.abi_version,sizeof(binding)) ||
      (binding.modulus_group!=1 && binding.modulus_group!=2 && binding.modulus_group!=4))return RNS8_INVALID_ARGUMENT;
  if(e.transfer)return RNS8_UNSUPPORTED_BACKEND;
  if(e.d || e.plan->backend!=RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(e.plan->desc.semantics) ||
      !e.plan->tile_schedule.empty())return RNS8_UNSUPPORTED_BACKEND;
  auto status=plan_operation_preflight(*e.ctx,*e.plan,RNS8_OP_NATIVE_GROUP,RNS8_EXEC_GROUP,RNS8_QUERY_NATIVE_RETAINED);
  if(status!=RNS8_SUCCESS)return status;
  status=validate_prepared(e);if(status!=RNS8_SUCCESS)return status;
  const bool u=e.plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
  const auto native=[u](const rns8_matrix& matrix) {return u?matrix.hip_native_u64:matrix.hip_native_i64;};
  for(const auto* matrix:{e.a,e.b}) {
    const auto bytes=u?matrix->hip_native_u64_bytes:matrix->hip_native_i64_bytes;
    if(!(matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !matrix->device_native_current || !native(*matrix) ||
        bytes!=uint64_t(matrix->desc.rows)*uint64_t(matrix->desc.cols)*8)return RNS8_INVALID_ARGUMENT;
  }
  const auto* cache=binding.encoded_b;
  if(cache && (cache->ctx!=e.ctx || cache->source!=e.b || cache->instance!=e.b->matrix_instance_id ||
      cache->generation!=e.b->content_generation || e.b->prefix!=e.plan->prefix || !cache->storage))return RNS8_INVALID_ARGUMENT;
  boost::multiprecision::cpp_int bound;rns8_exact_matrix_ranges ranges;
  status=prepared_range(e,bound,ranges);if(status!=RNS8_SUCCESS)return status;
  value.task={native(*e.a),native(*e.b),static_cast<int8_t*>(e.c->hip_residues),e.plan->desc.m,e.plan->desc.n,e.plan->desc.k,
      e.a->desc.cols,e.b->desc.cols,e.c->desc.cols,e.plan->prefix,binding.modulus_group,u?1u:0u,
      cache?static_cast<const uint32_t*>(cache->storage.get()):nullptr,
      cache?((uint64_t(e.plan->desc.k)+15)/16)*4:0};
  // Native encoding/WMMA is a different lowering from the ordinary RNS plan.
  // Recheck the shared per-modulus slab invariant once with its own identity.
  if(!e.plan->arithmetic_guard)return RNS8_INVALID_ARGUMENT;
  auto certificate=e.plan->arithmetic_guard->certificate;
  certificate.instruction.opcode="native_wmma_i8_16x16x16";
  certificate.instruction.lowering_identity="native-queue-v1;plan="+e.plan->arithmetic_guard->checked.sha256+
      ";group="+std::to_string(binding.modulus_group)+";encoded_b_layout="+(cache?"1":"0")+
      ";a_instance="+std::to_string(e.a->matrix_instance_id)+";a_generation="+std::to_string(e.a->content_generation)+
      ";b_instance="+std::to_string(e.b->matrix_instance_id)+";b_generation="+std::to_string(e.b->content_generation);
  value.arithmetic_guard=std::make_shared<const arithmetic::Guard>(std::move(certificate));
  if(!value.arithmetic_guard->checked.admitted())return RNS8_RANGE_ERROR;
  value.a_instance=e.a->matrix_instance_id;value.a_generation=e.a->content_generation;
  value.b_instance=e.b->matrix_instance_id;value.b_generation=e.b->content_generation;
  return RNS8_SUCCESS;
}
rns8_status native_group_values_current(const rns8_prepared_group& group) {
  if(!group.native_queue || group.native_values.size()!=group.executions.size())return RNS8_INVALID_ARGUMENT;
  for(size_t i=0;i<group.executions.size();++i) {
    const auto& e=*group.executions[i];const auto& value=group.native_values[i];
    const auto* cache=group.native_bindings[i].encoded_b;
    const bool u=e.plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    if(!value.arithmetic_guard || !value.arithmetic_guard->checked.admitted() || value.a_instance!=e.a->matrix_instance_id || value.a_generation!=e.a->content_generation ||
        value.b_instance!=e.b->matrix_instance_id || value.b_generation!=e.b->content_generation ||
        value.task.a!=(u?e.a->hip_native_u64:e.a->hip_native_i64) ||
        value.task.b!=(u?e.b->hip_native_u64:e.b->hip_native_i64) || value.task.c!=e.c->hip_residues ||
        !e.a->device_native_current || !e.b->device_native_current ||
        (cache && (cache->generation!=e.b->content_generation || value.task.encoded_b!=cache->storage.get())))
      return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}
}
static rns8_status prepare_native_group_impl(rns8_prepared_group* group,const rns8_native_group_desc* desc,
    const rns8_native_group_binding* bindings,uint32_t count,const rns8_locality_desc* locality) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!group || !desc || !bindings || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
        count!=group->executions.size() || count>4096 || desc->schedule_mode>2 ||
        !desc->claim_chunk || desc->claim_chunk>32 || !desc->workers || desc->workers>1024 ||
        desc->ordinary_grid_control>1)return RNS8_INVALID_ARGUMENT;
    if(locality && (!locality_descriptor_valid(*locality) || desc->schedule_mode || desc->ordinary_grid_control))return RNS8_INVALID_ARGUMENT;
    if(native_group_busy(*group))return RNS8_BUSY;
    // Copy before replacement: callers may pass this owner's current bindings.
    const auto config=*desc;
    std::vector<rns8_native_group_binding> retained(bindings,bindings+count);
    std::vector<rns8_prepared_group::NativeBinding> values(count);
    std::vector<NativeWmmaTask> tasks;tasks.reserve(count);
    std::unordered_set<const rns8_workspace*> workspaces;
    std::unordered_map<const rns8_encoded_b*,uint64_t> cache_refs;
    auto* ctx=group->executions.front()->ctx;
    for(uint32_t i=0;i<count;++i) {
      auto& e=*group->executions[i];
      if(e.ctx!=ctx || !workspaces.insert(e.workspace).second)return RNS8_INVALID_ARGUMENT;
      const auto status=admit_native_group_child(e,retained[i],values[i]);if(status!=RNS8_SUCCESS)return status;
      tasks.push_back(values[i].task);
      if(retained[i].encoded_b)++cache_refs[retained[i].encoded_b];
    }
    std::string proof_identity="native-queue-schedule-v1;mode="+std::to_string(config.schedule_mode)+
        ";chunk="+std::to_string(config.claim_chunk)+";workers="+std::to_string(config.workers)+
        ";ordinary="+std::to_string(config.ordinary_grid_control);
    for(const auto& value:values)proof_identity+=';'+value.arithmetic_guard->checked.sha256;
    if(locality)proof_identity+=";locality="+std::to_string(locality->order)+","+std::to_string(locality->target_id)+","+std::to_string(locality->topology_version)+","+
      std::to_string(locality->cache_domains)+","+std::to_string(locality->concurrent_groups_per_domain)+","+std::to_string(locality->cache_bytes_per_domain);
    const auto proof_hash=sha256(proof_identity);
    for(const auto& [cache,references]:cache_refs)
      if(cache->prepared_references>UINT64_MAX-references)return RNS8_RANGE_ERROR;
    const auto old_bytes=group->native_queue?group->native_queue->device_metadata_bytes():0;
    if(config.max_metadata_bytes<=old_bytes)return RNS8_WORKSPACE_TOO_SMALL;
    auto queue=std::make_unique<NativeWmmaPreparedQueue>();
    auto status=queue->prepare(ctx->device_id,tasks,config.workers,config.max_metadata_bytes-old_bytes,
        {1,static_cast<NativeWmmaScheduleMode>(config.schedule_mode),config.claim_chunk},locality);
    if(status!=RNS8_SUCCESS)return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device device(ctx->device_id);if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    // Events/stream are reusable owners. Partial setup never binds new values or
    // publishes outputs, and can be retried without descriptor replacement.
    for(auto* e:group->executions) {
      if(!e->completion_event && e->completion_event.create()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    }
    if(!group->native_stream && hipStreamCreateWithFlags(&group->native_stream,hipStreamNonBlocking)!=hipSuccess)
      return RNS8_BACKEND_FAILURE;
    status=close_native_group_graph(*group);if(status!=RNS8_SUCCESS)return status;
#endif
    if(group->native_queue) {status=group->native_queue->close();if(status!=RNS8_SUCCESS)return status;}
    for(const auto& binding:retained)if(binding.encoded_b)++binding.encoded_b->prepared_references;
    for(const auto& binding:group->native_bindings)if(binding.encoded_b)--binding.encoded_b->prepared_references;
    group->native_peak_device_bytes=old_bytes+queue->preparation_peak_device_metadata_bytes();
    group->native_queue=std::move(queue);group->native_values=std::move(values);
    group->native_bindings=std::move(retained);group->native_desc=config;group->native_locality=locality?*locality:rns8_locality_desc{};
    group->native_certificate_sha256=proof_hash;
    for(auto* e:group->executions) {bind_inputs(*e);e->hip_submission_prepared=true;}
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_prepare_native_group(rns8_prepared_group* group,const rns8_native_group_desc* desc,
    const rns8_native_group_binding* bindings,uint32_t count){return prepare_native_group_impl(group,desc,bindings,count,nullptr);}
rns8_status rns8_prepare_native_group_locality(rns8_prepared_group* group,const rns8_native_group_desc* desc,
    const rns8_native_group_binding* bindings,uint32_t count,const rns8_locality_desc* locality){
  if(!locality)return RNS8_INVALID_ARGUMENT;return prepare_native_group_impl(group,desc,bindings,count,locality);}
rns8_status rns8_get_native_group_locality_info(const rns8_prepared_group* group,rns8_locality_info* out){
  if(!group || !group->native_queue || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  const auto info=group->native_queue->locality_info();if(!info.struct_size)return RNS8_INVALID_ARGUMENT;*out=info;return RNS8_SUCCESS;}
rns8_status rns8_get_native_group_info(const rns8_prepared_group* group,rns8_native_group_info* info) {
  if(!group || !info || !group->native_queue || !rns8::detail::valid_abi(info->struct_size,info->abi_version,sizeof(*info)))
    return RNS8_INVALID_ARGUMENT;
  *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;
  const auto header=group->native_queue->schedule_header();
  info->task_count=header.task_count;info->work_count=header.work_count;info->schedule_mode=header.mode;
  info->claim_chunk=header.claim_chunk;info->workers=group->native_desc.workers;
  info->ordinary_grid_control=group->native_desc.ordinary_grid_control;
  info->retained_device_metadata_bytes=group->native_queue->device_metadata_bytes();
  info->retained_host_metadata_bytes=group->native_queue->retained_host_metadata_bytes()+
      group->native_values.capacity()*sizeof(rns8_prepared_group::NativeBinding)+
      group->native_bindings.capacity()*sizeof(rns8_native_group_binding);
  info->preparation_peak_device_metadata_bytes=group->native_peak_device_bytes;
  std::copy(group->native_certificate_sha256.begin(),group->native_certificate_sha256.end(),info->arithmetic_certificate_sha256);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  info->graph_ready=group->native_executable!=nullptr;
#endif
  return RNS8_SUCCESS;
}
static rns8_status submit_native_group(rns8_prepared_group* group,void* stream,
    rns8_completion_token* tokens,uint32_t capacity,uint32_t* out_submitted,bool graph) {
  return guard_api([&]() -> rns8_status {
    if(!group || !stream || !tokens || !out_submitted || capacity<group->executions.size())return RNS8_INVALID_ARGUMENT;
    *out_submitted=0;
    if(native_group_busy(*group))return RNS8_BUSY;
    auto status=native_group_values_current(*group);if(status!=RNS8_SUCCESS)return status;
    std::vector<PreparedHipProof> proofs(group->executions.size());
    std::unordered_map<const rns8_matrix*,uint64_t> readers;
    for(size_t i=0;i<group->executions.size();++i) {
      auto* e=group->executions[i];
      status=prove_prepared_submission(e,stream,false,proofs[i]);if(status!=RNS8_SUCCESS)return status;
      ++readers[e->a];if(e->a!=e->b)++readers[e->b];
    }
    for(const auto& [matrix,count]:readers)if(matrix->in_flight_readers>UINT64_MAX-count)return RNS8_RANGE_ERROR;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(graph && !group->native_executable)return RNS8_UNSUPPORTED_BACKEND;
    auto* ctx=group->executions.front()->ctx;
    rns8::detail::hip_scoped_device device(ctx->device_id);if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    // All throwing allocation/proofs and whole-batch hazards are complete.
    // Reserve EVERY child before the first device write; observation uses the
    // same single-operation completion machinery, including downstream DAGs.
    for(size_t i=0;i<group->executions.size();++i) {
      auto& e=*group->executions[i];e.pending_bound.swap(proofs[i].bound);e.pending_ranges=std::move(proofs[i].ranges);
      invalidate_output_currentness(*e.c);++e.a->in_flight_readers;if(e.a!=e.b)++e.b->in_flight_readers;
      ++e.c->prepared_in_flight_writers;e.c->in_flight_writer=true;e.workspace->in_flight=true;e.running=true;e.pending_stream=stream;
      e.async_failure={};e.cancel_requested=false;e.event_dependencies=false;e.event_dependency_depth=0;
      ++e.epoch;tokens[i]={e.id,e.epoch};++*out_submitted;begin_prepared_integrity(e,proofs[i].integrity_bound);
    }
    hipError_t error=hipSuccess;bool written=false;
    if(graph) {error=hipGraphLaunch(group->native_executable,static_cast<hipStream_t>(stream));
      status=error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;}
    else status=group->native_queue->enqueue_borrowed_stream(stream,group->native_desc.ordinary_grid_control!=0,written);
    auto stage=RNS8_ASYNC_FAILURE_ENQUEUE;
    if(status==RNS8_SUCCESS)for(auto* e:group->executions) {
      status=enqueue_prepared_integrity(*e,stream);if(status!=RNS8_SUCCESS)break;
      error=hipEventRecord(e->completion_event.get(),static_cast<hipStream_t>(stream));
      if(error!=hipSuccess) {status=RNS8_BACKEND_FAILURE;stage=RNS8_ASYNC_FAILURE_EVENT_RECORD;break;}
    }
    if(status!=RNS8_SUCCESS) {
      // An event-record failure after any earlier event invalidates the WHOLE
      // submission; earlier events must not accidentally publish success.
      const auto drain=hipStreamSynchronize(static_cast<hipStream_t>(stream));
      for(auto* e:group->executions) {
        record_async_failure(e->async_failure,status,stage,error,ctx->device_id,e->epoch);
        if(record_async_drain(*ctx,e->async_failure,drain))finish_prepared_hip(*e,status);
      }
    }
    return status;
#else
    (void)graph;return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_submit_native_group_hip(rns8_prepared_group* group,void* stream,
    rns8_completion_token* tokens,uint32_t capacity,uint32_t* submitted) {
  return submit_native_group(group,stream,tokens,capacity,submitted,false);
}
rns8_status rns8_submit_native_group_hip_graph(rns8_prepared_group* group,void* stream,
    rns8_completion_token* tokens,uint32_t capacity,uint32_t* submitted) {
  return submit_native_group(group,stream,tokens,capacity,submitted,true);
}
rns8_status rns8_capture_native_group_hip_graph(rns8_prepared_group* group) {
  return guard_api([&]() -> rns8_status {
    if(!group)return RNS8_INVALID_ARGUMENT;
    if(native_group_busy(*group))return RNS8_BUSY;
    auto status=native_group_values_current(*group);if(status!=RNS8_SUCCESS)return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    auto* ctx=group->executions.front()->ctx;
    rns8::detail::hip_scoped_device device(ctx->device_id);if(device.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    // Own the capture stream; public borrowed-stream submission rejects external capture.
    rns8::detail::hip_unique_graph replacement;
    if(replacement.reserve()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    rns8::detail::hip_capture_scope capture(group->native_stream,replacement);
    if(capture.begin()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    bool written=false;status=group->native_queue->enqueue_borrowed_stream(group->native_stream,
        group->native_desc.ordinary_grid_control!=0,written);
    const auto end=capture.end();
    if(status!=RNS8_SUCCESS || end!=hipSuccess)return status!=RNS8_SUCCESS?status:RNS8_BACKEND_FAILURE;
    rns8::detail::hip_unique_graph_exec executable;
    if(executable.instantiate(replacement.get())!=hipSuccess)return RNS8_BACKEND_FAILURE;
    status=close_native_group_graph(*group);
    if(status!=RNS8_SUCCESS)return status;
    group->native_graph=replacement.release();group->native_executable=executable.release();return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
static rns8_status execute_native_group_sync(rns8_prepared_group* group,rns8_completion_token* tokens,
    uint32_t capacity,uint32_t* executed) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=rns8_submit_native_group_hip(group,group->native_stream,tokens,capacity,executed);
  // Even a partial/failed launch may own resources. Drain every issued token;
  // never return early while a later child's reservations are still live.
  for(uint32_t i=0;i<*executed;++i) {
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    const auto wait=rns8_wait_prepared_hip(group->executions[i],tokens[i],&info);
    if(status==RNS8_SUCCESS)status=wait==RNS8_SUCCESS?info.operation_status:wait;
  }
  return status;
#else
  (void)group;(void)tokens;(void)capacity;(void)executed;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
