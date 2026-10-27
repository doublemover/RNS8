// Caller capture is an explicit composition contract. A body has no epoch until
// the finalized executable is launched through the ordinary prepared machinery.
struct rns8_caller_hip_graph {
  struct Member {
    rns8_prepared_execution* execution=nullptr;
    const void *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr;
  };
  std::vector<Member> members;
  rns8_prepared_group* group=nullptr;
  void* capture_stream=nullptr;
  unsigned long long capture_id=0;
  bool body_complete=false;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hipGraph_t source=nullptr;
  hipGraphExec_t executable=nullptr;
#endif
};
namespace {
struct CallerCaptureRegistry {
  std::mutex mutex;std::unordered_map<unsigned long long,const rns8_caller_hip_graph*> bindings;
};
CallerCaptureRegistry& caller_capture_registry(){static CallerCaptureRegistry registry;return registry;}
}
rns8_status rns8_capture_prepared_group_in_caller_hip_graph(
    rns8_prepared_execution* const* executions,uint32_t count,
    void* stream,rns8_caller_hip_graph** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!executions || !count || count>64 || !stream)return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<rns8_caller_hip_graph>();
    owner->members.reserve(count);
    std::unordered_set<rns8_workspace*> workspaces;
    for(uint32_t i=0;i<count;++i) {
      auto* e=executions[i];
      if(!e || e->closing)return RNS8_INVALID_ARGUMENT;
      if(e->running || e->caller_graph_references || e->producer_count || e->dependent_references || e->group_references)return RNS8_BUSY;
      if(e->ctx!=executions[0]->ctx)return RNS8_INVALID_ARGUMENT;
      if(e->workspace && !workspaces.insert(e->workspace).second)return RNS8_INVALID_ARGUMENT;
      if(!e->hip_submission_prepared)return RNS8_UNSUPPORTED_BACKEND;
      auto status=prepared_hip_support_status(*e);if(status!=RNS8_SUCCESS)return status;
      status=validate_prepared(*e);if(status!=RNS8_SUCCESS)return status;
      if(!inputs_match(*e))return RNS8_INVALID_ARGUMENT;
      owner->members.push_back({e,prepared_device_storage(*e->a),prepared_device_storage(*e->b),
          prepared_device_storage(*e->c),e->d?prepared_device_storage(*e->d):nullptr});
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    auto* first=executions[0];
    rns8::detail::hip_scoped_device selected(first->ctx->device_id);
    if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    auto capture=static_cast<hipStream_t>(stream);
    if(capture==hipStreamLegacy || capture==hipStreamPerThread)return RNS8_INVALID_ARGUMENT;
    hipDevice_t device=-1;if(hipStreamGetDevice(capture,&device)!=hipSuccess || device!=first->ctx->device_id)return RNS8_INVALID_ARGUMENT;
    hipStreamCaptureStatus state=hipStreamCaptureStatusNone;
    if(hipStreamGetCaptureInfo_v2(capture,&state,&owner->capture_id,&owner->source,nullptr,nullptr)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(state!=hipStreamCaptureStatusActive || !owner->source || !owner->capture_id)return RNS8_INVALID_ARGUMENT;
    owner->capture_stream=stream;
    // Validate cross-body matrix and external-lease hazards before any node.
    auto status=rns8_create_prepared_group(executions,count,&owner->group);
    if(status!=RNS8_SUCCESS)return status;
    std::unique_ptr<rns8_prepared_group,decltype(&rns8_destroy_prepared_group)> held_group(owner->group,&rns8_destroy_prepared_group);
    {auto& registry=caller_capture_registry();std::lock_guard<std::mutex> lock(registry.mutex);
      if(!registry.bindings.emplace(owner->capture_id,owner.get()).second) {
        return RNS8_INVALID_ARGUMENT;
      }}
    // Partial capture failures retain every resource referenced by appended nodes.
    for(auto& member:owner->members)++member.execution->caller_graph_references;
    held_group.release();*out=owner.release();
    for(auto& member:(*out)->members) {
      status=enqueue_prepared(*member.execution,stream);
      if(status!=RNS8_SUCCESS)return status;
    }
    (*out)->body_complete=true;return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_capture_prepared_in_caller_hip_graph(rns8_prepared_execution* e,
    void* stream,rns8_caller_hip_graph** out) {
  return rns8_capture_prepared_group_in_caller_hip_graph(&e,1,stream,out);
}
rns8_status rns8_finalize_caller_hip_graph(rns8_caller_hip_graph* owner,void* graph) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !graph || !owner->body_complete)return RNS8_INVALID_ARGUMENT;
    for(auto& member:owner->members)if(member.execution->running)return RNS8_BUSY;
    auto* e=owner->members.front().execution;
    if(e->ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(owner->source!=static_cast<hipGraph_t>(graph))return RNS8_INVALID_ARGUMENT;
    rns8::detail::hip_scoped_device selected(e->ctx->device_id);
    if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    hipStreamCaptureStatus state=hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(static_cast<hipStream_t>(owner->capture_stream),&state)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(state!=hipStreamCaptureStatusNone)return RNS8_BUSY;
    hipGraphExec_t replacement=nullptr;
    if(hipGraphInstantiate(&replacement,static_cast<hipGraph_t>(graph),nullptr,nullptr,0)!=hipSuccess){
      if(replacement)(void)hipGraphExecDestroy(replacement);return RNS8_BACKEND_FAILURE;
    }
    if(owner->executable && hipGraphExecDestroy(owner->executable)!=hipSuccess){
      (void)hipGraphExecDestroy(replacement);return RNS8_BACKEND_FAILURE;
    }
    owner->executable=replacement;return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_submit_caller_hip_graph_batch(rns8_caller_hip_graph* owner,void* stream,
    rns8_completion_token* tokens,uint32_t capacity,uint32_t* out_submitted) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !tokens || !out_submitted || !owner->body_complete || capacity<owner->members.size())return RNS8_INVALID_ARGUMENT;
    *out_submitted=0;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(!owner->executable)return RNS8_INVALID_ARGUMENT;
    const auto count=owner->members.size();
    std::vector<PreparedHipProof> proofs(count);
    std::unordered_map<const rns8_matrix*,uint64_t> readers;
    readers.reserve(count*3);
    for(size_t i=0;i<count;++i) {
      const auto& member=owner->members[i];auto* e=member.execution;
      if(member.a!=prepared_device_storage(*e->a) || member.b!=prepared_device_storage(*e->b) ||
          member.c!=prepared_device_storage(*e->c) || member.d!=(e->d?prepared_device_storage(*e->d):nullptr))return RNS8_INVALID_ARGUMENT;
      auto status=prove_prepared_submission(e,stream,false,proofs[i]);if(status!=RNS8_SUCCESS)return status;
      if(!e->transfer || e->transfer->reads_matrix()) {
        ++readers[e->a];if(e->b!=e->a)++readers[e->b];if(e->d && e->d!=e->a && e->d!=e->b)++readers[e->d];
      }
    }
    for(const auto& entry:readers)if(entry.first->in_flight_readers>UINT64_MAX-entry.second)return RNS8_RANGE_ERROR;
    auto* first=owner->members.front().execution;
    rns8::detail::hip_scoped_device selected(first->ctx->device_id);
    if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    // Reserve the complete graph before a single body can execute.
    for(size_t i=0;i<count;++i)begin_prepared_submission(owner->members[i].execution,stream,tokens+i,proofs[i]);
    *out_submitted=static_cast<uint32_t>(count);
    rns8_status status=RNS8_SUCCESS;hipError_t error=hipSuccess;
    for(auto& member:owner->members)if(member.execution->transfer) {
      status=member.execution->transfer->before_enqueue(stream);if(status!=RNS8_SUCCESS)break;
    }
    if(status==RNS8_SUCCESS) {
      error=hipGraphLaunch(owner->executable,static_cast<hipStream_t>(stream));
      if(error!=hipSuccess)status=RNS8_BACKEND_FAILURE;
    }
    if(status==RNS8_SUCCESS)for(auto& member:owner->members) {
      status=enqueue_prepared_integrity(*member.execution,stream);if(status!=RNS8_SUCCESS)break;
    }
    auto stage=RNS8_ASYNC_FAILURE_ENQUEUE;
    if(status==RNS8_SUCCESS)for(auto& member:owner->members) {
      error=hipEventRecord(member.execution->completion_event.get(),static_cast<hipStream_t>(stream));
      if(error!=hipSuccess){status=RNS8_BACKEND_FAILURE;stage=RNS8_ASYNC_FAILURE_EVENT_RECORD;break;}
    }
    if(status!=RNS8_SUCCESS) {
      for(auto& member:owner->members)record_async_failure(member.execution->async_failure,status,stage,error,
          member.execution->ctx->device_id,member.execution->epoch);
      const auto drained=hipStreamSynchronize(static_cast<hipStream_t>(stream));
      for(auto& member:owner->members) {
        auto* e=member.execution;
        if(record_async_drain(*e->ctx,e->async_failure,drained))finish_prepared_hip(*e,status);
      }
    }
    return status;
#else
    (void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_submit_caller_hip_graph(rns8_caller_hip_graph* owner,void* stream,rns8_completion_token* token) {
  if(!owner || owner->members.size()!=1)return RNS8_INVALID_ARGUMENT;
  uint32_t submitted=0;return rns8_submit_caller_hip_graph_batch(owner,stream,token,1,&submitted);
}
rns8_status rns8_destroy_caller_hip_graph(rns8_caller_hip_graph* owner) {
  if(!owner)return RNS8_SUCCESS;
  for(auto& member:owner->members)if(member.execution->running)return RNS8_BUSY;
    auto* e=owner->members.front().execution;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_scoped_device selected(e->ctx->device_id);
  if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(owner->executable && hipGraphExecDestroy(owner->executable)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  owner->executable=nullptr;
#endif
  {auto& registry=caller_capture_registry();std::lock_guard<std::mutex> lock(registry.mutex);
    registry.bindings.erase(owner->capture_id);}
  const auto status=rns8_destroy_prepared_group(owner->group);
  if(status!=RNS8_SUCCESS)return status;
  for(auto& member:owner->members)--member.execution->caller_graph_references;
  delete owner;return RNS8_SUCCESS;
}
