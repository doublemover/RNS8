// Caller capture is an explicit composition contract. A body has no epoch until
// the finalized executable is launched through the ordinary prepared machinery.
struct rns8_caller_hip_graph {
  rns8_prepared_execution* execution=nullptr;
  const void *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr;
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
rns8_status rns8_capture_prepared_in_caller_hip_graph(rns8_prepared_execution* e,
    void* stream,rns8_caller_hip_graph** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!e || !stream || e->closing)return RNS8_INVALID_ARGUMENT;
    if(e->running || e->caller_graph_references || e->producer_count || e->dependent_references || e->group_references)return RNS8_BUSY;
    if(!e->hip_submission_prepared)return RNS8_UNSUPPORTED_BACKEND;
    auto status=prepared_hip_support_status(*e);if(status!=RNS8_SUCCESS)return status;
    status=validate_prepared(*e);if(status!=RNS8_SUCCESS)return status;
    if(!inputs_match(*e))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device selected(e->ctx->device_id);
    if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    auto capture=static_cast<hipStream_t>(stream);
    if(capture==hipStreamLegacy || capture==hipStreamPerThread)return RNS8_INVALID_ARGUMENT;
    hipDevice_t device=-1;if(hipStreamGetDevice(capture,&device)!=hipSuccess || device!=e->ctx->device_id)return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<rns8_caller_hip_graph>();
    hipStreamCaptureStatus state=hipStreamCaptureStatusNone;
    if(hipStreamGetCaptureInfo_v2(capture,&state,&owner->capture_id,&owner->source,nullptr,nullptr)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(state!=hipStreamCaptureStatusActive || !owner->source || !owner->capture_id)return RNS8_INVALID_ARGUMENT;
    owner->execution=e;owner->capture_stream=stream;
    owner->a=prepared_device_storage(*e->a);owner->b=prepared_device_storage(*e->b);
    owner->c=prepared_device_storage(*e->c);owner->d=e->d?prepared_device_storage(*e->d):nullptr;
    {auto& registry=caller_capture_registry();std::lock_guard<std::mutex> lock(registry.mutex);
      // A graph with several independently reserved owners could launch the
      // unreserved bodies. Reject it; multi-operation composition uses the DAG.
      if(!registry.bindings.emplace(owner->capture_id,owner.get()).second)return RNS8_UNSUPPORTED_BACKEND;}
    // Retain before the first node may be appended, including partial failures.
    ++e->caller_graph_references;*out=owner.release();
    status=enqueue_prepared(*e,stream);
    (*out)->body_complete=status==RNS8_SUCCESS;return status;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_finalize_caller_hip_graph(rns8_caller_hip_graph* owner,void* graph) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !graph || !owner->body_complete)return RNS8_INVALID_ARGUMENT;
    auto* e=owner->execution;if(e->running)return RNS8_BUSY;
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
rns8_status rns8_submit_caller_hip_graph(rns8_caller_hip_graph* owner,void* stream,rns8_completion_token* token) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !token || !owner->body_complete)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    auto* e=owner->execution;
    if(!owner->executable || owner->a!=prepared_device_storage(*e->a) || owner->b!=prepared_device_storage(*e->b) ||
        owner->c!=prepared_device_storage(*e->c) || owner->d!=(e->d?prepared_device_storage(*e->d):nullptr))return RNS8_INVALID_ARGUMENT;
    PreparedHipProof proof;
    const auto status=prove_prepared_submission(e,stream,false,proof);if(status!=RNS8_SUCCESS)return status;
    return start_prepared_submission(e,stream,token,true,proof,owner->executable);
#else
    (void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_destroy_caller_hip_graph(rns8_caller_hip_graph* owner) {
  if(!owner)return RNS8_SUCCESS;
  auto* e=owner->execution;if(e->running)return RNS8_BUSY;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_scoped_device selected(e->ctx->device_id);
  if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(owner->executable && hipGraphExecDestroy(owner->executable)!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
  {auto& registry=caller_capture_registry();std::lock_guard<std::mutex> lock(registry.mutex);
    registry.bindings.erase(owner->capture_id);}
  --e->caller_graph_references;delete owner;return RNS8_SUCCESS;
}
