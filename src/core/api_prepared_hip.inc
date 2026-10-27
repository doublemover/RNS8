namespace {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#include "api_prepared_pipeline.inc"
#endif
uint32_t prepared_storage_planes(const rns8_prepared_execution& e) {
  if(e.transfer)return e.transfer->output_prefix();
  if(e.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64) return 8;
  return uses_finite_storage(e.plan->desc.semantics)?1:e.plan->prefix;
}
const void* prepared_device_storage(const rns8_matrix& matrix) {
  return matrix.desc.semantics==RNS8_WRAP_U64_MOD_2_64
      ? matrix.hip_byte_limbs : matrix.hip_residues;
}
rns8_status prepared_hip_support_status(const rns8_prepared_execution& e) {
  if (e.ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
  if(e.transfer){boost::multiprecision::cpp_int bound;return e.transfer->validate(nullptr,0,bound);}
  const auto provider=plan_operation_preflight(*e.ctx,*e.plan,e.d?RNS8_OP_EXACT_EPILOGUE:RNS8_OP_GEMM,
      RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH);
  if(provider!=RNS8_SUCCESS)return provider;
  if (!direct_hip_compiled() || (e.plan->backend != RNS8_BACKEND_HIP_DIRECT && e.plan->backend != RNS8_BACKEND_AMDGPU_BUILTINS && e.plan->backend != RNS8_BACKEND_CK && e.plan->backend != RNS8_BACKEND_ROCWMMA && e.plan->backend != RNS8_BACKEND_HIPBLASLT) ||
      (!rns8::detail::exact_wide_semantics(e.plan->desc.semantics) &&
       !uses_finite_storage(e.plan->desc.semantics) &&
       e.plan->desc.semantics!=RNS8_BOUNDED_I64 &&
       e.plan->desc.semantics!=RNS8_BOUNDED_U64 &&
       e.plan->desc.semantics!=RNS8_WRAP_U64_MOD_2_64) ||
      !e.plan->tile_schedule.empty() ||
      !rns8::detail::qualified_i32_gemm_address(e.plan->desc.m, e.plan->desc.n,
          e.plan->desc.k, e.a->desc.cols, e.b->desc.cols, e.c->desc.cols, prepared_storage_planes(e)))
    return RNS8_UNSUPPORTED_BACKEND;
  return RNS8_SUCCESS;
}
void finish_prepared_hip(rns8_prepared_execution &e, rns8_status operation) {
  if (operation == RNS8_SUCCESS && e.cancel_requested) operation = RNS8_CANCELED;
  operation=finish_prepared_integrity(e,operation);
  if(e.transfer) {
    operation=e.transfer->finish(operation,e.pending_bound);
    bind_inputs(e);
  } else {
  --e.a->in_flight_readers;
  if (e.a != e.b)
    --e.b->in_flight_readers;
  if (e.d && e.d != e.a && e.d != e.b)
    --e.d->in_flight_readers;
  --e.c->prepared_in_flight_writers;
  e.c->in_flight_writer = false;
  e.workspace->in_flight = false;
  if (operation == RNS8_SUCCESS) {
    if(e.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64) {
      mark_output_device_byte_limbs_current(*e.c);
      e.c->finite_modulus=0;
      e.c->exact_axis_ranges.rows.clear();e.c->exact_axis_ranges.cols.clear();
    } else {
      mark_output_device_residues_current(*e.c, e.plan);
      if(uses_finite_storage(e.plan->desc.semantics)) {
        e.c->finite_modulus=static_cast<uint16_t>(e.plan->desc.finite_modulus);
        e.c->exact_axis_ranges.rows.clear();e.c->exact_axis_ranges.cols.clear();
      } else rns8::detail::commit_exact_range(*e.c, *e.plan, e.pending_bound,&e.pending_ranges);
    }
  }
  e.c->in_flight_writer=e.c->prepared_in_flight_writers!=0;
  }
  e.native_integrity_report.c_generation=e.c->content_generation;
  e.completion = {};
  e.completion.operation_status = operation;
  e.completion.a_instance = e.a_instance;
  e.completion.a_generation = e.a_generation;
  e.completion.b_instance = e.b_instance;
  e.completion.b_generation = e.b_generation;
  e.completion.c_instance = e.c->matrix_instance_id;
  e.completion.c_generation = e.c->content_generation;
  e.completion.completed = 1;
  record_epilogue_completion(e);
  e.running = false;
  e.pending_stream = nullptr;
  if(e.event_dependencies) {release_prepared_dependency(e);e.event_dependencies=false;e.event_dependency_depth=0;}
}
rns8_status observe_prepared_hip(rns8_prepared_execution *e,
                                 rns8_completion_token token,
                                 rns8_completion_info *out, bool wait) {
  return guard_api([&]() -> rns8_status {
    if (!e || !out ||
        !rns8::detail::valid_abi(out->struct_size, out->abi_version,
                                 sizeof(*out)) ||
        !e->hip_submission_prepared || token.prepared_id != e->id ||
        !token.epoch || token.epoch != e->epoch)
      return RNS8_INVALID_ARGUMENT;
    if (!e->running)
      return rns8_get_prepared_completion(e, token, out);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device(e->ctx->device_id);
    if (!e->pending_stream || device.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    rns8_status dependency_operation=RNS8_SUCCESS;
    const auto dependency_status=observe_pipeline_inputs(*e,wait,dependency_operation);
    if(dependency_status!=RNS8_SUCCESS) return dependency_status;
    if (e->async_failure.stage != RNS8_ASYNC_FAILURE_NONE) {
      if (!wait) return RNS8_BACKEND_FAILURE;
      if (!record_async_drain(*e->ctx, e->async_failure,
              hipStreamSynchronize(static_cast<hipStream_t>(e->pending_stream))))
        return RNS8_BACKEND_FAILURE;
      finish_prepared_hip(*e, e->async_failure.operation_status);
      return rns8_get_prepared_completion(e, token, out);
    }
    const auto error = wait ? hipEventSynchronize(e->completion_event.get())
                            : hipEventQuery(e->completion_event.get());
    if (error == hipErrorNotReady)
      return RNS8_BUSY;
    if (error != hipSuccess) {
      record_async_failure(e->async_failure, RNS8_BACKEND_FAILURE,
          RNS8_ASYNC_FAILURE_EVENT_OBSERVE, error, e->ctx->device_id, e->epoch);
      if (!wait) {
        e->ctx->hip_failure_poisoned = true;
        return RNS8_BACKEND_FAILURE;
      }
      // An unsuccessful drain does not establish that kernels/DMA stopped.
      if (!record_async_drain(*e->ctx, e->async_failure,
              hipStreamSynchronize(static_cast<hipStream_t>(e->pending_stream))))
        return RNS8_BACKEND_FAILURE;
    }
    finish_prepared_hip(*e, error == hipSuccess ? dependency_operation
                                                : RNS8_BACKEND_FAILURE);
    return rns8_get_prepared_completion(e, token, out);
#else
    (void)wait;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
} // namespace
rns8_status rns8_prepare_hip_submission(rns8_prepared_execution *e) {
  return guard_api([&]() -> rns8_status {
    if (!e)
      return RNS8_INVALID_ARGUMENT;
    if (e->running)
      return RNS8_BUSY;
    const auto support = prepared_hip_support_status(*e);
    if (support != RNS8_SUCCESS) return support;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS)
      return status;
    if (!inputs_match(*e))
      return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device(e->ctx->device_id);
    if (device.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!e->completion_event && e->completion_event.create() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    if(e->plan && e->plan->backend==RNS8_BACKEND_HIPBLASLT && !e->hipblaslt_async){
      const auto prepared=rns8::detail::hipblaslt_prepare_async(e->ctx,e->plan,e->workspace,&e->hipblaslt_async);
      if(prepared!=RNS8_SUCCESS)return prepared;
    }
#endif
    e->hip_submission_prepared = true;
    return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
struct PreparedHipProof {
  boost::multiprecision::cpp_int bound;
  rns8_exact_matrix_ranges ranges;
  bool pipeline=false;
  uint64_t integrity_bound[3]{};
};
static rns8_status prove_prepared_submission(rns8_prepared_execution *e,
                                             void *stream, bool graph,
                                             PreparedHipProof &proof) {
  if (!e || !stream || e->closing)
    return RNS8_INVALID_ARGUMENT;
  if (e->running || e->dependent_references || (e->pool_owned && !e->pool_dispatch))
    return RNS8_BUSY;
  if(!proof.pipeline) {
    const auto dependency = resolve_prepared_dependency(*e);
    if (dependency != RNS8_SUCCESS) return dependency;
  }
  if (!e->hip_submission_prepared)
    return RNS8_UNSUPPORTED_BACKEND;
  if (e->epoch == UINT64_MAX || e->c->prepared_in_flight_writers==UINT64_MAX || e->a->in_flight_readers == UINT64_MAX ||
      e->b->in_flight_readers == UINT64_MAX ||
      (e->d && e->d->in_flight_readers == UINT64_MAX))
    return RNS8_RANGE_ERROR;
  rns8_status status=RNS8_SUCCESS;
  if(proof.pipeline) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    status=prove_pipeline_inputs(*e,proof.bound,proof.ranges);
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  } else {
    status=validate_prepared(*e);
    if(status==RNS8_SUCCESS && !inputs_match(*e)) status=RNS8_INVALID_ARGUMENT;
    if(status==RNS8_SUCCESS) status=prepared_range(*e,proof.bound,proof.ranges);
  }
  if(status!=RNS8_SUCCESS) return status;
  if(e->native_integrity_enabled) {
    status=admit_prepared_integrity(*e,proof.bound);if(status!=RNS8_SUCCESS)return status;
    const boost::multiprecision::cpp_int mask=(boost::multiprecision::cpp_int(1)<<64)-1;
    for(unsigned i=0;i<3;++i)proof.integrity_bound[i]=((proof.bound>>(64*i))&mask).convert_to<uint64_t>();
  }
  if (!e->transfer && !rns8::detail::qualified_i32_gemm_address(
          e->plan->desc.m, e->plan->desc.n, e->plan->desc.k, e->a->desc.cols,
          e->b->desc.cols, e->c->desc.cols, prepared_storage_planes(*e)))
    return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (graph &&
      (!e->graph_executable || e->graph_a != prepared_device_storage(*e->a) ||
       e->graph_b != prepared_device_storage(*e->b) || e->graph_c != prepared_device_storage(*e->c) ||
       (e->d && e->graph_d != prepared_device_storage(*e->d))))
    return RNS8_UNSUPPORTED_BACKEND;
  rns8::detail::hip_scoped_device device(e->ctx->device_id);
  if (device.status() != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  if (static_cast<hipStream_t>(stream) == hipStreamLegacy ||
      static_cast<hipStream_t>(stream) == hipStreamPerThread)
    return RNS8_INVALID_ARGUMENT;
  hipDevice_t stream_device = -1;
  if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &stream_device) !=
          hipSuccess ||
      stream_device != e->ctx->device_id)
    return RNS8_INVALID_ARGUMENT;
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture) !=
      hipSuccess)
    return RNS8_BACKEND_FAILURE;
  if (capture != hipStreamCaptureStatusNone)
    return RNS8_UNSUPPORTED_BACKEND;
  return RNS8_SUCCESS;
#else
  (void)graph;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
// All fallible admission and allocations precede this nonthrowing reservation.
static void begin_prepared_submission(rns8_prepared_execution* e,void* stream,
    rns8_completion_token* token,PreparedHipProof& proof) {
  e->pending_bound.swap(proof.bound);
  e->pending_ranges = std::move(proof.ranges);
  if(e->transfer)e->transfer->reserve();
  else {
  invalidate_output_currentness(*e->c);
  ++e->a->in_flight_readers;
  if (e->a != e->b)
    ++e->b->in_flight_readers;
  if (e->d && e->d != e->a && e->d != e->b)
    ++e->d->in_flight_readers;
  ++e->c->prepared_in_flight_writers;
  e->c->in_flight_writer = true;
  e->workspace->in_flight = true;
  }
  e->running = true;
  e->pending_stream = stream;
  e->async_failure = {};
  e->cancel_requested = false;
  ++e->epoch;
  *token = {e->id, e->epoch};
  begin_prepared_integrity(*e,proof.integrity_bound);
  e->event_dependencies=proof.pipeline;
  e->event_dependency_depth=0;
  if(proof.pipeline) for(uint32_t i=0;i<e->producer_count;++i)
    e->event_dependency_depth=std::max(e->event_dependency_depth,e->producers[i].producer->event_dependency_depth+1);
  if(!proof.pipeline) release_prepared_dependency(*e); // readers retain already published values
}
static rns8_status start_prepared_submission(rns8_prepared_execution *e,
                                             void *stream,
                                             rns8_completion_token *token,
                                             bool graph,
                                             PreparedHipProof &proof,
                                             void* caller_executable=nullptr) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_scoped_device device(e->ctx->device_id);
  if (device.status() != hipSuccess) return RNS8_BACKEND_FAILURE;
  begin_prepared_submission(e,stream,token,proof);
  hipError_t launch_error = hipSuccess;
  rns8_status status=RNS8_SUCCESS;
  if(proof.pipeline) {
    for(uint32_t i=0;i<e->producer_count;++i) {
      const auto* producer=e->producers[i].producer;
      if(!producer->running) continue;
      launch_error=hipStreamWaitEvent(static_cast<hipStream_t>(stream),producer->completion_event.get(),0);
      if(launch_error!=hipSuccess) {status=RNS8_BACKEND_FAILURE;break;}
    }
  }
  if(status==RNS8_SUCCESS && e->transfer)status=e->transfer->before_enqueue(stream);
  if (status==RNS8_SUCCESS && graph) {
    launch_error = hipGraphLaunch(caller_executable?static_cast<hipGraphExec_t>(caller_executable):e->graph_executable, static_cast<hipStream_t>(stream));
    status = launch_error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  } else if(status==RNS8_SUCCESS) {
    status = enqueue_prepared(*e, stream);
  }
  if(status==RNS8_SUCCESS)status=enqueue_prepared_integrity(*e,stream);
  if (status != RNS8_SUCCESS)
    record_async_failure(e->async_failure, status, RNS8_ASYNC_FAILURE_ENQUEUE,
                         launch_error, e->ctx->device_id, e->epoch);
  if (status == RNS8_SUCCESS) {
    const auto error = hipEventRecord(e->completion_event.get(), static_cast<hipStream_t>(stream));
    if (error != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
      record_async_failure(e->async_failure, status, RNS8_ASYNC_FAILURE_EVENT_RECORD,
                           error, e->ctx->device_id, e->epoch);
    }
  }
  if (status != RNS8_SUCCESS) {
    if (record_async_drain(*e->ctx, e->async_failure,
                          hipStreamSynchronize(static_cast<hipStream_t>(stream))) && !proof.pipeline)
      finish_prepared_hip(*e, status);
  }
  return status;
#else
  (void)e;
  (void)stream;
  (void)token;
  (void)graph;
  (void)proof;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
static rns8_status submit_prepared_hip_impl(rns8_prepared_execution *e,
                                            void *stream,
                                            rns8_completion_token *token,
                                            bool graph, bool pipeline=false) {
  return guard_api([&]() -> rns8_status {
    if (!token)
      return RNS8_INVALID_ARGUMENT;
    PreparedHipProof proof;proof.pipeline=pipeline;
    const auto status = prove_prepared_submission(e, stream, graph, proof);
    if (status != RNS8_SUCCESS)
      return status;
    return start_prepared_submission(e, stream, token, graph, proof);
  });
}
rns8_status rns8_submit_prepared_hip(rns8_prepared_execution *e, void *stream,
                                     rns8_completion_token *token) {
  return submit_prepared_hip_impl(e, stream, token, false);
}
rns8_status rns8_submit_prepared_hip_graph(rns8_prepared_execution *e,
                                           void *stream,
                                           rns8_completion_token *token) {
  return submit_prepared_hip_impl(e, stream, token, true);
}
rns8_status rns8_submit_prepared_hip_pipeline(rns8_prepared_execution* e,void* stream,rns8_completion_token* token) {
  return submit_prepared_hip_impl(e,stream,token,false,true);
}
rns8_status rns8_submit_prepared_hip_graph_pipeline(rns8_prepared_execution* e,void* stream,rns8_completion_token* token) {
  return submit_prepared_hip_impl(e,stream,token,true,true);
}
rns8_status rns8_capture_prepared_hip_graph(rns8_prepared_execution *e) {
  return guard_api([&]() -> rns8_status {
    const auto preparation = rns8_prepare_hip_submission(e);
    if (preparation != RNS8_SUCCESS)
      return preparation;
    if (!e->transfer && !rns8::detail::qualified_i32_gemm_address(
            e->plan->desc.m, e->plan->desc.n, e->plan->desc.k, e->a->desc.cols,
            e->b->desc.cols, e->c->desc.cols, prepared_storage_planes(*e)))
      return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device(e->ctx->device_id);
    if (device.status() != hipSuccess) return RNS8_BACKEND_FAILURE;
    rns8::detail::hip_unique_stream capture_stream;
    if (capture_stream.create_non_blocking() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (hipStreamBeginCapture(capture_stream.get(),
                              hipStreamCaptureModeThreadLocal) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    const auto body = enqueue_prepared(*e, capture_stream.get());
    hipGraph_t graph = nullptr;
    const auto end = hipStreamEndCapture(capture_stream.get(), &graph);
    if (body != RNS8_SUCCESS || end != hipSuccess) {
      if (graph)
        (void)hipGraphDestroy(graph);
      return body != RNS8_SUCCESS ? body : RNS8_BACKEND_FAILURE;
    }
    hipGraphExec_t executable = nullptr;
    const auto instantiated =
        hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (instantiated != hipSuccess) {
      if (executable)
        (void)hipGraphExecDestroy(executable);
      (void)hipGraphDestroy(graph);
      return RNS8_BACKEND_FAILURE;
    }
    // Keep the old graph intact until replacement capture/instantiation
    // succeeds.
    if (e->graph_executable)
      (void)hipGraphExecDestroy(e->graph_executable);
    if (e->graph)
      (void)hipGraphDestroy(e->graph);
    e->graph = graph;
    e->graph_executable = executable;
    e->graph_a = prepared_device_storage(*e->a);
    e->graph_b = prepared_device_storage(*e->b);
    e->graph_c = prepared_device_storage(*e->c);
    e->graph_d = e->d ? prepared_device_storage(*e->d) : nullptr;
    return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_query_prepared_hip(rns8_prepared_execution *e,
                                    rns8_completion_token token,
                                    rns8_completion_info *out) {
  return observe_prepared_hip(e, token, out, false);
}
rns8_status rns8_wait_prepared_hip(rns8_prepared_execution *e,
                                   rns8_completion_token token,
                                   rns8_completion_info *out) {
  return observe_prepared_hip(e, token, out, true);
}

rns8_status rns8_get_prepared_async_failure(const rns8_prepared_execution* e,
    rns8_completion_token token, rns8_async_failure_info* info) {
  if (!e || !info || !rns8::detail::valid_abi(info->struct_size, info->abi_version, sizeof(*info)) ||
      token.prepared_id != e->id || !token.epoch || token.epoch != e->epoch)
    return RNS8_INVALID_ARGUMENT;
  *info = e->async_failure;
  info->struct_size = sizeof(*info);
  info->abi_version = RNS8_ABI_VERSION;
  if (info->stage == RNS8_ASYNC_FAILURE_NONE) info->failed_node = UINT32_MAX;
  info->epoch = e->epoch;
  info->device_id = e->ctx->device_id;
  return RNS8_SUCCESS;
}

static rns8_status
submit_prepared_group_hip_impl(rns8_prepared_group *group, void *const *streams,
                               uint32_t count, rns8_completion_token *tokens,
                               uint32_t token_capacity, uint32_t *out_submitted,
                               bool graph) {
  return guard_api([&]() -> rns8_status {
    if (!group || !streams || !tokens || !out_submitted ||
        count != group->executions.size() || token_capacity < count)
      return RNS8_INVALID_ARGUMENT;
    *out_submitted = 0;
    if (group->running)
      return RNS8_BUSY;
    std::unordered_set<const rns8_workspace *> workspaces;
    std::unordered_map<const rns8_matrix *, uint64_t> readers;
    std::vector<PreparedHipProof> proofs(count);
    workspaces.reserve(count);
    readers.reserve(static_cast<size_t>(count) * 3);
    const auto device = group->executions.front()->ctx->device_id;
    for (uint32_t i = 0; i < count; ++i) {
      auto *e = group->executions[i];
      if (e->ctx->device_id != device)
        return RNS8_UNSUPPORTED_BACKEND;
      if (e->workspace && !workspaces.insert(e->workspace).second)
        return RNS8_INVALID_ARGUMENT;
      const auto status =
          prove_prepared_submission(e, streams[i], graph, proofs[i]);
      if (status != RNS8_SUCCESS)
        return status;
      if(!e->transfer || e->transfer->reads_matrix()) {
        ++readers[e->a];
        if(e->b!=e->a)++readers[e->b];
        if(e->d && e->d!=e->a && e->d!=e->b)++readers[e->d];
      }
    }
    for (const auto &[matrix, reservations] : readers)
      if (matrix->in_flight_readers > UINT64_MAX - reservations)
        return RNS8_RANGE_ERROR;
    // Whole-batch admission is complete. No allocation or proof recomputation
    // follows the first enqueue. Host calls sharing these handles are
    // serialized.
    struct RunningGuard {
      bool &running;
      explicit RunningGuard(bool &flag) : running(flag) { running = true; }
      ~RunningGuard() { running = false; }
    } running(group->running);
    for (uint32_t i = 0; i < count; ++i) {
      auto *e = group->executions[i];
      const auto prior_epoch = e->epoch;
      const auto status = start_prepared_submission(e, streams[i], tokens + i,
                                                    graph, proofs[i]);
      if (e->epoch != prior_epoch)
        ++*out_submitted;
      if (status != RNS8_SUCCESS)
        return status;
    }
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_submit_prepared_group_hip(rns8_prepared_group *group,
                                           void *const *streams, uint32_t count,
                                           rns8_completion_token *tokens,
                                           uint32_t token_capacity,
                                           uint32_t *out_submitted) {
  return submit_prepared_group_hip_impl(group, streams, count, tokens,
                                        token_capacity, out_submitted, false);
}
rns8_status rns8_submit_prepared_group_hip_graph(rns8_prepared_group *group,
                                                 void *const *streams,
                                                 uint32_t count,
                                                 rns8_completion_token *tokens,
                                                 uint32_t token_capacity,
                                                 uint32_t *out_submitted) {
  return submit_prepared_group_hip_impl(group, streams, count, tokens,
                                        token_capacity, out_submitted, true);
}


rns8_status rns8_cancel_prepared_hip(rns8_prepared_execution* e,
                                    rns8_completion_token token) {
  if (!e || !e->hip_submission_prepared || token.prepared_id != e->id ||
      !token.epoch || token.epoch != e->epoch) return RNS8_INVALID_ARGUMENT;
  if (e->running) e->cancel_requested = true;
  return RNS8_SUCCESS;
}


rns8_status rns8_wait_prepared_hip_for(rns8_prepared_execution* e,
    rns8_completion_token token, uint64_t timeout_us, rns8_completion_info* info) {
  return guard_api([&]() {
    return poll_async_until(timeout_us, [&]() { return rns8_query_prepared_hip(e, token, info); });
  });
}


rns8_status rns8_get_prepared_hip_support(const rns8_prepared_execution* e,
                                        rns8_async_support_info* info) {
  if (!e || !info || !rns8::detail::valid_abi(info->struct_size, info->abi_version, sizeof(*info)))
    return RNS8_INVALID_ARGUMENT;
  *info = async_support_info(prepared_hip_support_status(*e), e->ctx->device_id);
  info->supports_caller_graph_binding=info->support_status==RNS8_SUCCESS;
  return RNS8_SUCCESS;
}
