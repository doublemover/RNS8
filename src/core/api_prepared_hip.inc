namespace {
void finish_prepared_hip(rns8_prepared_execution &e, rns8_status operation) {
  if (operation == RNS8_SUCCESS && e.cancel_requested) operation = RNS8_CANCELED;
  --e.a->in_flight_readers;
  if (e.a != e.b)
    --e.b->in_flight_readers;
  if (e.d && e.d != e.a && e.d != e.b)
    --e.d->in_flight_readers;
  e.c->in_flight_writer = false;
  e.workspace->in_flight = false;
  if (operation == RNS8_SUCCESS) {
    mark_output_device_residues_current(*e.c);
    rns8::detail::commit_exact_range(*e.c, *e.plan, e.pending_bound,
                                     &e.pending_ranges);
  }
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
    finish_prepared_hip(*e, error == hipSuccess ? RNS8_SUCCESS
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
    if (e->plan->backend != RNS8_BACKEND_HIP_DIRECT ||
        !rns8::detail::exact_wide_semantics(e->plan->desc.semantics) ||
        !e->plan->tile_schedule.empty())
      return RNS8_UNSUPPORTED_BACKEND;
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
};
static rns8_status prove_prepared_submission(rns8_prepared_execution *e,
                                             void *stream, bool graph,
                                             PreparedHipProof &proof) {
  if (!e || !stream)
    return RNS8_INVALID_ARGUMENT;
  if (e->running)
    return RNS8_BUSY;
  if (!e->hip_submission_prepared)
    return RNS8_UNSUPPORTED_BACKEND;
  if (e->epoch == UINT64_MAX || e->a->in_flight_readers == UINT64_MAX ||
      e->b->in_flight_readers == UINT64_MAX ||
      (e->d && e->d->in_flight_readers == UINT64_MAX))
    return RNS8_RANGE_ERROR;
  auto status = validate_prepared(*e);
  if (status != RNS8_SUCCESS)
    return status;
  if (!inputs_match(*e))
    return RNS8_INVALID_ARGUMENT;
  status = prepared_range(*e, proof.bound, proof.ranges);
  if (status != RNS8_SUCCESS)
    return status;
  if (!rns8::detail::qualified_i32_gemm_address(
          e->plan->desc.m, e->plan->desc.n, e->plan->desc.k, e->a->desc.cols,
          e->b->desc.cols, e->c->desc.cols, e->plan->prefix))
    return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (graph &&
      (!e->graph_executable || e->graph_a != e->a->hip_residues ||
       e->graph_b != e->b->hip_residues || e->graph_c != e->c->hip_residues ||
       (e->d && e->graph_d != e->d->hip_residues)))
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
static rns8_status start_prepared_submission(rns8_prepared_execution *e,
                                             void *stream,
                                             rns8_completion_token *token,
                                             bool graph,
                                             PreparedHipProof &proof) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_scoped_device device(e->ctx->device_id);
  if (device.status() != hipSuccess) return RNS8_BACKEND_FAILURE;
  e->pending_bound.swap(proof.bound);
  e->pending_ranges = std::move(proof.ranges);
  invalidate_output_currentness(*e->c);
  ++e->a->in_flight_readers;
  if (e->a != e->b)
    ++e->b->in_flight_readers;
  if (e->d && e->d != e->a && e->d != e->b)
    ++e->d->in_flight_readers;
  e->c->in_flight_writer = true;
  e->workspace->in_flight = true;
  e->running = true;
  e->pending_stream = stream;
  e->async_failure = {};
  e->cancel_requested = false;
  ++e->epoch;
  *token = {e->id, e->epoch};
  hipError_t launch_error = hipSuccess;
  rns8_status status;
  if (graph) {
    launch_error = hipGraphLaunch(e->graph_executable, static_cast<hipStream_t>(stream));
    status = launch_error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  } else {
    status = enqueue_prepared(*e, stream);
  }
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
                          hipStreamSynchronize(static_cast<hipStream_t>(stream))))
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
                                            bool graph) {
  return guard_api([&]() -> rns8_status {
    if (!token)
      return RNS8_INVALID_ARGUMENT;
    PreparedHipProof proof;
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
rns8_status rns8_capture_prepared_hip_graph(rns8_prepared_execution *e) {
  return guard_api([&]() -> rns8_status {
    const auto preparation = rns8_prepare_hip_submission(e);
    if (preparation != RNS8_SUCCESS)
      return preparation;
    if (!rns8::detail::qualified_i32_gemm_address(
            e->plan->desc.m, e->plan->desc.n, e->plan->desc.k, e->a->desc.cols,
            e->b->desc.cols, e->c->desc.cols, e->plan->prefix))
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
    e->graph_a = e->a->hip_residues;
    e->graph_b = e->b->hip_residues;
    e->graph_c = e->c->hip_residues;
    e->graph_d = e->d ? e->d->hip_residues : nullptr;
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
      if (!workspaces.insert(e->workspace).second)
        return RNS8_INVALID_ARGUMENT;
      const auto status =
          prove_prepared_submission(e, streams[i], graph, proofs[i]);
      if (status != RNS8_SUCCESS)
        return status;
      ++readers[e->a];
      if (e->b != e->a)
        ++readers[e->b];
      if (e->d && e->d != e->a && e->d != e->b)
        ++readers[e->d];
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
