namespace {
void finish_resident_graph_hip(rns8_resident_graph &graph, rns8_status status,
                               uint32_t failed) {
  for (const auto &input : graph.inputs)
    --input.matrix->in_flight_readers;
  for (auto *workspace : graph.hip_workspaces)
    workspace->in_flight = false;
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &node = graph.nodes[i];
    node.c->in_flight_writer = false;
    if (status == RNS8_SUCCESS) {
      mark_output_device_residues_current(*node.c);
      rns8::detail::commit_exact_range(*node.c, *node.plan,
                                       graph.pending_proofs[i].bound,
                                       &graph.pending_proofs[i].ranges);
    }
  }
  graph.completion = {};
  graph.completion.struct_size = sizeof(graph.completion);
  graph.completion.abi_version = RNS8_ABI_VERSION;
  graph.completion.epoch = graph.epoch;
  graph.completion.operation_status = status;
  graph.completion.completed_nodes =
      status == RNS8_SUCCESS ? static_cast<uint32_t>(graph.nodes.size()) : 0;
  graph.completion.failed_node = failed;
  graph.running = false;
  graph.pending_stream = nullptr;
}

rns8_status observe_resident_graph_hip(rns8_resident_graph *graph,
                                       rns8_graph_completion_token token,
                                       rns8_graph_execution_info *info,
                                       bool wait) {
  return guard_api([&]() -> rns8_status {
    if (!graph || !info ||
        !rns8::detail::valid_abi(info->struct_size, info->abi_version,
                                 sizeof(*info)) ||
        !graph->hip_prepared || token.graph_id != graph->id || !token.epoch ||
        token.epoch != graph->epoch)
      return RNS8_INVALID_ARGUMENT;
    if (!graph->running) {
      if (graph->completion.epoch != token.epoch)
        return RNS8_INVALID_ARGUMENT;
      *info = graph->completion;
      return RNS8_SUCCESS;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (!graph->pending_stream ||
        hipSetDevice(graph->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    const auto error = wait ? hipEventSynchronize(graph->completion_event.get())
                            : hipEventQuery(graph->completion_event.get());
    if (error == hipErrorNotReady)
      return RNS8_BUSY;
    if (error != hipSuccess)
      (void)hipStreamSynchronize(
          static_cast<hipStream_t>(graph->pending_stream));
    finish_resident_graph_hip(
        *graph, error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE,
        UINT32_MAX);
    *info = graph->completion;
    return RNS8_SUCCESS;
#else
    (void)wait;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
} // namespace

rns8_status rns8_prepare_resident_graph_hip(rns8_resident_graph *graph) {
  return guard_api([&]() -> rns8_status {
    if (!graph)
      return RNS8_INVALID_ARGUMENT;
    if (graph->running)
      return RNS8_BUSY;
    if (graph->ctx->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    std::unordered_set<rns8_workspace *> seen;
    std::vector<rns8_workspace *> workspaces;
    workspaces.reserve(graph->nodes.size());
    for (const auto &node : graph->nodes) {
      if ((node.kind != RNS8_RESIDENT_MATMUL &&
           node.kind != RNS8_RESIDENT_EXACT_AXPBY) ||
          node.plan->backend != RNS8_BACKEND_HIP_DIRECT ||
          !node.plan->tile_schedule.empty() ||
          !rns8::detail::qualified_i32_gemm_address(
              node.plan->desc.m, node.plan->desc.n, node.plan->desc.k,
              node.a->desc.cols, node.b->desc.cols, node.c->desc.cols,
              node.plan->prefix))
        return RNS8_UNSUPPORTED_BACKEND;
      if (seen.insert(node.workspace).second)
        workspaces.push_back(node.workspace);
    }
    uint32_t failed = UINT32_MAX;
    const auto status = prove_graph(*graph, failed, true);
    if (status != RNS8_SUCCESS)
      return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hipSetDevice(graph->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!graph->completion_event &&
        graph->completion_event.create() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    graph->hip_workspaces.swap(workspaces);
    graph->hip_prepared = true;
    return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}

rns8_status rns8_submit_resident_graph_hip(rns8_resident_graph *graph,
                                           void *stream,
                                           rns8_graph_completion_token *token) {
  return guard_api([&]() -> rns8_status {
    if (!graph || !stream || !token)
      return RNS8_INVALID_ARGUMENT;
    if (graph->running)
      return RNS8_BUSY;
    if (!graph->hip_prepared)
      return RNS8_UNSUPPORTED_BACKEND;
    if (graph->epoch == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    for (const auto &input : graph->inputs)
      if (input.matrix->in_flight_readers == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    uint32_t failed = UINT32_MAX;
    std::vector<rns8_resident_graph::OutputProof> proofs;
    auto status = prove_graph(*graph, failed, true, &proofs);
    if (status != RNS8_SUCCESS)
      return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hipSetDevice(graph->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    const auto hip_stream = static_cast<hipStream_t>(stream);
    hipDevice_t device = -1;
    if (hipStreamGetDevice(hip_stream, &device) != hipSuccess ||
        device != graph->ctx->device_id)
      return RNS8_INVALID_ARGUMENT;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(hip_stream, &capture) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (capture != hipStreamCaptureStatusNone)
      return RNS8_UNSUPPORTED_BACKEND;
    graph->pending_proofs.swap(proofs);
    for (const auto &input : graph->inputs)
      ++input.matrix->in_flight_readers;
    for (auto *workspace : graph->hip_workspaces)
      workspace->in_flight = true;
    for (const auto &node : graph->nodes)
      node.c->in_flight_writer = true;
    graph->pending_stream = stream;
    graph->running = true;
    *token = {graph->id, ++graph->epoch};
    // One ordered stream provides the DAG dependencies. Internal values remain
    // unpublished until the final event; these private enqueues do no
    // admission.
    for (uint32_t i = 0; i < graph->nodes.size(); ++i) {
      const auto &node = graph->nodes[i];
      invalidate_output_currentness(*node.c);
      status = node.kind == RNS8_RESIDENT_EXACT_AXPBY
                   ? rns8::detail::hip_direct_enqueue_axpby_no_publish(
                         node.plan, node.a, node.b, node.d, node.c, node.alpha,
                         node.beta, stream)
                   : rns8::detail::hip_direct_enqueue_rns_no_publish(
                         node.plan, node.a, node.b, node.c, stream);
      if (status != RNS8_SUCCESS) {
        failed = i;
        break;
      }
    }
    if (status == RNS8_SUCCESS &&
        hipEventRecord(graph->completion_event.get(), hip_stream) != hipSuccess)
      status = RNS8_BACKEND_FAILURE;
    if (status != RNS8_SUCCESS) {
      (void)hipStreamSynchronize(hip_stream);
      finish_resident_graph_hip(*graph, status, failed);
    }
    return status;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}

rns8_status rns8_query_resident_graph_hip(rns8_resident_graph *graph,
                                          rns8_graph_completion_token token,
                                          rns8_graph_execution_info *info) {
  return observe_resident_graph_hip(graph, token, info, false);
}
rns8_status rns8_wait_resident_graph_hip(rns8_resident_graph *graph,
                                         rns8_graph_completion_token token,
                                         rns8_graph_execution_info *info) {
  return observe_resident_graph_hip(graph, token, info, true);
}
