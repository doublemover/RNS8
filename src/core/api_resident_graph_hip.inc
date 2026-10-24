namespace {
rns8_status resident_graph_hip_support_status(const rns8_resident_graph& graph) {
  if (graph.ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
  if (!direct_hip_compiled() || graph.ctx->backend != RNS8_BACKEND_HIP_DIRECT)
    return RNS8_UNSUPPORTED_BACKEND;
  for (const auto& node : graph.nodes) {
    if ((node.kind != RNS8_RESIDENT_MATMUL && node.kind != RNS8_RESIDENT_EXACT_AXPBY && node.kind != RNS8_RESIDENT_COMPLEX_MATMUL) ||
        node.plan->backend != RNS8_BACKEND_HIP_DIRECT || !node.plan->tile_schedule.empty() ||
        (!rns8::detail::exact_wide_semantics(node.plan->desc.semantics) &&
         !(node.kind==RNS8_RESIDENT_MATMUL && (uses_finite_storage(node.plan->desc.semantics) ||
             node.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64 || bounded_graph_semantics(node.plan->desc.semantics)))) ||
        !rns8::detail::qualified_i32_gemm_address(node.plan->desc.m, node.plan->desc.n,
            node.plan->desc.k, node.a->desc.cols, node.b->desc.cols, node.c->desc.cols, node.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64?8:
            uses_finite_storage(node.plan->desc.semantics)?1:node.plan->prefix))
      return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_SUCCESS;
}
// The graph already owns every output writer reservation. Clearing currentness
// must not re-enter the generic writable check, which rejects those reservations.
// Keep reservations intact, and invalidate only outputs whose writes are attempted.
void invalidate_owned_graph_output(rns8_matrix& output) {
  clear_residue_current(output);
  clear_byte_limb_current(output);
  clear_native_current(output);
}
rns8_status enqueue_resident_node(rns8_resident_graph& graph, size_t index, void *stream) {
  const auto& node = graph.nodes[index];
  if (node.kind == RNS8_RESIDENT_COMPLEX_MATMUL) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return rns8::detail::hip_direct_enqueue_complex_no_publish(node,
        graph.complex_resources[index], graph.complex_storage.get(), stream);
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  if(node.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64)
    return rns8::detail::wrap64_hip_enqueue_byte_limbs(
        node.a->hip_byte_limbs,node.b->hip_byte_limbs,node.c->hip_byte_limbs,
        node.plan->desc.m,node.plan->desc.n,node.plan->desc.k,stream);
  if(uses_finite_storage(node.plan->desc.semantics))
    return rns8::detail::hip_direct_enqueue_finite_no_publish(node.plan,node.a,node.b,node.c,stream);
  return node.kind == RNS8_RESIDENT_EXACT_AXPBY
             ? rns8::detail::hip_direct_enqueue_axpby_no_publish(
                   node.plan, node.a, node.b, node.d, node.c, node.alpha,
                   node.beta, stream)
             : rns8::detail::hip_direct_enqueue_rns_no_publish(
                   node.plan, node.a, node.b, node.c, stream);
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
std::array<const void *, 8>
resident_node_addresses(const rns8_resident_graph& graph, size_t index) {
  const auto& node = graph.nodes[index];
  if(node.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64)
    return {node.a->hip_byte_limbs,node.b->hip_byte_limbs,node.c->hip_byte_limbs,nullptr};
  return {node.a->hip_residues, node.b->hip_residues, node.c->hip_residues,
          node.d ? node.d->hip_residues : nullptr,
          node.a_imag ? node.a_imag->hip_residues : nullptr,
          node.b_imag ? node.b_imag->hip_residues : nullptr,
          node.c_imag ? node.c_imag->hip_residues : nullptr,
          node.c_imag ? graph.complex_storage.get() : nullptr};
}
#endif
void invalidate_owned_graph_outputs(const rns8_resident_node& node) {
  invalidate_owned_graph_output(*node.c);
  if (node.c_imag) invalidate_owned_graph_output(*node.c_imag);
}
void finish_resident_graph_hip(rns8_resident_graph &graph, rns8_status status,
                               uint32_t failed) {
  if (status == RNS8_SUCCESS && graph.cancel_requested) status = RNS8_CANCELED;
  for (const auto &input : graph.inputs)
    --input.matrix->in_flight_readers;
  for (auto *workspace : graph.hip_workspaces)
    workspace->in_flight = false;
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &node = graph.nodes[i];
    node.c->in_flight_writer = false;
    if (node.c_imag) node.c_imag->in_flight_writer = false;
    if (status == RNS8_SUCCESS) {
      if(node.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64) {
        mark_output_device_byte_limbs_current(*node.c);
        node.c->finite_modulus=0;
        node.c->exact_axis_ranges.rows.clear();node.c->exact_axis_ranges.cols.clear();
        continue;
      }
      mark_output_device_residues_current(*node.c, node.plan);
      if (node.c_imag) {
        mark_output_device_residues_current(*node.c_imag, node.plan);
        rns8::detail::commit_exact_range(*node.c_imag, *node.plan, graph.pending_proofs[i].imag_bound);
      }
      if(uses_finite_storage(node.plan->desc.semantics)) {
        node.c->finite_modulus=static_cast<uint16_t>(node.plan->desc.finite_modulus);
        node.c->exact_axis_ranges.rows.clear();node.c->exact_axis_ranges.cols.clear();
      } else rns8::detail::commit_exact_range(*node.c,*node.plan,
          graph.pending_proofs[i].bound,&graph.pending_proofs[i].ranges);
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
    rns8::detail::hip_scoped_device device_scope(graph->ctx->device_id);
    if (!graph->pending_stream || device_scope.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (graph->async_failure.stage != RNS8_ASYNC_FAILURE_NONE) {
      if (!wait) return RNS8_BACKEND_FAILURE;
      if (!record_async_drain(*graph->ctx, graph->async_failure,
              hipStreamSynchronize(static_cast<hipStream_t>(graph->pending_stream))))
        return RNS8_BACKEND_FAILURE;
      finish_resident_graph_hip(*graph, graph->async_failure.operation_status,
                               graph->async_failure.failed_node);
      *info = graph->completion;
      return RNS8_SUCCESS;
    }
    const auto error = wait ? hipEventSynchronize(graph->completion_event.get())
                            : hipEventQuery(graph->completion_event.get());
    if (error == hipErrorNotReady)
      return RNS8_BUSY;
    if (error != hipSuccess) {
      record_async_failure(graph->async_failure, RNS8_BACKEND_FAILURE,
          RNS8_ASYNC_FAILURE_EVENT_OBSERVE, error, graph->ctx->device_id, graph->epoch);
      if (!wait) {
        graph->ctx->hip_failure_poisoned = true;
        return RNS8_BACKEND_FAILURE;
      }
      if (!record_async_drain(*graph->ctx, graph->async_failure,
              hipStreamSynchronize(static_cast<hipStream_t>(graph->pending_stream))))
        return RNS8_BACKEND_FAILURE;
    }
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
    const auto support = resident_graph_hip_support_status(*graph);
    if (support != RNS8_SUCCESS) return support;
    std::unordered_set<rns8_workspace *> seen;
    std::vector<rns8_workspace *> workspaces;
    workspaces.reserve(graph->nodes.size());
    for (const auto &node : graph->nodes) {
      if (seen.insert(node.workspace).second)
        workspaces.push_back(node.workspace);
    }
    uint32_t failed = UINT32_MAX;
    std::vector<rns8_resident_graph::OutputProof> proofs;
    const auto status = prove_graph(*graph, failed, true, &proofs);
    if (status != RNS8_SUCCESS)
      return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device_scope(graph->ctx->device_id);
    if (device_scope.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!graph->completion_event &&
        graph->completion_event.create() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!graph->hip_prepared) {
      std::vector<rns8::detail::ComplexResidueScratch> resources;
      resources.reserve(proofs.size());
      uint64_t bytes = 0;
      for (const auto& proof : proofs) {
        resources.push_back(proof.complex_scratch);
        bytes = std::max(bytes, proof.complex_scratch.scratch_bytes);
      }
      // Nodes execute serially, so a single maximum-sized arena suffices.
      // Once prepared, never move its address: captured graphs refer to it.
      if (bytes) {
        if (!graph->synchronous_stream.get() &&
            graph->synchronous_stream.create_non_blocking() != hipSuccess)
          return RNS8_BACKEND_FAILURE;
        const auto allocation = graph->complex_storage.allocate(graph->ctx->device_id, bytes);
        if (allocation != RNS8_SUCCESS) return allocation;
      }
      graph->complex_resources.swap(resources);
    }
    graph->hip_workspaces.swap(workspaces);
    graph->hip_prepared = true;
    return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}

static rns8_status submit_resident_graph_hip(rns8_resident_graph *graph,
                                             void *stream,
                                             rns8_graph_completion_token *token,
                                             bool replay) {
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
    if (replay) {
      if (!graph->captured_executable ||
          graph->captured_addresses.size() != graph->nodes.size())
        return RNS8_UNSUPPORTED_BACKEND;
      for (size_t i = 0; i < graph->nodes.size(); ++i)
        if (resident_node_addresses(*graph, i) !=
            graph->captured_addresses[i])
          return RNS8_UNSUPPORTED_BACKEND;
    }
    rns8::detail::hip_scoped_device device_scope(graph->ctx->device_id);
    if (device_scope.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    const auto hip_stream = static_cast<hipStream_t>(stream);
    if (hip_stream == hipStreamLegacy || hip_stream == hipStreamPerThread)
      return RNS8_INVALID_ARGUMENT;
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
    for (const auto &node : graph->nodes) {
      node.c->in_flight_writer = true;
      if (node.c_imag) node.c_imag->in_flight_writer = true;
    }
    graph->pending_stream = stream;
    graph->async_failure = {};
    graph->cancel_requested = false;
    graph->running = true;
    *token = {graph->id, ++graph->epoch};
    // One ordered stream provides the DAG dependencies. Internal values remain
    // unpublished until the final event; these private enqueues do no
    // admission.
    hipError_t launch_error = hipSuccess;
    if (replay) {
      for (const auto &node : graph->nodes)
        invalidate_owned_graph_outputs(node);
      launch_error = hipGraphLaunch(graph->captured_executable, hip_stream);
      status = launch_error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
    } else
      for (uint32_t i = 0; i < graph->nodes.size(); ++i) {
        const auto &node = graph->nodes[i];
        invalidate_owned_graph_outputs(node);
        status = enqueue_resident_node(*graph, i, stream);
        if (status != RNS8_SUCCESS) {
          failed = i;
          break;
        }
      }
    if (status != RNS8_SUCCESS)
      record_async_failure(graph->async_failure, status, RNS8_ASYNC_FAILURE_ENQUEUE,
                           launch_error, graph->ctx->device_id, graph->epoch, failed);
    if (status == RNS8_SUCCESS) {
      const auto error = hipEventRecord(graph->completion_event.get(), hip_stream);
      if (error != hipSuccess) {
        status = RNS8_BACKEND_FAILURE;
        record_async_failure(graph->async_failure, status, RNS8_ASYNC_FAILURE_EVENT_RECORD,
                             error, graph->ctx->device_id, graph->epoch);
      }
    }
    if (status != RNS8_SUCCESS) {
      if (record_async_drain(*graph->ctx, graph->async_failure, hipStreamSynchronize(hip_stream)))
        finish_resident_graph_hip(*graph, status, failed);
    }
    return status;
#else
    (void)replay;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}

rns8_status rns8_submit_resident_graph_hip(rns8_resident_graph *graph,
                                           void *stream,
                                           rns8_graph_completion_token *token) {
  return submit_resident_graph_hip(graph, stream, token, false);
}
rns8_status rns8_replay_resident_graph_hip(rns8_resident_graph *graph,
                                           void *stream,
                                           rns8_graph_completion_token *token) {
  return submit_resident_graph_hip(graph, stream, token, true);
}
rns8_status rns8_capture_resident_graph_hip(rns8_resident_graph *graph) {
  return guard_api([&]() -> rns8_status {
    const auto status = rns8_prepare_resident_graph_hip(graph);
    if (status != RNS8_SUCCESS)
      return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device_scope(graph->ctx->device_id);
    if (device_scope.status() != hipSuccess) return RNS8_BACKEND_FAILURE;
    std::vector<std::array<const void *, 8>> addresses;
    addresses.reserve(graph->nodes.size());
    for (size_t i = 0; i < graph->nodes.size(); ++i)
      addresses.push_back(resident_node_addresses(*graph, i));
    rns8::detail::hip_unique_stream capture_stream;
    if (capture_stream.create_non_blocking() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (hipStreamBeginCapture(capture_stream.get(),
                              hipStreamCaptureModeThreadLocal) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    rns8_status body = RNS8_SUCCESS;
    for (size_t i = 0; i < graph->nodes.size(); ++i) {
      body = enqueue_resident_node(*graph, i, capture_stream.get());
      if (body != RNS8_SUCCESS)
        break;
    }
    hipGraph_t captured = nullptr;
    const auto ended = hipStreamEndCapture(capture_stream.get(), &captured);
    if (body != RNS8_SUCCESS || ended != hipSuccess) {
      if (captured)
        (void)hipGraphDestroy(captured);
      return body != RNS8_SUCCESS ? body : RNS8_BACKEND_FAILURE;
    }
    hipGraphExec_t executable = nullptr;
    if (hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0) !=
        hipSuccess) {
      if (executable)
        (void)hipGraphExecDestroy(executable);
      (void)hipGraphDestroy(captured);
      return RNS8_BACKEND_FAILURE;
    }
    if (graph->captured_executable)
      (void)hipGraphExecDestroy(graph->captured_executable);
    if (graph->captured_graph)
      (void)hipGraphDestroy(graph->captured_graph);
    graph->captured_graph = captured;
    graph->captured_executable = executable;
    graph->captured_addresses.swap(addresses);
    return RNS8_SUCCESS;
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

rns8_status rns8_get_resident_graph_async_failure(const rns8_resident_graph* graph,
    rns8_graph_completion_token token, rns8_async_failure_info* info) {
  if (!graph || !info || !rns8::detail::valid_abi(info->struct_size, info->abi_version, sizeof(*info)) ||
      token.graph_id != graph->id || !token.epoch || token.epoch != graph->epoch)
    return RNS8_INVALID_ARGUMENT;
  *info = graph->async_failure;
  info->struct_size = sizeof(*info);
  info->abi_version = RNS8_ABI_VERSION;
  if (info->stage == RNS8_ASYNC_FAILURE_NONE) info->failed_node = UINT32_MAX;
  info->epoch = graph->epoch;
  info->device_id = graph->ctx->device_id;
  return RNS8_SUCCESS;
}


rns8_status rns8_cancel_resident_graph_hip(rns8_resident_graph* graph,
                                         rns8_graph_completion_token token) {
  if (!graph || !graph->hip_prepared || token.graph_id != graph->id ||
      !token.epoch || token.epoch != graph->epoch) return RNS8_INVALID_ARGUMENT;
  if (graph->running) graph->cancel_requested = true;
  return RNS8_SUCCESS;
}


rns8_status rns8_wait_resident_graph_hip_for(rns8_resident_graph* graph,
    rns8_graph_completion_token token, uint64_t timeout_us, rns8_graph_execution_info* info) {
  return guard_api([&]() {
    return poll_async_until(timeout_us, [&]() { return rns8_query_resident_graph_hip(graph, token, info); });
  });
}


rns8_status rns8_get_resident_graph_hip_support(const rns8_resident_graph* graph,
                                             rns8_async_support_info* info) {
  if (!graph || !info || !rns8::detail::valid_abi(info->struct_size, info->abi_version, sizeof(*info)))
    return RNS8_INVALID_ARGUMENT;
  *info = async_support_info(resident_graph_hip_support_status(*graph), graph->ctx->device_id);
  return RNS8_SUCCESS;
}
