namespace {
rns8_status validate_resident_transfer_envelopes(const rns8_resident_graph& graph,
    const std::vector<rns8_resident_graph::OutputProof>& proofs) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(proofs.size()!=graph.nodes.size() || graph.transfers.size()!=graph.nodes.size())return RNS8_INTERNAL_ERROR;
  for(size_t i=0;i<graph.nodes.size();++i)if(graph.nodes[i].kind==RNS8_RESIDENT_BASE_EXTEND) {
    const auto& transfer=graph.transfers[i];
    if(proofs[i].source_prefix<transfer.source || proofs[i].bound>transfer.envelope)return RNS8_RANGE_ERROR;
  }
#else
  (void)graph;(void)proofs;
#endif
  return RNS8_SUCCESS;
}
rns8_status resident_graph_hip_support_status(const rns8_resident_graph& graph) {
  if(graph.retiring)return RNS8_BUSY;
  if (graph.ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
  if (!direct_hip_compiled() || graph.ctx->backend != RNS8_BACKEND_HIP_DIRECT)
    return RNS8_UNSUPPORTED_BACKEND;
  for (const auto& node : graph.nodes) {
    if(node.kind==RNS8_RESIDENT_BASE_EXTEND || node.kind==RNS8_RESIDENT_EXPORT_LIMBS || node.kind==RNS8_RESIDENT_NATIVE_TO_RNS) {
      if(!rns8::detail::exact_wide_semantics(node.c->desc.semantics) ||
          !rns8::detail::qualified_i32_matrix_address(node.c->desc.rows,node.c->desc.cols,node.c->desc.cols,node.c->prefix))
        return RNS8_UNSUPPORTED_BACKEND;
      if(node.kind==RNS8_RESIDENT_EXPORT_LIMBS && !rns8::detail::qualified_i32_matrix_address(
          node.c->desc.rows,node.c->desc.cols,node.c->desc.cols,1,uint64_t(node.export_limb_count)*8))return RNS8_UNSUPPORTED_BACKEND;
      continue;
    }
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
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(node.kind==RNS8_RESIDENT_NATIVE_TO_RNS)
    return graph.native_imports[index]->enqueue(graph.complex_storage.get(),graph.complex_storage.bytes(),
        graph.transfers[index].host.get(),stream);
  if(node.kind==RNS8_RESIDENT_BASE_EXTEND) {
    const auto& transfer=graph.transfers[index];
    if(!transfer.layout.device_bytes)return RNS8_SUCCESS;
    return rns8::detail::hip_direct_enqueue_exact_lift(graph.ctx->device_id,node.c->hip_residues,
        node.c->desc.rows*node.c->desc.cols,transfer.source,transfer.target,
        node.c->desc.semantics==RNS8_EXACT_WIDE_SIGNED,transfer.bound.data(),graph.complex_storage.get(),
        graph.complex_storage.bytes(),transfer.host.get(),false,stream);
  }
  if(node.kind==RNS8_RESIDENT_EXPORT_LIMBS) {
    const auto& transfer=graph.transfers[index];
    return rns8::detail::hip_direct_enqueue_limb_export_to_host(graph.ctx->device_id,node.c->hip_residues,
        node.c->desc.rows,node.c->desc.cols,node.plan->prefix,node.plan->desc.semantics==RNS8_EXACT_WIDE_SIGNED,
        node.export_limb_count,graph.complex_storage.get(),graph.complex_storage.bytes(),
        transfer.host.get(),transfer.host.capacity(),stream);
  }
#endif
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
std::array<const void *, 12>
resident_node_addresses(const rns8_resident_graph& graph, size_t index) {
  const auto& node = graph.nodes[index];
  if(node.kind==RNS8_RESIDENT_NATIVE_TO_RNS)
    return {node.c->hip_residues,graph.complex_storage.get(),graph.transfers[index].host.get(),
        graph.native_imports[index]->source(),graph.native_imports[index]->retained_native()};
  if(node.kind==RNS8_RESIDENT_BASE_EXTEND || node.kind==RNS8_RESIDENT_EXPORT_LIMBS)
    return {node.c->hip_residues,graph.complex_storage.get(),graph.transfers[index].host.get(),node.export_limbs};
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
  if(node.kind==RNS8_RESIDENT_BASE_EXTEND || node.kind==RNS8_RESIDENT_EXPORT_LIMBS)return;
  invalidate_owned_graph_output(*node.c);
  if (node.c_imag) invalidate_owned_graph_output(*node.c_imag);
}
void finish_resident_graph_hip(rns8_resident_graph &graph, rns8_status status,
                               uint32_t failed) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  graph.complex_storage.mark_completed();
  for(size_t i=0;i<graph.transfers.size();++i) {
    const auto& node=graph.nodes[i];auto& transfer=graph.transfers[i];
    transfer.host.mark_completed();
    if(status!=RNS8_SUCCESS || !transfer.layout.pinned_bytes)continue;
    const int flag=*static_cast<const int*>(transfer.host.get());
    if(flag) {
      status=node.kind==RNS8_RESIDENT_BASE_EXTEND?(flag==1?RNS8_INVALID_ARGUMENT:flag==2?RNS8_RANGE_ERROR:RNS8_BACKEND_FAILURE):
          flag==int(RNS8_RANGE_ERROR)?RNS8_RANGE_ERROR:flag==int(RNS8_INVALID_ARGUMENT)?RNS8_INVALID_ARGUMENT:RNS8_BACKEND_FAILURE;
      failed=uint32_t(i);
    }
  }
#endif
  if (status == RNS8_SUCCESS && graph.cancel_requested) status = RNS8_CANCELED;
  for(auto* input:graph.native_readers)input->release_external(status==RNS8_SUCCESS);
  for (const auto &input : graph.inputs)
    --input.matrix->in_flight_readers;
  for (auto *workspace : graph.hip_workspaces)
    workspace->in_flight = false;
  for(auto* matrix:graph.hip_outputs)matrix->in_flight_writer=false;
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &node = graph.nodes[i];
    if (status == RNS8_SUCCESS) {
      if(node.kind==RNS8_RESIDENT_NATIVE_TO_RNS) {
        mark_output_device_residues_current(*node.c);
        node.c->exact_max_magnitude.swap(graph.pending_proofs[i].bound);
        node.c->exact_range_prefix=node.c->materialized_rns_prefix=node.target_prefix;
        node.c->exact_axis_ranges.rows.clear();node.c->exact_axis_ranges.cols.clear();
        if(graph.native_imports[i]->retained_native())mark_device_native_current(*node.c);
        continue;
      }
      if(node.kind==RNS8_RESIDENT_BASE_EXTEND) {
        const auto target=graph.pending_proofs[i].target_prefix;
        if(target>node.c->exact_range_prefix)node.c->host_residues_current=false;
        node.c->exact_range_prefix=std::max(node.c->exact_range_prefix,target);
        node.c->materialized_rns_prefix=std::max(node.c->materialized_rns_prefix,target);
        continue;
      }
      if(node.kind==RNS8_RESIDENT_EXPORT_LIMBS) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
        const auto* source=static_cast<const uint8_t*>(graph.transfers[i].host.get())+8;
        const auto row_bytes=uint64_t(node.c->desc.cols)*node.export_limb_count*8;
        for(int64_t row=0;row<node.c->desc.rows;++row)
          std::memcpy(node.export_limbs+uint64_t(row)*node.export_ld*node.export_limb_count,source+uint64_t(row)*row_bytes,size_t(row_bytes));
#endif
        continue;
      }
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
      if (node.workspace && seen.insert(node.workspace).second)
        workspaces.push_back(node.workspace);
    }
    uint32_t failed = UINT32_MAX;
    std::vector<rns8_resident_graph::OutputProof> proofs;
    const auto status = prove_graph(*graph, failed, true, &proofs);
    if (status != RNS8_SUCCESS)
      return status;
    if(graph->hip_prepared) {
      const auto envelope=validate_resident_transfer_envelopes(*graph,proofs);
      if(envelope!=RNS8_SUCCESS)return envelope;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_scoped_device device_scope(graph->ctx->device_id);
    if (device_scope.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!graph->completion_event &&
        graph->completion_event.create() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!graph->hip_prepared) {
      std::vector<rns8::detail::ComplexResidueScratch> resources;
      std::vector<rns8_resident_graph::Transfer> transfers(graph->nodes.size());
      std::vector<rns8_matrix*> outputs;std::unordered_set<rns8_matrix*> writers;
      resources.reserve(proofs.size());uint64_t bytes=0,pinned_bytes=0;
      for(size_t i=0;i<graph->nodes.size();++i) {
        const auto& node=graph->nodes[i];const auto& proof=proofs[i];auto& transfer=transfers[i];
        resources.push_back(proof.complex_scratch);bytes=std::max(bytes,proof.complex_scratch.scratch_bytes);
        if(node.kind!=RNS8_RESIDENT_EXPORT_LIMBS) {
          if(writers.insert(node.c).second)outputs.push_back(node.c);
          if(node.c_imag && writers.insert(node.c_imag).second)outputs.push_back(node.c_imag);
        }
        if(node.kind==RNS8_RESIDENT_BASE_EXTEND) {
          transfer.source=proof.source_prefix;transfer.target=proof.target_prefix;
          const auto product=rns8::detail::modulus_product(transfer.source);
          transfer.envelope=node.c->desc.semantics==RNS8_EXACT_WIDE_SIGNED?
              boost::multiprecision::cpp_int((product-1)/2):boost::multiprecision::cpp_int(product-1);
          transfer.layout=rns8::detail::resident_lift_layout(node.c->desc.rows*node.c->desc.cols,transfer.source,transfer.target);
          auto bound=transfer.envelope;
          for(auto& limb:transfer.bound){limb=static_cast<uint64_t>(bound&UINT64_MAX);bound>>=64;}
          if(bound!=0)return RNS8_RANGE_ERROR;
        } else if(node.kind==RNS8_RESIDENT_NATIVE_TO_RNS) {
          transfer.layout=rns8::detail::resident_import_layout(node.c->desc.rows,node.c->desc.cols,node.target_prefix);
          transfer.target=node.target_prefix;transfer.envelope=proof.bound;
        } else if(node.kind==RNS8_RESIDENT_EXPORT_LIMBS)
          transfer.layout=rns8::detail::resident_export_layout(node.c->desc.rows,node.c->desc.cols,node.export_limb_count);
        else continue;
        if(transfer.layout.status!=RNS8_SUCCESS)return transfer.layout.status;
        if(transfer.layout.device_bytes>UINT64_MAX-transfer.layout.pinned_bytes ||
            transfer.layout.device_bytes+transfer.layout.pinned_bytes>node.max_transfer_storage_bytes)return RNS8_WORKSPACE_TOO_SMALL;
        if(node.kind==RNS8_RESIDENT_NATIVE_TO_RNS) {
          const auto native_bytes=graph->native_imports[i]->required_native_bytes();
          if(native_bytes>node.max_transfer_storage_bytes-transfer.layout.device_bytes-transfer.layout.pinned_bytes)return RNS8_WORKSPACE_TOO_SMALL;
          const auto prepared=graph->native_imports[i]->prepare_retained_native();if(prepared!=RNS8_SUCCESS)return prepared;
        }
        if(transfer.layout.pinned_bytes>uint64_t(PTRDIFF_MAX)-pinned_bytes)return RNS8_RANGE_ERROR;
        pinned_bytes+=transfer.layout.pinned_bytes;
        bytes=std::max(bytes,transfer.layout.device_bytes);
        if(transfer.layout.pinned_bytes && transfer.host.ensure(graph->ctx->device_id,size_t(transfer.layout.pinned_bytes))!=hipSuccess)
          return RNS8_BACKEND_FAILURE;
      }
      // Ordered nodes share scratch; each host export owns a separate staging
      // payload until the final event. Addresses remain fixed after preparation.
      if(!graph->synchronous_stream.get() && graph->synchronous_stream.create_non_blocking()!=hipSuccess)return RNS8_BACKEND_FAILURE;
      if(bytes) {const auto allocation=graph->complex_storage.allocate(graph->ctx->device_id,size_t(bytes));if(allocation!=RNS8_SUCCESS)return allocation;}
      graph->complex_resources.swap(resources);graph->transfers.swap(transfers);graph->hip_outputs.swap(outputs);
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
    status=validate_resident_transfer_envelopes(*graph,proofs);
    if(status!=RNS8_SUCCESS)return status;
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
    for(auto* input:graph->native_readers)input->reserve_external();
    for(auto* output:graph->hip_outputs)output->in_flight_writer=true;
    graph->complex_storage.mark_pending();
    for(auto& transfer:graph->transfers)if(transfer.layout.pinned_bytes) {
      transfer.host.mark_pending();*static_cast<int*>(transfer.host.get())=-1;
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
    for(auto* input:graph->native_readers) {
      status=input->before_enqueue(stream);if(status!=RNS8_SUCCESS)break;
    }
    if (status==RNS8_SUCCESS && replay) {
      for (const auto &node : graph->nodes)
        invalidate_owned_graph_outputs(node);
      launch_error = hipGraphLaunch(graph->captured_executable, hip_stream);
      status = launch_error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
    } else if(status==RNS8_SUCCESS)
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
    std::vector<std::array<const void *, 12>> addresses;
    addresses.reserve(graph->nodes.size());
    for (size_t i = 0; i < graph->nodes.size(); ++i)
      addresses.push_back(resident_node_addresses(*graph, i));
    if(graph->captured_executable)
      return addresses==graph->captured_addresses?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;
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
