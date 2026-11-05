return "not_applicable";
}

void print_plan_packing_json(std::ostream& output, const BenchmarkResult& result) {
  if (!result.packing_info_available) {
    output << "  \"plan_packing\": null,\n";
    return;
  }
  const auto& packing = result.packing_info;
  output << "  \"plan_packing\": {\n";
  output << "    \"source\": \"rns8_get_plan_packing_info\",\n";
  output << "    \"backend\": \"" << backend_name(packing.backend) << "\",\n";
  output << "    \"semantics\": \"" << c_semantics_name(packing.semantics) << "\",\n";
  output << "    \"uses_resident_matrix_inputs\": "
         << (packing.uses_resident_matrix_inputs ? "true" : "false") << ",\n";
  output << "    \"uses_transient_pack_workspace\": "
         << (packing.uses_transient_pack_workspace ? "true" : "false") << ",\n";
  output << "    \"uses_matrix_engine_pack_layout\": "
         << (packing.uses_matrix_engine_pack_layout ? "true" : "false") << ",\n";
  output << "    \"reusable_prepack_cache_available\": "
         << (packing.reusable_prepack_cache_available ? "true" : "false") << ",\n";
  output << "    \"production_prepack_cache_available\": "
         << (packing.production_prepack_cache_available ? "true" : "false") << ",\n";
  output << "    \"input_domain_name\": \"" << json_escape(packing.input_domain_name) << "\",\n";
  output << "    \"output_domain_name\": \"" << json_escape(packing.output_domain_name) << "\",\n";
  output << "    \"output_host_current\": " << (packing.output_host_current ? "true" : "false")
         << ",\n";
  output << "    \"output_device_current\": " << (packing.output_device_current ? "true" : "false")
         << ",\n";
  output << "    \"source_versioned_inputs\": "
         << ((packing.flags & RNS8_PLAN_PACKING_SOURCE_VERSIONED_INPUTS) ? "true" : "false")
         << ",\n";
  output << "    \"same_source_version_pack_elision_available\": "
         << ((packing.flags & RNS8_PLAN_PACKING_SAME_SOURCE_VERSION_PACK_ELISION) ? "true"
                                                                                  : "false")
         << ",\n";
  output << "    \"flags\": " << packing.flags << ",\n";
  output << "    \"next_op_flags\": " << packing.next_op_flags << ",\n";
  output << "    \"next_op_hint\": \"" << json_escape(packing.next_op_hint) << "\",\n";
  output << "    \"a_pack_workspace_bytes\": " << packing.a_pack_workspace_bytes << ",\n";
  output << "    \"b_pack_workspace_bytes\": " << packing.b_pack_workspace_bytes << ",\n";
  output << "    \"accumulator_workspace_bytes\": " << packing.accumulator_workspace_bytes << ",\n";
  output << "    \"library_workspace_bytes\": " << packing.library_workspace_bytes << ",\n";
  output << "    \"total_transient_workspace_bytes\": " << packing.total_transient_workspace_bytes
         << ",\n";
  output << "    \"a_layout_version\": \"" << json_escape(packing.a_layout_version) << "\",\n";
  output << "    \"b_layout_version\": \"" << json_escape(packing.b_layout_version) << "\",\n";
  output << "    \"output_layout_version\": \"" << json_escape(packing.output_layout_version)
         << "\",\n";
  output << "    \"prepack_cache_scope\": \"" << json_escape(packing.prepack_cache_scope)
         << "\",\n";
  output << "    \"detail\": \"" << json_escape(packing.detail) << "\"\n";
  output << "  },\n";
}

void print_plan_lowering_json(std::ostream& output, const BenchmarkResult& result) {
  if (!result.lowering_info_available) {
    output << "  \"plan_lowering\": null,\n";
    return;
  }
  const auto& lowering = result.lowering_info;
  output << "  \"plan_lowering\": {\n";
  output
      << "    \"source\": \"rns8_get_plan_backend_info+rns8_get_plan_packing_info+rns8_get_plan_schedule_info\",\n";
  output << "    \"operation\": \"" << json_escape(lowering.operation) << "\",\n";
  output << "    \"semantic_contract\": \"" << json_escape(lowering.semantic_contract) << "\",\n";
  output << "    \"backend_family\": \"" << json_escape(lowering.backend_family) << "\",\n";
  output << "    \"input_domain\": \"" << json_escape(lowering.input_domain) << "\",\n";
  output << "    \"output_domain\": \"" << json_escape(lowering.output_domain) << "\",\n";
  output << "    \"desired_output\": \"" << json_escape(lowering.desired_output) << "\",\n";
  output << "    \"schedule_strategy\": \"" << json_escape(lowering.schedule_strategy) << "\",\n";
  output << "    \"packing_strategy\": \"" << json_escape(lowering.packing_strategy) << "\",\n";
  output << "    \"reuse_strategy\": \"" << json_escape(lowering.reuse_strategy) << "\",\n";
  output << "    \"conversion_strategy\": \"" << json_escape(lowering.conversion_strategy)
         << "\",\n";
  output << "    \"lowering_path\": \"" << json_escape(lowering.lowering_path) << "\",\n";
  output << "    \"final_export_available\": "
         << (lowering.final_export_available ? "true" : "false") << ",\n";
  output << "    \"rns_continuation_available\": "
         << (lowering.rns_continuation_available ? "true" : "false") << ",\n";
  output << "    \"native_continuation_available\": "
         << (lowering.native_continuation_available ? "true" : "false") << ",\n";
  output << "    \"native_to_rns_available\": "
         << (lowering.native_to_rns_available ? "true" : "false") << ",\n";
  output << "    \"reusable_b_prepack_available\": "
         << (lowering.reusable_b_prepack_available ? "true" : "false") << "\n";
  output << "  },\n";
}

void print_requested_next_op_json(std::ostream& output, const Args& args,
                                  const BenchmarkResult& result) {
  const std::string resolved = resolved_next_op_hint(args, result);
  output << "  \"requested_next_op\": {\n";
  output << "    \"requested\": \"" << next_op_hint_name(args.next_op_hint) << "\",\n";
  output << "    \"resolved\": \"" << json_escape(resolved) << "\",\n";
  output << "    \"source\": \"" << next_op_hint_source(args) << "\",\n";
  output << "    \"final_export_available\": "
         << (result.lowering_info_available && result.lowering_info.final_export_available
                 ? "true"
                 : "false")
         << ",\n";
  output << "    \"rns_continuation_available\": "
         << (result.lowering_info_available && result.lowering_info.rns_continuation_available
                 ? "true"
                 : "false")
         << ",\n";
  output << "    \"native_continuation_available\": "
         << (result.lowering_info_available && result.lowering_info.native_continuation_available
                 ? "true"
                 : "false")
         << ",\n";
  output << "    \"native_to_rns_available\": "
         << (result.lowering_info_available && result.lowering_info.native_to_rns_available
                 ? "true"
                 : "false")
         << ",\n";
  output << "    \"reusable_b_prepack_available\": "
         << (result.lowering_info_available && result.lowering_info.reusable_b_prepack_available
                 ? "true"
                 : "false")
         << "\n";
  output << "  },\n";
}

void print_output_policy_json(std::ostream& output, const Args& args, int64_t output_ld) {
  output << "  \"output_policy\": {\n";
  output << "    \"completion_boundary\": \""
         << (args.grouped_compact_output ? "compact_task_major_host_slab"
                                         : "per_task_host_destinations")
         << "\",\n";
  output << "    \"destination_materialization\": \""
         << (args.grouped_compact_output ? "outside_timing_for_oracle_only"
                                         : "inside_each_timed_export")
         << "\",\n";
  output << "    \"destination_layout\": \"" << output_destination_layout(args) << "\",\n";
  output << "    \"logical_ld\": " << output_ld << ",\n";
  output << "    \"ld_padding\": " << args.output_ld_padding << ",\n";
  output << "    \"per_repeat_logical_export\": "
         << (residue_current_output_mode(args) ? "false" : "true") << ",\n";
  output << "    \"final_checksum_export_after_repeats\": "
         << (residue_current_output_mode(args) ? "true" : "false") << ",\n";
  output << "    \"status_handling\": \"" << output_status_handling(args) << "\",\n";
  output << "    \"status_event_policy\": \"" << output_status_event_policy(args) << "\"\n";
  output << "  },\n";
}

void print_target_variant_json(std::ostream& output, const Args& args, const rns8_device_info& info,
                               const BenchmarkResult& result, const char* selected_backend) {
  const std::string target_id = benchmark_key_target_id(result);
  const std::string target_namespace = target_namespace_for_id(target_id);
  const uint32_t visible_device_count = visible_device_count_from_environment();
  const uint32_t node_gpu_count = benchmark_node_gpu_count();
  output << "  \"target_variant\": {\n";
  output << "    \"target_id\": \"" << json_escape(target_id) << "\",\n";
  output << "    \"target_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  output << "    \"target_cache_key\": \"" << json_escape(target_cache_key(info, result))
         << "\",\n";
  output << "    \"target_instance_id\": \"" << json_escape(target_instance_id(info, result))
         << "\",\n";
  output << "    \"target_namespace\": \"" << json_escape(target_namespace) << "\",\n";
  output << "    \"review_group_key\": \""
         << json_escape(target_review_group_key(args, info, result, selected_backend)) << "\",\n";
  output << "    \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS)
         << "\",\n";
  output << "    \"device_index\": " << info.device_id << ",\n";
  output << "    \"device_name\": \"" << json_escape(info.name) << "\",\n";
  output << "    \"visible_device_count\": " << visible_device_count << ",\n";
  output << "    \"node_gpu_count\": " << node_gpu_count << ",\n";
  output << "    \"hip_enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  output << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  output << "    \"hip_driver_version\": " << info.hip_driver_version << "\n";
  output << "  },\n";
}

void print_counter_object(std::ostream& output,
                          const rns8::detail::hip_direct_allocation_counters& counters) {
  output << "{"
         << "\"allocate_calls\": " << counters.allocate_calls << ", "
         << "\"free_calls\": " << counters.free_calls << ", "
         << "\"allocated_bytes\": " << counters.allocated_bytes
         << ", \"h2d_calls\": " << counters.h2d_calls << ", \"h2d_bytes\": " << counters.h2d_bytes
         << ", \"d2h_calls\": " << counters.d2h_calls << ", \"d2h_bytes\": " << counters.d2h_bytes
         << ", \"d2d_calls\": " << counters.d2d_calls << ", \"d2d_bytes\": " << counters.d2d_bytes
         << "}";
}

rns8::detail::hip_direct_allocation_counters
allocation_delta(const rns8::detail::hip_direct_allocation_counters& before,
                 const rns8::detail::hip_direct_allocation_counters& after) {
  rns8::detail::hip_direct_allocation_counters delta{};
  delta.allocate_calls = after.allocate_calls >= before.allocate_calls
                             ? after.allocate_calls - before.allocate_calls
                             : 0;
  delta.free_calls =
      after.free_calls >= before.free_calls ? after.free_calls - before.free_calls : 0;
  delta.allocated_bytes = after.allocated_bytes >= before.allocated_bytes
                              ? after.allocated_bytes - before.allocated_bytes
                              : 0;
  delta.h2d_calls = after.h2d_calls >= before.h2d_calls ? after.h2d_calls - before.h2d_calls : 0;
  delta.h2d_bytes = after.h2d_bytes >= before.h2d_bytes ? after.h2d_bytes - before.h2d_bytes : 0;
  delta.d2h_calls = after.d2h_calls >= before.d2h_calls ? after.d2h_calls - before.d2h_calls : 0;
  delta.d2h_bytes = after.d2h_bytes >= before.d2h_bytes ? after.d2h_bytes - before.d2h_bytes : 0;
  delta.d2d_calls = after.d2d_calls >= before.d2d_calls ? after.d2d_calls - before.d2d_calls : 0;
  delta.d2d_bytes = after.d2d_bytes >= before.d2d_bytes ? after.d2d_bytes - before.d2d_bytes : 0;
  return delta;
}

const char* benchmark_setup_scope(const Args& args) {
  if (args.oneshot) {
    return "transient_api_setup_inside_measured_repeat";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "benchmark_experimental_residue_channel_fusion_persistent_output";
  }
  if (hip_graph_replay_full_bounded_pack_export_requested(args)) {
    return "benchmark_hip_graph_replay_bounded_pack_gemm_export";
  }
  if (hip_graph_replay_full_finite_pack_export_requested(args)) {
    return "benchmark_hip_graph_replay_finite_u8_pack_gemm_export";
  }
  if (hip_graph_replay_full_exact_wide_pack_export_requested(args)) {
    return "benchmark_hip_graph_replay_exact_wide_pack_gemm_export";
  }
  if (hip_graph_replay_full_wrap64_pack_export_requested(args)) {
    return "benchmark_hip_graph_replay_wrap64_pack_gemm_export";
  }
  if (hip_graph_replay_resident_chain_requested(args)) {
    return "benchmark_hip_graph_replay_resident_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "benchmark_grouped_dispatch_one_shared_plan_persistent_resident_tasks";
  }
  if (args.vector_alu_baseline || runtime_vector_alu_backend(args) ||
      bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "benchmark_owned_transient_native_input_buffers";
  }
  if (args.reuse_packed_inputs) {
    return "persistent_plan_workspace_prepacked_reuse";
  }
  return "persistent_plan_workspace_resident_matrices";
}

void print_device_allocation_json(std::ostream& output, const Args& args,
                                  const BenchmarkResult& result) {
  output << "  \"device_allocation\": {\n";
  output << "    \"tracking_available\": "
         << (result.allocation_tracking_available ? "true" : "false") << ",\n";
  output << "    \"source\": \"hip_direct_allocation_counters_snapshot\",\n";
  output << "    \"transfer_scope\": \"direct_backend_successful_synchronous_api_payload\",\n";
  output << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  output
      << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  output << "    \"before\": ";
  print_counter_object(output, result.allocation_before);
  output << ",\n";
  output << "    \"after_warmups\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(output, result.allocation_after_warmups);
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"after_repeats\": ";
  print_counter_object(output, result.allocation_after_repeats);
  output << ",\n";
  output << "    \"setup_delta\": ";
  print_counter_object(output,
                       allocation_delta(result.allocation_before, result.allocation_after_warmups));
  output << ",\n";
  output << "    \"measured_repeat_delta\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(
        output, allocation_delta(result.allocation_after_warmups, result.allocation_after_repeats));
  } else {
    output << "null";
  }
  output << "\n";
  output << "  },\n";
}

void print_auto_selector_json(std::ostream& output, const Args& args, const rns8_device_info& info,
                              const BenchmarkResult& result, const char* selected_backend) {
  const auto& trace = result.auto_trace;
  const std::string selected_key = result.backend_info.autotune_key;
  output << "  \"auto_selector\": {\n";
  output << "    \"source\": \"immutable_plan_selection_trace_v1\",\n";
  output << "    \"requested_backend\": \"" << requested_backend_name(args) << "\",\n";
  output << "    \"selected_backend\": \"" << selected_backend << "\",\n";
  output << "    \"selected_key\": ";
  print_nullable_std_string(output, selected_key);
  output << ",\n";
  output << "    \"trace_recorded\": " << (result.auto_trace_available ? "true" : "false") << ",\n";
  output << "    \"selection_performed\": " << (trace.selection_performed ? "true" : "false")
         << ",\n";
  output << "    \"cache_path\": \"" << json_escape(trace.cache_path) << "\",\n";
  output << "    \"cache_exists\": " << (trace.cache_exists ? "true" : "false") << ",\n";
  output << "    \"cache_loaded\": " << (trace.cache_loaded ? "true" : "false") << ",\n";
  output << "    \"cache_entry_count\": " << trace.cache_entry_count << ",\n";
  output << "    \"runtime_target_id\": \"" << json_escape(benchmark_key_target_id(result))
         << "\",\n";
  output << "    \"runtime_version\": \"" << json_escape(result.backend_info.accelerator_version)
         << "\",\n";
  output << "    \"exact_hit\": " << (trace.selected_exact_entry ? "true" : "false") << ",\n";
  output << "    \"validated_hit\": " << (trace.selected_validated_entry ? "true" : "false")
         << ",\n";
  output << "    \"fallback_reason\": \""
         << json_escape(result.auto_trace_available ? trace.decision
                                                    : "unknown; plan trace not recorded")
         << "\",\n";
  output
      << "    \"rejection_reason_vocabulary\": [\"unsupported semantics\",\"unsupported modulus\",\"per-tile unsupported\",\"backend not compiled\",\"probe failed\",\"no exact entry\",\"unvalidated entry\",\"identity/runtime mismatch\",\"workspace mismatch\",\"slower than selected\",\"tied with selected\",\"arithmetic proof rejected\"],\n";
  output << "    \"candidate_decisions\": [";
  bool first = true;
  for (const auto& d : result.auto_decisions) {
    output << (first ? "\n" : ",\n");
    first = false;
    output << "      {\"backend\":\"" << backend_name(static_cast<rns8_backend_kind>(d.backend))
           << "\",\"disposition\":" << d.disposition << ",\"reason_code\":" << d.reason_code
           << ",\"reason\":\"" << json_escape(d.reason)
           << "\",\"exact_entry_present\":" << (d.exact_entry_present ? "true" : "false")
           << ",\"validated_entry_present\":" << (d.validated_entry_present ? "true" : "false")
           << ",\"required_workspace_bytes\":" << d.required_workspace_bytes
           << ",\"reviewed_median_us\":";
    if (d.reason_code == RNS8_AUTO_SLOWER || d.reason_code == RNS8_AUTO_TIED ||
        d.reason_code == RNS8_AUTO_SELECTED)
      output << d.reviewed_median_us;
    else
      output << "null";
    output << ",\"key_sha256\":\"" << d.key_sha256 << "\"}";
  }
  output << "\n    ],\n    \"rejected_candidates\": [";
  first = true;
  for (const auto& d : result.auto_decisions)
    if (d.disposition == 0) {
      output << (first ? "\n" : ",\n");
      first = false;
      output << "      {\"backend\":\"" << backend_name(static_cast<rns8_backend_kind>(d.backend))
             << "\",\"reason\":\"" << json_escape(d.reason) << "\"}";
    }
  output << "\n    ]\n  },\n";
  (void)info;
}

const char* reuse_operand_role(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "none";
  }
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "A+B";
  }
  if (args.reuse_packed_a) {
    return "A";
  }
  if (args.reuse_packed_b) {
    return "B";
  }
  return "none";
}

const char* shape_family_bucket(const Args& args) {
  if (args.m <= 128 && args.n <= 128 && args.k <= 128) {
    return "small";
  }
  if (args.m <= 1024 && args.n <= 1024 && args.k <= 1024) {
    return "medium";
  }
  if (args.m == args.n && args.n == args.k) {
    return "large_square";
  }
  if (args.n <= 8) {
    return "skinny_n";
  }
  return "large_general";
}

std::string output_domain_contract_name(const Args& args) {
  if (residue_current_output_mode(args)) {
    return "rns_residue_current";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "finite_u8_host";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_u64_host";
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return "exact_wide_limb_host";
  }
  return "native_i64_u64_host";
}

uint32_t estimated_modulus_product_bits(const Args& args, const BenchmarkResult& result) {
  if (finite_benchmark_semantics(args.semantics)) {
    return args.finite_modulus <= 1 ? 0u : 8u;
  }
  const uint32_t prefix = selected_execution_prefix(args, result);
  return prefix * 8u;
}

void print_prepack_setup_breakdown_json(std::ostream& output, const BenchmarkResult& result,
                                        const char* indent) {
  if (!result.prepack_setup_available) {
    output << "null";
    return;
  }
  output << "{\n";
  output << indent << "  \"pack_a\": " << result.prepack_setup_pack_a_us << ",\n";
  output << indent << "  \"pack_b\": " << result.prepack_setup_pack_b_us << ",\n";
  output << indent << "  \"runtime_cache\": " << result.prepack_setup_runtime_cache_us << ",\n";
  output << indent << "  \"unclassified\": " << result.prepack_setup_unclassified_us << "\n";
  output << indent << "}";
}

void print_graph_caller_observation_json(std::ostream& output, const Args& args,
                                         const GraphCallerObservation& observation) {
  output << "{\"total_us\": " << observation.total_us << ", \"input_staging_us\": ";
  if (!args.production_host_timing)
    output << observation.phases_us[0];
  else
    output << "null";
  output << ", \"replay_us\": ";
  if (!args.production_host_timing)
    output << observation.phases_us[1];
  else
    output << "null";
  output << ", \"caller_delivery_us\": ";
  if (!args.production_host_timing)
    output << observation.phases_us[2];
  else
    output << "null";
  output << "}";
}

void print_graph_caller_transaction_json(std::ostream& output, const Args& args,
                                         const BenchmarkResult& result) {
  if (!result.hip_graph_caller_transaction) {
    output << "null";
    return;
  }
  output
      << "{\"schema\": \"rns8_graph_caller_transaction_v1\", "
      << "\"completion_boundary\": \"per_task_host_destinations\", "
      << "\"input_staging\": \"every_completed_call\", \"caller_output_delivery\": \"every_completed_call\", "
      << "\"common_setup_excluded\": [\"planning_workspace\", \"matrix_allocation\"], "
      << "\"phase_clocks\": " << (args.production_host_timing ? "false" : "true")
      << ", \"additional_preparation_us\": " << result.hip_graph_owner_prepare_us
      << ", \"stream_preparation_us\": " << result.hip_graph_stream_prepare_us
      << ", \"teardown_us\": " << result.hip_graph_owner_teardown_us
      << ", \"input_a_bytes_per_call\": " << result.hip_graph_caller_payload_bytes[0]
      << ", \"input_b_bytes_per_call\": " << result.hip_graph_caller_payload_bytes[1]
      << ", \"output_bytes_per_call\": " << result.hip_graph_caller_payload_bytes[2]
      << ", \"first_replay\": ";
  if (result.hip_graph_first_caller_observation_available)
    print_graph_caller_observation_json(output, args, result.hip_graph_first_caller_observation);
  else
    output << "null";
  output << ", \"observations\": [";
  for (std::size_t i = 0; i < result.hip_graph_caller_samples.size(); ++i) {
    if (i)
      output << ", ";
    print_graph_caller_observation_json(output, args, result.hip_graph_caller_samples[i]);
  }
  output << "]}";
}

void print_first_call_setup_inventory_json(std::ostream& output, const Args& args,
                                           const BenchmarkResult& result) {
  // These parents are disjoint; prepack's A/B/cache subphases must not be added again.
  const bool supported =
      !args.oneshot && !args.transient_uniform_small_inputs && !args.hip_graph_replay &&
      !args.streaming_overlap && !args.native_to_rns_bridge && !args.resident_lifetime &&
      !args.workspace_arena && !args.wrap64_rocwmma_candidate && !args.vector_to_rns_chain &&
      !args.vector_alu_baseline && !runtime_vector_alu_backend(args) && !args.sparse_a_4_to_2 &&
      !args.residue_channel_fusion && args.residue_chain_length == 1 &&
      !host_api_batch_requested(args) && !grouped_dispatch_requested(args) &&
      args.resident_redesign_candidate.empty() &&
      (!args.reuse_packed_inputs || result.prepack_setup_available);
  output
      << "  \"first_call_setup_inventory\": {\"schema\": \"rns8_first_call_setup_inventory_v1\", "
      << "\"boundary\": \"caller_buffers_ready_after_context_creation\", \"supported\": "
      << (supported ? "true" : "false")
      << ", \"components\": {\"planning_workspace\": \"raw_timings_us.planning\", "
      << "\"matrix_allocation\": \"raw_timings_us.matrix_alloc\", \"prepack\": "
      << (args.reuse_packed_inputs ? "\"prepack_setup_us\"" : "null")
      << ", \"completed_call\": \"raw_timings_us.end_to_end\"}, "
      << "\"excluded\": [\"context_creation\", \"caller_buffer_allocation\", \"input_generation\", \"independent_oracle\"]},\n";
}

void print_reuse_contract_json(std::ostream& output, const Args& args,
                               const BenchmarkResult& result, const char* selected_backend,
                               const char* selected_kernel) {
  const bool reuse_enabled = args.reuse_packed_inputs || args.residue_chain_length > 1 ||
                             host_api_batch_requested(args) || grouped_dispatch_requested(args);
  output << "  \"reuse_contract\": {\n";
  output << "    \"enabled\": " << (reuse_enabled ? "true" : "false") << ",\n";
  output << "    \"operand_role\": \"" << reuse_operand_role(args) << "\",\n";
  output
      << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  output << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  output << "    \"setup_cost_us\": ";
  if (result.prepack_setup_available) {
    output << result.prepack_setup_us;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"setup_breakdown_us\": ";
  print_prepack_setup_breakdown_json(output, result, "    ");
  output << ",\n";
  const bool setup_inclusive_available =
      result.prepack_setup_available && args.repeats > 0 && !result.samples.end_to_end_us.empty();
  const double setup_amortized_us =
      setup_inclusive_available
          ? static_cast<double>(result.prepack_setup_us) / static_cast<double>(args.repeats)
          : 0.0;
  const double repeat_median_end_to_end_us =
      setup_inclusive_available ? median(result.samples.end_to_end_us) : 0.0;
  output << "    \"setup_amortized_us\": ";
  if (setup_inclusive_available) {
    output << setup_amortized_us;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"repeat_median_end_to_end_us\": ";
  if (setup_inclusive_available) {
    output << repeat_median_end_to_end_us;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"setup_inclusive_median_end_to_end_us\": ";
  if (setup_inclusive_available) {
    output << (repeat_median_end_to_end_us + setup_amortized_us);
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"setup_inclusive_policy\": \"one_time_setup_amortized_over_measured_repeats\",\n";
  output << "    \"measured_repeat_count\": " << args.repeats << ",\n";
  output << "    \"break_even_repeat_count\": null,\n";
  output << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args))
         << "\",\n";
  output << "    \"next_op\": \"" << json_escape(resolved_next_op_hint(args, result)) << "\",\n";
  output << "    \"target_fingerprint\": \"" << json_escape(benchmark_key_target_id(result))
         << "\",\n";
  output << "    \"backend_fingerprint\": \"" << json_escape(selected_backend) << "\",\n";
  output << "    \"kernel_fingerprint\": ";
  if (selected_kernel) {
    output << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"workspace_fingerprint\": \"" << result.backend_info.workspace_required_bytes
         << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  output << "    \"runtime_prepack_cache\": ";
  if (!result.prepack_cache_info_available) {
    output << "null";
  } else {
    const auto& cache = result.prepack_cache_info;
    output << "{\n";
    output << "      \"source\": \"rns8_get_prepack_cache_info\",\n";
    output << "      \"backend\": \"" << backend_name(cache.backend) << "\",\n";
    output << "      \"semantics\": \"" << c_semantics_name(cache.semantics) << "\",\n";
    output << "      \"operand_role\": \"" << (cache.operand_role == RNS8_OPERAND_B ? "B" : "A")
           << "\",\n";
    output << "      \"cache_key_valid\": " << (cache.cache_key_valid ? "true" : "false") << ",\n";
    output << "      \"reusable_prepack_cache_available\": "
           << (cache.reusable_prepack_cache_available ? "true" : "false") << ",\n";
    output << "      \"production_prepack_cache_available\": "
           << (cache.production_prepack_cache_available ? "true" : "false") << ",\n";
    output << "      \"hip_device_id\": " << cache.hip_device_id << ",\n";
    output << "      \"matrix_rows\": " << cache.matrix_rows << ",\n";
    output << "      \"matrix_cols\": " << cache.matrix_cols << ",\n";
    output << "      \"k\": " << cache.k << ",\n";
    output << "      \"max_prefix\": " << cache.max_prefix << ",\n";
    output << "      \"finite_modulus\": " << cache.finite_modulus << ",\n";
    output << "      \"source_version\": " << cache.source_version << ",\n";
    output << "      \"plan_fingerprint\": " << cache.plan_fingerprint << ",\n";
    output << "      \"cache_key_hash\": " << cache.cache_key_hash << ",\n";
    output << "      \"device_bytes\": " << cache.device_bytes << ",\n";
    output << "      \"operand_pack_bytes\": " << cache.operand_pack_bytes << ",\n";
    output << "      \"matrix_layout_version\": \"" << json_escape(cache.matrix_layout_version)
           << "\",\n";
    output << "      \"operand_layout_version\": \"" << json_escape(cache.operand_layout_version)
           << "\",\n";
    output << "      \"cache_scope\": \"" << json_escape(cache.cache_scope) << "\",\n";
    output << "      \"cache_key\": \"" << json_escape(cache.cache_key) << "\",\n";
    output << "      \"detail\": \"" << json_escape(cache.detail) << "\"\n";
    output << "    }";
  }
  output << ",\n";
  const bool production_runtime_cache =
      result.prepack_cache_info_available &&
      result.prepack_cache_info.production_prepack_cache_available != 0;
  output << "    \"production_runtime_prepack_cache_available\": "
         << (production_runtime_cache ? "true" : "false") << ",\n";
  output << "    \"setup_inclusive_cache_promotion_candidate\": "
         << (reuse_enabled && production_runtime_cache ? "true" : "false") << ",\n";
  output << "    \"promotion_eligible\": " << (!reuse_enabled ? "true" : "false") << ",\n";
  output << "    \"invalidation_reasons\": [";
  bool first = true;
  if (args.reuse_packed_inputs) {
    output << "\"source_version_changed\"";
    first = false;
  }
  if (grouped_dispatch_requested(args)) {
    output << (first ? "" : ", ") << "\"descriptor_identity_changed\"";
    first = false;
  }
  if (args.hip_graph_replay) {
    output << (first ? "" : ", ") << "\"graph_capture_descriptor_changed\"";
    first = false;
  }
  (void)first;
  output << "]\n";
  output << "  },\n";
}

void print_exact_output_contract_json(std::ostream& output, const Args& args,
                                      const BenchmarkResult& result, int64_t output_ld,
                                      const char* selected_kernel) {
  const std::string export_kernel =
      result.export_plan_info_available
          ? result.export_selected_kernel
          : (selected_kernel ? std::string(selected_kernel) : std::string());
  output << "  \"exact_output_contract\": {\n";
  output << "    \"requested_final_output\": \"" << json_escape(output_domain_contract_name(args))
         << "\",\n";
  output << "    \"limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    output << args.exact_wide_limb_count;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"output_logical_ld\": " << output_ld << ",\n";
  output << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  output << "    \"kernel_identity\": ";
  print_nullable_std_string(output, export_kernel);
  output << ",\n";
  output << "    \"output_domain_after_measured_repeats\": \""
         << (residue_current_output_mode(args) ? "rns_residue_current"
                                               : json_escape(output_domain_contract_name(args)))
         << "\",\n";
  output << "    \"final_checksum_export_after_repeats\": "
         << (residue_current_output_mode(args) ? "true" : "false") << "\n";
  output << "  },\n";
  (void)result;
}

void print_export_variant_json(std::ostream& output, const Args& args,
                               const BenchmarkResult& result, const char* selected_kernel) {
  const bool default_variant = args.export_variant == "default";
  const bool reviewable_fixed_limb_variant =
      args.export_variant == "exact-wide-fixed-limb-export" &&
      exact_wide_benchmark_semantics(args.semantics) && args.exact_wide_limb_count > 0 &&
      args.reconstruction_variant == "default_garner";
  const bool promotion_eligible = default_variant || reviewable_fixed_limb_variant;
  const std::string export_kernel =
      result.export_plan_info_available
          ? result.export_selected_kernel
          : (selected_kernel ? std::string(selected_kernel) : std::string());
  output << "  \"export_variant\": {\n";
  output << "    \"name\": \"" << json_escape(args.export_variant) << "\",\n";
  output << "    \"source\": \""
         << (default_variant
                 ? "current_backend_export_path"
                 : (reviewable_fixed_limb_variant ? "reviewable_exact_wide_fixed_limb_selector"
                                                  : "benchmark_cli_evidence_mode"))
         << "\",\n";
  output << "    \"selector_source\": \""
         << json_escape(result.export_plan_info_available ? result.export_plan_source
                                                          : "legacy_selected_kernel")
         << "\",\n";
  output << "    \"selector_key\": ";
  print_nullable_std_string(output, result.export_plan_info_available ? result.export_selector_key
                                                                      : std::string());
  output << ",\n";
  output << "    \"selector_policy\": \""
         << json_escape(result.export_plan_info_available ? result.export_selector_policy
                                                          : "unknown")
         << "\",\n";
  output << "    \"semantic_contract\": \""
         << json_escape(result.export_plan_info_available ? result.export_semantic_contract
                                                          : semantics_name(args.semantics))
         << "\",\n";
  output << "    \"backend\": \""
         << json_escape(result.export_plan_info_available ? result.export_backend
                                                          : backend_name(args.backend))
         << "\",\n";
  output << "    \"target_id\": \""
         << json_escape(result.export_plan_info_available ? result.export_target_id
                                                          : benchmark_key_target_id(result))
         << "\",\n";
  output << "    \"prefix_contract\": \""
         << json_escape(result.export_plan_info_available ? result.export_prefix_contract
                                                          : schedule_hash_string(args, result))
         << "\",\n";
  output << "    \"signedness\": \""
         << json_escape(result.export_plan_info_available ? result.export_signedness : "unknown")
         << "\",\n";
  output << "    \"output_layout\": \""
         << json_escape(result.export_plan_info_available ? result.export_output_layout : "unknown")
         << "\",\n";
  output << "    \"limb_count\": ";
  if (result.export_plan_info_available && result.export_limb_count != 0) {
    output << result.export_limb_count;
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    output << args.exact_wide_limb_count;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  output << "    \"selector_status_policy\": \""
         << json_escape(result.export_plan_info_available ? result.export_selector_status_policy
                                                          : "unknown")
         << "\",\n";
  output << "    \"d2h_policy\": \""
         << json_escape(result.export_plan_info_available ? result.export_d2h_policy : "unknown")
         << "\",\n";
  output << "    \"final_output_mode\": \""
         << (residue_current_output_mode(args)
                 ? "residue_chain_no_final_export"
                 : json_escape(result.export_plan_info_available ? result.export_final_output_mode
                                                                 : "final_host_output"))
         << "\",\n";
  output << "    \"cache_visibility\": \""
         << json_escape(result.export_plan_info_available ? result.export_cache_visibility
                                                          : "legacy_capture")
         << "\",\n";
  output << "    \"stale_entry_reason\": ";
  print_nullable_std_string(
      output, result.export_plan_info_available ? result.export_stale_entry_reason : std::string());
  output << ",\n";
  output << "    \"status_elision_reason\": ";
  print_nullable_std_string(output, result.export_plan_info_available
                                        ? result.export_status_elision_reason
                                        : std::string());
  output << ",\n";
  output << "    \"requires_tile_metadata\": "
         << (result.export_plan_info_available && result.export_requires_tile_metadata ? "true"
                                                                                       : "false")
         << ",\n";
  output << "    \"all_zero_tiled_output\": "
         << (result.export_plan_info_available && result.export_all_zero_tiled_output ? "true"
                                                                                      : "false")
         << ",\n";
  output << "    \"selected_kernel\": ";
  print_nullable_std_string(output, export_kernel);
  output << ",\n";
  output << "    \"constants_placement\": \"backend_default\",\n";
  output << "    \"promotion_eligible\": " << (promotion_eligible ? "true" : "false") << ",\n";
  output << "    \"promotion_blocker\": ";
  print_nullable_std_string(
      output, promotion_eligible ? std::string() : std::string("experimental_export_variant"));
  output << "\n";
  output << "  },\n";
}

void print_reconstruction_variant_json(std::ostream& output, const Args& args,
                                       const BenchmarkResult& result, const char* selected_kernel) {
  const bool default_variant = args.reconstruction_variant == "default_garner";
  const std::string export_kernel =
      result.export_plan_info_available
          ? result.export_selected_kernel
          : (selected_kernel ? std::string(selected_kernel) : std::string());
  output << "  \"reconstruction_variant\": {\n";
  output << "    \"name\": \"" << json_escape(args.reconstruction_variant) << "\",\n";
  output << "    \"family\": \""
         << (default_variant ? "garner_fixed_prefix" : "benchmark_reconstruction_zoo") << "\",\n";
  output << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  output << "    \"kernel_identity\": ";
  print_nullable_std_string(output, export_kernel);
  output << ",\n";
  output << "    \"controller\": \""
         << (default_variant ? "public_export_api"
                             : (args.reconstruction_variant == "tree_crt_candidate"
                                    ? "benchmark_internal_tree_crt_export_kernel"
                                    : "benchmark_metadata_only"))
         << "\",\n";
  output << "    \"promotion_eligible\": " << (default_variant ? "true" : "false") << ",\n";
  output << "    \"promotion_blocker\": ";
  print_nullable_std_string(
      output, default_variant ? std::string() : std::string("experimental_reconstruction_variant"));
  output << "\n";
  output << "  },\n";
}

void print_modulus_and_residue_policy_json(std::ostream& output, const Args& args,
                                           const BenchmarkResult& result) {
  const bool experimental = args.modulus_set != "default";
  output << "  \"modulus_set\": {\n";
  output << "    \"name\": \"" << json_escape(args.modulus_set) << "\",\n";
  output << "    \"source\": \""
         << (experimental ? "benchmark_experimental_ladder" : "rns8_default_modulus_ladder")
         << "\",\n";
  output << "    \"execution_ladder\": \""
         << (finite_benchmark_semantics(args.semantics) ? "finite_single_modulus"
                                                        : "rns8_default_8bit_coprime_ladder")
         << "\",\n";
  output << "    \"experimental\": " << (experimental ? "true" : "false") << ",\n";
  output << "    \"runtime_selectable\": " << (experimental ? "false" : "true") << ",\n";
  output << "    \"search_report_required\": " << (experimental ? "true" : "false") << ",\n";
  output << "    \"product_bits\": " << estimated_modulus_product_bits(args, result) << ",\n";
  output << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  output
      << "    \"pairwise_coprime_proof\": \"schema_declared_current_ladder_or_offline_search_report\",\n";
  output << "    \"reducer_cost_hint\": \""
         << (experimental ? "offline_search_required" : "backend_default") << "\",\n";
  output
      << "    \"default_change_gate\": \"spec_cache_schema_proof_and_same_target_review_required\",\n";
  output << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(output,
                            experimental ? std::string("experimental_modulus_set") : std::string());
  output << "\n";
  output << "  },\n";
  output << "  \"residue_count_policy\": {\n";
  output << "    \"policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  output << "    \"requested_prefix\": " << benchmark_prefix(args) << ",\n";
  output << "    \"selected_prefix\": " << selected_execution_prefix(args, result) << ",\n";
  output << "    \"minimum_range_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  output << "    \"redundant_residue_count\": "
         << (selected_execution_prefix(args, result) > result.schedule_info.min_required_prefix
                 ? selected_execution_prefix(args, result) -
                       result.schedule_info.min_required_prefix
                 : 0)
         << ",\n";
  output << "    \"autotune_scope\": \""
         << (experimental ? "evidence_only_non_promoting" : "current_exact_cache") << "\",\n";
  output << "    \"promotion_eligible\": " << (experimental ? "false" : "true") << ",\n";
  output << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(output, experimental ? std::string("experimental_residue_count_policy")
                                                 : std::string());
  output << "\n";
  output << "  },\n";
}

void print_tile_shape_variant_json(std::ostream& output, const Args& args,
                                   const BenchmarkResult& result, const char* selected_kernel) {
  const bool default_variant = args.tile_shape_variant == "default";
  const int64_t k_block = benchmark_k_block_size(args, result);
  output << "  \"tile_shape_variant\": {\n";
  output << "    \"name\": \"" << json_escape(args.tile_shape_variant) << "\",\n";
  output << "    \"tile_m\": " << args.tile_m << ",\n";
  output << "    \"tile_n\": " << args.tile_n << ",\n";
  output << "    \"tile_k\": " << k_block << ",\n";
  output << "    \"k_block_policy\": \"" << json_escape(args.k_block_policy) << "\",\n";
  output << "    \"split_k_mode\": \"single_gpu_no_split_k\",\n";
  output << "    \"accumulator_safety_key\": \"k_block_size=" << k_block << ";k_block_cap="
         << (result.backend_info_available ? result.backend_info.accumulator_k_block_cap : 0)
         << ";safe_for_k_block="
         << (result.backend_info_available && result.backend_info.accumulator_safe_for_k_block
                 ? "true"
                 : "false")
         << "\",\n";
  output << "    \"selected_kernel_identity\": ";
  if (selected_kernel) {
    output << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"resource_report_key\": \"tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
         << ";tile_k=" << k_block
         << ";kernel=" << (selected_kernel ? json_escape(selected_kernel) : "unknown") << "\",\n";
  output << "    \"shape_family_bucket\": \"" << shape_family_bucket(args) << "\",\n";
  output
      << "    \"resource_report_required\": \"isa_or_counter_for_non_default_k_block_policy\",\n";
  output << "    \"stale_kernel_rejection\": \"selected_kernel_identity_must_match_capture\"\n";
  output << "  },\n";
  (void)default_variant;
}

void print_resident_redesign_json(std::ostream& output, const Args& args) {
  const bool enabled = !args.resident_redesign_candidate.empty();
  output << "  \"resident_redesign\": {\n";
  output << "    \"enabled\": " << (enabled ? "true" : "false") << ",\n";
  output << "    \"candidate\": ";
  if (enabled) {
    output << "\"" << json_escape(args.resident_redesign_candidate) << "\"";
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"dimensions\": [";
  for (std::size_t index = 0; index < args.resident_redesign_dimensions.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << "\"" << json_escape(args.resident_redesign_dimensions[index]) << "\"";
  }
  output << "],\n";
  output << "    \"policy\": \"benchmark_only_resident_route_candidate_requires_rank51_report\",\n";
  output << "    \"resource_evidence_required\": true,\n";
  output << "    \"promotion_eligible\": false,\n";
  output << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(
      output, enabled ? std::string("resident_redesign_candidate_not_reviewed") : std::string());
  output << "\n";
  output << "  },\n";
}
