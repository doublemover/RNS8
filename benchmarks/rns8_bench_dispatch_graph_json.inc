struct GroupedDispatchDescriptorContract {
  bool requested = false;
  uint64_t task_count = 0;
  std::string descriptor_identity;
  std::string source_hash;
  std::string output_hash = "final_checksum_u64";
  std::string setup_scope;
  std::string execution_strategy = "not_requested";
  bool batched_export_enabled = false;
  uint64_t device_output_slab_bytes = 0;
  std::string capture_status = "not_requested";
  bool promotion_eligible = false;
  uint32_t schema_version = 1;
  std::string descriptor_layout = "not_requested";
  std::string bucket_policy = "not_requested";
  uint32_t bucket_count = 0;
  bool same_shape_required = false;
  std::string shape_key;
  std::string semantics;
  std::string output_domain;
  std::string source_version_policy = "not_requested";
  std::string workspace_policy = "not_requested";
  std::string matrix_ownership_policy = "not_requested";
  std::string descriptor_reuse_policy = "not_requested";
  std::string stride_policy = "not_requested";
  std::string output_currentness_policy = "not_requested";
  std::string lifetime_policy = "not_requested";
  std::string checksum_policy = "not_requested";
  std::string status_policy = "not_requested";
  std::string device_descriptor_policy = "not_requested";
};

GroupedDispatchDescriptorContract make_grouped_dispatch_descriptor_contract(
    const Args& args,
    const BenchmarkResult& result) {
  GroupedDispatchDescriptorContract contract{};
  contract.requested = grouped_dispatch_requested(args);
  contract.task_count = args.grouped_dispatch_tasks;
  contract.descriptor_identity =
      "same_shape_m=" + std::to_string(args.m) + ";n=" + std::to_string(args.n) +
      ";k=" + std::to_string(args.k) + ";semantics=" + semantics_name(args.semantics);
  contract.source_hash = std::to_string(args.seed);
  contract.setup_scope = benchmark_setup_scope(args);
  contract.shape_key =
      "m=" + std::to_string(args.m) + ";n=" + std::to_string(args.n) +
      ";k=" + std::to_string(args.k) + ";tile_m=" + std::to_string(args.tile_m) +
      ";tile_n=" + std::to_string(args.tile_n) + ";prefix=" +
      std::to_string(selected_execution_prefix(args, result));
  contract.semantics = semantics_name(args.semantics);
  contract.output_domain = output_domain_contract_name(args);
  contract.execution_strategy =
      contract.requested && result.grouped_dispatch_execution_strategy == "not_requested"
          ? "host_phase_loop_per_task_export"
          : result.grouped_dispatch_execution_strategy;
  contract.batched_export_enabled = contract.requested && result.grouped_dispatch_batched_export_enabled;
  contract.device_output_slab_bytes =
      contract.requested ? result.grouped_dispatch_device_output_slab_bytes : 0;
  contract.capture_status = contract.requested ? "executed" : "not_requested";
  if (!contract.requested) {
    contract.execution_strategy = "not_requested";
    return contract;
  }
  const bool grouped_device_descriptors = contract.execution_strategy.rfind("device_grouped_", 0) == 0;
  contract.descriptor_layout = "same_shape_resident_task_triplets_v1";
  contract.bucket_policy = "single_same_shape_bucket";
  contract.bucket_count = 1;
  contract.same_shape_required = true;
  contract.source_version_policy = "per_task_monotonic_source_version_repack";
  contract.workspace_policy = "one_workspace_per_task_shared_plan";
  contract.matrix_ownership_policy = "benchmark_owns_all_task_triplets_until_capture_end";
  contract.descriptor_reuse_policy = "reuse_after_shape_workspace_source_validation";
  contract.stride_policy = "matrix_ld_matches_logical_shape_host_output_ld_explicit";
  contract.output_currentness_policy = "device_residue_current_after_grouped_gemm_host_output_after_export";
  contract.lifetime_policy = "task_matrices_and_workspaces_destroyed_after_capture";
  contract.checksum_policy = "combined_per_task_checksum_u64";
  contract.status_policy = "fail_fast_per_task_operation_status";
  contract.device_descriptor_policy =
      grouped_device_descriptors ? "device_pointer_tables_and_compact_slabs" : "host_resident_task_loop";
  return contract;
}

void print_dispatch_and_graph_json(std::ostream& output, const Args& args, const BenchmarkResult& result) {
  const GroupedDispatchDescriptorContract grouped_contract =
      make_grouped_dispatch_descriptor_contract(args, result);
  const bool grouped = grouped_contract.requested;
  const bool adaptive_grouped = args.adaptive_grouped_scheduler;
  const bool adaptive_grouped_executed =
      adaptive_grouped && result.backend_info_available &&
      result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT && result.schedule_info_available &&
      result.schedule_info.tile_count != 0 &&
      std::string(result.backend_info.selected_kernel) ==
          "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3";
  output << "  \"grouped_dispatch\": {\n";
  output << "    \"requested\": " << (grouped ? "true" : "false") << ",\n";
  output << "    \"task_count\": " << grouped_contract.task_count << ",\n";
  output << "    \"descriptor_identity\": \"" << json_escape(grouped_contract.descriptor_identity) << "\",\n";
  output << "    \"source_hash\": \"" << json_escape(grouped_contract.source_hash) << "\",\n";
  output << "    \"output_hash\": \"" << json_escape(grouped_contract.output_hash) << "\",\n";
  output << "    \"setup_scope\": \"" << json_escape(grouped_contract.setup_scope) << "\",\n";
  output << "    \"execution_strategy\": \"" << json_escape(grouped_contract.execution_strategy) << "\",\n";
  output << "    \"batched_export_enabled\": "
            << (grouped_contract.batched_export_enabled ? "true" : "false") << ",\n";
  output << "    \"device_output_slab_bytes\": "
            << grouped_contract.device_output_slab_bytes << ",\n";
  output << "    \"capture_status\": \"" << json_escape(grouped_contract.capture_status) << "\",\n";
  output << "    \"unsupported_reason\": ";
  print_nullable_std_string(output, std::string());
  output << ",\n";
  output << "    \"promotion_eligible\": "
            << (grouped_contract.promotion_eligible ? "true" : "false") << ",\n";
  output << "    \"task_descriptor_contract\": {\n";
  output << "      \"schema_version\": " << grouped_contract.schema_version << ",\n";
  output << "      \"descriptor_layout\": \"" << json_escape(grouped_contract.descriptor_layout) << "\",\n";
  output << "      \"bucket_policy\": \"" << json_escape(grouped_contract.bucket_policy) << "\",\n";
  output << "      \"bucket_count\": " << grouped_contract.bucket_count << ",\n";
  output << "      \"task_count\": " << grouped_contract.task_count << ",\n";
  output << "      \"same_shape_required\": "
            << (grouped_contract.same_shape_required ? "true" : "false") << ",\n";
  output << "      \"shape_key\": \"" << json_escape(grouped_contract.shape_key) << "\",\n";
  output << "      \"semantics\": \"" << json_escape(grouped_contract.semantics) << "\",\n";
  output << "      \"output_domain\": \"" << json_escape(grouped_contract.output_domain) << "\",\n";
  output << "      \"source_version_policy\": \""
            << json_escape(grouped_contract.source_version_policy) << "\",\n";
  output << "      \"workspace_policy\": \"" << json_escape(grouped_contract.workspace_policy) << "\",\n";
  output << "      \"matrix_ownership_policy\": \""
            << json_escape(grouped_contract.matrix_ownership_policy) << "\",\n";
  output << "      \"descriptor_reuse_policy\": \""
            << json_escape(grouped_contract.descriptor_reuse_policy) << "\",\n";
  output << "      \"stride_policy\": \"" << json_escape(grouped_contract.stride_policy) << "\",\n";
  output << "      \"output_currentness_policy\": \""
            << json_escape(grouped_contract.output_currentness_policy) << "\",\n";
  output << "      \"lifetime_policy\": \"" << json_escape(grouped_contract.lifetime_policy) << "\",\n";
  output << "      \"checksum_policy\": \"" << json_escape(grouped_contract.checksum_policy) << "\",\n";
  output << "      \"status_policy\": \"" << json_escape(grouped_contract.status_policy) << "\",\n";
  output << "      \"device_descriptor_policy\": \""
            << json_escape(grouped_contract.device_descriptor_policy) << "\",\n";
  output << "      \"promotion_eligible\": "
            << (grouped_contract.promotion_eligible ? "true" : "false") << "\n";
  output << "    }\n";
  output << "  },\n";
  output << "  \"adaptive_grouped_scheduler\": {\n";
  output << "    \"requested\": " << (adaptive_grouped ? "true" : "false") << ",\n";
  output << "    \"strategy\": \""
            << (adaptive_grouped ? "prefix_tile_zero_mask_grouped_descriptors"
                                 : "current_compact_active_prefix_schedule")
            << "\",\n";
  output << "    \"descriptor_identity\": \"prefix=" << selected_execution_prefix(args, result)
            << ";tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
            << ";zero_tiles=" << result.zero_output_tile_count << "\",\n";
  output << "    \"group_count\": "
            << (adaptive_grouped ? std::max<uint64_t>(1, result.schedule_info.prefix_group_count) : 0) << ",\n";
  output << "    \"active_prefix_count\": "
            << (adaptive_grouped ? result.adaptive_grouped_active_prefix_count : 0) << ",\n";
  output << "    \"active_tile_count\": " << result.schedule_info.tile_count << ",\n";
  output << "    \"active_entry_count\": "
            << (adaptive_grouped ? result.adaptive_grouped_active_entry_count : 0) << ",\n";
  output << "    \"zero_tile_count\": " << result.zero_output_tile_count << ",\n";
  output << "    \"independent_launch_count_model\": "
            << (adaptive_grouped ? result.adaptive_grouped_independent_launch_count : 0) << ",\n";
  output << "    \"aggregate_launch_count_model\": "
            << (adaptive_grouped ? result.adaptive_grouped_aggregate_launch_count : 0) << ",\n";
  output << "    \"launch_reduction_ratio\": ";
  if (adaptive_grouped && result.adaptive_grouped_aggregate_launch_count != 0) {
    output << static_cast<double>(result.adaptive_grouped_independent_launch_count) /
                     static_cast<double>(result.adaptive_grouped_aggregate_launch_count);
  } else {
    output << 0.0;
  }
  output << ",\n";
  output << "    \"event_scope\": \""
            << (adaptive_grouped_executed ? "aggregate_rns_gemm_kernel_group_per_measured_repeat"
                                          : "not_available")
            << "\",\n";
  output << "    \"selected_prefix_histogram\": \"min=" << result.schedule_info.min_selected_prefix
            << ";max=" << result.schedule_info.max_selected_prefix << "\",\n";
  output << "    \"capture_status\": \""
            << (adaptive_grouped_executed ? "executed"
                                          : adaptive_grouped ? "metadata_only_unsupported_for_execution_path"
                                                             : "not_requested")
            << "\",\n";
  output << "    \"unsupported_reason\": ";
  print_nullable_std_string(output, adaptive_grouped && !adaptive_grouped_executed
          ? std::string("adaptive_grouped_scheduler_not_executed_by_current_path")
          : std::string());
  output << ",\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
}

void print_residency_arena_overlap_json(std::ostream& output, const Args& args, const BenchmarkResult& result) {
  const bool residency = args.resident_lifetime || args.reuse_packed_inputs || args.residue_chain_length > 1;
  const bool arena = args.workspace_arena;
  const bool streaming_executed = result.streaming_overlap_executed;
  const uint64_t arena_size =
      result.workspace_arena_info_available ? result.workspace_arena_size_bytes : result.backend_info.workspace_required_bytes;
  const uint64_t arena_high_water =
      result.workspace_arena_info_available ? result.workspace_arena_high_water_mark_bytes : result.backend_info.workspace_required_bytes;
  const uint32_t arena_suballocations =
      result.workspace_arena_info_available ? result.workspace_arena_suballocation_count : 5u;
  const std::string arena_policy =
      result.workspace_arena_info_available ? result.workspace_arena_policy : std::string("benchmark_declared_workspace_bytes");
  const std::string arena_target =
      result.workspace_arena_info_available ? result.workspace_arena_target_id : result.target_id;
  const bool repeat_alloc_free =
      result.allocation_tracking_available &&
      (result.allocation_after_repeats.allocate_calls == result.allocation_after_warmups.allocate_calls) &&
      (result.allocation_after_repeats.free_calls == result.allocation_after_warmups.free_calls);
  output << "  \"resident_lifetime\": {\n";
  output << "    \"enabled\": " << (residency ? "true" : "false") << ",\n";
  output << "    \"matrix_roles\": \"" << (residency ? "A/B/C explicit benchmark resident roles" : "transient") << "\",\n";
  output << "    \"source_version_policy\": \"matrix_instance_plus_monotonic_content_generation\",\n";
  output << "    \"current_storage_state\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  output << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  output << "    \"contract_api\": \"rns8_get_resident_lifetime_info\",\n";
  output << "    \"role_binding_policy\": \"A=m_by_k;B=k_by_n;C=m_by_n\",\n";
  output << "    \"workspace_identity\": \"" << result.backend_info.workspace_required_bytes
            << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  output << "    \"workspace_binding_policy\": \"plan_fingerprint_plus_workspace_identity_schedule_backend\",\n";
  output << "    \"output_currentness_policy\": \"explicit_host_or_device_current_domain_no_implicit_semantic_inference\",\n";
  output << "    \"stale_source_rejection\": \"source_version_descriptor_semantic_prefix_target_workspace_mismatch\",\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
  output << "  \"workspace_arena\": {\n";
  output << "    \"enabled\": " << (arena ? "true" : "false") << ",\n";
  output << "    \"arena_identity\": \"" << json_escape(result.backend_info.autotune_key) << "|"
            << json_escape(result.backend_info.workspace_mode) << "|target=" << json_escape(arena_target)
            << "|policy=" << json_escape(arena_policy) << "\",\n";
  output << "    \"size_bytes\": " << arena_size << ",\n";
  output << "    \"high_water_mark_bytes\": " << arena_high_water << ",\n";
  output << "    \"suballocation_count\": " << (arena ? arena_suballocations : 0) << ",\n";
  output << "    \"measured_repeat_allocation_free\": "
            << (repeat_alloc_free ? "true" : "false") << ",\n";
  output << "    \"setup_allocation_delta\": ";
  print_counter_object(output, allocation_delta(result.allocation_before, result.allocation_after_warmups));
  output << ",\n";
  output << "    \"measured_repeat_allocation_delta\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(output, allocation_delta(result.allocation_after_warmups, result.allocation_after_repeats));
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"source_version_policy\": \"plan_target_backend_semantic_shape_prefix_output_policy\",\n";
  output << "    \"stream_safety\": \"" << (args.streaming_overlap ? "event_guarded_pipeline_lanes" : "single_stream_owner")
            << "\",\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
  output << "  \"streaming_overlap\": {\n";
  output << "    \"requested\": " << (args.streaming_overlap ? "true" : "false") << ",\n";
  output << "    \"pipeline\": \""
            << (streaming_executed ? (args.streaming_serial_control ? "serialized_same_pipeline_stages" : "pack_next_gemm_current_export_previous") : args.streaming_overlap
                          ? "pack_next_gemm_current_export_previous"
                          : "serial_default_stream")
            << "\",\n";
  output << "    \"buffering\": \"" << (streaming_executed ? "double_buffered_benchmark_only"
                                              : args.streaming_overlap ? "requested_but_not_executed"
                                                                       : "none")
            << "\",\n";
  output << "    \"dependency_contract\": \"pack_before_gemm;gemm_before_export;status_before_host_read;final_sync_before_checksum\",\n";
  output << "    \"transfer_policy\": \"compact_or_padded_output_policy_declared_by_output_policy\",\n";
  output << "    \"stream_count\": " << result.streaming_overlap_stream_count << ",\n";
  output << "    \"buffer_count\": " << result.streaming_overlap_buffer_count << ",\n";
  output << "    \"measured_repeat_count\": " << result.streaming_overlap_measured_repeat_count << ",\n";
  output << "    \"final_exact_reference_passed\": " << (streaming_executed ? "true" : "false") << ",\n";
  output << "    \"observation_contract_version\": 2,\n";
  output << "    \"schedule\": \"" << (args.streaming_serial_control ? "serial_dependency_chain" : "overlapped_three_stream") << "\",\n";
  output << "    \"fill_drain_policy\": \"fresh_pinned_stream_event_setup_separate;full_fill_drain_status_and_final_host_copy_inside_each_batch\",\n";
  output << "    \"warmup_batch_count\": " << result.streaming_warmup_batch_count << ",\n";
  output << "    \"warmup_calls_per_batch\": " << (streaming_executed ? std::max(2u,args.repeats) : 0) << ",\n";
  output << "    \"first_use_batch_wall_us\": " << result.streaming_first_use_batch_us << ",\n";
  output << "    \"per_call_latency_available\": false,\n";
  output << "    \"batches\": [";
  for(std::size_t i=0;i<result.streaming_observations.size();++i) {
    const auto& batch=result.streaming_observations[i];if(i)output << ",";
    output << "{\"calls\":" << batch.calls << ",\"duration_us\":" << batch.duration_us
              << ",\"setup_us\":" << batch.setup_us << ",\"first_source_version\":" << batch.first_source_version
              << ",\"device_allocate_calls_inside_batch\":" << batch.device_allocate_calls
              << ",\"device_allocated_bytes_inside_batch\":" << batch.device_allocated_bytes << "}";
  }
  output << "],\n";
  output << "    \"batch_wall_us\": " << result.streaming_overlap_batch_wall_us << ",\n";
  output << "    \"per_repeat_pipeline_us\": " << result.streaming_overlap_per_repeat_pipeline_us << ",\n";
  output << "    \"explicit_dependency_events\": " << (streaming_executed ? "true" : "false") << ",\n";
  output << "    \"stage_event_scope\": \""
            << (streaming_executed ? "direct_hip_streaming_overlap_multistream_operation_groups"
                                   : "not_available")
            << "\",\n";
  output << "    \"capture_status\": \""
            << (streaming_executed ? "executed"
                                   : args.streaming_overlap ? "metadata_only_unsupported_for_execution_path"
                                                            : "not_requested")
            << "\",\n";
  output << "    \"unsupported_reason\": ";
  print_nullable_std_string(output, args.streaming_overlap && !streaming_executed
                                ? std::string("streaming_overlap_not_executed_by_current_path")
                                : std::string());
  output << ",\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
}

void print_workload_proxy_json(std::ostream& output, const Args& args) {
  const bool enabled = args.workload_proxy != "none";
  std::string family = "not_requested";
  if (enabled) {
    family = (args.workload_proxy.find("fhe") != std::string::npos ||
              args.workload_proxy.find("lattice") != std::string::npos ||
              args.workload_proxy.find("ckks") != std::string::npos)
                 ? "fhe_lattice_proxy"
                 : "dense_exact_arithmetic_proxy";
  }
  output << "  \"workload_proxy\": {\n";
  output << "    \"enabled\": " << (enabled ? "true" : "false") << ",\n";
  output << "    \"label\": \"" << json_escape(args.workload_proxy) << "\",\n";
  output << "    \"family\": \"" << json_escape(family) << "\",\n";
  output << "    \"tower_role\": \"" << (enabled ? "dense_gemm_adjacent_proxy" : "none") << "\",\n";
  output << "    \"reuse_profile\": \"" << (args.reuse_packed_inputs ? reuse_operand_role(args) : "none") << "\",\n";
  output << "    \"transform_role\": \"" << (enabled ? "not_a_public_fhe_backend" : "none") << "\",\n";
  output << "    \"output_domain_requirement\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  output << "    \"compatibility_claim\": false\n";
  output << "  },\n";
}

const char* benchmark_final_comparison_status(const Args& args,const BenchmarkResult& result) {
  const bool compared=rns8::bench::complete_integer_reference(result.native_reference,uint64_t(args.m),uint64_t(args.n),
      std::max(args.host_api_batch_size,args.grouped_dispatch_tasks));
  return compared?"exact_cpu_reference_compared":"checksum_recorded_reference_required";
}
void print_release_and_verification_json(std::ostream& output, const Args& args, const BenchmarkResult& result) {
  const char* final_comparison_status=benchmark_final_comparison_status(args,result);
  const bool gate_requested = args.release_gate != "none";
  const bool amortized = args.verification_amortization != "none";
  const bool error_detecting = args.error_detection_policy != "none";
  const bool probabilistic =
      error_detecting &&
      (args.error_detection_policy.find("freivalds") != std::string::npos ||
       args.error_detection_policy.find("probabilistic") != std::string::npos ||
       args.error_detection_policy.find("cra") != std::string::npos);
  output << "  \"release_gate\": {\n";
  output << "    \"name\": \"" << json_escape(args.release_gate) << "\",\n";
  output << "    \"requested\": " << (gate_requested ? "true" : "false") << ",\n";
  output << "    \"classification_tier\": \""
            << (gate_requested ? "cpu_backed_release_candidate_pending_review" : "not_requested") << "\",\n";
  output << "    \"cpu_reference_policy\": \"chunked_when_large_fixed_seed_checksum_recorded\",\n";
  output << "    \"memory_cap_policy\": \"declared_by_sweep_runner_or_not_applicable\",\n";
  output << "    \"resume_policy\": \"scenario_id_and_capture_path_stable_under_temp\",\n";
  output << "    \"review_status\": \""
            << (gate_requested ? "pending_reviewed_summary" : "not_requested") << "\",\n";
  output << "    \"cache_eligible\": false,\n";
  output << "    \"blockers\": [";
  if (gate_requested) {
    output << "\"reviewed_summary_missing\", \"release_A_B_margin_missing\"";
  }
  output << "]\n";
  output << "  },\n";
  output << "  \"verification_amortization\": {\n";
  output << "    \"enabled\": " << (amortized ? "true" : "false") << ",\n";
  output << "    \"policy\": \"" << json_escape(args.verification_amortization) << "\",\n";
  output << "    \"reused_reference_structure\": \""
            << (amortized ? "shape_seed_semantic_reference_inputs" : "none") << "\",\n";
  output << "    \"final_exact_comparison_required\": true,\n";
  output << "    \"final_exact_comparison_status\": \""
            << final_comparison_status << "\",\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
  output << "  \"error_detection_policy\": {\n";
  output << "    \"enabled\": " << (error_detecting ? "true" : "false") << ",\n";
  output << "    \"policy\": \"" << json_escape(args.error_detection_policy) << "\",\n";
  output << "    \"mode\": \""
            << (probabilistic ? "probabilistic_product_check"
                              : (error_detecting ? "deterministic_error_check" : "not_requested"))
            << "\",\n";
  output << "    \"verification_basis\": \""
            << (error_detecting ? "benchmark_metadata_only_final_exact_cpu_reference_required" : "none")
            << "\",\n";
  output << "    \"false_negative_policy\": \""
            << (probabilistic ? "bounded_by_recorded_rounds_seed_and_reference_field_not_default_exact_api"
                              : (error_detecting ? "deterministic_check_no_probabilistic_false_negative_claim"
                                                 : "none"))
            << "\",\n";
  output << "    \"verification_rounds\": " << (probabilistic ? 2 : 0) << ",\n";
  output << "    \"rng_seed_recorded\": " << (probabilistic ? "true" : "false") << ",\n";
  output << "    \"final_exact_comparison_required\": true,\n";
  output << "    \"final_exact_comparison_status\": \""
            << final_comparison_status
            << "\",\n";
  output << "    \"research_only\": " << (error_detecting ? "true" : "false") << ",\n";
  output << "    \"default_exact_api_unchanged\": true,\n";
  output << "    \"runtime_routing_allowed\": false,\n";
  output << "    \"cache_eligible\": false,\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
}

void print_cpu_selector_and_incremental_cache_json(std::ostream& output, const Args& args,
    const BenchmarkResult& result,
    const std::string& selected_backend) {
  const bool cpu_selector_enabled = args.cpu_small_shape_selector != "none";
  const bool incremental_enabled = args.incremental_result_cache != "none";
  const bool public_incremental_policy = public_incremental_result_cache_contract_requested(args);
  const bool public_incremental_available = result.incremental_result_cache_public_contract_available;
  const bool cpu_backend = selected_backend == "cpu-reference" || selected_backend == "cpu";
  const std::string incremental_candidate_role =
      incremental_result_cache_candidate_role(args, result, selected_backend);
  const std::string boundary_key =
      std::string("semantics=") + semantics_name(args.semantics) +
      ";m=" + std::to_string(args.m) +
      ";n=" + std::to_string(args.n) +
      ";k=" + std::to_string(args.k) +
      ";modulus=" + std::to_string(args.finite_modulus) +
      ";layout=row_major;output=" + output_domain_contract_name(args) +
      ";target=" + json_escape(result.target_id.empty() ? std::string("unknown") : result.target_id);
  const std::string checksum_status = benchmark_final_comparison_status(args,result);
  output << "  \"cpu_small_shape_selector\": {\n";
  output << "    \"enabled\": " << (cpu_selector_enabled ? "true" : "false") << ",\n";
  output << "    \"policy\": \"" << json_escape(args.cpu_small_shape_selector) << "\",\n";
  output << "    \"candidate_role\": \""
            << (cpu_selector_enabled ? (cpu_backend ? "cpu_baseline" : "comparison_candidate")
                                     : "not_requested")
            << "\",\n";
  output << "    \"boundary_key\": \"" << json_escape(boundary_key) << "\",\n";
  output << "    \"threshold_scope\": \""
            << (cpu_selector_enabled
                    ? "semantic_layout_output_target_family_explicit_review_only"
                    : "not_requested")
            << "\",\n";
  output << "    \"selector_explanation\": \""
            << (cpu_selector_enabled
                    ? "non_routing_auto_explanation_metadata_only"
                    : "not_requested")
            << "\",\n";
  output << "    \"cpu_reference_required\": true,\n";
  output << "    \"release_review_required\": " << (cpu_selector_enabled ? "true" : "false") << ",\n";
  output << "    \"runtime_routing_allowed\": false,\n";
  output << "    \"cache_eligible\": false,\n";
  output << "    \"promotion_eligible\": false\n";
  output << "  },\n";
  output << "  \"incremental_result_cache\": {\n";
  output << "    \"enabled\": " << (incremental_enabled ? "true" : "false") << ",\n";
  output << "    \"policy\": \"" << json_escape(args.incremental_result_cache) << "\",\n";
  output << "    \"candidate_role\": \"" << json_escape(incremental_candidate_role) << "\",\n";
  output << "    \"source_identity_policy\": \""
            << incremental_result_cache_source_identity_policy(args, public_incremental_policy) << "\",\n";
  output << "    \"source_version_policy\": \""
            << incremental_result_cache_source_version_policy(args, public_incremental_policy) << "\",\n";
  output << "    \"dirty_region_policy\": \""
            << incremental_result_cache_dirty_region_policy(args, public_incremental_policy) << "\",\n";
  output << "    \"result_lifetime_policy\": \""
            << incremental_result_cache_result_lifetime_policy(args, public_incremental_policy) << "\",\n";
  output << "    \"checksum_policy\": \""
            << (incremental_enabled ? "final_exact_cpu_comparison_required" : "none") << "\",\n";
  output << "    \"partial_recompute_policy\": \""
            << incremental_result_cache_partial_recompute_policy(args, public_incremental_policy) << "\",\n";
  output << "    \"final_exact_comparison_required\": true,\n";
  output << "    \"final_exact_comparison_status\": \"" << checksum_status << "\",\n";
  output << "    \"public_contract_available\": "
            << (public_incremental_available ? "true" : "false") << ",\n";
  output << "    \"default_gemm_unchanged\": true,\n";
  output << "    \"runtime_routing_allowed\": "
            << (public_incremental_available ? "true" : "false") << ",\n";
  output << "    \"cache_eligible\": " << (public_incremental_available ? "true" : "false") << ",\n";
  output << "    \"promotion_eligible\": " << (public_incremental_available ? "true" : "false");
  if (public_incremental_available && result.incremental_result_cache_info_available) {
    output << ",\n";
    output << "    \"status\": \"" << json_escape(result.incremental_result_cache_status) << "\",\n";
    output << "    \"a_matrix_instance_id\": "
              << result.incremental_result_cache_a_matrix_instance_id << ",\n";
    output << "    \"b_matrix_instance_id\": "
              << result.incremental_result_cache_b_matrix_instance_id << ",\n";
    output << "    \"c_matrix_instance_id\": "
              << result.incremental_result_cache_c_matrix_instance_id << ",\n";
    output << "    \"a_source_version\": " << result.incremental_result_cache_a_source_version << ",\n";
    output << "    \"b_source_version\": " << result.incremental_result_cache_b_source_version << ",\n";
    output << "    \"c_source_version\": " << result.incremental_result_cache_c_source_version << ",\n";
    output << "    \"result_cache_key_fingerprint\": "
              << result.incremental_result_cache_key_fingerprint << ",\n";
    output << "    \"dirty_region_count\": "
              << result.incremental_result_cache_dirty_region_count << ",\n";
    output << "    \"recomputed_region_count\": "
              << result.incremental_result_cache_recomputed_region_count << ",\n";
    output << "    \"copied_from_cache_bytes\": "
              << result.incremental_result_cache_copied_from_cache_bytes << ",\n";
    output << "    \"recomputed_cell_count\": "
              << result.incremental_result_cache_recomputed_cell_count << ",\n";
    output << "    \"cache_allocation_bytes\": "
              << result.incremental_result_cache_allocation_bytes << ",\n";
    output << "    \"snapshot_device_bytes\": "
              << result.incremental_result_cache_snapshot_device_bytes << ",\n";
    output << "    \"last_full_fallback\": "
              << (result.incremental_result_cache_full_fallback != 0 ? "true" : "false") << ",\n";
    output << "    \"last_cache_hit\": "
              << (result.incremental_result_cache_last_cache_hit != 0 ? "true" : "false") << ",\n";
    output << "    \"last_cache_miss\": "
              << (result.incremental_result_cache_last_cache_miss != 0 ? "true" : "false") << ",\n";
    output << "    \"stale_rejection_covered\": "
              << (result.incremental_result_cache_stale_rejection_covered ? "true" : "false") << ",\n";
    output << "    \"stale_reason\": \""
              << json_escape(result.incremental_result_cache_stale_reason) << "\",\n";
    output << "    \"fallback_reason\": \""
              << json_escape(result.incremental_result_cache_fallback_reason) << "\",\n";
    output << "    \"detail\": \""
              << json_escape(result.incremental_result_cache_detail) << "\",\n";
    output << "    \"dirty_regions\": [";
    for (std::size_t index = 0; index < result.incremental_result_cache_dirty_regions.size(); ++index) {
      const auto& region = result.incremental_result_cache_dirty_regions[index];
      if (index != 0) {
        output << ", ";
      }
      output << "{\"row_offset\": " << region.row_offset
                << ", \"col_offset\": " << region.col_offset
                << ", \"row_extent\": " << region.row_extent
                << ", \"col_extent\": " << region.col_extent << "}";
    }
    output << "]\n";
  } else {
    output << "\n";
  }
  output << "  },\n";
}

