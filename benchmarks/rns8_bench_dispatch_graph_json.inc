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

void print_dispatch_and_graph_json(const Args& args, const BenchmarkResult& result) {
  const GroupedDispatchDescriptorContract grouped_contract =
      make_grouped_dispatch_descriptor_contract(args, result);
  const bool grouped = grouped_contract.requested;
  const bool adaptive_grouped = args.adaptive_grouped_scheduler;
  std::cout << "  \"grouped_dispatch\": {\n";
  std::cout << "    \"requested\": " << (grouped ? "true" : "false") << ",\n";
  std::cout << "    \"task_count\": " << grouped_contract.task_count << ",\n";
  std::cout << "    \"descriptor_identity\": \"" << json_escape(grouped_contract.descriptor_identity) << "\",\n";
  std::cout << "    \"source_hash\": \"" << json_escape(grouped_contract.source_hash) << "\",\n";
  std::cout << "    \"output_hash\": \"" << json_escape(grouped_contract.output_hash) << "\",\n";
  std::cout << "    \"setup_scope\": \"" << json_escape(grouped_contract.setup_scope) << "\",\n";
  std::cout << "    \"execution_strategy\": \"" << json_escape(grouped_contract.execution_strategy) << "\",\n";
  std::cout << "    \"batched_export_enabled\": "
            << (grouped_contract.batched_export_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"device_output_slab_bytes\": "
            << grouped_contract.device_output_slab_bytes << ",\n";
  std::cout << "    \"capture_status\": \"" << json_escape(grouped_contract.capture_status) << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": "
            << (grouped_contract.promotion_eligible ? "true" : "false") << ",\n";
  std::cout << "    \"task_descriptor_contract\": {\n";
  std::cout << "      \"schema_version\": " << grouped_contract.schema_version << ",\n";
  std::cout << "      \"descriptor_layout\": \"" << json_escape(grouped_contract.descriptor_layout) << "\",\n";
  std::cout << "      \"bucket_policy\": \"" << json_escape(grouped_contract.bucket_policy) << "\",\n";
  std::cout << "      \"bucket_count\": " << grouped_contract.bucket_count << ",\n";
  std::cout << "      \"task_count\": " << grouped_contract.task_count << ",\n";
  std::cout << "      \"same_shape_required\": "
            << (grouped_contract.same_shape_required ? "true" : "false") << ",\n";
  std::cout << "      \"shape_key\": \"" << json_escape(grouped_contract.shape_key) << "\",\n";
  std::cout << "      \"semantics\": \"" << json_escape(grouped_contract.semantics) << "\",\n";
  std::cout << "      \"output_domain\": \"" << json_escape(grouped_contract.output_domain) << "\",\n";
  std::cout << "      \"source_version_policy\": \""
            << json_escape(grouped_contract.source_version_policy) << "\",\n";
  std::cout << "      \"workspace_policy\": \"" << json_escape(grouped_contract.workspace_policy) << "\",\n";
  std::cout << "      \"matrix_ownership_policy\": \""
            << json_escape(grouped_contract.matrix_ownership_policy) << "\",\n";
  std::cout << "      \"descriptor_reuse_policy\": \""
            << json_escape(grouped_contract.descriptor_reuse_policy) << "\",\n";
  std::cout << "      \"stride_policy\": \"" << json_escape(grouped_contract.stride_policy) << "\",\n";
  std::cout << "      \"output_currentness_policy\": \""
            << json_escape(grouped_contract.output_currentness_policy) << "\",\n";
  std::cout << "      \"lifetime_policy\": \"" << json_escape(grouped_contract.lifetime_policy) << "\",\n";
  std::cout << "      \"checksum_policy\": \"" << json_escape(grouped_contract.checksum_policy) << "\",\n";
  std::cout << "      \"status_policy\": \"" << json_escape(grouped_contract.status_policy) << "\",\n";
  std::cout << "      \"device_descriptor_policy\": \""
            << json_escape(grouped_contract.device_descriptor_policy) << "\",\n";
  std::cout << "      \"promotion_eligible\": "
            << (grouped_contract.promotion_eligible ? "true" : "false") << "\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  std::cout << "  \"adaptive_grouped_scheduler\": {\n";
  std::cout << "    \"requested\": " << (adaptive_grouped ? "true" : "false") << ",\n";
  std::cout << "    \"strategy\": \""
            << (adaptive_grouped ? "prefix_tile_zero_mask_grouped_descriptors"
                                 : "current_compact_active_prefix_schedule")
            << "\",\n";
  std::cout << "    \"descriptor_identity\": \"prefix=" << selected_execution_prefix(args, result)
            << ";tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
            << ";zero_tiles=" << result.zero_output_tile_count << "\",\n";
  std::cout << "    \"group_count\": "
            << (adaptive_grouped ? std::max<uint64_t>(1, result.schedule_info.prefix_group_count) : 0) << ",\n";
  std::cout << "    \"active_tile_count\": " << result.schedule_info.tile_count << ",\n";
  std::cout << "    \"zero_tile_count\": " << result.zero_output_tile_count << ",\n";
  std::cout << "    \"selected_prefix_histogram\": \"min=" << result.schedule_info.min_selected_prefix
            << ";max=" << result.schedule_info.max_selected_prefix << "\",\n";
  std::cout << "    \"capture_status\": \""
            << (adaptive_grouped ? "metadata_only_unsupported_for_execution_path" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(adaptive_grouped ? std::string("adaptive_grouped_scheduler_not_executed_by_current_path")
                                             : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
}

void print_residency_arena_overlap_json(const Args& args, const BenchmarkResult& result) {
  const bool residency = args.resident_lifetime || args.reuse_packed_inputs || args.residue_chain_length > 1;
  const bool arena = args.workspace_arena;
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
  std::cout << "  \"resident_lifetime\": {\n";
  std::cout << "    \"enabled\": " << (residency ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_roles\": \"" << (residency ? "A/B/C explicit benchmark resident roles" : "transient") << "\",\n";
  std::cout << "    \"source_version_policy\": \"monotonic_per_import_or_pack\",\n";
  std::cout << "    \"current_storage_state\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"workspace_identity\": \"" << result.backend_info.workspace_required_bytes
            << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  std::cout << "    \"stale_source_rejection\": \"source_version_descriptor_semantic_prefix_target_workspace_mismatch\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
  std::cout << "  \"workspace_arena\": {\n";
  std::cout << "    \"enabled\": " << (arena ? "true" : "false") << ",\n";
  std::cout << "    \"arena_identity\": \"" << json_escape(result.backend_info.autotune_key) << "|"
            << json_escape(result.backend_info.workspace_mode) << "|target=" << json_escape(arena_target)
            << "|policy=" << json_escape(arena_policy) << "\",\n";
  std::cout << "    \"size_bytes\": " << arena_size << ",\n";
  std::cout << "    \"high_water_mark_bytes\": " << arena_high_water << ",\n";
  std::cout << "    \"suballocation_count\": " << (arena ? arena_suballocations : 0) << ",\n";
  std::cout << "    \"measured_repeat_allocation_free\": "
            << (repeat_alloc_free ? "true" : "false") << ",\n";
  std::cout << "    \"setup_allocation_delta\": ";
  print_counter_object(allocation_delta(result.allocation_before, result.allocation_after_warmups));
  std::cout << ",\n";
  std::cout << "    \"measured_repeat_allocation_delta\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(allocation_delta(result.allocation_after_warmups, result.allocation_after_repeats));
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"source_version_policy\": \"plan_target_backend_semantic_shape_prefix_output_policy\",\n";
  std::cout << "    \"stream_safety\": \"" << (args.streaming_overlap ? "event_guarded_pipeline_lanes" : "single_stream_owner")
            << "\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
  std::cout << "  \"streaming_overlap\": {\n";
  std::cout << "    \"requested\": " << (args.streaming_overlap ? "true" : "false") << ",\n";
  std::cout << "    \"pipeline\": \""
            << (args.streaming_overlap ? "pack_next_gemm_current_export_previous" : "serial_default_stream")
            << "\",\n";
  std::cout << "    \"buffering\": \"" << (args.streaming_overlap ? "double_buffered_benchmark_only" : "none") << "\",\n";
  std::cout << "    \"dependency_contract\": \"pack_before_gemm;gemm_before_export;status_before_host_read;final_sync_before_checksum\",\n";
  std::cout << "    \"transfer_policy\": \"compact_or_padded_output_policy_declared_by_output_policy\",\n";
  std::cout << "    \"capture_status\": \""
            << (args.streaming_overlap ? "metadata_only_unsupported_for_execution_path" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(args.streaming_overlap ? std::string("streaming_overlap_not_executed_by_current_path")
                                                   : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
}

void print_workload_proxy_json(const Args& args) {
  const bool enabled = args.workload_proxy != "none";
  std::string family = "not_requested";
  if (enabled) {
    family = (args.workload_proxy.find("fhe") != std::string::npos ||
              args.workload_proxy.find("lattice") != std::string::npos ||
              args.workload_proxy.find("ckks") != std::string::npos)
                 ? "fhe_lattice_proxy"
                 : "dense_exact_arithmetic_proxy";
  }
  std::cout << "  \"workload_proxy\": {\n";
  std::cout << "    \"enabled\": " << (enabled ? "true" : "false") << ",\n";
  std::cout << "    \"label\": \"" << json_escape(args.workload_proxy) << "\",\n";
  std::cout << "    \"family\": \"" << json_escape(family) << "\",\n";
  std::cout << "    \"tower_role\": \"" << (enabled ? "dense_gemm_adjacent_proxy" : "none") << "\",\n";
  std::cout << "    \"reuse_profile\": \"" << (args.reuse_packed_inputs ? reuse_operand_role(args) : "none") << "\",\n";
  std::cout << "    \"transform_role\": \"" << (enabled ? "not_a_public_fhe_backend" : "none") << "\",\n";
  std::cout << "    \"output_domain_requirement\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"compatibility_claim\": false\n";
  std::cout << "  },\n";
}

void print_release_and_verification_json(const Args& args, const BenchmarkResult& result) {
  const bool gate_requested = args.release_gate != "none";
  const bool amortized = args.verification_amortization != "none";
  std::cout << "  \"release_gate\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.release_gate) << "\",\n";
  std::cout << "    \"requested\": " << (gate_requested ? "true" : "false") << ",\n";
  std::cout << "    \"classification_tier\": \""
            << (gate_requested ? "cpu_backed_release_candidate_pending_review" : "not_requested") << "\",\n";
  std::cout << "    \"cpu_reference_policy\": \"chunked_when_large_fixed_seed_checksum_recorded\",\n";
  std::cout << "    \"memory_cap_policy\": \"declared_by_sweep_runner_or_not_applicable\",\n";
  std::cout << "    \"resume_policy\": \"scenario_id_and_capture_path_stable_under_temp\",\n";
  std::cout << "    \"review_status\": \""
            << (gate_requested ? "pending_reviewed_summary" : "not_requested") << "\",\n";
  std::cout << "    \"cache_eligible\": false,\n";
  std::cout << "    \"blockers\": [";
  if (gate_requested) {
    std::cout << "\"reviewed_summary_missing\", \"release_A_B_margin_missing\"";
  }
  std::cout << "]\n";
  std::cout << "  },\n";
  std::cout << "  \"verification_amortization\": {\n";
  std::cout << "    \"enabled\": " << (amortized ? "true" : "false") << ",\n";
  std::cout << "    \"policy\": \"" << json_escape(args.verification_amortization) << "\",\n";
  std::cout << "    \"reused_reference_structure\": \""
            << (amortized ? "shape_seed_semantic_reference_inputs" : "none") << "\",\n";
  std::cout << "    \"final_exact_comparison_required\": true,\n";
  std::cout << "    \"final_exact_comparison_status\": \""
            << (result.checksum != 0 ? "checksum_recorded_reference_required" : "reference_required") << "\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
}

