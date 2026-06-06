  return "not_applicable";
}

void print_plan_packing_json(const BenchmarkResult& result) {
  if (!result.packing_info_available) {
    std::cout << "  \"plan_packing\": null,\n";
    return;
  }
  const auto& packing = result.packing_info;
  std::cout << "  \"plan_packing\": {\n";
  std::cout << "    \"source\": \"rns8_get_plan_packing_info\",\n";
  std::cout << "    \"backend\": \"" << backend_name(packing.backend) << "\",\n";
  std::cout << "    \"semantics\": \"" << c_semantics_name(packing.semantics) << "\",\n";
  std::cout << "    \"uses_resident_matrix_inputs\": "
            << (packing.uses_resident_matrix_inputs ? "true" : "false") << ",\n";
  std::cout << "    \"uses_transient_pack_workspace\": "
            << (packing.uses_transient_pack_workspace ? "true" : "false") << ",\n";
  std::cout << "    \"uses_matrix_engine_pack_layout\": "
            << (packing.uses_matrix_engine_pack_layout ? "true" : "false") << ",\n";
  std::cout << "    \"reusable_prepack_cache_available\": "
            << (packing.reusable_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "    \"production_prepack_cache_available\": "
            << (packing.production_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "    \"input_domain_name\": \"" << json_escape(packing.input_domain_name) << "\",\n";
  std::cout << "    \"output_domain_name\": \"" << json_escape(packing.output_domain_name) << "\",\n";
  std::cout << "    \"output_host_current\": " << (packing.output_host_current ? "true" : "false") << ",\n";
  std::cout << "    \"output_device_current\": " << (packing.output_device_current ? "true" : "false") << ",\n";
  std::cout << "    \"next_op_flags\": " << packing.next_op_flags << ",\n";
  std::cout << "    \"next_op_hint\": \"" << json_escape(packing.next_op_hint) << "\",\n";
  std::cout << "    \"a_pack_workspace_bytes\": " << packing.a_pack_workspace_bytes << ",\n";
  std::cout << "    \"b_pack_workspace_bytes\": " << packing.b_pack_workspace_bytes << ",\n";
  std::cout << "    \"accumulator_workspace_bytes\": " << packing.accumulator_workspace_bytes << ",\n";
  std::cout << "    \"library_workspace_bytes\": " << packing.library_workspace_bytes << ",\n";
  std::cout << "    \"total_transient_workspace_bytes\": " << packing.total_transient_workspace_bytes << ",\n";
  std::cout << "    \"a_layout_version\": \"" << json_escape(packing.a_layout_version) << "\",\n";
  std::cout << "    \"b_layout_version\": \"" << json_escape(packing.b_layout_version) << "\",\n";
  std::cout << "    \"output_layout_version\": \"" << json_escape(packing.output_layout_version) << "\",\n";
  std::cout << "    \"prepack_cache_scope\": \"" << json_escape(packing.prepack_cache_scope) << "\",\n";
  std::cout << "    \"detail\": \"" << json_escape(packing.detail) << "\"\n";
  std::cout << "  },\n";
}

void print_plan_lowering_json(const BenchmarkResult& result) {
  if (!result.lowering_info_available) {
    std::cout << "  \"plan_lowering\": null,\n";
    return;
  }
  const auto& lowering = result.lowering_info;
  std::cout << "  \"plan_lowering\": {\n";
  std::cout << "    \"source\": \"rns8_get_plan_backend_info+rns8_get_plan_packing_info+rns8_get_plan_schedule_info\",\n";
  std::cout << "    \"operation\": \"" << json_escape(lowering.operation) << "\",\n";
  std::cout << "    \"semantic_contract\": \"" << json_escape(lowering.semantic_contract) << "\",\n";
  std::cout << "    \"backend_family\": \"" << json_escape(lowering.backend_family) << "\",\n";
  std::cout << "    \"input_domain\": \"" << json_escape(lowering.input_domain) << "\",\n";
  std::cout << "    \"output_domain\": \"" << json_escape(lowering.output_domain) << "\",\n";
  std::cout << "    \"desired_output\": \"" << json_escape(lowering.desired_output) << "\",\n";
  std::cout << "    \"schedule_strategy\": \"" << json_escape(lowering.schedule_strategy) << "\",\n";
  std::cout << "    \"packing_strategy\": \"" << json_escape(lowering.packing_strategy) << "\",\n";
  std::cout << "    \"reuse_strategy\": \"" << json_escape(lowering.reuse_strategy) << "\",\n";
  std::cout << "    \"conversion_strategy\": \"" << json_escape(lowering.conversion_strategy) << "\",\n";
  std::cout << "    \"lowering_path\": \"" << json_escape(lowering.lowering_path) << "\",\n";
  std::cout << "    \"final_export_available\": " << (lowering.final_export_available ? "true" : "false") << ",\n";
  std::cout << "    \"rns_continuation_available\": "
            << (lowering.rns_continuation_available ? "true" : "false") << ",\n";
  std::cout << "    \"native_continuation_available\": "
            << (lowering.native_continuation_available ? "true" : "false") << ",\n";
  std::cout << "    \"native_to_rns_available\": "
            << (lowering.native_to_rns_available ? "true" : "false") << ",\n";
  std::cout << "    \"reusable_b_prepack_available\": "
            << (lowering.reusable_b_prepack_available ? "true" : "false") << "\n";
  std::cout << "  },\n";
}

void print_requested_next_op_json(const Args& args, const BenchmarkResult& result) {
  const std::string resolved = resolved_next_op_hint(args, result);
  std::cout << "  \"requested_next_op\": {\n";
  std::cout << "    \"requested\": \"" << next_op_hint_name(args.next_op_hint) << "\",\n";
  std::cout << "    \"resolved\": \"" << json_escape(resolved) << "\",\n";
  std::cout << "    \"source\": \"" << next_op_hint_source(args) << "\",\n";
  std::cout << "    \"final_export_available\": "
            << (result.lowering_info_available && result.lowering_info.final_export_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"rns_continuation_available\": "
            << (result.lowering_info_available && result.lowering_info.rns_continuation_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"native_continuation_available\": "
            << (result.lowering_info_available && result.lowering_info.native_continuation_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"native_to_rns_available\": "
            << (result.lowering_info_available && result.lowering_info.native_to_rns_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"reusable_b_prepack_available\": "
            << (result.lowering_info_available && result.lowering_info.reusable_b_prepack_available ? "true" : "false")
            << "\n";
  std::cout << "  },\n";
}

void print_output_policy_json(const Args& args, int64_t output_ld) {
  std::cout << "  \"output_policy\": {\n";
  std::cout << "    \"destination_layout\": \"" << output_destination_layout(args) << "\",\n";
  std::cout << "    \"logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"ld_padding\": " << args.output_ld_padding << ",\n";
  std::cout << "    \"per_repeat_logical_export\": " << (residue_current_output_mode(args) ? "false" : "true") << ",\n";
  std::cout << "    \"final_checksum_export_after_repeats\": "
            << (residue_current_output_mode(args) ? "true" : "false") << ",\n";
  std::cout << "    \"status_handling\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"status_event_policy\": \"" << output_status_event_policy(args) << "\"\n";
  std::cout << "  },\n";
}

void print_target_variant_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const std::string target_id = benchmark_key_target_id(result);
  const std::string target_namespace = target_namespace_for_id(target_id);
  const uint32_t visible_device_count = visible_device_count_from_environment();
  const uint32_t node_gpu_count = benchmark_node_gpu_count();
  std::cout << "  \"target_variant\": {\n";
  std::cout << "    \"target_id\": \"" << json_escape(target_id) << "\",\n";
  std::cout << "    \"target_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "    \"target_cache_key\": \"" << json_escape(target_cache_key(info, result)) << "\",\n";
  std::cout << "    \"target_instance_id\": \"" << json_escape(target_instance_id(info, result)) << "\",\n";
  std::cout << "    \"target_namespace\": \"" << json_escape(target_namespace) << "\",\n";
  std::cout << "    \"review_group_key\": \""
            << json_escape(target_review_group_key(args, info, result, selected_backend)) << "\",\n";
  std::cout << "    \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  std::cout << "    \"device_index\": " << info.device_id << ",\n";
  std::cout << "    \"device_name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "    \"visible_device_count\": " << visible_device_count << ",\n";
  std::cout << "    \"node_gpu_count\": " << node_gpu_count << ",\n";
  std::cout << "    \"hip_enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  std::cout << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "    \"hip_driver_version\": " << info.hip_driver_version << "\n";
  std::cout << "  },\n";
}

void print_counter_object(const rns8::detail::hip_direct_allocation_counters& counters) {
  std::cout << "{"
            << "\"allocate_calls\": " << counters.allocate_calls << ", "
            << "\"free_calls\": " << counters.free_calls << ", "
            << "\"allocated_bytes\": " << counters.allocated_bytes
            << "}";
}

rns8::detail::hip_direct_allocation_counters allocation_delta(
    const rns8::detail::hip_direct_allocation_counters& before,
    const rns8::detail::hip_direct_allocation_counters& after) {
  rns8::detail::hip_direct_allocation_counters delta{};
  delta.allocate_calls = after.allocate_calls >= before.allocate_calls ? after.allocate_calls - before.allocate_calls : 0;
  delta.free_calls = after.free_calls >= before.free_calls ? after.free_calls - before.free_calls : 0;
  delta.allocated_bytes = after.allocated_bytes >= before.allocated_bytes ? after.allocated_bytes - before.allocated_bytes : 0;
  return delta;
}

const char* benchmark_setup_scope(const Args& args) {
  if (args.oneshot) {
    return "transient_api_setup_inside_measured_repeat";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "benchmark_experimental_residue_channel_fusion_persistent_output";
  }
  if (hip_graph_replay_requested(args)) {
    return "benchmark_hip_graph_replay_resident_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "benchmark_grouped_dispatch_one_shared_plan_persistent_resident_tasks";
  }
  if (args.vector_alu_baseline || runtime_vector_alu_backend(args) || bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "benchmark_owned_transient_native_input_buffers";
  }
  if (args.reuse_packed_inputs) {
    return "persistent_plan_workspace_prepacked_reuse";
  }
  return "persistent_plan_workspace_resident_matrices";
}

void print_device_allocation_json(const Args& args, const BenchmarkResult& result) {
  std::cout << "  \"device_allocation\": {\n";
  std::cout << "    \"tracking_available\": " << (result.allocation_tracking_available ? "true" : "false") << ",\n";
  std::cout << "    \"source\": \"hip_direct_allocation_counters_snapshot\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  std::cout << "    \"before\": ";
  print_counter_object(result.allocation_before);
  std::cout << ",\n";
  std::cout << "    \"after_warmups\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(result.allocation_after_warmups);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"after_repeats\": ";
  print_counter_object(result.allocation_after_repeats);
  std::cout << ",\n";
  std::cout << "    \"setup_delta\": ";
  print_counter_object(allocation_delta(result.allocation_before, result.allocation_after_warmups));
  std::cout << ",\n";
  std::cout << "    \"measured_repeat_delta\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(allocation_delta(result.allocation_after_warmups, result.allocation_after_repeats));
  } else {
    std::cout << "null";
  }
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_auto_selector_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const auto snapshot = rns8::detail::read_autotune_cache();
  const std::string selected_key =
      result.backend_info.autotune_key[0] != '\0' ? std::string(result.backend_info.autotune_key) : std::string();
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  runtime.target_id = benchmark_key_target_id(result);
  runtime.hip_sdk_or_library_version =
      result.backend_info.accelerator_version[0] != '\0'
          ? std::string(result.backend_info.accelerator_version)
          : std::string(RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  const auto* exact_hit = selected_key.empty() ? nullptr : rns8::detail::find_exact_autotune_entry(snapshot, selected_key);
  const auto* validated_hit =
      selected_key.empty() ? nullptr : rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, selected_key, runtime);
  std::cout << "  \"auto_selector\": {\n";
  std::cout << "    \"source\": \"private_benchmark_selector_report\",\n";
  std::cout << "    \"requested_backend\": \"" << requested_backend_name(args) << "\",\n";
  std::cout << "    \"selected_backend\": \"" << selected_backend << "\",\n";
  std::cout << "    \"selected_key\": ";
  print_nullable_std_string(selected_key);
  std::cout << ",\n";
  std::cout << "    \"cache_path\": \"" << json_escape(snapshot.path.string()) << "\",\n";
  std::cout << "    \"cache_exists\": " << (snapshot.exists ? "true" : "false") << ",\n";
  std::cout << "    \"cache_loaded\": " << (snapshot.loaded ? "true" : "false") << ",\n";
  std::cout << "    \"cache_entry_count\": " << snapshot.entries.size() << ",\n";
  std::cout << "    \"runtime_target_id\": \"" << json_escape(runtime.target_id) << "\",\n";
  std::cout << "    \"runtime_version\": \"" << json_escape(runtime.hip_sdk_or_library_version) << "\",\n";
  std::cout << "    \"exact_hit\": " << (exact_hit ? "true" : "false") << ",\n";
  std::cout << "    \"validated_hit\": " << (validated_hit ? "true" : "false") << ",\n";
  std::cout << "    \"fallback_reason\": \""
            << json_escape(
                   selected_key.empty()
                       ? "no exact entry"
                       : rns8::detail::autotune_selection_rationale(snapshot, selected_key, selected_backend, runtime))
            << "\",\n";
  std::cout << "    \"rejection_reason_vocabulary\": ["
            << "\"unsupported semantics\", "
            << "\"per-tile unsupported\", "
            << "\"backend not compiled\", "
            << "\"probe failed\", "
            << "\"no exact entry\", "
            << "\"unvalidated entry\", "
            << "\"identity/runtime mismatch\", "
            << "\"workspace mismatch\", "
            << "\"slower than selected\"],\n";
  std::cout << "    \"rejected_candidates\": [";
  if (args.backend == RNS8_BACKEND_AUTO) {
    const std::array<const char*, 6> candidates = {
        "cpu-reference", "hip-direct", "hipblaslt", "ck", "rocwmma", "hip-vector-alu-int64"};
    bool first = true;
    for (const char* candidate : candidates) {
      if (std::string(candidate) == selected_backend) {
        continue;
      }
      std::cout << (first ? "\n" : ",\n");
      std::cout << "      {\"backend\": \"" << candidate << "\", \"reason\": \"no exact entry\"}";
      first = false;
    }
    if (!first) {
      std::cout << "\n    ";
    }
  }
  std::cout << "]\n";
  std::cout << "  },\n";
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

void print_reuse_contract_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_backend,
    const char* selected_kernel) {
  const bool reuse_enabled =
      args.reuse_packed_inputs || args.residue_chain_length > 1 || host_api_batch_requested(args) ||
      grouped_dispatch_requested(args);
  std::cout << "  \"reuse_contract\": {\n";
  std::cout << "    \"enabled\": " << (reuse_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"operand_role\": \"" << reuse_operand_role(args) << "\",\n";
  std::cout << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"setup_cost_us\": ";
  if (result.prepack_setup_available) {
    std::cout << result.prepack_setup_us;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"measured_repeat_count\": " << args.repeats << ",\n";
  std::cout << "    \"break_even_repeat_count\": null,\n";
  std::cout << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"next_op\": \"" << json_escape(resolved_next_op_hint(args, result)) << "\",\n";
  std::cout << "    \"target_fingerprint\": \"" << json_escape(benchmark_key_target_id(result)) << "\",\n";
  std::cout << "    \"backend_fingerprint\": \"" << json_escape(selected_backend) << "\",\n";
  std::cout << "    \"kernel_fingerprint\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"workspace_fingerprint\": \"" << result.backend_info.workspace_required_bytes
            << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  std::cout << "    \"promotion_eligible\": " << (!reuse_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"invalidation_reasons\": [";
  bool first = true;
  if (args.reuse_packed_inputs) {
    std::cout << "\"source_version_changed\"";
    first = false;
  }
  if (grouped_dispatch_requested(args)) {
    std::cout << (first ? "" : ", ") << "\"descriptor_identity_changed\"";
    first = false;
  }
  if (args.hip_graph_replay) {
    std::cout << (first ? "" : ", ") << "\"graph_capture_descriptor_changed\"";
    first = false;
  }
  (void)first;
  std::cout << "]\n";
  std::cout << "  },\n";
}

void print_exact_output_contract_json(
    const Args& args,
    const BenchmarkResult& result,
    int64_t output_ld,
    const char* selected_kernel) {
  const std::string export_kernel =
      result.export_plan_info_available ? result.export_selected_kernel
                                        : (selected_kernel ? std::string(selected_kernel) : std::string());
  std::cout << "  \"exact_output_contract\": {\n";
  std::cout << "    \"requested_final_output\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"output_logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"kernel_identity\": ";
  print_nullable_std_string(export_kernel);
  std::cout << ",\n";
  std::cout << "    \"output_domain_after_measured_repeats\": \""
            << (residue_current_output_mode(args) ? "rns_residue_current" : json_escape(output_domain_contract_name(args)))
            << "\",\n";
  std::cout << "    \"final_checksum_export_after_repeats\": "
            << (residue_current_output_mode(args) ? "true" : "false") << "\n";
  std::cout << "  },\n";
  (void)result;
}

void print_export_variant_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_kernel) {
  const bool default_variant = args.export_variant == "default";
  const bool reviewable_fixed_limb_variant =
      args.export_variant == "exact-wide-fixed-limb-export" &&
      exact_wide_benchmark_semantics(args.semantics) &&
      args.exact_wide_limb_count > 0 &&
      args.reconstruction_variant == "default_garner";
  const bool promotion_eligible = default_variant || reviewable_fixed_limb_variant;
  const std::string export_kernel =
      result.export_plan_info_available ? result.export_selected_kernel
                                        : (selected_kernel ? std::string(selected_kernel) : std::string());
  std::cout << "  \"export_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.export_variant) << "\",\n";
  std::cout << "    \"source\": \""
            << (default_variant ? "current_backend_export_path"
                                : (reviewable_fixed_limb_variant ? "reviewable_exact_wide_fixed_limb_selector"
                                                                 : "benchmark_cli_evidence_mode"))
            << "\",\n";
  std::cout << "    \"selector_source\": \""
            << json_escape(result.export_plan_info_available ? result.export_plan_source : "legacy_selected_kernel")
            << "\",\n";
  std::cout << "    \"selector_key\": ";
  print_nullable_std_string(result.export_plan_info_available ? result.export_selector_key : std::string());
  std::cout << ",\n";
  std::cout << "    \"selector_policy\": \""
            << json_escape(result.export_plan_info_available ? result.export_selector_policy : "unknown")
            << "\",\n";
  std::cout << "    \"semantic_contract\": \""
            << json_escape(result.export_plan_info_available ? result.export_semantic_contract : semantics_name(args.semantics))
            << "\",\n";
  std::cout << "    \"backend\": \""
            << json_escape(result.export_plan_info_available ? result.export_backend : backend_name(args.backend))
            << "\",\n";
  std::cout << "    \"target_id\": \""
            << json_escape(result.export_plan_info_available ? result.export_target_id : benchmark_key_target_id(result))
            << "\",\n";
  std::cout << "    \"prefix_contract\": \""
            << json_escape(result.export_plan_info_available ? result.export_prefix_contract : schedule_hash_string(args, result))
            << "\",\n";
  std::cout << "    \"signedness\": \""
            << json_escape(result.export_plan_info_available ? result.export_signedness : "unknown") << "\",\n";
  std::cout << "    \"output_layout\": \""
            << json_escape(result.export_plan_info_available ? result.export_output_layout : "unknown") << "\",\n";
  std::cout << "    \"limb_count\": ";
  if (result.export_plan_info_available && result.export_limb_count != 0) {
    std::cout << result.export_limb_count;
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"selector_status_policy\": \""
            << json_escape(result.export_plan_info_available ? result.export_selector_status_policy : "unknown")
            << "\",\n";
  std::cout << "    \"d2h_policy\": \""
            << json_escape(result.export_plan_info_available ? result.export_d2h_policy : "unknown") << "\",\n";
  std::cout << "    \"final_output_mode\": \""
            << (residue_current_output_mode(args) ? "residue_chain_no_final_export"
                                                  : json_escape(result.export_plan_info_available
                                                                    ? result.export_final_output_mode
                                                                    : "final_host_output"))
            << "\",\n";
  std::cout << "    \"cache_visibility\": \""
            << json_escape(result.export_plan_info_available ? result.export_cache_visibility : "legacy_capture")
            << "\",\n";
  std::cout << "    \"stale_entry_reason\": ";
  print_nullable_std_string(result.export_plan_info_available ? result.export_stale_entry_reason : std::string());
  std::cout << ",\n";
  std::cout << "    \"status_elision_reason\": ";
  print_nullable_std_string(result.export_plan_info_available ? result.export_status_elision_reason : std::string());
  std::cout << ",\n";
  std::cout << "    \"requires_tile_metadata\": "
            << (result.export_plan_info_available && result.export_requires_tile_metadata ? "true" : "false")
            << ",\n";
  std::cout << "    \"all_zero_tiled_output\": "
            << (result.export_plan_info_available && result.export_all_zero_tiled_output ? "true" : "false")
            << ",\n";
  std::cout << "    \"selected_kernel\": ";
  print_nullable_std_string(export_kernel);
  std::cout << ",\n";
  std::cout << "    \"constants_placement\": \"backend_default\",\n";
  std::cout << "    \"promotion_eligible\": " << (promotion_eligible ? "true" : "false") << ",\n";
  std::cout << "    \"promotion_blocker\": ";
  print_nullable_std_string(promotion_eligible ? std::string() : std::string("experimental_export_variant"));
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_reconstruction_variant_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_kernel) {
  const bool default_variant = args.reconstruction_variant == "default_garner";
  const std::string export_kernel =
      result.export_plan_info_available ? result.export_selected_kernel
                                        : (selected_kernel ? std::string(selected_kernel) : std::string());
  std::cout << "  \"reconstruction_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.reconstruction_variant) << "\",\n";
  std::cout << "    \"family\": \"" << (default_variant ? "garner_fixed_prefix" : "benchmark_reconstruction_zoo") << "\",\n";
  std::cout << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"kernel_identity\": ";
  print_nullable_std_string(export_kernel);
  std::cout << ",\n";
  std::cout << "    \"controller\": \""
            << (default_variant
                    ? "public_export_api"
                    : (args.reconstruction_variant == "tree_crt_candidate"
                           ? "benchmark_internal_tree_crt_export_kernel"
                           : "benchmark_metadata_only"))
            << "\",\n";
  std::cout << "    \"promotion_eligible\": " << (default_variant ? "true" : "false") << ",\n";
  std::cout << "    \"promotion_blocker\": ";
  print_nullable_std_string(default_variant ? std::string() : std::string("experimental_reconstruction_variant"));
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_modulus_and_residue_policy_json(const Args& args, const BenchmarkResult& result) {
  const bool experimental = args.modulus_set != "default";
  std::cout << "  \"modulus_set\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.modulus_set) << "\",\n";
  std::cout << "    \"source\": \"" << (experimental ? "benchmark_experimental_ladder" : "rns8_default_modulus_ladder")
            << "\",\n";
  std::cout << "    \"execution_ladder\": \"" << (finite_benchmark_semantics(args.semantics) ? "finite_single_modulus"
                                                                               : "rns8_default_8bit_coprime_ladder")
            << "\",\n";
  std::cout << "    \"experimental\": " << (experimental ? "true" : "false") << ",\n";
  std::cout << "    \"product_bits\": " << estimated_modulus_product_bits(args, result) << ",\n";
  std::cout << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"pairwise_coprime_proof\": \"schema_declared_current_ladder_or_offline_search_report\",\n";
  std::cout << "    \"reducer_cost_hint\": \"" << (experimental ? "offline_search_required" : "backend_default")
            << "\",\n";
  std::cout << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(experimental ? std::string("experimental_modulus_set") : std::string());
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"residue_count_policy\": {\n";
  std::cout << "    \"policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  std::cout << "    \"requested_prefix\": " << benchmark_prefix(args) << ",\n";
  std::cout << "    \"selected_prefix\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"minimum_range_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  std::cout << "    \"redundant_residue_count\": "
            << (selected_execution_prefix(args, result) > result.schedule_info.min_required_prefix
                    ? selected_execution_prefix(args, result) - result.schedule_info.min_required_prefix
                    : 0)
            << ",\n";
  std::cout << "    \"autotune_scope\": \"" << (experimental ? "evidence_only_non_promoting" : "current_exact_cache")
            << "\",\n";
  std::cout << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(experimental ? std::string("experimental_residue_count_policy") : std::string());
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_tile_shape_variant_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_kernel) {
  const bool default_variant = args.tile_shape_variant == "default";
  const int64_t k_block = benchmark_k_block_size(args, result);
  std::cout << "  \"tile_shape_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.tile_shape_variant) << "\",\n";
  std::cout << "    \"tile_m\": " << args.tile_m << ",\n";
  std::cout << "    \"tile_n\": " << args.tile_n << ",\n";
  std::cout << "    \"tile_k\": " << k_block << ",\n";
  std::cout << "    \"k_block_policy\": \"" << json_escape(args.k_block_policy) << "\",\n";
  std::cout << "    \"split_k_mode\": \"single_gpu_no_split_k\",\n";
  std::cout << "    \"accumulator_safety_key\": \"k_block_size=" << k_block
            << ";k_block_cap="
            << (result.backend_info_available ? result.backend_info.accumulator_k_block_cap : 0)
            << ";safe_for_k_block="
            << (result.backend_info_available && result.backend_info.accumulator_safe_for_k_block ? "true" : "false")
            << "\",\n";
  std::cout << "    \"selected_kernel_identity\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"resource_report_key\": \"tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
            << ";tile_k=" << k_block << ";kernel=" << (selected_kernel ? json_escape(selected_kernel) : "unknown")
            << "\",\n";
  std::cout << "    \"shape_family_bucket\": \"" << shape_family_bucket(args) << "\",\n";
  std::cout << "    \"resource_report_required\": \"isa_or_counter_for_non_default_k_block_policy\",\n";
  std::cout << "    \"stale_kernel_rejection\": \"selected_kernel_identity_must_match_capture\"\n";
  std::cout << "  },\n";
  (void)default_variant;
}

