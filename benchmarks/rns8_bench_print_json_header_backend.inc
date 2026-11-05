  output << "  \"schema_version\": " << kBenchmarkSchemaVersion << ",\n";
  output << "  \"process_id\": " << rns8::tools::process_id() << ",\n";
  output << "  \"execution_environment_witness\": ";
  execution_environment.print(output);
  output << ",\n";
  output << "  \"profiler_ranges\": ";
  rns8::detail::trace::write_snapshot(output);
  output << ",\n";
  output << "  \"device_identity_witness\": ";
  device_identity.print(output);
  output << ",\n";
  output << "  \"benchmark\": \"" << benchmark_name(args) << "\",\n";
  output << "  \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args, result) << "\",\n";
  output << "  \"backend_requested\": \"" << requested_backend_name(args) << "\",\n";
  output << "  \"backend_selected\": \"" << selected_backend << "\",\n";
  const char* selected_kernel = selected_kernel_name(args, info, result);
  output << "  \"selected_kernel\": ";
  if (selected_kernel) {
    output << "\"" << selected_kernel << "\"";
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"backend_metadata\": {\n";
  output << "    \"source\": \"" << backend_metadata_source(args, result) << "\",\n";
  output << "    \"selected_kernel\": ";
  print_json_string_or_null(output, result.backend_info.selected_kernel);
  output << ",\n";
  output << "    \"accelerator_backend\": "
            << (result.backend_info.is_accelerator ? "true" : "false") << ",\n";
  output << "    \"correctness_backend\": "
            << (result.backend_info.is_correctness_backend ? "true" : "false") << ",\n";
  output << "    \"matrix_engine_backend\": "
            << (result.backend_info.is_matrix_engine_backend ? "true" : "false") << ",\n";
  output << "    \"compiled_kernel_available\": "
            << (result.backend_info.compiled_kernel_available ? "true" : "false") << ",\n";
  output << "    \"exact_differential_validated\": "
            << (result.backend_info.exact_differential_validated ? "true" : "false") << ",\n";
  output << "    \"performance_validated\": "
            << (result.backend_info.performance_validated ? "true" : "false") << ",\n";
  output << "    \"accelerator_library\": ";
  print_json_string_or_null(output, result.backend_info.accelerator_library);
  output << ",\n";
  output << "    \"accelerator_version\": ";
  print_json_string_or_null(output, result.backend_info.accelerator_version);
  output << ",\n";
  output << "    \"capability_status\": ";
  print_json_string_or_null(output, result.backend_info.capability_status);
  output << ",\n";
  output << "    \"epilogue_mode\": ";
  print_json_string_or_null(output, result.backend_info.epilogue_mode);
  output << ",\n";
  output << "    \"workspace_mode\": ";
  print_json_string_or_null(output, result.backend_info.workspace_mode);
  output << ",\n";
  output << "    \"workspace_required_bytes\": " << result.backend_info.workspace_required_bytes << ",\n";
  output << "    \"isa_evidence\": ";
  print_json_string_or_null(output, result.backend_info.isa_evidence);
  output << ",\n";
  output << "    \"autotune_key\": ";
  print_json_string_or_null(output, result.backend_info.autotune_key);
  output << ",\n";
  if (result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    const char* matrix_family = amdgpu_builtin_matrix_instruction_family(selected_kernel);
    const char* matrix_shape = amdgpu_builtin_matrix_instruction_shape(selected_kernel);
    const char* matrix_dtype = amdgpu_builtin_matrix_instruction_dtype(selected_kernel);
    const char* matrix_sparsity = amdgpu_builtin_matrix_sparsity(selected_kernel);
    const char* matrix_operand_signedness = amdgpu_builtin_matrix_operand_signedness(selected_kernel);
    const char* matrix_a_value_contract = amdgpu_builtin_matrix_a_value_contract(args, selected_kernel);
    const char* matrix_b_value_contract = amdgpu_builtin_matrix_b_value_contract(args, selected_kernel);
    const char* matrix_sparse_contract = amdgpu_builtin_matrix_sparse_contract(selected_kernel);
    const char* matrix_sparse_dense_operand = amdgpu_builtin_matrix_sparse_dense_operand(selected_kernel);
    const char* matrix_sparse_index_layout = amdgpu_builtin_matrix_sparse_index_layout(selected_kernel);
    output << "    \"matrix_instruction_family\": ";
    print_json_string_or_null(output, matrix_family);
    output << ",\n";
    output << "    \"matrix_instruction_shape\": ";
    print_json_string_or_null(output, matrix_shape);
    output << ",\n";
    output << "    \"matrix_instruction_dtype\": ";
    print_json_string_or_null(output, matrix_dtype);
    output << ",\n";
    output << "    \"matrix_instruction_sparsity\": ";
    print_json_string_or_null(output, matrix_sparsity);
    output << ",\n";
    output << "    \"matrix_operand_signedness\": ";
    print_json_string_or_null(output, matrix_operand_signedness);
    output << ",\n";
    output << "    \"matrix_a_value_contract\": ";
    print_json_string_or_null(output, matrix_a_value_contract);
    output << ",\n";
    output << "    \"matrix_b_value_contract\": ";
    print_json_string_or_null(output, matrix_b_value_contract);
    output << ",\n";
    output << "    \"matrix_sparse_contract\": ";
    print_json_string_or_null(output, matrix_sparse_contract);
    output << ",\n";
    output << "    \"matrix_sparse_dense_operand\": ";
    print_json_string_or_null(output, matrix_sparse_dense_operand);
    output << ",\n";
    output << "    \"matrix_sparse_a_compression_index_layout\": ";
    print_json_string_or_null(output, matrix_sparse_index_layout);
    output << ",\n";
    output << "    \"matrix_rdna_integer_modifier_policy\": ";
    if (amdgpu_builtin_matrix_uses_rdna_integer_modifiers(selected_kernel)) {
      output << "{\n";
      output << "      \"NEG[0]\": \"A operand signedness: 0 unsigned, 1 signed\",\n";
      output << "      \"NEG[1]\": \"B operand signedness: 0 unsigned, 1 signed\",\n";
      output << "      \"NEG[2]\": \"must be zero for integer WMMA/SWMMAC\",\n";
      output << "      \"NEG_HI\": \"must be zero for integer WMMA/SWMMAC\"\n";
      output << "    }";
    } else {
      output << "null";
    }
    output << ",\n";
  }
  output << "    \"accumulator_safety\": {\n";
  output << "      \"input_domain\": ";
  print_json_string_or_null(output, result.backend_info.accumulator_input_domain);
  output << ",\n";
  output << "      \"signedness\": ";
  print_json_string_or_null(output, result.backend_info.accumulator_signedness);
  output << ",\n";
  output << "      \"accumulator_type\": ";
  print_json_string_or_null(output, result.backend_info.accumulator_type);
  output << ",\n";
  output << "      \"modulus_policy\": ";
  print_json_string_or_null(output, result.backend_info.accumulator_modulus_policy);
  output << ",\n";
  output << "      \"modulus\": " << result.backend_info.accumulator_modulus << ",\n";
  output << "      \"uses_int32_inner_product\": "
            << (result.backend_info.accumulator_uses_int32_inner_product ? "true" : "false") << ",\n";
  output << "      \"k_block_size\": " << result.backend_info.accumulator_k_block_size << ",\n";
  output << "      \"k_block_cap\": " << result.backend_info.accumulator_k_block_cap << ",\n";
  output << "      \"max_lhs_abs\": " << result.backend_info.accumulator_max_lhs_abs << ",\n";
  output << "      \"max_rhs_abs\": " << result.backend_info.accumulator_max_rhs_abs << ",\n";
  output << "      \"max_product\": " << result.backend_info.accumulator_max_product << ",\n";
  output << "      \"safe_for_k_block\": "
            << (result.backend_info.accumulator_safe_for_k_block ? "true" : "false") << ",\n";
  output << "      \"status\": ";
  print_json_string_or_null(output, result.backend_info.accumulator_safety_status);
  output << "\n";
  output << "    }\n";
  output << "  },\n";
  output << "  \"cpu_parallel\": {\n";
  output << "    \"available\": " << (rns8::detail::cpu_parallel_runtime_available() ? "true" : "false") << ",\n";
  output << "    \"enabled\": " << (rns8::detail::cpu_parallel_enabled() ? "true" : "false") << ",\n";
  output << "    \"runtime\": \"" << rns8::detail::cpu_parallel_runtime_name() << "\",\n";
  output << "    \"requested_threads\": " << rns8::detail::cpu_parallel_requested_threads() << ",\n";
  output << "    \"max_threads\": " << rns8::detail::cpu_parallel_max_threads() << ",\n";
  output << "    \"effective_threads\": " << rns8::detail::cpu_parallel_effective_threads() << ",\n";
  output << "    \"threshold_ops\": " << rns8::detail::cpu_parallel_threshold_ops() << ",\n";
  output << "    \"progress\": " << (args.progress ? "true" : "false") << ",\n";
  output << "    \"reference_mode\": \"" << json_escape(args.cpu_reference_mode) << "\",\n";
  output << "    \"correctness_anchor\": " << (args.cpu_reference_mode == "correctness-anchor" ? "true" : "false") << ",\n";
  output << "    \"timed_cpu_baseline\": " << (args.cpu_reference_mode == "timed-baseline" ? "true" : "false") << "\n";
  output << "  },\n";
  print_plan_packing_json(output, result);
  print_plan_lowering_json(output, result);
  print_requested_next_op_json(output, args, result);
  print_output_policy_json(output, args, output_ld);
  print_target_variant_json(output, args, info, result, selected_backend);
  print_auto_selector_json(output, args, info, result, selected_backend);
  print_device_allocation_json(output, args, result);
  output << "  \"hipblaslt_metadata\": {\n";
  output << "    \"abi_version\": 2, \"policy\": \"" << (args.hipblaslt_metadata_per_block_control ? "per_block_control" : "per_call_distinct_geometry") << "\",\n";
  output << "    \"scope\": \"calling_thread_library_API_attempt_counts_not_allocator_bytes_or_device_work\",\n";
  output << "    \"cross_call_raw_descriptor_retention\": false,\n";
  auto print_metadata_counters=[&output](const rns8::detail::HipblasLtMetadataCounters& v) {
    output << "{";
    output << "\"descriptor_create_calls\":" << v.descriptor_create_calls << ",";
    output << "\"descriptor_destroy_calls\":" << v.descriptor_destroy_calls << ",";
    output << "\"attribute_set_calls\":" << v.attribute_set_calls << ",";
    output << "\"preference_create_calls\":" << v.preference_create_calls << ",";
    output << "\"preference_destroy_calls\":" << v.preference_destroy_calls << ",";
    output << "\"preference_attribute_calls\":" << v.preference_attribute_calls << ",";
    output << "\"heuristic_calls\":" << v.heuristic_calls << ",";
    output << "\"algorithm_cache_hits\":" << v.algorithm_cache_hits << ",";
    output << "\"descriptor_set_hits\":" << v.descriptor_set_hits << ",";
    output << "\"descriptor_sets_created\":" << v.descriptor_sets_created << ",";
    output << "\"matmul_block_calls\":" << v.matmul_block_calls << ",";
    output << "\"algorithm_directory_lookups\":" << v.algorithm_directory_lookups << ",";
    output << "\"call_local_algorithm_hits\":" << v.call_local_algorithm_hits << ",";
    output << "\"maximum_live_sets\":" << v.maximum_live_sets << "}";
  };
  output << "    \"after_warmups\": "; print_metadata_counters(result.metadata_after_warmups); output << ",\n";
  output << "    \"after_measured_repeats\": "; print_metadata_counters(result.metadata_after_repeats); output << "\n  },\n";
  const auto print_algorithms=[&output](const rns8::detail::HipblasLtAlgorithmInventory& inventory) {
    output << "{\"available\":" << (inventory.available?"true":"false")
              << ",\"complete\":" << (inventory.complete?"true":"false")
              << ",\"index_queries\":" << inventory.index_queries << ",\"entries\":[";
    for(uint32_t i=0;i<inventory.size;++i) {
      const auto& entry=inventory.entries[i];const auto& key=entry.key;
      if(i)output << ',';
      output << "{\"device_id\":" << key.device_id << ",\"m\":" << key.m << ",\"n\":" << key.n
                << ",\"k\":" << key.k << ",\"scratch_ld\":" << key.scratch_ld
                << ",\"workspace_bytes\":" << key.workspace_bytes << ",\"attribute_abi\":" << key.attribute_abi
                << ",\"library_version\":" << key.library_version << ",\"runtime_version\":" << key.runtime_version
                << ",\"driver_version\":" << key.driver_version << ",\"context_configuration\":" << key.context_configuration
                << ",\"target_identity\":" << key.target_identity << ",\"algorithm_index\":";
      if(entry.algorithm_index<0)output << "null";else output << entry.algorithm_index;
      output << ",\"matmul_calls\":" << entry.calls << '}';
    }
    output << "]}";
  };
  output << "  \"hipblaslt_algorithms\": {\"abi_version\":2,"
            << "\"scope\":\"library_context_device_padded_geometry_algorithm_index\","
            << "\"query_policy\":\"opt_in_once_per_cached_algorithm\",\"after_warmups\":";
  print_algorithms(result.algorithms_after_warmups);
  output << ",\"after_measured_repeats\":";print_algorithms(result.algorithms_after_repeats);
  output << "},\n";
  output << "  \"rocwmma_packing\": {\"abi_version\":1,\"policy\":\"" << args.rocwmma_packing_policy
      << "\",\"scope\":\"calling_thread_successful_submissions_not_bytes_or_time\",\n";
  const auto print_rocwmma=[&output](const rns8::detail::RocwmmaPackingStats& v) {
    output << "{\"a_pack_launches\":" << v.a_pack_launches << ",\"b_pack_launches\":" << v.b_pack_launches
        << ",\"direct_a_blocks\":" << v.direct_a_blocks << ",\"a_panel_hits\":" << v.a_panel_hits
        << ",\"b_panel_hits\":" << v.b_panel_hits << "}";
  };
  output << "    \"after_warmups\":";print_rocwmma(result.rocwmma_after_warmups);
  output << ",\n    \"after_measured_repeats\":";print_rocwmma(result.rocwmma_after_repeats);output << "\n  },\n";
  output << "  \"hipblaslt_operand_prepack\": {\n    \"abi_version\":1,\"policy\":\"" << args.hipblaslt_prepack_policy
      << "\",\"scope\":\"calling_thread_successful_pack_submissions_and_cache_matches_not_bytes_or_time\",\n";
  auto print_prepack_counters=[&output](const rns8::detail::HipblasLtPrepackCounters& v) {
    output << "{\"a_retained_builds\":" << v.a_retained_builds << ",\"b_retained_builds\":" << v.b_retained_builds
      << ",\"a_retained_hits\":" << v.a_retained_hits << ",\"b_retained_hits\":" << v.b_retained_hits
      << ",\"a_retained_planes_packed\":" << v.a_retained_planes_packed << ",\"b_retained_planes_packed\":" << v.b_retained_planes_packed
      << ",\"a_transient_planes_packed\":" << v.a_transient_planes_packed << ",\"b_transient_planes_packed\":" << v.b_transient_planes_packed << "}";
  };
  output << "    \"after_warmups\":";print_prepack_counters(result.prepack_after_warmups);
  output << ",\n    \"after_measured_repeats\":";print_prepack_counters(result.prepack_after_repeats);
  output << "\n  },\n";

  output << "  \"immutable_schedule_work\": {\"abi_version\":1,\"scope\":\"calling_thread_source_work_counters_not_latency\",\n";
  auto print_schedule_work=[&output](const rns8::detail::api::plan_validation_counters& v) {
    output << "{\"core_exhaustive_validations\":" << v.exhaustive_validations
      << ",\"fingerprint_traversals\":" << v.fingerprint_traversals
      << ",\"canonical_equivalent_comparisons\":" << v.equivalent_comparisons
      << ",\"backend_schedule_seals\":" << v.backend_schedule_seals
      << ",\"backend_sealed_tile_visits\":" << v.backend_schedule_sealed_tile_visits
      << ",\"backend_fast_submissions\":" << v.backend_schedule_fast_submissions
      << ",\"backend_raw_submissions\":" << v.backend_schedule_raw_submissions << "}";
  };
  output << "    \"after_warmups\":";print_schedule_work(result.schedule_after_warmups);
  output << ",\n    \"after_measured_repeats\":";print_schedule_work(result.schedule_after_repeats);
  output << "\n  },\n";

  print_reuse_contract_json(output, args, result, selected_backend, selected_kernel);
  print_exact_output_contract_json(output, args, result, output_ld, selected_kernel);
  print_export_variant_json(output, args, result, selected_kernel);
  print_reconstruction_variant_json(output, args, result, selected_kernel);
  print_modulus_and_residue_policy_json(output, args, result);
  print_tile_shape_variant_json(output, args, result, selected_kernel);
  print_resident_redesign_json(output, args);
  print_dispatch_and_graph_json(output, args, result);
  print_residency_arena_overlap_json(output, args, result);
  print_workload_proxy_json(output, args);
  print_release_and_verification_json(output, args, result);
  print_cpu_selector_and_incremental_cache_json(output, args, result, selected_backend);
  output << "  \"semantics\": \"" << semantics_name(args.semantics) << "\",\n";
  output << "  \"bound_kind\": \"" << bound_kind_name(args) << "\",\n";
  output << "  \"bound_mode\": \"" << bound_mode_name(args.bound_mode) << "\",\n";
  output << "  \"bound\": " << bound << ",\n";
  output << "  \"bound_source\": \"" << bound_source_name(args) << "\",\n";
  output << "  \"bound_discovery\": ";
  if (bounded_benchmark_semantics(args.semantics)) {
    output << "{\n";
    output << "    \"source\": \"" << bound_discovery_source_name(args, global_bound_scan_available) << "\",\n";
    output << "    \"static_bound\": "
              << (result.effective_bound_available ? result.static_bound : bound) << ",\n";
    output << "    \"selected_bound\": " << bound << ",\n";
    output << "    \"discovered_global_bound\": ";
    if (global_bound_scan_available) {
      output << result.discovered_global_bound;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"candidate_row_sum_col_max\": ";
    if (global_bound_scan_available) {
      output << result.bound_candidate_row_sum_col_max;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"candidate_row_max_col_sum\": ";
    if (global_bound_scan_available) {
      output << result.bound_candidate_row_max_col_sum;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"row_abs_sum_max\": ";
    if (global_bound_scan_available) {
      output << result.row_abs_sum_max;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"row_abs_max\": ";
    if (global_bound_scan_available) {
      output << result.row_abs_max;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"col_abs_sum_max\": ";
    if (global_bound_scan_available) {
      output << result.col_abs_sum_max;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"col_abs_max\": ";
    if (global_bound_scan_available) {
      output << result.col_abs_max;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"zero_row_count\": ";
    if (global_bound_scan_available) {
      output << result.zero_row_count;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "    \"zero_col_count\": ";
    if (global_bound_scan_available) {
      output << result.zero_col_count;
    } else {
      output << "null";
    }
    output << "\n";
    output << "  },\n";
  } else {
    output << "null,\n";
  }
