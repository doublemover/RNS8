  std::cout << "  \"schema_version\": " << kBenchmarkSchemaVersion << ",\n";
  std::cout << "  \"benchmark\": \"" << benchmark_name(args) << "\",\n";
  std::cout << "  \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args, result) << "\",\n";
  std::cout << "  \"backend_requested\": \"" << requested_backend_name(args) << "\",\n";
  std::cout << "  \"backend_selected\": \"" << selected_backend << "\",\n";
  const char* selected_kernel = selected_kernel_name(args, info, result);
  std::cout << "  \"selected_kernel\": ";
  if (selected_kernel) {
    std::cout << "\"" << selected_kernel << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"backend_metadata\": {\n";
  std::cout << "    \"source\": \"" << backend_metadata_source(args, result) << "\",\n";
  std::cout << "    \"selected_kernel\": ";
  print_json_string_or_null(result.backend_info.selected_kernel);
  std::cout << ",\n";
  std::cout << "    \"accelerator_backend\": "
            << (result.backend_info.is_accelerator ? "true" : "false") << ",\n";
  std::cout << "    \"correctness_backend\": "
            << (result.backend_info.is_correctness_backend ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_engine_backend\": "
            << (result.backend_info.is_matrix_engine_backend ? "true" : "false") << ",\n";
  std::cout << "    \"compiled_kernel_available\": "
            << (result.backend_info.compiled_kernel_available ? "true" : "false") << ",\n";
  std::cout << "    \"exact_differential_validated\": "
            << (result.backend_info.exact_differential_validated ? "true" : "false") << ",\n";
  std::cout << "    \"performance_validated\": "
            << (result.backend_info.performance_validated ? "true" : "false") << ",\n";
  std::cout << "    \"accelerator_library\": ";
  print_json_string_or_null(result.backend_info.accelerator_library);
  std::cout << ",\n";
  std::cout << "    \"accelerator_version\": ";
  print_json_string_or_null(result.backend_info.accelerator_version);
  std::cout << ",\n";
  std::cout << "    \"capability_status\": ";
  print_json_string_or_null(result.backend_info.capability_status);
  std::cout << ",\n";
  std::cout << "    \"epilogue_mode\": ";
  print_json_string_or_null(result.backend_info.epilogue_mode);
  std::cout << ",\n";
  std::cout << "    \"workspace_mode\": ";
  print_json_string_or_null(result.backend_info.workspace_mode);
  std::cout << ",\n";
  std::cout << "    \"workspace_required_bytes\": " << result.backend_info.workspace_required_bytes << ",\n";
  std::cout << "    \"isa_evidence\": ";
  print_json_string_or_null(result.backend_info.isa_evidence);
  std::cout << ",\n";
  std::cout << "    \"autotune_key\": ";
  print_json_string_or_null(result.backend_info.autotune_key);
  std::cout << ",\n";
  std::cout << "    \"accumulator_safety\": {\n";
  std::cout << "      \"input_domain\": ";
  print_json_string_or_null(result.backend_info.accumulator_input_domain);
  std::cout << ",\n";
  std::cout << "      \"signedness\": ";
  print_json_string_or_null(result.backend_info.accumulator_signedness);
  std::cout << ",\n";
  std::cout << "      \"accumulator_type\": ";
  print_json_string_or_null(result.backend_info.accumulator_type);
  std::cout << ",\n";
  std::cout << "      \"modulus_policy\": ";
  print_json_string_or_null(result.backend_info.accumulator_modulus_policy);
  std::cout << ",\n";
  std::cout << "      \"modulus\": " << result.backend_info.accumulator_modulus << ",\n";
  std::cout << "      \"uses_int32_inner_product\": "
            << (result.backend_info.accumulator_uses_int32_inner_product ? "true" : "false") << ",\n";
  std::cout << "      \"k_block_size\": " << result.backend_info.accumulator_k_block_size << ",\n";
  std::cout << "      \"k_block_cap\": " << result.backend_info.accumulator_k_block_cap << ",\n";
  std::cout << "      \"max_lhs_abs\": " << result.backend_info.accumulator_max_lhs_abs << ",\n";
  std::cout << "      \"max_rhs_abs\": " << result.backend_info.accumulator_max_rhs_abs << ",\n";
  std::cout << "      \"max_product\": " << result.backend_info.accumulator_max_product << ",\n";
  std::cout << "      \"safe_for_k_block\": "
            << (result.backend_info.accumulator_safe_for_k_block ? "true" : "false") << ",\n";
  std::cout << "      \"status\": ";
  print_json_string_or_null(result.backend_info.accumulator_safety_status);
  std::cout << "\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  std::cout << "  \"cpu_parallel\": {\n";
  std::cout << "    \"available\": " << (rns8::detail::cpu_parallel_runtime_available() ? "true" : "false") << ",\n";
  std::cout << "    \"enabled\": " << (rns8::detail::cpu_parallel_enabled() ? "true" : "false") << ",\n";
  std::cout << "    \"runtime\": \"" << rns8::detail::cpu_parallel_runtime_name() << "\",\n";
  std::cout << "    \"requested_threads\": " << rns8::detail::cpu_parallel_requested_threads() << ",\n";
  std::cout << "    \"max_threads\": " << rns8::detail::cpu_parallel_max_threads() << ",\n";
  std::cout << "    \"effective_threads\": " << rns8::detail::cpu_parallel_effective_threads() << ",\n";
  std::cout << "    \"threshold_ops\": " << rns8::detail::cpu_parallel_threshold_ops() << ",\n";
  std::cout << "    \"progress\": " << (args.progress ? "true" : "false") << ",\n";
  std::cout << "    \"reference_mode\": \"" << json_escape(args.cpu_reference_mode) << "\",\n";
  std::cout << "    \"correctness_anchor\": " << (args.cpu_reference_mode == "correctness-anchor" ? "true" : "false") << ",\n";
  std::cout << "    \"timed_cpu_baseline\": " << (args.cpu_reference_mode == "timed-baseline" ? "true" : "false") << "\n";
  std::cout << "  },\n";
  print_plan_packing_json(result);
  print_plan_lowering_json(result);
  print_requested_next_op_json(args, result);
  print_output_policy_json(args, output_ld);
  print_target_variant_json(args, info, result, selected_backend);
  print_auto_selector_json(args, info, result, selected_backend);
  print_device_allocation_json(args, result);
  print_reuse_contract_json(args, result, selected_backend, selected_kernel);
  print_exact_output_contract_json(args, result, output_ld, selected_kernel);
  print_export_variant_json(args, result, selected_kernel);
  print_reconstruction_variant_json(args, result, selected_kernel);
  print_modulus_and_residue_policy_json(args, result);
  print_tile_shape_variant_json(args, result, selected_kernel);
  print_resident_redesign_json(args);
  print_dispatch_and_graph_json(args, result);
  print_residency_arena_overlap_json(args, result);
  print_workload_proxy_json(args);
  print_release_and_verification_json(args, result);
  print_cpu_selector_and_incremental_cache_json(args, result, selected_backend);
  std::cout << "  \"semantics\": \"" << semantics_name(args.semantics) << "\",\n";
  std::cout << "  \"bound_kind\": \"" << bound_kind_name(args) << "\",\n";
  std::cout << "  \"bound_mode\": \"" << bound_mode_name(args.bound_mode) << "\",\n";
  std::cout << "  \"bound\": " << bound << ",\n";
  std::cout << "  \"bound_source\": \"" << bound_source_name(args) << "\",\n";
  std::cout << "  \"bound_discovery\": ";
  if (bounded_benchmark_semantics(args.semantics)) {
    std::cout << "{\n";
    std::cout << "    \"source\": \"" << bound_discovery_source_name(args, global_bound_scan_available) << "\",\n";
    std::cout << "    \"static_bound\": "
              << (result.effective_bound_available ? result.static_bound : bound) << ",\n";
    std::cout << "    \"selected_bound\": " << bound << ",\n";
    std::cout << "    \"discovered_global_bound\": ";
    if (global_bound_scan_available) {
      std::cout << result.discovered_global_bound;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"candidate_row_sum_col_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.bound_candidate_row_sum_col_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"candidate_row_max_col_sum\": ";
    if (global_bound_scan_available) {
      std::cout << result.bound_candidate_row_max_col_sum;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"row_abs_sum_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.row_abs_sum_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"row_abs_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.row_abs_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"col_abs_sum_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.col_abs_sum_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"col_abs_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.col_abs_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"zero_row_count\": ";
    if (global_bound_scan_available) {
      std::cout << result.zero_row_count;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"zero_col_count\": ";
    if (global_bound_scan_available) {
      std::cout << result.zero_col_count;
    } else {
      std::cout << "null";
    }
    std::cout << "\n";
    std::cout << "  },\n";
  } else {
    std::cout << "null,\n";
  }
