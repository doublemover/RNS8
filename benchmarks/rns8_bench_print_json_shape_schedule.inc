  output << "  \"exact_proof_policy\": \"" << (args.exact_scalar_first ? "scalar_first_v1" : "eager_moments_v1") << "\",\n";
  output << "  \"m\": " << args.m << ",\n";
  output << "  \"n\": " << args.n << ",\n";
  output << "  \"k\": " << args.k << ",\n";
  output << "  \"output_logical_ld\": " << output_ld << ",\n";
  output << "  \"output_ld_padding\": " << args.output_ld_padding << ",\n";
  output << "  \"prefix\": " << prefix << ",\n";
  output << "  \"selected_prefix\": " << selected_prefix << ",\n";
  output << "  \"requested_max_prefix\": " << prefix << ",\n";
  output << "  \"contract_prefix_policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  output << "  \"residue_planes_requested\": " << prefix << ",\n";
  output << "  \"residue_planes_selected\": " << selected_prefix << ",\n";
  output << "  \"residue_planes_skipped\": " << residue_planes_skipped << ",\n";
  output << "  \"residue_plane_skip_fraction\": " << residue_plane_skip_fraction << ",\n";
  output << "  \"finite_modulus\": ";
  if (finite_benchmark_semantics(args.semantics)) {
    output << args.finite_modulus;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"tile_m\": " << args.tile_m << ",\n";
  output << "  \"tile_n\": " << args.tile_n << ",\n";
  output << "  \"layout\": \"row_major\",\n";
  output << "  \"k_block_size\": "
            << benchmark_k_block_size(args, result)
            << ",\n";
  output << "  \"adaptive_tile_size\": null,\n";
  output << "  \"tile_bounds_u64\": ";
  if (args.bound_mode == BoundMode::PerTile) {
    output << "{\n";
    output << "    \"source\": \"exact_seeded_input_prepass\",\n";
    output << "    \"pattern\": \"" << tile_bound_pattern(args) << "\",\n";
    output << "    \"order\": \"row_major_output_tiles\",\n";
    output << "    \"count\": " << result.tile_bounds.size() << ",\n";
    output << "    \"min\": " << result.tile_bound_min << ",\n";
    output << "    \"max\": " << result.tile_bound_max << ",\n";
    output << "    \"hash_u64\": " << result.tile_bound_hash << "\n";
    output << "  },\n";
  } else {
    output << "null,\n";
  }
  output << "  \"schedule_metadata\": {\n";
  output << "    \"source\": \"" << json_escape(result.schedule_source) << "\",\n";
  output << "    \"bound_kind\": \"" << bound_kind_name(result.schedule_info.bound_kind) << "\",\n";
  output << "    \"effective_bound\": " << result.schedule_info.effective_bound << ",\n";
  output << "    \"lhs_bound\": " << result.schedule_info.lhs_bound << ",\n";
  output << "    \"rhs_bound\": " << result.schedule_info.rhs_bound << ",\n";
  output << "    \"bound_contract\": \"" << json_escape(result.schedule_info.bound_contract) << "\",\n";
  output << "    \"tile_m\": " << result.schedule_info.tile_m << ",\n";
  output << "    \"tile_n\": " << result.schedule_info.tile_n << ",\n";
  output << "    \"tile_rows\": " << result.schedule_info.tile_rows << ",\n";
  output << "    \"tile_cols\": " << result.schedule_info.tile_cols << ",\n";
  output << "    \"tile_count\": " << result.schedule_info.tile_count << ",\n";
  output << "    \"min_required_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  output << "    \"max_required_prefix\": " << result.schedule_info.max_required_prefix << ",\n";
  output << "    \"min_selected_prefix\": " << result.schedule_info.min_selected_prefix << ",\n";
  output << "    \"max_selected_prefix\": " << result.schedule_info.max_selected_prefix << ",\n";
  output << "    \"prefix_group_count\": " << result.schedule_info.prefix_group_count << ",\n";
  output << "    \"adaptive_prefix_active\": "
            << (result.schedule_info.adaptive_prefix_active ? "true" : "false") << ",\n";
  output << "    \"adaptive_skip_active\": "
            << (result.schedule_info.adaptive_skip_active ? "true" : "false") << ",\n";
  output << "    \"adaptive_execution_applied\": " << (adaptive_applied ? "true" : "false") << ",\n";
  output << "    \"flags\": " << result.schedule_info.flags << ",\n";
  output << "    \"zero_output_tile_count\": " << result.zero_output_tile_count << ",\n";
  const double zero_output_tile_fraction =
      result.schedule_info.tile_count == 0
          ? 0.0
          : static_cast<double>(result.zero_output_tile_count) /
                static_cast<double>(result.schedule_info.tile_count);
  output << "    \"zero_output_tile_fraction\": ";
  output << zero_output_tile_fraction;
  output << ",\n";
  output << "    \"zero_output_selected_residue_planes\": "
            << result.zero_output_selected_residue_plane_count << ",\n";
  output << "    \"zero_output_skip_active\": "
            << (result.zero_output_tile_count != 0 ? "true" : "false") << ",\n";
  output << "    \"zero_a_row_proof_count\": " << result.zero_a_row_proof_count << ",\n";
  output << "    \"zero_b_col_proof_count\": " << result.zero_b_col_proof_count << ",\n";
  output << "    \"zero_row_col_product_count\": " << result.zero_row_col_product_count << ",\n";
  output << "    \"planner_zero_a_row_count\": " << result.schedule_info.zero_a_row_count << ",\n";
  output << "    \"planner_zero_b_col_count\": " << result.schedule_info.zero_b_col_count << ",\n";
  output << "    \"planner_zero_row_col_product_count\": "
            << result.schedule_info.zero_row_col_product_count << ",\n";
  output << "    \"range_bit_length\": " << result.schedule_info.range_bit_length << "\n";
  output << "  },\n";
  output << "  \"epilogue_type\": \"" << epilogue_type(args) << "\",\n";
  output << "  \"exact_wide_limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    output << args.exact_wide_limb_count;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"exact_wide_magnitude\": ";
  if (result.exact_wide_magnitude.available) {
    const auto& magnitude = result.exact_wide_magnitude;
    output << "{\"schema\": \"exact_wide_magnitude_v1\", \"oracle\": \"boost_cpp_int_full_integer_gemm_all_limbs\", "
              << "\"oracle_complete\": " << (magnitude.oracle_complete ? "true" : "false")
              << ", \"a_max_abs\": \"" << magnitude.a_max_abs << "\", \"b_max_abs\": \"" << magnitude.b_max_abs
              << "\", \"admitted_abs_bound\": \"" << magnitude.admitted_abs_bound << "\", \"basis_product\": \"" << magnitude.basis_product
              << "\", \"a_max_magnitude_bits\": " << magnitude.a_max_magnitude_bits
              << ", \"b_max_magnitude_bits\": " << magnitude.b_max_magnitude_bits
              << ", \"output_min_magnitude_bits\": " << magnitude.output_min_magnitude_bits
              << ", \"output_max_magnitude_bits\": " << magnitude.output_max_magnitude_bits
              << ", \"output_max_magnitude_limbs\": " << (magnitude.output_max_magnitude_bits + 63u) / 64u
              << ", \"storage_bits\": " << args.exact_wide_limb_count * 64u
              << ", \"checked_outputs\": " << magnitude.checked_outputs
              << ", \"checked_limbs\": " << magnitude.checked_limbs
              << ", \"negative_outputs\": " << magnitude.negative_outputs
              << ", \"admission_us\": " << magnitude.admission_us << ", \"oracle_us\": " << magnitude.oracle_us
              << ", \"timing_scope\": \"outside_all_repeated_public_call_samples\"}";
  } else output << "null";
  output << ",\n";
  output << "  \"exact_wide_export_status_check\": ";
  print_json_string_or_null(output, exact_wide_export_status_check_name(args));
  output << ",\n";
  const auto& reference = result.native_reference;
  output << "  \"native_reference_verification\": {\"method\": \"" << reference.method
            << "\", \"cache_key\": \"" << reference.key << "\", \"payload_sha256\": \"" << reference.payload_sha256
            << "\", \"checked_cells\": " << reference.checked_cells << ", \"elapsed_us\": " << reference.elapsed_us
            << ", \"inside_production_interval\": false},\n";
  output << "  \"sustained_production\": {\"requested_batches\": " << args.sustained_batches
            << ", \"min_calls\": " << args.sustained_min_calls << ", \"max_calls\": " << args.sustained_max_calls
            << ", \"min_duration_us\": " << args.sustained_min_ms * 1000.0
            << ", \"boundary\": \"same_completed_output_as_latency_lane\", \"instrumentation\": \"hooks_disabled\", "
            << "\"oracle\": \"full_integer_postflight_outside_timed_interval\", \"observations\": [";
  for (std::size_t i = 0; i < result.sustained_observations.size(); ++i) {
    const auto& observation = result.sustained_observations[i];
    if (i) output << ", ";
    output << "{\"calls\": " << observation.calls << ", \"tasks\": " << observation.tasks
              << ", \"duration_us\": " << observation.duration_us
              << ", \"per_call_us\": " << observation.duration_us / observation.calls
              << ", \"termination\": \"" << (observation.duration_reached ? "duration_and_minimum_calls" : "hard_call_cap") << "\"}";
  }
  output << "]},\n";
  output << "  \"residue_chain_length\": " << args.residue_chain_length << ",\n";
  output << "  \"residue_chain_final_export\": " << (chain_final_export ? "true" : "false") << ",\n";
  output << "  \"residue_chain_independent_final_export\": "
            << (chain_independent_final_export ? "true" : "false") << ",\n";
  output << "  \"residue_output_mode\": \"" << residue_output_mode_name(args) << "\",\n";
  output << "  \"host_api_batch\": {\n";
  output << "    \"enabled\": " << (host_batch ? "true" : "false") << ",\n";
  output << "    \"batch_size\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  output << "    \"tasks_per_measured_repeat\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  output << "    \"total_measured_tasks\": "
            << static_cast<uint64_t>(host_batch ? args.host_api_batch_size : 1) * static_cast<uint64_t>(args.repeats)
            << ",\n";
  output << "    \"setup_scope\": \""
            << (host_batch
                    ? "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
                    : "single_task_default_benchmark_mode")
            << "\",\n";
  output << "    \"timing_policy\": \""
            << (host_batch ? "aggregate_batch_totals_per_measured_repeat" : "single_call_totals_per_measured_repeat")
            << "\",\n";
  output << "    \"checksum_policy\": \""
            << (host_batch ? "fnv1a_over_final_task_output_checksums" : "single_final_output_checksum")
            << "\"\n";
  output << "  },\n";
  output << "  \"hip_graph_replay\": {\n";
  output << "    \"requested\": " << (result.hip_graph_replay_requested ? "true" : "false") << ",\n";
  output << "    \"available\": " << (result.hip_graph_replay_available ? "true" : "false") << ",\n";
  output << "    \"used\": " << (result.hip_graph_replay_used ? "true" : "false") << ",\n";
  output << "    \"status\": \"" << json_escape(result.hip_graph_replay_status) << "\",\n";
  output << "    \"scope\": \"" << json_escape(result.hip_graph_replay_scope) << "\",\n";
  output << "    \"descriptor_identity\": \"fixed_plan_workspace_descriptor:m=" << args.m << ";n=" << args.n
            << ";k=" << args.k << "\",\n";
  output << "    \"plan_identity\": \"" << json_escape(result.backend_info.autotune_key) << "\",\n";
  output << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  output << "    \"capture_status\": \""
            << (result.hip_graph_replay_used ? "replayed" : "not_requested") << "\",\n";
  output << "    \"unsupported_reason\": ";
  print_nullable_std_string(output, result.hip_graph_replay_requested && !result.hip_graph_replay_used ? result.hip_graph_replay_caveat
                                                                         : std::string());
  output << ",\n";
  output << "    \"promotion_eligible\": false,\n";
  output << "    \"caller_transaction\": ";
  print_graph_caller_transaction_json(output, args, result);
  output << ",\n";
  output << "    \"capture_us\": " << result.hip_graph_capture_us << ",\n";
  output << "    \"instantiate_us\": " << result.hip_graph_instantiate_us << ",\n";
  output << "    \"graph_launches_per_measured_repeat\": "
            << (result.hip_graph_replay_used ? 1 : 0) << ",\n";
  output << "    \"total_graph_launches\": " << result.hip_graph_launch_count << ",\n";
  const bool graph_full_bounded_pack_export = hip_graph_replay_full_bounded_pack_export_used(result);
  const bool graph_full_finite_pack_export =
      result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export";
  const bool graph_full_exact_wide_pack_export =
      result.hip_graph_replay_scope == "direct_hip_exact_wide_pack_gemm_export";
  const bool graph_full_wrap64_pack_export =
      result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export";
  const bool graph_full_pack_export =
      graph_full_bounded_pack_export || graph_full_finite_pack_export ||
      graph_full_exact_wide_pack_export || graph_full_wrap64_pack_export;
  output << "    \"captured_chain_length\": "
            << (result.hip_graph_replay_used ? (graph_full_pack_export ? 1 : args.residue_chain_length) : 0)
            << ",\n";
  output << "    \"timing_policy\": \""
            << (result.hip_graph_replay_used
                     ? (graph_full_finite_pack_export
                            ? "raw_timings_us.end_to_end_measure_one_full_finite_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                            : (graph_full_exact_wide_pack_export
                                   ? "raw_timings_us.end_to_end_measure_one_full_exact_wide_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                                   : (graph_full_wrap64_pack_export
                                          ? "raw_timings_us.end_to_end_measure_one_full_wrap64_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                                          : (graph_full_bounded_pack_export
                                                 ? "raw_timings_us.end_to_end_measure_one_full_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                                                 : "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync"))))
                    : "not_applicable")
            << "\",\n";
  output << "    \"setup_policy\": \""
            << (result.hip_graph_replay_used
                    ? (graph_full_pack_export ? "capture_and_instantiate_before_warmups_no_prepack_reuse"
                                              : "A_B_prepack_before_capture_capture_and_instantiate_before_warmups")
                    : "not_applicable")
            << "\",\n";
  output << "    \"final_export_policy\": \""
            << (result.hip_graph_replay_used
                     ? (graph_full_finite_pack_export
                            ? "finite_canonical_export_and_output_d2h_captured_inside_graph_each_repeat"
                            : (graph_full_exact_wide_pack_export
                                   ? "exact_wide_limb_export_and_output_d2h_captured_inside_graph_each_repeat"
                                   : (graph_full_wrap64_pack_export
                                          ? "wrap64_low64_export_and_output_d2h_captured_inside_graph_each_repeat"
                                          : (graph_full_bounded_pack_export
                                                 ? "logical_export_and_output_d2h_captured_inside_graph_each_repeat"
                                                 : "one_final_logical_export_after_measured_repeats_for_checksum_only"))))
                    : "not_applicable")
            << "\",\n";
  output << "    \"caveat\": ";
  print_nullable_std_string(output, result.hip_graph_replay_caveat);
  output << "\n";
  output << "  },\n";
  output << "  \"packed_layout_version\": ";
  print_json_string_or_null(output, packed_layout_version(args));
  output << ",\n";
  output << "  \"seed\": " << args.seed << ",\n";
  output << "  \"warmups\": " << args.warmups << ",\n";
  output << "  \"repeats\": " << args.repeats << ",\n";
  output << "  \"reuse_packed_inputs\": " << (args.reuse_packed_inputs ? "true" : "false") << ",\n";
  output << "  \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  output << "  \"prepack_reuse_operands\": ";
  print_string_array(output, prepack_reuse_operands(args));
  output << ",\n";
  output << "  \"prepack_reuse_strategy\": \"" << prepack_reuse_strategy_name(result.prepack_reuse_strategy)
            << "\",\n";
  output << "  \"prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    output << result.prepack_setup_us;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"prepack_setup_breakdown_us\": ";
  print_prepack_setup_breakdown_json(output, result, "  ");
  output << ",\n";
  output << "  \"input_profile\": \"" << input_profile_name(args) << "\",\n";
  output << "  \"input_distribution\": \"" << input_distribution(args) << "\",\n";
  output << "  \"command_line\": \"" << json_escape(cmdline) << "\",\n";
  output << "  \"git_commit\": \"" << json_escape(runtime_git_commit()) << "\",\n";
#ifdef RNS8_BUILD_MANIFEST_JSON
  output << "  \"build_manifest\": " << RNS8_BUILD_MANIFEST_JSON << ",\n";
#endif
  output << "  \"compiler\": {\n";
  output << "    \"id\": \"" << compiler_id() << "\",\n";
  output << "    \"version\": \"" << compiler_version() << "\"\n";
  output << "  },\n";
  output << "  \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  output << "  \"hip_toolchain\": {\n";
  output << "    \"enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  output << "    \"hip_root\": ";
  print_nullable_string(output, RNS8_CONFIGURED_HIP_ROOT);
  output << ",\n";
  output << "    \"hipcc_path\": ";
  print_nullable_string(output, RNS8_CONFIGURED_HIPCC_PATH);
  output << ",\n";
  output << "    \"hipcc_version\": ";
  print_nullable_string(output, RNS8_CONFIGURED_HIPCC_VERSION);
  output << ",\n";
  output << "    \"hip_sdk_or_rocm_version\": ";
  print_nullable_string(output, RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  output << ",\n";
  output << "    \"version_source\": "
            << (RNS8_CONFIGURED_HIP_ENABLED && RNS8_CONFIGURED_HIPCC_VERSION[0] != '\0'
                    ? "\"hipcc --version\""
                    : "null")
            << "\n";
  output << "  },\n";
  output << "  \"runtime_environment\": {\n";
  output << "    \"HIP_VISIBLE_DEVICES\": ";
  print_json_string_or_null(output, environment_value("HIP_VISIBLE_DEVICES"));
  output << ",\n";
  output << "    \"ROCR_VISIBLE_DEVICES\": ";
  print_json_string_or_null(output, environment_value("ROCR_VISIBLE_DEVICES"));
  output << ",\n";
  output << "    \"GPU_DEVICE_ORDINAL\": ";
  print_json_string_or_null(output, environment_value("GPU_DEVICE_ORDINAL"));
  output << ",\n";
  output << "    \"ROCM_PATH\": ";
  print_json_string_or_null(output, environment_value("ROCM_PATH"));
  output << ",\n";
  output << "    \"HIP_PATH\": ";
  print_json_string_or_null(output, environment_value("HIP_PATH"));
  output << ",\n";
  output << "    \"LD_LIBRARY_PATH\": ";
  print_json_string_or_null(output, environment_value("LD_LIBRARY_PATH"));
  output << ",\n";
  output << "    \"RNS8_MULTI_GPU_MODE\": ";
  print_json_string_or_null(output, environment_value("RNS8_MULTI_GPU_MODE"));
  output << ",\n";
  output << "    \"RNS8_RANK\": ";
  print_json_string_or_null(output, environment_value("RNS8_RANK"));
  output << ",\n";
  output << "    \"RNS8_WORLD_SIZE\": ";
  print_json_string_or_null(output, environment_value("RNS8_WORLD_SIZE"));
  output << "\n";
  output << "  },\n";
  const uint32_t visible_device_count = visible_device_count_from_environment();
  const uint32_t node_gpu_count = benchmark_node_gpu_count();
  output << "  \"device\": {\n";
  output << "    \"device_id\": " << info.device_id << ",\n";
  output << "    \"device_index\": " << info.device_id << ",\n";
  output << "    \"name\": \"" << json_escape(info.name) << "\",\n";
  output << "    \"device_name\": \"" << json_escape(info.name) << "\",\n";
  output << "    \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  output << "    \"target_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  output << "    \"target_cache_key\": \"" << json_escape(target_cache_key(info, result)) << "\",\n";
  output << "    \"target_instance_id\": \"" << json_escape(target_instance_id(info, result)) << "\",\n";
  output << "    \"hip_available\": " << info.hip_available << ",\n";
  output << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  output << "    \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  output << "    \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  output << "    \"visible_device_count\": " << visible_device_count << ",\n";
  output << "    \"node_gpu_count\": " << node_gpu_count << "\n";
  output << "  },\n";
  output << "  \"clock_power_settings\": null,\n";
  print_comparison_baseline(output, args, info, result);
  output << "  \"derived_tops_equivalent\": null,\n";
  output << "  \"timing_source\": \"std::chrono::steady_clock\",\n";
  if (graph_full_wrap64_pack_export) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP wrap64 full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, byte-limb pack kernels, one byte-GEMM36 kernel, "
              << "low64 export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export "
                 "are zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (graph_full_finite_pack_export) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP finite-u8 full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, finite pack kernels, one modular GEMM, canonical "
              << "finite export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export "
                 "are zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (graph_full_bounded_pack_export) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP bounded full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, pack kernels, one RNS GEMM, export status handling, "
              << "export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export are "
                 "zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (hip_graph_replay_requested(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP residue-current RNS GEMM chain; A and B are packed once before graph capture, "
              << "capture_us and instantiate_us record one-time graph setup before warmups, each measured repeat "
              << "runs one hipGraphLaunch plus stream synchronization containing " << args.residue_chain_length
              << " resident RNS GEMM launches, raw_timings_us.pack and raw_timings_us.crt_export are zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_residue_output) {
    output << "  \"timing_note\": \"host wall-clock timings for a residue-current RNS GEMM chain; "
              << "each measured repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_independent_final_export) {
    output << "  \"timing_note\": \"host wall-clock timings for an independent final-output RNS GEMM chain; "
              << "each measured repeat performs " << args.residue_chain_length
              << " GEMM calls, exports every intermediate host output, repacks non-final intermediates as the next "
                 "native input, and includes all materialization work in raw_timings_us.pack, raw_timings_us."
                 "crt_export, and end_to_end\",\n";
  } else if (chain_final_export) {
    output << "  \"timing_note\": \"host wall-clock timings for a final-output RNS GEMM chain; each measured "
              << "repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls with intermediate outputs kept in RNS form, then exports the final "
                 "logical output inside the measured repeat so raw_timings_us.crt_export and end_to_end include "
                 "the requested host-output contract\",\n";
  } else if (args.oneshot) {
    if (finite_benchmark_semantics(args.semantics)) {
      output << "  \"timing_note\": \"host wall-clock timings for the public finite-u8 one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_finite_ring_u8_oneshot or rns8_gemm_finite_field_u8_oneshot call, while "
                   "raw_timings_us.pack and raw_timings_us.crt_export are zero because native uint8 input copies, "
                   "direct-HIP native-input finite GEMM work, canonical export, and teardown happen inside the "
                   "measured API call\",\n";
    } else {
      output << "  \"timing_note\": \"host wall-clock timings for the public bounded one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_i64_oneshot or rns8_gemm_u64_oneshot call, while raw_timings_us.pack and "
                   "raw_timings_us.crt_export are zero because transient input copies, any fused native-input "
                   "direct-HIP GEMM work, logical export, and teardown happen inside the measured API call\",\n";
    }
  } else if (bounded_residue_channel_fusion_requested(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for a benchmark-only direct-HIP residue-channel "
                 "fusion experiment over width-3 groups from the first fixed prefix-9 moduli; this capture is "
                 "not AUTO-selected and is evidence-only for pack/layout comparison\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for an explicit benchmark-owned direct-HIP "
                 "uniform-small native-input path; each measured repeat copies A and B as row-major int8 inputs, "
                 "runs the existing prefix-9 colpair centered-residue GEMM kernel group, and exports through the "
                 "normal CRT output path\",\n";
  } else if (all_zero_direct_hip_pack_elided) {
    output << "  \"timing_note\": \"host wall-clock timings for a trusted all-zero direct-HIP adaptive "
                 "per-tile bounded capture; raw_timings_us.pack is zero because exact tile-bound scheduling "
                 "proved every output tile zero before measurement, so the backend materializes resident RNS "
                 "zero output without reading A or B\",\n";
  } else if (rns_residue_chain_requested(args)) {
    output << "      \"matrix_alloc\": \"one-time persistent A/B/C matrix allocation plus one scratch matrix for alternating chained RNS outputs\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for a vector-native-to-Direct-RNS chain; "
                 "each measured repeat packs producer A/B into the HIP vector-ALU backend, ";
    if (args.reuse_packed_b) {
      output << "reuses a Direct-HIP consumer B input that was packed once before warmups, ";
    } else {
      output << "packs a second Direct-HIP RNS input, ";
    }
    output << "runs vector GEMM to native device output, materializes that native output into Direct-HIP RNS "
                 "storage, runs the Direct-HIP consumer RNS GEMM, and exports the final logical output";
    if (args.reuse_packed_b) {
      output << "; per-repeat end_to_end excludes one-time prepack_setup_us";
    }
    output << "\",\n";
  } else if (native_to_rns_bridge_requested(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for an explicit AUTO/direct-HIP native-to-RNS "
                 "bridge benchmark; each measured repeat packs bounded inputs into direct-HIP matrices, forces "
                 "resident RNS input residues stale while keeping native device buffers current, runs the existing "
                 "native-to-RNS conversion kernels inside rns8_gemm_rns, then runs the normal direct-HIP RNS GEMM "
                 "and CRT export\",\n";
  } else if (args.reuse_packed_inputs) {
    if (use_prepacked_b_cache) {
      output << "  \"timing_note\": \"host wall-clock timings with a reusable rocWMMA B prepack cache; "
                   "prepack_setup_us records explicit cache materialization before warmups, "
                   "end_to_end includes per-repeat A packing, cached-B rns_gemm, and export\",\n";
    } else if (reuses_all_packed_inputs(args)) {
      output << "  \"timing_note\": \"host wall-clock timings for repeated GEMM/export against persistent "
                   "packed A/B inputs; pack is a zero-valued per-repeat phase and prepack_setup_us records the "
                   "one-time pack before warmups, which end_to_end excludes\",\n";
    } else {
      output << "  \"timing_note\": \"host wall-clock timings with persistent packed "
                << prepack_reuse_operand_text(args) << " input and per-repeat packing of "
                << per_repeat_pack_operand_text(args)
                << "; prepack_setup_us records the one-time pack before warmups and end_to_end includes "
                   "the per-repeat pack phase for the non-reused operand\",\n";
    }
  } else if (grouped_dispatch) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-owned grouped dispatch evidence; each "
              << "measured repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate grouped totals, avg_*_per_task_us fields "
                 "divide those totals by grouped_dispatch.task_count";
    if (result.grouped_dispatch_batched_export_enabled) {
      output << "; grouped export kernels write into one shared device output slab before one compact D2H copy";
    }
    output << ", and this capture remains non-promotable until a real device grouped dispatcher exists\",\n";
  } else if (host_batch) {
    output << "  \"timing_note\": \"host wall-clock timings for benchmark-owned host API batching; each measured "
              << "repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate batch totals, and avg_*_per_task_us fields "
                 "divide those totals by host_api_batch.batch_size\",\n";
  } else if (args.vector_alu_baseline) {
    output << "  \"timing_note\": \"host wall-clock timings for the benchmark-only HIP vector-ALU exact "
                 "int64 baseline; phases are raw input H2D copies, one 192-bit-limb exact output kernel, "
                 "and direct output D2H export with range-status validation\",\n";
  } else if (runtime_vector_alu_backend(args)) {
    output << "  \"timing_note\": \"host wall-clock timings for the public HIP vector-ALU bounded runtime "
                 "backend; phases are native device packing, one 192-bit-limb exact output kernel, and direct "
                 "native output export with range-status validation\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    output << "  \"timing_note\": \"host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 "
                 "candidate; GPU event timing uses direct-HIP byte-limb pack/export labels plus one candidate "
                 "operation-group label and this path is not public or AUTO-selected\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64 && selected_backend_kind == RNS8_BACKEND_HIP_DIRECT) {
    output << "  \"timing_note\": \"host wall-clock timings for the direct-HIP wrap64 tiled byte-limb "
                 "path; GPU event timing uses wrap64-specific tiled byte-GEMM/export labels plus "
                 "current rns_gemm/crt_export aggregate phase labels\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    output << "  \"timing_note\": \"host wall-clock timings for the CPU wrap64 byte-limb reference; "
                 "no GPU event timing is requested for this backend\",\n";
  } else if (finite_benchmark_semantics(args.semantics)) {
    output << "  \"timing_note\": \"host wall-clock timings for persistent finite-u8 packing, finite residue "
                 "GEMM, and canonical uint8 export with an explicit benchmark modulus\",\n";
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    output << "  \"timing_note\": \"host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, "
                 "and fixed-width little-endian limb export; GPU event timing names exact-wide export operation "
                 "groups when backend hooks are available";
    if (!exact_wide_export_status_check_required(args)) {
      output << "; the selected limb count covers the backend 192-bit reconstruction width, so exact-wide "
                   "range-status memset and status D2H are elided and reported as zero-valued event phases";
    }
    output << "\",\n";
  } else if (adaptive_applied && info.backend == RNS8_BACKEND_CPU_REFERENCE) {
    output << "  \"timing_note\": \"host wall-clock timings for the CPU adaptive per-tile bounded "
                 "reference path; no GPU event timing is requested for this backend\",\n";
  } else if (adaptive_applied && accelerator_deep_kernel_events) {
    output << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator "
                 "path; GPU event timing combines direct-HIP pack/export labels with one accelerator GEMM "
                 "operation-group label\",\n";
  } else if (adaptive_applied && gpu_events_available) {
    output << "  \"timing_note\": \"host wall-clock timings for the direct-HIP adaptive per-tile bounded "
                 "correctness path; GPU event timing aggregates all selected-prefix tiled GEMM launches and tiled "
                 "CRT export work into backend operation-group labels\",\n";
  } else if (adaptive_applied) {
    output << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator path; "
                 "GPU event timing was not captured for this selected backend/path\",\n";
  } else if (info.backend == RNS8_BACKEND_HIPBLASLT) {
    output << "  \"timing_note\": \"host wall-clock timings for the hipBLASLt baseline path; GPU event timing "
                 "separates persistent RNS packing, hipBLASLt INT8-to-INT32 matmul, separate centered-residue "
                 "reduction, and CRT export operation groups\",\n";
  } else {
    output << "  \"timing_note\": \"host wall-clock timings; direct-HIP calls include current backend "
                 "synchronization, first-use persistent buffer allocation, copies, kernel launches, fused reduction, "
                 "and GPU bounded export when using HIP\",\n";
  }

  output << "  \"locality_schedule\": ";
  if(result.locality_info.struct_size) {
    const auto& v=result.locality_info;
    output << "{\"version\":1,\"representation\":\"resident_residue8\",\"modulus_group_size\":1,"
      << "\"requested_order\":" << v.requested_order << ",\"applied_order\":" << v.applied_order
      << ",\"target_id\":" << v.target_id << ",\"topology_version\":" << v.topology_version
      << ",\"applied\":" << (v.applied?"true":"false") << ",\"physical_affinity_established\":false"
      << ",\"cache_domains_model\":" << v.cache_domains << ",\"concurrent_groups_per_domain_model\":" << v.concurrent_groups_per_domain
      << ",\"cache_bytes_per_domain_model\":" << v.cache_bytes_per_domain << ",\"active_entries\":" << v.active_entries
      << ",\"directory_bytes\":" << v.directory_bytes << ",\"k_phases\":" << v.k_phases
      << ",\"requested_source_bytes\":" << v.requested_source_bytes << ",\"unique_source_bytes\":" << v.unique_source_bytes
      << ",\"output_read_write_bytes\":" << v.output_read_write_bytes << ",\"permutation_fingerprint\":" << v.permutation_fingerprint
      << ",\"traffic_scope\":\"active_arithmetic_only_excludes_pack_zero_fill_export\",\"actual_hbm_bytes\":null,\"measured_cache_hit_rate\":null,\"promotion_eligible\":false}";
  } else output << "null";
  output << ",\n";
