  std::cout << "  \"m\": " << args.m << ",\n";
  std::cout << "  \"n\": " << args.n << ",\n";
  std::cout << "  \"k\": " << args.k << ",\n";
  std::cout << "  \"output_logical_ld\": " << output_ld << ",\n";
  std::cout << "  \"output_ld_padding\": " << args.output_ld_padding << ",\n";
  std::cout << "  \"prefix\": " << prefix << ",\n";
  std::cout << "  \"selected_prefix\": " << selected_prefix << ",\n";
  std::cout << "  \"requested_max_prefix\": " << prefix << ",\n";
  std::cout << "  \"contract_prefix_policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  std::cout << "  \"residue_planes_requested\": " << prefix << ",\n";
  std::cout << "  \"residue_planes_selected\": " << selected_prefix << ",\n";
  std::cout << "  \"residue_planes_skipped\": " << residue_planes_skipped << ",\n";
  std::cout << "  \"residue_plane_skip_fraction\": " << residue_plane_skip_fraction << ",\n";
  std::cout << "  \"finite_modulus\": ";
  if (finite_benchmark_semantics(args.semantics)) {
    std::cout << args.finite_modulus;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"tile_m\": " << args.tile_m << ",\n";
  std::cout << "  \"tile_n\": " << args.tile_n << ",\n";
  std::cout << "  \"layout\": \"row_major\",\n";
  std::cout << "  \"k_block_size\": "
            << benchmark_k_block_size(args, result)
            << ",\n";
  std::cout << "  \"adaptive_tile_size\": null,\n";
  std::cout << "  \"tile_bounds_u64\": ";
  if (args.bound_mode == BoundMode::PerTile) {
    std::cout << "{\n";
    std::cout << "    \"source\": \"exact_seeded_input_prepass\",\n";
    std::cout << "    \"pattern\": \"" << tile_bound_pattern(args) << "\",\n";
    std::cout << "    \"order\": \"row_major_output_tiles\",\n";
    std::cout << "    \"count\": " << result.tile_bounds.size() << ",\n";
    std::cout << "    \"min\": " << result.tile_bound_min << ",\n";
    std::cout << "    \"max\": " << result.tile_bound_max << ",\n";
    std::cout << "    \"hash_u64\": " << result.tile_bound_hash << "\n";
    std::cout << "  },\n";
  } else {
    std::cout << "null,\n";
  }
  std::cout << "  \"schedule_metadata\": {\n";
  std::cout << "    \"source\": \"" << json_escape(result.schedule_source) << "\",\n";
  std::cout << "    \"bound_kind\": \"" << bound_kind_name(result.schedule_info.bound_kind) << "\",\n";
  std::cout << "    \"effective_bound\": " << result.schedule_info.effective_bound << ",\n";
  std::cout << "    \"lhs_bound\": " << result.schedule_info.lhs_bound << ",\n";
  std::cout << "    \"rhs_bound\": " << result.schedule_info.rhs_bound << ",\n";
  std::cout << "    \"bound_contract\": \"" << json_escape(result.schedule_info.bound_contract) << "\",\n";
  std::cout << "    \"tile_m\": " << result.schedule_info.tile_m << ",\n";
  std::cout << "    \"tile_n\": " << result.schedule_info.tile_n << ",\n";
  std::cout << "    \"tile_rows\": " << result.schedule_info.tile_rows << ",\n";
  std::cout << "    \"tile_cols\": " << result.schedule_info.tile_cols << ",\n";
  std::cout << "    \"tile_count\": " << result.schedule_info.tile_count << ",\n";
  std::cout << "    \"min_required_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  std::cout << "    \"max_required_prefix\": " << result.schedule_info.max_required_prefix << ",\n";
  std::cout << "    \"min_selected_prefix\": " << result.schedule_info.min_selected_prefix << ",\n";
  std::cout << "    \"max_selected_prefix\": " << result.schedule_info.max_selected_prefix << ",\n";
  std::cout << "    \"prefix_group_count\": " << result.schedule_info.prefix_group_count << ",\n";
  std::cout << "    \"adaptive_prefix_active\": "
            << (result.schedule_info.adaptive_prefix_active ? "true" : "false") << ",\n";
  std::cout << "    \"adaptive_skip_active\": "
            << (result.schedule_info.adaptive_skip_active ? "true" : "false") << ",\n";
  std::cout << "    \"adaptive_execution_applied\": " << (adaptive_applied ? "true" : "false") << ",\n";
  std::cout << "    \"flags\": " << result.schedule_info.flags << ",\n";
  std::cout << "    \"zero_output_tile_count\": " << result.zero_output_tile_count << ",\n";
  const double zero_output_tile_fraction =
      result.schedule_info.tile_count == 0
          ? 0.0
          : static_cast<double>(result.zero_output_tile_count) /
                static_cast<double>(result.schedule_info.tile_count);
  std::cout << "    \"zero_output_tile_fraction\": ";
  std::cout << zero_output_tile_fraction;
  std::cout << ",\n";
  std::cout << "    \"zero_output_selected_residue_planes\": "
            << result.zero_output_selected_residue_plane_count << ",\n";
  std::cout << "    \"zero_output_skip_active\": "
            << (result.zero_output_tile_count != 0 ? "true" : "false") << ",\n";
  std::cout << "    \"zero_a_row_proof_count\": " << result.zero_a_row_proof_count << ",\n";
  std::cout << "    \"zero_b_col_proof_count\": " << result.zero_b_col_proof_count << ",\n";
  std::cout << "    \"zero_row_col_product_count\": " << result.zero_row_col_product_count << ",\n";
  std::cout << "    \"planner_zero_a_row_count\": " << result.schedule_info.zero_a_row_count << ",\n";
  std::cout << "    \"planner_zero_b_col_count\": " << result.schedule_info.zero_b_col_count << ",\n";
  std::cout << "    \"planner_zero_row_col_product_count\": "
            << result.schedule_info.zero_row_col_product_count << ",\n";
  std::cout << "    \"range_bit_length\": " << result.schedule_info.range_bit_length << "\n";
  std::cout << "  },\n";
  std::cout << "  \"epilogue_type\": \"" << epilogue_type(args) << "\",\n";
  std::cout << "  \"exact_wide_limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"exact_wide_export_status_check\": ";
  print_json_string_or_null(exact_wide_export_status_check_name(args));
  std::cout << ",\n";
  std::cout << "  \"residue_chain_length\": " << args.residue_chain_length << ",\n";
  std::cout << "  \"residue_chain_final_export\": " << (chain_final_export ? "true" : "false") << ",\n";
  std::cout << "  \"residue_chain_independent_final_export\": "
            << (chain_independent_final_export ? "true" : "false") << ",\n";
  std::cout << "  \"residue_output_mode\": \"" << residue_output_mode_name(args) << "\",\n";
  std::cout << "  \"host_api_batch\": {\n";
  std::cout << "    \"enabled\": " << (host_batch ? "true" : "false") << ",\n";
  std::cout << "    \"batch_size\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"tasks_per_measured_repeat\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"total_measured_tasks\": "
            << static_cast<uint64_t>(host_batch ? args.host_api_batch_size : 1) * static_cast<uint64_t>(args.repeats)
            << ",\n";
  std::cout << "    \"setup_scope\": \""
            << (host_batch
                    ? "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
                    : "single_task_default_benchmark_mode")
            << "\",\n";
  std::cout << "    \"timing_policy\": \""
            << (host_batch ? "aggregate_batch_totals_per_measured_repeat" : "single_call_totals_per_measured_repeat")
            << "\",\n";
  std::cout << "    \"checksum_policy\": \""
            << (host_batch ? "fnv1a_over_final_task_output_checksums" : "single_final_output_checksum")
            << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"hip_graph_replay\": {\n";
  std::cout << "    \"requested\": " << (result.hip_graph_replay_requested ? "true" : "false") << ",\n";
  std::cout << "    \"available\": " << (result.hip_graph_replay_available ? "true" : "false") << ",\n";
  std::cout << "    \"used\": " << (result.hip_graph_replay_used ? "true" : "false") << ",\n";
  std::cout << "    \"status\": \"" << json_escape(result.hip_graph_replay_status) << "\",\n";
  std::cout << "    \"scope\": \"" << json_escape(result.hip_graph_replay_scope) << "\",\n";
  std::cout << "    \"descriptor_identity\": \"fixed_plan_workspace_descriptor:m=" << args.m << ";n=" << args.n
            << ";k=" << args.k << "\",\n";
  std::cout << "    \"plan_identity\": \"" << json_escape(result.backend_info.autotune_key) << "\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"capture_status\": \""
            << (result.hip_graph_replay_used ? "replayed" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(
      result.hip_graph_replay_requested && !result.hip_graph_replay_used ? result.hip_graph_replay_caveat
                                                                         : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false,\n";
  std::cout << "    \"capture_us\": " << result.hip_graph_capture_us << ",\n";
  std::cout << "    \"instantiate_us\": " << result.hip_graph_instantiate_us << ",\n";
  std::cout << "    \"graph_launches_per_measured_repeat\": "
            << (result.hip_graph_replay_used ? 1 : 0) << ",\n";
  std::cout << "    \"total_graph_launches\": " << result.hip_graph_launch_count << ",\n";
  const bool graph_full_bounded_pack_export = hip_graph_replay_full_bounded_pack_export_used(result);
  const bool graph_full_finite_pack_export =
      result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export";
  const bool graph_full_wrap64_pack_export =
      result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export";
  const bool graph_full_pack_export =
      graph_full_bounded_pack_export || graph_full_finite_pack_export || graph_full_wrap64_pack_export;
  std::cout << "    \"captured_chain_length\": "
            << (result.hip_graph_replay_used ? (graph_full_pack_export ? 1 : args.residue_chain_length) : 0)
            << ",\n";
  std::cout << "    \"timing_policy\": \""
            << (result.hip_graph_replay_used
                     ? (graph_full_finite_pack_export
                            ? "raw_timings_us.end_to_end_measure_one_full_finite_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                            : (graph_full_wrap64_pack_export
                                   ? "raw_timings_us.end_to_end_measure_one_full_wrap64_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                                   : (graph_full_bounded_pack_export
                                          ? "raw_timings_us.end_to_end_measure_one_full_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
                                          : "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync")))
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"setup_policy\": \""
            << (result.hip_graph_replay_used
                    ? (graph_full_pack_export ? "capture_and_instantiate_before_warmups_no_prepack_reuse"
                                              : "A_B_prepack_before_capture_capture_and_instantiate_before_warmups")
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"final_export_policy\": \""
            << (result.hip_graph_replay_used
                     ? (graph_full_finite_pack_export
                            ? "finite_canonical_export_and_output_d2h_captured_inside_graph_each_repeat"
                            : (graph_full_wrap64_pack_export
                                   ? "wrap64_low64_export_and_output_d2h_captured_inside_graph_each_repeat"
                                   : (graph_full_bounded_pack_export
                                          ? "logical_export_and_output_d2h_captured_inside_graph_each_repeat"
                                          : "one_final_logical_export_after_measured_repeats_for_checksum_only")))
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"caveat\": ";
  print_nullable_std_string(result.hip_graph_replay_caveat);
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"packed_layout_version\": ";
  print_json_string_or_null(packed_layout_version(args));
  std::cout << ",\n";
  std::cout << "  \"seed\": " << args.seed << ",\n";
  std::cout << "  \"warmups\": " << args.warmups << ",\n";
  std::cout << "  \"repeats\": " << args.repeats << ",\n";
  std::cout << "  \"reuse_packed_inputs\": " << (args.reuse_packed_inputs ? "true" : "false") << ",\n";
  std::cout << "  \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  std::cout << "  \"prepack_reuse_operands\": ";
  print_string_array(prepack_reuse_operands(args));
  std::cout << ",\n";
  std::cout << "  \"prepack_reuse_strategy\": \"" << prepack_reuse_strategy_name(result.prepack_reuse_strategy)
            << "\",\n";
  std::cout << "  \"prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    std::cout << result.prepack_setup_us;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"input_distribution\": \"" << input_distribution(args) << "\",\n";
  std::cout << "  \"command_line\": \"" << json_escape(cmdline) << "\",\n";
  std::cout << "  \"git_commit\": \"" << json_escape(runtime_git_commit()) << "\",\n";
  std::cout << "  \"compiler\": {\n";
  std::cout << "    \"id\": \"" << compiler_id() << "\",\n";
  std::cout << "    \"version\": \"" << compiler_version() << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  std::cout << "  \"hip_toolchain\": {\n";
  std::cout << "    \"enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  std::cout << "    \"hip_root\": ";
  print_nullable_string(RNS8_CONFIGURED_HIP_ROOT);
  std::cout << ",\n";
  std::cout << "    \"hipcc_path\": ";
  print_nullable_string(RNS8_CONFIGURED_HIPCC_PATH);
  std::cout << ",\n";
  std::cout << "    \"hipcc_version\": ";
  print_nullable_string(RNS8_CONFIGURED_HIPCC_VERSION);
  std::cout << ",\n";
  std::cout << "    \"hip_sdk_or_rocm_version\": ";
  print_nullable_string(RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  std::cout << ",\n";
  std::cout << "    \"version_source\": "
            << (RNS8_CONFIGURED_HIP_ENABLED && RNS8_CONFIGURED_HIPCC_VERSION[0] != '\0'
                    ? "\"hipcc --version\""
                    : "null")
            << "\n";
  std::cout << "  },\n";
  std::cout << "  \"runtime_environment\": {\n";
  std::cout << "    \"HIP_VISIBLE_DEVICES\": ";
  print_json_string_or_null(environment_value("HIP_VISIBLE_DEVICES"));
  std::cout << ",\n";
  std::cout << "    \"ROCR_VISIBLE_DEVICES\": ";
  print_json_string_or_null(environment_value("ROCR_VISIBLE_DEVICES"));
  std::cout << ",\n";
  std::cout << "    \"GPU_DEVICE_ORDINAL\": ";
  print_json_string_or_null(environment_value("GPU_DEVICE_ORDINAL"));
  std::cout << ",\n";
  std::cout << "    \"ROCM_PATH\": ";
  print_json_string_or_null(environment_value("ROCM_PATH"));
  std::cout << ",\n";
  std::cout << "    \"HIP_PATH\": ";
  print_json_string_or_null(environment_value("HIP_PATH"));
  std::cout << ",\n";
  std::cout << "    \"LD_LIBRARY_PATH\": ";
  print_json_string_or_null(environment_value("LD_LIBRARY_PATH"));
  std::cout << ",\n";
  std::cout << "    \"RNS8_MULTI_GPU_MODE\": ";
  print_json_string_or_null(environment_value("RNS8_MULTI_GPU_MODE"));
  std::cout << ",\n";
  std::cout << "    \"RNS8_RANK\": ";
  print_json_string_or_null(environment_value("RNS8_RANK"));
  std::cout << ",\n";
  std::cout << "    \"RNS8_WORLD_SIZE\": ";
  print_json_string_or_null(environment_value("RNS8_WORLD_SIZE"));
  std::cout << "\n";
  std::cout << "  },\n";
  const uint32_t visible_device_count = visible_device_count_from_environment();
  const uint32_t node_gpu_count = benchmark_node_gpu_count();
  std::cout << "  \"device\": {\n";
  std::cout << "    \"device_id\": " << info.device_id << ",\n";
  std::cout << "    \"device_index\": " << info.device_id << ",\n";
  std::cout << "    \"name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "    \"device_name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "    \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "    \"target_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "    \"target_cache_key\": \"" << json_escape(target_cache_key(info, result)) << "\",\n";
  std::cout << "    \"target_instance_id\": \"" << json_escape(target_instance_id(info, result)) << "\",\n";
  std::cout << "    \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "    \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "    \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  std::cout << "    \"visible_device_count\": " << visible_device_count << ",\n";
  std::cout << "    \"node_gpu_count\": " << node_gpu_count << "\n";
  std::cout << "  },\n";
  std::cout << "  \"clock_power_settings\": null,\n";
  print_comparison_baseline(args, info, result);
  std::cout << "  \"derived_tops_equivalent\": null,\n";
  std::cout << "  \"timing_source\": \"std::chrono::steady_clock\",\n";
  if (graph_full_wrap64_pack_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP wrap64 full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, byte-limb pack kernels, one byte-GEMM36 kernel, "
              << "low64 export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export "
                 "are zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (graph_full_finite_pack_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP finite-u8 full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, finite pack kernels, one modular GEMM, canonical "
              << "finite export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export "
                 "are zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (graph_full_bounded_pack_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP bounded full pack/GEMM/export path; capture_us and instantiate_us record one-time "
              << "graph setup before warmups, each measured repeat runs one hipGraphLaunch plus stream "
              << "synchronization containing A/B H2D copies, pack kernels, one RNS GEMM, export status handling, "
              << "export kernel, and output D2H, while raw_timings_us.pack and raw_timings_us.crt_export are "
                 "zero because the graph launch is measured as one end-to-end unit\",\n";
  } else if (hip_graph_replay_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP residue-current RNS GEMM chain; A and B are packed once before graph capture, "
              << "capture_us and instantiate_us record one-time graph setup before warmups, each measured repeat "
              << "runs one hipGraphLaunch plus stream synchronization containing " << args.residue_chain_length
              << " resident RNS GEMM launches, raw_timings_us.pack and raw_timings_us.crt_export are zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_residue_output) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a residue-current RNS GEMM chain; "
              << "each measured repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_independent_final_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an independent final-output RNS GEMM chain; "
              << "each measured repeat performs " << args.residue_chain_length
              << " GEMM calls, exports every intermediate host output, repacks non-final intermediates as the next "
                 "native input, and includes all materialization work in raw_timings_us.pack, raw_timings_us."
                 "crt_export, and end_to_end\",\n";
  } else if (chain_final_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a final-output RNS GEMM chain; each measured "
              << "repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls with intermediate outputs kept in RNS form, then exports the final "
                 "logical output inside the measured repeat so raw_timings_us.crt_export and end_to_end include "
                 "the requested host-output contract\",\n";
  } else if (args.oneshot) {
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "  \"timing_note\": \"host wall-clock timings for the public finite-u8 one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_finite_ring_u8_oneshot or rns8_gemm_finite_field_u8_oneshot call, while "
                   "raw_timings_us.pack and raw_timings_us.crt_export are zero because native uint8 input copies, "
                   "direct-HIP native-input finite GEMM work, canonical export, and teardown happen inside the "
                   "measured API call\",\n";
    } else {
      std::cout << "  \"timing_note\": \"host wall-clock timings for the public bounded one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_i64_oneshot or rns8_gemm_u64_oneshot call, while raw_timings_us.pack and "
                   "raw_timings_us.crt_export are zero because transient input copies, any fused native-input "
                   "direct-HIP GEMM work, logical export, and teardown happen inside the measured API call\",\n";
    }
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a benchmark-only direct-HIP residue-channel "
                 "fusion experiment over width-3 groups from the first fixed prefix-9 moduli; this capture is "
                 "not AUTO-selected and is evidence-only for pack/layout comparison\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an explicit benchmark-owned direct-HIP "
                 "uniform-small native-input path; each measured repeat copies A and B as row-major int8 inputs, "
                 "runs the existing prefix-9 colpair centered-residue GEMM kernel group, and exports through the "
                 "normal CRT output path\",\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a trusted all-zero direct-HIP adaptive "
                 "per-tile bounded capture; raw_timings_us.pack is zero because exact tile-bound scheduling "
                 "proved every output tile zero before measurement, so the backend materializes resident RNS "
                 "zero output without reading A or B\",\n";
  } else if (rns_residue_chain_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time persistent A/B/C matrix allocation plus one scratch matrix for alternating chained RNS outputs\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a vector-native-to-Direct-RNS chain; "
                 "each measured repeat packs producer A/B into the HIP vector-ALU backend, ";
    if (args.reuse_packed_b) {
      std::cout << "reuses a Direct-HIP consumer B input that was packed once before warmups, ";
    } else {
      std::cout << "packs a second Direct-HIP RNS input, ";
    }
    std::cout << "runs vector GEMM to native device output, materializes that native output into Direct-HIP RNS "
                 "storage, runs the Direct-HIP consumer RNS GEMM, and exports the final logical output";
    if (args.reuse_packed_b) {
      std::cout << "; per-repeat end_to_end excludes one-time prepack_setup_us";
    }
    std::cout << "\",\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an explicit AUTO/direct-HIP native-to-RNS "
                 "bridge benchmark; each measured repeat packs bounded inputs into direct-HIP matrices, forces "
                 "resident RNS input residues stale while keeping native device buffers current, runs the existing "
                 "native-to-RNS conversion kernels inside rns8_gemm_rns, then runs the normal direct-HIP RNS GEMM "
                 "and CRT export\",\n";
  } else if (args.reuse_packed_inputs) {
    if (use_prepacked_b_cache) {
      std::cout << "  \"timing_note\": \"host wall-clock timings with a reusable rocWMMA B prepack cache; "
                   "prepack_setup_us records persistent B packing plus cache materialization before warmups, "
                   "end_to_end includes per-repeat A packing, cached-B rns_gemm, and export\",\n";
    } else if (reuses_all_packed_inputs(args)) {
      std::cout << "  \"timing_note\": \"host wall-clock timings for repeated GEMM/export against persistent "
                   "packed A/B inputs; pack is a zero-valued per-repeat phase and prepack_setup_us records the "
                   "one-time pack before warmups, which end_to_end excludes\",\n";
    } else {
      std::cout << "  \"timing_note\": \"host wall-clock timings with persistent packed "
                << prepack_reuse_operand_text(args) << " input and per-repeat packing of "
                << per_repeat_pack_operand_text(args)
                << "; prepack_setup_us records the one-time pack before warmups and end_to_end includes "
                   "the per-repeat pack phase for the non-reused operand\",\n";
    }
  } else if (grouped_dispatch) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-owned grouped dispatch evidence; each "
              << "measured repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate grouped totals, avg_*_per_task_us fields "
                 "divide those totals by grouped_dispatch.task_count";
    if (result.grouped_dispatch_batched_export_enabled) {
      std::cout << "; grouped export kernels write into one shared device output slab before one compact D2H copy";
    }
    std::cout << ", and this capture remains non-promotable until a real device grouped dispatcher exists\",\n";
  } else if (host_batch) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-owned host API batching; each measured "
              << "repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate batch totals, and avg_*_per_task_us fields "
                 "divide those totals by host_api_batch.batch_size\",\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the benchmark-only HIP vector-ALU exact "
                 "int64 baseline; phases are raw input H2D copies, one 192-bit-limb exact output kernel, "
                 "and direct output D2H export with range-status validation\",\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the public HIP vector-ALU bounded runtime "
                 "backend; phases are native device packing, one 192-bit-limb exact output kernel, and direct "
                 "native output export with range-status validation\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 "
                 "candidate; GPU event timing uses direct-HIP byte-limb pack/export labels plus one candidate "
                 "operation-group label and this path is not public or AUTO-selected\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64 && selected_backend_kind == RNS8_BACKEND_HIP_DIRECT) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP wrap64 tiled byte-limb "
                 "path; GPU event timing uses wrap64-specific tiled byte-GEMM/export labels plus "
                 "current rns_gemm/crt_export aggregate phase labels\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the CPU wrap64 byte-limb reference; "
                 "no GPU event timing is requested for this backend\",\n";
  } else if (finite_benchmark_semantics(args.semantics)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for persistent finite-u8 packing, finite residue "
                 "GEMM, and canonical uint8 export with an explicit benchmark modulus\",\n";
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, "
                 "and fixed-width little-endian limb export; GPU event timing names exact-wide export operation "
                 "groups when backend hooks are available";
    if (!exact_wide_export_status_check_required(args)) {
      std::cout << "; the selected limb count covers the backend 192-bit reconstruction width, so exact-wide "
                   "range-status memset and status D2H are elided and reported as zero-valued event phases";
    }
    std::cout << "\",\n";
  } else if (adaptive_applied && info.backend == RNS8_BACKEND_CPU_REFERENCE) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the CPU adaptive per-tile bounded "
                 "reference path; no GPU event timing is requested for this backend\",\n";
  } else if (adaptive_applied && accelerator_deep_kernel_events) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator "
                 "path; GPU event timing combines direct-HIP pack/export labels with one accelerator GEMM "
                 "operation-group label\",\n";
  } else if (adaptive_applied && gpu_events_available) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP adaptive per-tile bounded "
                 "correctness path; GPU event timing aggregates all selected-prefix tiled GEMM launches and tiled "
                 "CRT export work into backend operation-group labels\",\n";
  } else if (adaptive_applied) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator path; "
                 "GPU event timing was not captured for this selected backend/path\",\n";
  } else if (info.backend == RNS8_BACKEND_HIPBLASLT) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the hipBLASLt baseline path; GPU event timing "
                 "separates persistent RNS packing, hipBLASLt INT8-to-INT32 matmul, separate centered-residue "
                 "reduction, and CRT export operation groups\",\n";
  } else {
    std::cout << "  \"timing_note\": \"host wall-clock timings; direct-HIP calls include current backend "
                 "synchronization, first-use persistent buffer allocation, copies, kernel launches, fused reduction, "
                 "and GPU bounded export when using HIP\",\n";
  }
