  std::cout << "  \"timing_metadata\": {\n";
  std::cout << "    \"unit\": \"microseconds\",\n";
  std::cout << "    \"source\": \"std::chrono::steady_clock\",\n";
  std::cout << "    \"source_scope\": \"host_wall_clock\",\n";
  std::cout << "    \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args, result) << "\",\n";
  std::cout << "    \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  std::cout << "    \"host_api_batch_enabled\": " << (host_batch ? "true" : "false") << ",\n";
  std::cout << "    \"host_api_batch_size\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"grouped_dispatch_enabled\": " << (grouped_dispatch ? "true" : "false") << ",\n";
  std::cout << "    \"grouped_dispatch_task_count\": " << args.grouped_dispatch_tasks << ",\n";
  std::cout << "    \"grouped_dispatch_execution_strategy\": \""
            << (grouped_dispatch
                    ? json_escape(result.grouped_dispatch_execution_strategy == "not_requested"
                                      ? "host_phase_loop_per_task_export"
                                      : result.grouped_dispatch_execution_strategy)
                    : "not_requested")
            << "\",\n";
  std::cout << "    \"grouped_dispatch_batched_export_enabled\": "
            << (grouped_dispatch && result.grouped_dispatch_batched_export_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"grouped_dispatch_device_output_slab_bytes\": "
            << (grouped_dispatch ? result.grouped_dispatch_device_output_slab_bytes : 0) << ",\n";
  std::cout << "    \"hip_graph_replay_enabled\": " << (result.hip_graph_replay_used ? "true" : "false") << ",\n";
  std::cout << "    \"hip_graph_replay_status\": \"" << json_escape(result.hip_graph_replay_status) << "\",\n";
  std::cout << "    \"hip_graph_replay_scope\": \"" << json_escape(result.hip_graph_replay_scope) << "\",\n";
  std::cout << "    \"hip_graph_capture_us\": " << result.hip_graph_capture_us << ",\n";
  std::cout << "    \"hip_graph_instantiate_us\": " << result.hip_graph_instantiate_us << ",\n";
  std::cout << "    \"hip_graph_total_launches\": " << result.hip_graph_launch_count << ",\n";
  std::cout << "    \"native_to_rns_bridge_forced\": "
            << (native_to_rns_bridge_requested(args) ? "true" : "false") << ",\n";
  std::cout << "    \"vector_to_rns_chain\": "
            << (vector_to_rns_chain_requested(args) ? "true" : "false") << ",\n";
  std::cout << "    \"vector_to_rns_chain_control_mode\": ";
  if (vector_to_rns_chain_requested(args)) {
    std::cout << "\""
              << (vector_to_rns_chain_host_repack_control_requested(args)
                      ? "host_export_repack_control"
                      : "fused_device_native_to_rns")
              << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"vector_to_rns_chain_producer_backend\": "
            << (vector_to_rns_chain_requested(args) ? "\"hip-vector-alu-int64\"" : "null") << ",\n";
  std::cout << "    \"vector_to_rns_chain_consumer_backend\": "
            << (vector_to_rns_chain_requested(args) ? "\"hip-direct\"" : "null") << ",\n";
  std::cout << "    \"vector_to_rns_chain_consumer_k\": ";
  if (vector_to_rns_chain_requested(args)) {
    std::cout << args.n;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"residue_chain_final_export\": " << (chain_final_export ? "true" : "false") << ",\n";
  std::cout << "    \"residue_chain_independent_final_export\": "
            << (chain_independent_final_export ? "true" : "false") << ",\n";
  std::cout << "    \"pack_layout\": \"" << json_escape(benchmark_pack_layout(args, result)) << "\",\n";
  std::cout << "    \"fusion_mode\": \"" << benchmark_fusion_mode(args) << "\",\n";
  std::cout << "    \"residue_group_width\": " << benchmark_residue_group_width(args) << ",\n";
  std::cout << "    \"residue_group_layout\": \"" << benchmark_residue_group_layout(args) << "\",\n";
  std::cout << "    \"generated_reducer_identity\": \"" << json_escape(generated_reducer_identity(args, result)) << "\",\n";
  std::cout << "    \"direct_hip_export_staging_policy\": \""
            << direct_hip_export_staging_policy(args, selected_backend_kind) << "\",\n";
  std::cout << "    \"direct_hip_pinned_export_staging_threshold_bytes\": "
            << kDirectHipPinnedExportStagingThresholdBytes << ",\n";
  std::cout << "    \"benchmark_output_destination_layout\": \"" << output_destination_layout(args) << "\",\n";
  std::cout << "    \"benchmark_output_logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"benchmark_output_ld_padding\": " << args.output_ld_padding << ",\n";
  std::cout << "    \"prepack_reuse_operands\": ";
  print_string_array(prepack_reuse_operands(args));
  std::cout << ",\n";
  std::cout << "    \"prepack_reuse_strategy\": \"" << prepack_reuse_strategy_name(result.prepack_reuse_strategy)
            << "\",\n";
  std::cout << "    \"gpu_event_timing\": " << (gpu_events_available ? "true" : "false") << ",\n";
  std::cout << "    \"gpu_event_timing_reason\": \"" << gpu_event_reason << "\",\n";
  std::cout << "    \"gpu_event_timing_status\": \"" << gpu_event_status << "\",\n";
  std::cout << "    \"gpu_event_timing_source\": "
            << (gpu_events_available ? "\"hipEventElapsedTime\"" : "null") << ",\n";
  std::cout << "    \"gpu_event_timing_source_scope\": " << gpu_event_scope << ",\n";
  std::cout << "    \"gpu_event_timing_caveat\": " << gpu_event_caveat << ",\n";
  if (!gpu_events_available && !result.gpu_events.unavailable_reasons.empty()) {
    std::cout << "    \"gpu_event_timing_unavailable_reasons\": ";
    print_string_array(result.gpu_events.unavailable_reasons);
    std::cout << ",\n";
  }
  std::cout
      << "    \"phase_order\": [";
  if (global_bound_scan_available) {
    std::cout << "\"global_bound_scan\", ";
  }
  std::cout << "\"planning\", \"scheduling\", ";
  if (result.tile_bound_scan_available) {
    std::cout << "\"tile_bound_scan\", ";
  }
  std::cout << "\"matrix_alloc\", \"pack\", \"rns_gemm\", \"crt_export\", \"end_to_end\"],\n";
  std::cout << "    \"gpu_event_phase_order\": ";
  if (gpu_events_available) {
    print_string_array(event_phase_order);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"phase_notes\": {\n";
  if (global_bound_scan_available) {
    std::cout << "      \"global_bound_scan\": \"one-time exact seeded input prepass that computes row/column absolute-summary global bounds before plan creation\",\n";
  }
  if (args.oneshot) {
    std::cout << "      \"planning\": \"one-time metadata-only rns8_create_plan timing; each measured one-shot API call performs its own internal setup inside rns_gemm/end_to_end\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"planning\": \"one-time benchmark-owned metadata initialization for the internal rocWMMA wrap64 candidate\",\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"planning\": \"one-time rns8_create_plan schedule validation for the same bounded semantic contract\",\n";
  } else {
    std::cout << "      \"planning\": \"one-time rns8_create_plan plus rns8_create_workspace host timing\",\n";
  }
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"scheduling\": \"one-time fixed 16x16 WMMA candidate schedule derivation from the matrix shape\",\n";
  } else {
    std::cout << "      \"scheduling\": \"one-time rns8_get_plan_schedule_info host timing\",\n";
  }
  if (result.tile_bound_scan_available) {
    std::cout << "      \"tile_bound_scan\": \"one-time exact seeded input prepass that computes per-output-tile bounds before plan creation\",\n";
  }
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned compact byte-limb HIP device buffer allocation host timing\",\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned HIP device buffer allocation host timing\",\n";
  } else if (args.oneshot) {
    std::cout << "      \"matrix_alloc\": \"zero-valued external phase; transient API allocations, if any, are inside the measured one-shot call\",\n";
  } else if (grouped_dispatch) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned allocation of one resident A/B/C matrix triplet and workspace per grouped dispatch task\",\n";
  } else if (host_batch) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned allocation of one resident A/B/C matrix triplet and workspace per host API batch task\",\n";
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned native int8 A/B HIP buffers plus resident RNS output matrix allocation for the experimental width-3 residue-channel fusion comparison\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned native int8 A/B HIP buffers plus resident RNS output matrix allocation host timing\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time vector-ALU producer matrix allocation plus Direct-HIP RNS consumer matrix allocation host timing\",\n";
  } else {
    std::cout << "      \"matrix_alloc\": \"one-time persistent matrix allocation host timing\",\n";
  }
  if (hip_graph_replay_full_pack_export_used(result)) {
    std::cout << "      \"pack\": \"captured inside one HIP Graph launch; raw per-phase GPU events are unavailable for graph replay so raw_timings_us.pack is zero and setup-inclusive comparisons use end_to_end plus graph setup\",\n";
    if (result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export") {
      std::cout << "      \"rns_gemm\": \"captured inside one HIP Graph launch with the Direct-HIP finite-u8 pack and canonical export work\",\n";
      std::cout << "      \"crt_export\": \"captured inside one HIP Graph launch including finite export kernel and output D2H\",\n";
    } else if (result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export") {
      std::cout << "      \"rns_gemm\": \"captured inside one HIP Graph launch with Direct-HIP wrap64 byte-limb pack and low64 export work\",\n";
      std::cout << "      \"crt_export\": \"captured inside one HIP Graph launch including wrap64 low64 export kernel and output D2H\",\n";
    } else {
      std::cout << "      \"rns_gemm\": \"captured inside one HIP Graph launch with the Direct-HIP bounded pack and export work\",\n";
      std::cout << "      \"crt_export\": \"captured inside one HIP Graph launch including status memset, export kernel, status D2H, and output D2H\",\n";
    }
  } else if (hip_graph_replay_requested(args)) {
    std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before HIP Graph capture\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one hipGraphLaunch plus stream synchronization containing "
              << args.residue_chain_length
              << " captured Direct-HIP resident RNS GEMM launches\",\n";
    std::cout << "      \"crt_export\": \"zero-valued per-repeat phase; residue-current graph replay defers host logical export until one final checksum export after measured repeats\",\n";
  } else if (chain_residue_output) {
    if (args.reuse_packed_inputs) {
      if (reuses_all_packed_inputs(args)) {
        std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before warmups\",\n";
      } else {
        std::cout << "      \"pack\": \"per-repeat host timing for packing " << per_repeat_pack_operand_text(args)
                  << "; " << prepack_reuse_operand_text(args)
                  << " was packed once into a persistent RNS matrix before warmups\",\n";
      }
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    }
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for " << args.residue_chain_length
              << " chained rns8_gemm_rns calls that keep the intermediate output resident in RNS form\",\n";
    std::cout << "      \"crt_export\": \"zero-valued per-repeat phase; residue-current chain mode defers host logical export until one final checksum export after measured repeats\",\n";
  } else if (chain_independent_final_export) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing original A/B plus repacking each non-final host intermediate output into the next RNS input\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for " << args.residue_chain_length
              << " independent rns8_gemm_rns calls separated by host export and repack materialization\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for exporting/reconstructing every intermediate and final chained logical output inside the measured repeat\",\n";
  } else if (chain_final_export) {
    if (args.reuse_packed_inputs) {
      if (reuses_all_packed_inputs(args)) {
        std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before warmups\",\n";
      } else {
        std::cout << "      \"pack\": \"per-repeat host timing for packing " << per_repeat_pack_operand_text(args)
                  << "; " << prepack_reuse_operand_text(args)
                  << " was packed once into a persistent RNS matrix before warmups\",\n";
      }
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    }
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for " << args.residue_chain_length
              << " chained rns8_gemm_rns calls that keep intermediate outputs resident in RNS form\",\n";
    if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat host timing for exporting the final chained exact-wide limb output inside the measured repeat\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat host timing for exporting/reconstructing the final chained logical output inside the measured repeat\",\n";
    }
  } else if (args.oneshot) {
    std::cout << "      \"pack\": \"zero-valued external phase; native input copies and any backend-local transformation are inside the measured one-shot API call\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one complete public one-shot API call\",\n";
    std::cout << "      \"crt_export\": \"zero-valued external phase; logical output export is inside the measured one-shot API call\",\n";
  } else if (grouped_dispatch) {
    std::cout << "      \"pack\": \"per-repeat aggregate grouped-dispatch host timing for packing A and B for "
              << task_count
              << " independent resident tasks through one shared plan\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat aggregate grouped-dispatch host timing for "
              << task_count
              << " independent resident rns8_gemm calls through one shared plan and per-task workspace\",\n";
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for canonical finite-u8 export of every grouped task\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      if (result.grouped_dispatch_batched_export_enabled) {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for fixed-width exact-wide limb export kernels into one shared device slab plus one compact D2H copy for every grouped task\",\n";
      } else {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for fixed-width exact-wide limb export of every grouped task\",\n";
      }
    } else {
      if (result.grouped_dispatch_batched_export_enabled) {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for bounded CRT export kernels into one shared device slab plus one compact D2H copy for every grouped task\",\n";
      } else {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for CRT export/reconstruction of every grouped task\",\n";
      }
    }
  } else if (host_batch) {
    std::cout << "      \"pack\": \"per-repeat aggregate host timing for packing A and B for "
              << task_count
              << " independent resident tasks through one shared plan\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat aggregate host timing for "
              << task_count
              << " independent resident rns8_gemm calls through one shared plan and per-task workspace\",\n";
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for canonical finite-u8 export of every batch task\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for fixed-width exact-wide limb export of every batch task\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for CRT export/reconstruction of every batch task\",\n";
    }
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for copying uniform-small A and B into benchmark-owned native int8 HIP buffers for the experimental width-3 residue-channel fusion path\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the benchmark-only direct-HIP residue-channel fusion comparison kernel group over first-prefix9 width-3 groups\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for copying uniform-small A and B into benchmark-owned native int8 HIP buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the direct-HIP prefix-9 uniform-small int8 A/B colpair GEMM kernel group writing resident RNS residues\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "      \"pack\": \"zero-valued per-repeat phase; exact per-tile input scanning proved every output tile zero, so direct-HIP does not pack or read A/B for this capture\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for direct-HIP all-zero resident RNS output materialization\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for exporting the already-zero direct-HIP output\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    if (args.reuse_packed_b) {
      std::cout << "      \"pack\": \"per-repeat host timing for copying vector producer A/B into native HIP buffers; the Direct-HIP consumer B input was packed once before warmups\",\n";
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for copying vector producer A/B into native HIP buffers and packing the second Direct-HIP RNS input\",\n";
    }
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for vector-ALU native GEMM, native-to-RNS materialization into Direct-HIP storage, and Direct-HIP consumer RNS GEMM\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction of the final Direct-HIP consumer output\",\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into direct-HIP matrices with both RNS residues and native device buffers populated\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for forced native-to-RNS device conversion of A and B followed by direct-HIP rns8_gemm_rns\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
  } else if (args.reuse_packed_inputs) {
    if (reuses_all_packed_inputs(args)) {
      std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent matrices before warmups\",\n";
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for packing " << per_repeat_pack_operand_text(args)
                << "; " << prepack_reuse_operand_text(args)
                << " was packed once into a persistent matrix before warmups\",\n";
    }
    if (args.wrap64_rocwmma_candidate) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate against reused compact byte-limb inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for direct-HIP low-64-bit byte-limb export\",\n";
    } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_wrap_u64 against reused byte-limb inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for low-64-bit rns8_export_wrap_u64\",\n";
    } else if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_finite_u8 against reused finite-u8 inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for canonical uint8 rns8_export_finite_u8\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns against reused exact-wide persistent RNS inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export\",\n";
    } else {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns against reused persistent RNS inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
    }
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"pack\": \"per-repeat host timing for raw bounded inputs copied to benchmark-owned HIP buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one exact 192-bit-limb vector-ALU output kernel\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for range-status D2H plus direct logical output D2H\",\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into native int64/uint64 HIP device matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns dispatch to one exact 192-bit-limb vector-ALU output kernel\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for native output D2H plus range validation\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"pack\": \"per-repeat host timing for direct-HIP packing of A and B into compact byte-limb device buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for direct-HIP low-64-bit byte-limb export\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent byte-limb matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_wrap_u64\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for low-64-bit rns8_export_wrap_u64\",\n";
  } else if (finite_benchmark_semantics(args.semantics)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent finite-u8 matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_finite_u8\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for canonical uint8 rns8_export_finite_u8\",\n";
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into exact-wide persistent RNS matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
    if (exact_wide_export_status_check_required(args)) {
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export with range-status memset and status D2H\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export; selected limb width covers backend 192-bit reconstruction so range-status memset and status D2H are elided\",\n";
    }
  } else {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
  }
  if (hip_graph_replay_full_pack_export_used(result)) {
    if (result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export") {
      std::cout << "      \"end_to_end\": \"measured duration for one hipGraphLaunch plus stream synchronization containing the complete finite-u8 pack, modular GEMM, canonical export, and D2H path; excludes one-time hip_graph_replay.capture_us and hip_graph_replay.instantiate_us only\"\n";
    } else if (result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export") {
      std::cout << "      \"end_to_end\": \"measured duration for one hipGraphLaunch plus stream synchronization containing the complete wrap64 byte-limb pack, byte-GEMM36, low64 export, and D2H path; excludes one-time hip_graph_replay.capture_us and hip_graph_replay.instantiate_us only\"\n";
    } else {
      std::cout << "      \"end_to_end\": \"measured duration for one hipGraphLaunch plus stream synchronization containing the complete bounded pack, RNS GEMM, export, status, and D2H path; excludes one-time hip_graph_replay.capture_us and hip_graph_replay.instantiate_us only\"\n";
    }
  } else if (hip_graph_replay_requested(args)) {
    std::cout << "      \"end_to_end\": \"same measured duration as rns_gemm for one hipGraphLaunch plus stream synchronization; excludes one-time prepack_setup_us, hip_graph_replay.capture_us, hip_graph_replay.instantiate_us, and the final checksum-only logical export\"\n";
  } else if (chain_residue_output) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only logical export";
    if (args.reuse_packed_inputs) {
      std::cout << " and one-time prepack_setup_us";
    }
    std::cout << "\"\n";
  } else if (chain_independent_final_export) {
    std::cout << "      \"end_to_end\": \"per-repeat pack of original inputs plus intermediate repacks, "
              << args.residue_chain_length
              << " independent rns_gemm calls, and every intermediate/final logical export host timing\"\n";
  } else if (chain_final_export) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus " << args.residue_chain_length
              << " chained rns_gemm calls plus final logical export host timing";
    if (args.reuse_packed_inputs) {
      std::cout << "; excludes one-time prepack_setup_us";
    }
    std::cout << "\"\n";
  } else if (args.oneshot) {
    std::cout << "      \"end_to_end\": \"same measured duration as rns_gemm for one complete public one-shot API call\"\n";
  } else if (grouped_dispatch) {
    std::cout << "      \"end_to_end\": \"per-repeat aggregate grouped-dispatch pack plus rns_gemm plus crt_export host timing for "
              << task_count << " independent grouped tasks\"\n";
  } else if (host_batch) {
    std::cout << "      \"end_to_end\": \"per-repeat aggregate pack plus rns_gemm plus crt_export host timing for "
              << task_count << " independent batch tasks\"\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "      \"end_to_end\": \"per-repeat direct-HIP all-zero output materialization plus export host timing; pack is intentionally elided by the trusted all-zero schedule\"\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"end_to_end\": \"per-repeat vector producer pack";
    if (!args.reuse_packed_b) {
      std::cout << " plus Direct-HIP input pack";
    }
    std::cout << ", vector GEMM, native-to-RNS materialization, Direct-HIP consumer GEMM, and final CRT export host timing";
    if (args.reuse_packed_b) {
      std::cout << "; excludes one-time prepack_setup_us for the Direct-HIP consumer B input";
    }
    std::cout << "\"\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus native-to-RNS input conversion plus direct-HIP rns_gemm plus crt_export host timing\"\n";
  } else if (args.reuse_packed_inputs) {
    if (reuses_all_packed_inputs(args)) {
      std::cout << "      \"end_to_end\": \"per-repeat rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us\"\n";
    } else {
      std::cout << "      \"end_to_end\": \"per-repeat pack of non-reused input plus rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us\"\n";
    }
  } else {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus rns_gemm plus crt_export host timing\"\n";
  }
  std::cout << "    },\n";
  std::cout << "    \"phase_availability\": {\n";
  if (global_bound_scan_available) {
    std::cout << "      \"global_bound_scan\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"global_bound_scan\",\n";
    std::cout << "        \"scope\": \"input_row_column_abs_summary\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around seeded input row/column absolute-summary bound discovery before plan creation\"\n";
    std::cout << "      },\n";
  }
  std::cout << "      \"scheduling\": {\n";
  std::cout << "        \"timed\": true,\n";
  std::cout << "        \"timing_key\": \"scheduling\",\n";
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "        \"scope\": \"benchmark_static_wrap64_rocwmma_candidate_schedule\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization\"\n";
  } else {
    std::cout << "        \"scope\": \"one_time_schedule_info_query\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around rns8_get_plan_schedule_info\"\n";
  }
  std::cout << "      },\n";
  if (result.tile_bound_scan_available) {
    std::cout << "      \"tile_bound_scan\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"tile_bound_scan\",\n";
    std::cout << "        \"scope\": \"exact_seeded_input_prepass\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around exact per-output-tile bound computation before plan creation\"\n";
    std::cout << "      },\n";
  }
  std::cout << "      \"prepack_setup\": {\n";
  std::cout << "        \"timed\": " << (result.prepack_setup_available ? "true" : "false") << ",\n";
  std::cout << "        \"timing_key\": "
            << (result.prepack_setup_available ? "\"prepack_setup_us\"" : "null") << ",\n";
  std::cout << "        \"scope\": \""
            << (result.prepack_setup_available ? "one_time_before_warmups" : "not_requested_per_repeat_repack")
            << "\",\n";
  std::cout << "        \"reason\": \""
            << (result.prepack_setup_available
                    ? (use_prepacked_b_cache
                           ? "packed B into persistent matrix storage, materialized a reusable rocWMMA B cache before warmups, and reused that cache for every measured repeat"
                           : "prepacked " + prepack_reuse_operand_text(args) +
                                 " once before warmups and reused for every measured repeat")
                    : (args.oneshot
                           ? "one-shot benchmark mode does not expose a benchmark-side prepack phase"
                           : "benchmark mode packs A and B inside every measured repeat"))
            << "\"\n";
  std::cout << "      },\n";
  if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"native_to_rns_bridge\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"rns_gemm\",\n";
    std::cout << "        \"scope\": \"device_native_to_rns_conversion_inside_rns_gemm\",\n";
    std::cout << "        \"reason\": \"measured as explicit native_i64_to_rns_kernel or native_u64_to_rns_kernel GPU event phases inside rns_gemm\"\n";
    std::cout << "      },\n";
  }
  if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"vector_to_rns_chain\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"rns_gemm\",\n";
    std::cout << "        \"scope\": \""
              << (vector_to_rns_chain_host_repack_control_requested(args)
                      ? "vector_native_output_host_export_repack_to_direct_rns_consumer"
                      : "vector_native_output_to_direct_rns_consumer")
              << "\",\n";
    std::cout << "        \"reason\": \""
              << (vector_to_rns_chain_host_repack_control_requested(args)
                      ? "measured as vector_alu_i64_kernel or vector_alu_u64_kernel, vector_alu_output_d2h, vector_to_rns_host_repack_a, and rns_gemm_kernel_group GPU event phases inside rns_gemm"
                      : "measured as vector_alu_i64_kernel or vector_alu_u64_kernel, native_i64_to_rns_kernel or native_u64_to_rns_kernel, and rns_gemm_kernel_group GPU event phases inside rns_gemm")
              << "\"\n";
    std::cout << "      },\n";
  }
  std::cout << "      \"reduction\": {\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_wrap64_byte_limb\",\n";
    std::cout << "        \"reason\": \"strict wrap64 byte-limb captures do not use centered RNS residue reduction\"\n";
  } else if (std::string(selected_backend) == "hipblaslt") {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"separate_hipblaslt_i32_scratch_residue_reduce\",\n";
    std::cout << "        \"reason\": \"hipBLASLt baseline GEMM writes INT32 scratch and runs a separate centered-residue reduction inside rns_gemm\"\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_direct_int64_export\",\n";
    std::cout << "        \"reason\": \"benchmark-only vector-ALU baseline computes exact logical outputs directly and does not use centered RNS residue reduction\"\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_native_vector_output\",\n";
    std::cout << "        \"reason\": \"runtime vector-ALU backend computes exact native int64/uint64 outputs directly and does not use centered RNS residue reduction\"\n";
  } else {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"fused_into_rns_gemm\",\n";
    std::cout << "        \"reason\": \"current CPU and direct-HIP RNS GEMM paths reduce to centered residues inside rns_gemm, so no separate reduction kernel timing exists\"\n";
  }
  std::cout << "      }\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  if (gpu_events_available) {
    print_gpu_event_timings(event_phase_order, result.gpu_events);
    print_gpu_event_timing_summary(event_phase_order, result.gpu_events);
  } else {
