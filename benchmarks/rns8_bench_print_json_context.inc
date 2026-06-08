  const double avg_pack_us = average(result.samples.pack_us);
  const bool pack_a_timing_available = !result.samples.pack_a_us.empty();
  const bool pack_b_timing_available = !result.samples.pack_b_us.empty();
  const double avg_pack_a_us = pack_a_timing_available ? average(result.samples.pack_a_us) : 0.0;
  const double avg_pack_b_us = pack_b_timing_available ? average(result.samples.pack_b_us) : 0.0;
  const double avg_gemm_us = average(result.samples.gemm_us);
  const double avg_export_us = average(result.samples.export_us);
  const double avg_end_to_end_us = average(result.samples.end_to_end_us);
  const uint32_t prefix = benchmark_prefix(args);
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  const int64_t output_ld = output_logical_ld(args);
  const bool global_bound_scan_available = result.global_bound_scan_available;
  const uint32_t residue_planes_skipped = prefix > selected_prefix ? prefix - selected_prefix : 0;
  const double residue_plane_skip_fraction =
      prefix == 0 ? 0.0 : static_cast<double>(residue_planes_skipped) / static_cast<double>(prefix);
  const bool adaptive_applied = adaptive_execution_applied(args, info, result);
  const bool per_modulus_estimate_applicable =
      selected_prefix > 0 && args.bound_mode != BoundMode::PerTile && !adaptive_applied && !args.oneshot &&
      !args.vector_alu_baseline && !runtime_vector_alu_backend(args) && !vector_to_rns_chain_requested(args) &&
      !grouped_task_executor_requested(args) && !hip_graph_replay_requested(args) &&
      !result.streaming_overlap_executed;
  const double avg_per_modulus_gemm_estimate_us =
      per_modulus_estimate_applicable ? avg_gemm_us / static_cast<double>(selected_prefix) : avg_gemm_us;
  const bool gpu_events_available = gpu_event_timing_available(args, result);
  const rns8_backend_kind selected_backend_kind = selected_backend_for_events(args, result);
  const bool use_prepacked_b_cache = uses_runtime_b_prepack_cache(result);
  const bool chain_residue_output = residue_current_output_mode(args);
  const bool chain_final_export = residue_chain_final_export_requested(args);
  const bool chain_independent_final_export = residue_chain_independent_final_export_requested(args);
  const bool host_batch = host_api_batch_requested(args);
  const bool grouped_dispatch = grouped_dispatch_requested(args);
  const uint32_t task_count = measured_task_count(args);
  const double task_count_denominator = static_cast<double>(task_count);
  const auto event_phase_order =
      gpu_events_available ? gpu_event_phase_order(args, result, selected_backend_kind, use_prepacked_b_cache)
                           : std::vector<std::string>{};
  const bool wrap64_rocwmma_candidate_events = gpu_events_available && args.wrap64_rocwmma_candidate;
  const bool wrap64_hip_events =
      gpu_events_available && args.semantics == BenchSemantics::WrapU64Mod2_64 && !args.wrap64_rocwmma_candidate;
  const bool adaptive_hip_events = gpu_events_available && adaptive_applied;
  const char* selected_backend = selected_backend_name(args, info, &result);
  const std::string selected_backend_string(selected_backend);
  const bool hipblaslt_events = gpu_events_available && selected_backend_string == "hipblaslt";
  const bool accelerator_events =
      gpu_events_available && (selected_backend_string == "ck" || selected_backend_string == "rocwmma");
  const bool finite_accelerator_operation_group_events =
      accelerator_events && finite_benchmark_semantics(args.semantics);
  const bool accelerator_deep_kernel_events = accelerator_events && !finite_accelerator_operation_group_events;
  const bool vector_alu_events = gpu_events_available && selected_backend_string == "hip-vector-alu-int64";
  const bool oneshot_hip_events =
      gpu_events_available && args.oneshot && selected_backend_kind == RNS8_BACKEND_HIP_DIRECT;
  const bool oneshot_resident_fallback_events =
      gpu_events_available &&
      direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend_kind);
  const bool native_to_rns_bridge_events =
      gpu_events_available && native_to_rns_bridge_path(args, result, selected_backend_kind);
  const bool vector_to_rns_chain_events =
      gpu_events_available && vector_to_rns_chain_path(args, result, selected_backend_kind);
  const bool streaming_overlap_events = gpu_events_available && result.streaming_overlap_executed;
  const bool all_zero_direct_hip_pack_elided =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend_kind);
  const char* gpu_event_reason = "backend_has_no_gpu_event_hooks";
  const char* gpu_event_status = "not_requested_for_selected_backend";
  const char* gpu_event_scope = "null";
  const char* gpu_event_caveat = "null";
  if (gpu_events_available) {
    gpu_event_status = "available";
    gpu_event_scope = "\"direct_hip_default_stream_backend_operation_groups\"";
    gpu_event_caveat =
        "\"HIP event timings record backend default-stream operation groups only; host wall-clock timings remain "
        "required for CPU scheduling overhead, API dispatch, allocations, and any synchronous host-side copy overhead "
        "not represented on the HIP stream\"";
    if (streaming_overlap_events) {
      gpu_event_reason = "captured_by_direct_hip_streaming_overlap_hooks";
      gpu_event_scope = "\"direct_hip_streaming_overlap_multistream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record benchmark-owned nonblocking pack, GEMM, and export streams with explicit "
          "dependency events. The end_to_end timing samples are per-repeat pipeline throughput from the measured "
          "batch wall clock, not a serial sum of pack, GEMM, and export phase timings\"";
    } else if (chain_residue_output) {
      gpu_event_reason = "captured_by_residue_current_chain_backend_hooks";
      if (hipblaslt_events) {
        gpu_event_scope = "\"hipblaslt_baseline_default_stream_backend_operation_groups\"";
      } else if (accelerator_events) {
        gpu_event_scope = "\"accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export\"";
      } else {
        gpu_event_scope = "\"direct_hip_default_stream_backend_operation_groups\"";
      }
      if (hipblaslt_events) {
        gpu_event_caveat =
            "\"HIP event timings record per-repeat pack operation groups and the chained hipBLASLt rns_gemm "
            "operation groups that keep intermediate outputs resident in RNS form; hipBLASLt library operations use "
            "a benchmark-only synchronization before stop-event recording to make library work event-visible; the "
            "final checksum-only logical export runs after measured repeats and is intentionally absent from "
            "gpu_event_phase_order\"";
      } else {
        gpu_event_caveat =
            "\"HIP event timings record per-repeat pack operation groups and the chained rns_gemm backend operation "
            "groups that keep intermediate outputs resident in RNS form; the final checksum-only logical export runs "
            "after measured repeats and is intentionally absent from gpu_event_phase_order\"";
      }
    } else if (oneshot_hip_events) {
      if (oneshot_resident_fallback_events) {
        gpu_event_reason = "captured_by_direct_hip_oneshot_resident_fallback_api_hooks";
        gpu_event_scope = "\"direct_hip_oneshot_resident_fallback_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record the public bounded one-shot API's resident fallback pack, direct-HIP GEMM "
            "kernel group, and logical export operation groups; host wall-clock timings remain required for plan "
            "creation, transient matrix/workspace allocation, API dispatch, CPU scheduling, teardown, and synchronous "
            "host-side overhead not represented on the HIP stream\"";
      } else {
        gpu_event_reason = "captured_by_direct_hip_oneshot_api_hooks";
        gpu_event_scope = "\"direct_hip_oneshot_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record the public one-shot API's native input H2D copies, direct-HIP GEMM "
            "kernel group, and logical export operation groups; host wall-clock timings remain required for plan "
            "creation, transient allocations, API dispatch, CPU scheduling, teardown, and synchronous host-side "
            "overhead not represented on the HIP stream\"";
      }
    } else if (native_to_rns_bridge_events) {
      gpu_event_reason = "captured_by_direct_hip_native_to_rns_bridge_hooks";
      gpu_event_scope = "\"direct_hip_native_to_rns_bridge_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP pack/export operation groups plus the forced native-to-RNS "
          "device conversion kernels inside rns_gemm; host wall-clock timings remain required for API dispatch, "
          "CPU scheduling, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (vector_to_rns_chain_events) {
      gpu_event_reason = "captured_by_vector_native_to_direct_rns_chain_hooks";
      if (vector_to_rns_chain_host_repack_control_requested(args)) {
        gpu_event_scope = "\"direct_hip_vector_native_host_repack_chain_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record vector-ALU producer packing and GEMM events, vector native host export, "
            "Direct-HIP host repack of the produced intermediate, Direct-HIP second-input pack/export operation "
            "groups, and the Direct-HIP consumer RNS GEMM operation group; host wall-clock timings remain required "
            "for API dispatch, CPU scheduling, allocations, and synchronous host-side overhead not represented on "
            "the HIP stream\"";
      } else {
        gpu_event_scope = "\"direct_hip_vector_native_to_rns_chain_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record vector-ALU producer packing and GEMM events, Direct-HIP second-input "
            "pack/export operation groups, the native-to-RNS device materialization kernel, and the Direct-HIP "
            "consumer RNS GEMM operation group; host wall-clock timings remain required for API dispatch, CPU "
            "scheduling, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
      }
    } else if (hipblaslt_events) {
      gpu_event_reason = "captured_by_hipblaslt_backend_hooks";
      gpu_event_scope = "\"hipblaslt_baseline_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record hipBLASLt baseline operation groups; hipBLASLt library operations use a "
          "benchmark-only synchronization before stop-event recording to make library work event-visible. Host "
          "wall-clock timings remain required for API dispatch, descriptor setup, allocations, and synchronous "
          "host-side overhead not represented on the HIP stream\"";
    } else if (wrap64_rocwmma_candidate_events) {
      gpu_event_reason = "captured_by_internal_rocwmma_wrap64_candidate_hooks";
      gpu_event_scope = "\"rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP byte-limb pack/export operation groups plus one internal rocWMMA "
          "wrap64 byte-GEMM36 candidate operation group; host wall-clock timings remain required for CPU scheduling "
          "overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (wrap64_hip_events) {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
      gpu_event_scope = "\"direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record backend default-stream operation groups only; wrap64 uses a tiled byte-limb "
          "GEMM kernel and current rns_gemm/crt_export aggregate phase labels; host wall-clock timings remain "
          "required for CPU scheduling overhead, API dispatch, allocations, and synchronous host-side overhead not "
          "represented on the HIP stream\"";
    } else if (finite_accelerator_operation_group_events) {
      gpu_event_reason = "captured_by_accelerator_backend_operation_group_hooks";
      gpu_event_scope = "\"accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP finite-u8 pack/export operation groups plus one accelerator GEMM "
          "operation group; host wall-clock timings remain required for API dispatch, CPU scheduling, allocations, "
          "and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (accelerator_deep_kernel_events) {
      gpu_event_reason = "captured_by_accelerator_backend_deep_kernel_hooks";
      gpu_event_scope = "\"accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP pack/export operation groups plus accelerator GEMM group, aggregate "
          "kernel-class labels, and selected-prefix labels, with synchronized host fallback only if a platform event "
          "read fails; host wall-clock timings remain required for API dispatch, CPU scheduling, allocations, and "
          "synchronous host-side overhead not represented on the HIP stream\"";
    } else if (vector_alu_events) {
      gpu_event_reason = "captured_by_vector_alu_native_backend_hooks";
      gpu_event_scope = "\"vector_alu_default_stream_native_int64_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record benchmark/API vector-ALU native-buffer operation groups; host wall-clock "
          "timings remain required for CPU staging, range checks, API dispatch, allocations, and synchronous "
          "host-side overhead not represented on the HIP stream\"";
    } else if (adaptive_hip_events) {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
      gpu_event_scope = "\"direct_hip_bounded_adaptive_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record backend default-stream operation groups only; adaptive bounded captures "
          "aggregate all selected-prefix tile launches and tiled export kernels rather than exposing per-tile or "
          "per-prefix timings; host wall-clock timings remain required for scheduling overhead, API dispatch, "
          "allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
    }
  } else if (result.gpu_events.requested) {
    gpu_event_reason = "backend_event_capture_incomplete";
    gpu_event_status = "unavailable_missing_expected_events";
  } else if (result.hip_graph_replay_requested) {
    gpu_event_reason = "hip_graph_replay_wall_clock_only";
    gpu_event_status = result.hip_graph_replay_used ? "not_requested_graph_replay" : "not_requested_graph_replay_unavailable";
  } else if (chain_residue_output) {
    gpu_event_reason = "selected_chain_backend_has_no_gpu_event_hooks";
    gpu_event_status = "not_requested_for_selected_chain_backend";
  }

  std::cout << "{\n";
