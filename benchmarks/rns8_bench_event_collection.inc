bool backend_supports_gpu_event_capture(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_DIRECT || backend == RNS8_BACKEND_HIPBLASLT ||
         backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA ||
         backend == RNS8_BACKEND_AMDGPU_BUILTINS || backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

rns8_backend_kind selected_backend_for_events(const Args& args, const BenchmarkResult& result) {
  return result.backend_info_available ? result.backend_info.backend : args.backend;
}

bool gpu_event_capture_requested(const Args& args, rns8_backend_kind selected_backend) {
  if (hip_graph_replay_requested(args)) {
    return false;
  }
  return backend_supports_gpu_event_capture(selected_backend);
}

void add_unavailable_reason(GpuEventSamples& events, const std::string& reason) {
  events.complete = false;
  if (std::find(events.unavailable_reasons.begin(), events.unavailable_reasons.end(), reason) ==
      events.unavailable_reasons.end()) {
    events.unavailable_reasons.push_back(reason);
  }
}

void push_gpu_event_value(GpuEventSamples& events, const std::string& label, double value) {
  if (events.complete) {
    events.timings_us[label].push_back(value);
  }
}

double sum_event_label(
    GpuEventSamples& events,
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* phase,
    const char* label) {
  bool found = false;
  double total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      found = true;
      total += sample.microseconds;
    }
  }
  if (!found) {
    add_unavailable_reason(events, std::string(phase) + " missing backend HIP event label " + label);
  }
  return total;
}

double optional_event_label(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label) {
  double total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      total += sample.microseconds;
    }
  }
  return total;
}

bool sum_event_label_if_present(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label,
    double& total) {
  bool found = false;
  total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      found = true;
      total += sample.microseconds;
    }
  }
  return found;
}

std::vector<double> event_label_values(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label) {
  std::vector<double> values;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      values.push_back(sample.microseconds);
    }
  }
  return values;
}

double sum_event_values(const std::vector<double>& values) {
  double total = 0.0;
  for (const double value : values) {
    total += value;
  }
  return total;
}

double sum_alternating_event_values(const std::vector<double>& values, std::size_t offset) {
  double total = 0.0;
  for (std::size_t index = offset; index < values.size(); index += 2) {
    total += values[index];
  }
  return total;
}

bool collect_split_direct_hip_pack_gpu_events(
    const Args& args,
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    GpuEventSamples& events,
    std::size_t expected_values,
    const char* label_scope) {
  const bool finite = finite_benchmark_semantics(args.semantics);
  const char* h2d_label = finite ? "finite_pack_h2d" : "pack_h2d";
  const char* kernel_label = finite ? "finite_pack_kernel" : "pack_kernel";
  const char* pack_a_h2d_label = finite ? "finite_pack_a_h2d" : "pack_a_h2d";
  const char* pack_a_kernel_label = finite ? "finite_pack_a_kernel" : "pack_a_kernel";
  const char* pack_b_h2d_label = finite ? "finite_pack_b_h2d" : "pack_b_h2d";
  const char* pack_b_kernel_label = finite ? "finite_pack_b_kernel" : "pack_b_kernel";
  const auto h2d_values = event_label_values(samples, h2d_label);
  const auto kernel_values = event_label_values(samples, kernel_label);
  if (h2d_values.size() != expected_values || kernel_values.size() != expected_values) {
    add_unavailable_reason(
        events,
        std::string(label_scope) + " pack expected " + std::to_string(expected_values) + " " + h2d_label +
            " and " + kernel_label + " samples, got " + std::to_string(h2d_values.size()) + " and " +
            std::to_string(kernel_values.size()));
    return true;
  }

  const double a_h2d = sum_alternating_event_values(h2d_values, 0);
  const double b_h2d = sum_alternating_event_values(h2d_values, 1);
  const double a_kernel = sum_alternating_event_values(kernel_values, 0);
  const double b_kernel = sum_alternating_event_values(kernel_values, 1);
  const double h2d = sum_event_values(h2d_values);
  const double kernel = sum_event_values(kernel_values);
  if (events.complete) {
    push_gpu_event_value(events, pack_a_h2d_label, a_h2d);
    push_gpu_event_value(events, pack_a_kernel_label, a_kernel);
    push_gpu_event_value(events, pack_b_h2d_label, b_h2d);
    push_gpu_event_value(events, pack_b_kernel_label, b_kernel);
    push_gpu_event_value(events, h2d_label, h2d);
    push_gpu_event_value(events, kernel_label, kernel);
    push_gpu_event_value(events, "pack", h2d + kernel);
  }
  return true;
}

bool collect_direct_hip_host_batch_pack_gpu_events(
    const Args& args,
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    GpuEventSamples& events) {
  return collect_split_direct_hip_pack_gpu_events(
      args, samples, events, static_cast<std::size_t>(measured_task_count(args)) * 2u, "host-batch");
}

bool collect_direct_hip_grouped_dispatch_pack_gpu_events(
    const Args& args,
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    GpuEventSamples& events) {
  return collect_split_direct_hip_pack_gpu_events(args, samples, events, 2u, "grouped-dispatch");
}

void collect_pack_gpu_events(const Args& args, rns8_backend_kind selected_backend, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  if (args.sparse_a_4_to_2 && !args.sparse_a_4_to_2_dense_baseline &&
      (finite_benchmark_semantics(args.semantics) || bounded_benchmark_semantics(args.semantics))) {
    const double sparse_values_h2d = sum_event_label(events, samples, "pack", "sparse_a_values_h2d");
    const double sparse_indices_h2d = sum_event_label(events, samples, "pack", "sparse_a_indices_h2d");
    const bool finite = finite_benchmark_semantics(args.semantics);
    const char* dense_b_h2d_label = finite ? "finite_pack_h2d" : "pack_h2d";
    const char* dense_b_kernel_label = finite ? "finite_pack_kernel" : "pack_kernel";
    const double dense_b_h2d = sum_event_label(events, samples, "pack", dense_b_h2d_label);
    const double dense_b_kernel = sum_event_label(events, samples, "pack", dense_b_kernel_label);
    if (events.complete) {
      push_gpu_event_value(events, "sparse_a_values_h2d", sparse_values_h2d);
      push_gpu_event_value(events, "sparse_a_indices_h2d", sparse_indices_h2d);
      push_gpu_event_value(events, dense_b_h2d_label, dense_b_h2d);
      push_gpu_event_value(events, dense_b_kernel_label, dense_b_kernel);
      push_gpu_event_value(events, "pack", sparse_values_h2d + sparse_indices_h2d + dense_b_h2d + dense_b_kernel);
    }
    return;
  }
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const auto h2d_samples = event_label_values(samples, "residue_h2d_sync");
    std::size_t h2d_index = 0;
    const auto operand_h2d = [&](bool reused, const char* label) -> double {
      if (reused) {
        return 0.0;
      }
      const bool has_fallback = h2d_index < h2d_samples.size();
      const double fallback = has_fallback ? h2d_samples[h2d_index] : 0.0;
      ++h2d_index;
      double explicit_total = 0.0;
      if (sum_event_label_if_present(samples, label, explicit_total)) {
        return explicit_total;
      }
      if (has_fallback) {
        return fallback;
      }
      add_unavailable_reason(
          events, std::string("pack missing backend HIP event label ") + label + " or residue_h2d_sync");
      return 0.0;
    };
    const double a_h2d = operand_h2d(args.reuse_packed_a, "vector_alu_pack_a_h2d");
    const double b_h2d = operand_h2d(args.reuse_packed_b, "vector_alu_pack_b_h2d");
    if (events.complete) {
      push_gpu_event_value(events, "vector_alu_pack_a_h2d", a_h2d);
      push_gpu_event_value(events, "vector_alu_pack_b_h2d", b_h2d);
      push_gpu_event_value(events, "pack", a_h2d + b_h2d);
    }
    return;
  }
  if (grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    if (collect_direct_hip_grouped_dispatch_pack_gpu_events(args, samples, events)) {
      return;
    }
  }
  if (host_api_batch_requested(args) && !grouped_dispatch_requested(args) &&
      selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    if (collect_direct_hip_host_batch_pack_gpu_events(args, samples, events)) {
      return;
    }
  }
  const bool finite = finite_benchmark_semantics(args.semantics);
  const char* h2d_label = finite ? "finite_pack_h2d" : "pack_h2d";
  const char* kernel_label = finite ? "finite_pack_kernel" : "pack_kernel";
  const double h2d = sum_event_label(events, samples, "pack", h2d_label);
  const double kernel = sum_event_label(events, samples, "pack", kernel_label);
  if (events.complete) {
    push_gpu_event_value(events, h2d_label, h2d);
    push_gpu_event_value(events, kernel_label, kernel);
    push_gpu_event_value(events, "pack", h2d + kernel);
  }
}

void record_reused_pack_gpu_events(const Args& args, rns8_backend_kind selected_backend, GpuEventSamples& events) {
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    push_gpu_event_value(events, "vector_alu_pack_a_h2d", 0.0);
    push_gpu_event_value(events, "vector_alu_pack_b_h2d", 0.0);
    push_gpu_event_value(events, "pack", 0.0);
    return;
  }
  const bool finite = finite_benchmark_semantics(args.semantics);
  if (events.complete) {
    push_gpu_event_value(events, finite ? "finite_pack_h2d" : "pack_h2d", 0.0);
    push_gpu_event_value(events, finite ? "finite_pack_kernel" : "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", 0.0);
  }
}

void collect_gemm_gpu_events(
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache,
    bool direct_hip_zero_fill) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group";
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const double kernel_group =
      all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  const double zero_fill =
      direct_hip_zero_fill ? sum_event_label(events, samples, "rns_gemm", "direct_hip_zero_output_tile_memset") : 0.0;
  if (events.complete) {
    push_gpu_event_value(events, label, kernel_group);
    if (direct_hip_zero_fill) {
      push_gpu_event_value(events, "direct_hip_zero_output_tile_memset", zero_fill);
    }
    push_gpu_event_value(events, "rns_gemm", kernel_group + zero_fill);
  }
}

void collect_bounded_direct_hip_skinny_gemv_n1_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "rns_gemv_n1_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, "rns_gemv_n1_kernel_group", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_direct_hip_skinny_gemv_small_n_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "rns_gemv_small_n_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, "rns_gemv_small_n_kernel_group", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_incremental_result_cache_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool finite = finite_benchmark_semantics(args.semantics);
  const char* region_label = finite ? "finite_resident_gemm_region_kernel" : "rns_gemm_region_kernel_group";
  const double cache_restore = sum_event_label(events, samples, "rns_gemm", "residue_d2d_sync");
  const double region_kernel = sum_event_label(events, samples, "rns_gemm", region_label);
  if (events.complete) {
    push_gpu_event_value(events, "residue_d2d_sync", cache_restore);
    push_gpu_event_value(events, region_label, region_kernel);
    push_gpu_event_value(events, "rns_gemm", cache_restore + region_kernel);
  }
}

void collect_native_to_rns_bridge_gemm_gpu_events(
    const Args& args,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* conversion_label = native_to_rns_bridge_event_label(args);
  const double conversion = sum_event_label(events, samples, "rns_gemm", conversion_label);
  const double kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, conversion_label, conversion);
    push_gpu_event_value(events, "rns_gemm_kernel_group", kernel_group);
    push_gpu_event_value(events, "rns_gemm", conversion + kernel_group);
  }
}

void collect_vector_to_rns_chain_pack_gpu_events(
    const Args& args,
    GpuEventSamples& events,
    bool consumer_b_reused) {
  (void)args;
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const auto h2d_samples = event_label_values(samples, "residue_h2d_sync");
  std::size_t h2d_index = 0;
  const auto vector_operand_h2d = [&](const char* label) -> double {
    const bool has_fallback = h2d_index < h2d_samples.size();
    const double fallback = has_fallback ? h2d_samples[h2d_index] : 0.0;
    ++h2d_index;
    double explicit_total = 0.0;
    if (sum_event_label_if_present(samples, label, explicit_total)) {
      return explicit_total;
    }
    if (has_fallback) {
      return fallback;
    }
    add_unavailable_reason(
        events, std::string("pack missing backend HIP event label ") + label + " or residue_h2d_sync");
    return 0.0;
  };
  const double vector_a_h2d = vector_operand_h2d("vector_alu_pack_a_h2d");
  const double vector_b_h2d = vector_operand_h2d("vector_alu_pack_b_h2d");
  const double direct_h2d = consumer_b_reused ? 0.0 : sum_event_label(events, samples, "pack", "pack_h2d");
  const double direct_pack_kernel =
      consumer_b_reused ? 0.0 : sum_event_label(events, samples, "pack", "pack_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_pack_a_h2d", vector_a_h2d);
    push_gpu_event_value(events, "vector_alu_pack_b_h2d", vector_b_h2d);
    push_gpu_event_value(events, "pack_h2d", direct_h2d);
    push_gpu_event_value(events, "pack_kernel", direct_pack_kernel);
    push_gpu_event_value(events, "pack", vector_a_h2d + vector_b_h2d + direct_h2d + direct_pack_kernel);
  }
}

void collect_vector_to_rns_chain_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* vector_kernel_label = vector_to_rns_chain_kernel_label(args);
  const char* conversion_label = native_to_rns_bridge_event_label(args);
  const double status_memset = optional_event_label(samples, "vector_alu_status_memset");
  const double vector_kernel = sum_event_label(events, samples, "rns_gemm", vector_kernel_label);
  const double status_d2h = optional_event_label(samples, "vector_alu_status_d2h");
  const double conversion = sum_event_label(events, samples, "rns_gemm", conversion_label);
  const double direct_kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_status_memset", status_memset);
    push_gpu_event_value(events, vector_kernel_label, vector_kernel);
    push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
    push_gpu_event_value(events, conversion_label, conversion);
    push_gpu_event_value(events, "rns_gemm_kernel_group", direct_kernel_group);
    push_gpu_event_value(
        events,
        "rns_gemm",
        status_memset + vector_kernel + status_d2h + conversion + direct_kernel_group);
  }
}

void collect_vector_to_rns_chain_host_repack_control_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* vector_kernel_label = vector_to_rns_chain_kernel_label(args);
  const double status_memset = optional_event_label(samples, "vector_alu_status_memset");
  const double vector_kernel = sum_event_label(events, samples, "rns_gemm", vector_kernel_label);
  const double status_d2h = optional_event_label(samples, "vector_alu_status_d2h");
  const double output_d2h = sum_event_label(events, samples, "rns_gemm", "vector_alu_output_d2h");
  const double host_repack = sum_event_label(events, samples, "rns_gemm", "vector_to_rns_host_repack_a");
  const double direct_kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_status_memset", status_memset);
    push_gpu_event_value(events, vector_kernel_label, vector_kernel);
    push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
    push_gpu_event_value(events, "vector_alu_output_d2h", output_d2h);
    push_gpu_event_value(events, "vector_to_rns_host_repack_a", host_repack);
    push_gpu_event_value(events, "rns_gemm_kernel_group", direct_kernel_group);
    push_gpu_event_value(
        events,
        "rns_gemm",
        status_memset + vector_kernel + status_d2h + output_d2h + host_repack + direct_kernel_group);
  }
}

void collect_finite_direct_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "finite_resident_gemm_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "finite_resident_gemm_kernel", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_amdgpu_builtin_gemm_gpu_events(const Args& args, const BenchmarkResult& result, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = amdgpu_builtin_matrix_core_event_label(args, result);
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_finite_native_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", "finite_native_a_h2d");
  if (events.complete) {
    push_gpu_event_value(events, "finite_pack_h2d", h2d);
    push_gpu_event_value(events, "finite_pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_finite_native_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "finite_native_a_gemm_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "finite_native_a_gemm_kernel", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_native_a_pack_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", bounded_native_a_reuse_b_pack_h2d_label(args));
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_native_a_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_native_a_reuse_b_event_label(args);
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_reuse_a_pack_h2d_label());
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_uniform_small_i8_ab_reuse_a_event_label();
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double a_h2d =
      sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_transient_a_h2d_label());
  const double b_h2d =
      sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_transient_b_h2d_label());
  if (events.complete) {
    push_gpu_event_value(events, bounded_uniform_small_i8_ab_transient_a_h2d_label(), a_h2d);
    push_gpu_event_value(events, bounded_uniform_small_i8_ab_transient_b_h2d_label(), b_h2d);
    push_gpu_event_value(events, "pack", a_h2d + b_h2d);
  }
}

void collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_uniform_small_i8_ab_transient_event_label();
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_native_b_reuse_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  double h2d = 0.0;
  if (!sum_event_label_if_present(samples, bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label(), h2d) &&
      !sum_event_label_if_present(samples, "residue_h2d_sync", h2d)) {
    add_unavailable_reason(
        events,
        std::string("pack missing backend HIP event label ") +
            bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label() + " or residue_h2d_sync");
  }
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_native_b_reuse_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_native_b_reuse_a_u64_large_colpair_event_label();
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_hipblaslt_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool expect_pack_transpose = expects_hipblaslt_pack_transpose_event(args);
  const double pack = expect_pack_transpose
      ? sum_event_label(events, samples, "rns_gemm", "hipblaslt_pack_transpose_centered")
      : 0.0;
  const double matmul = sum_event_label(events, samples, "rns_gemm", "hipblaslt_int8_i32_matmul");
  const double reduce = sum_event_label(events, samples, "rns_gemm", "hipblaslt_i32_to_residue_reduce");
  if (events.complete) {
    if (expect_pack_transpose) {
      push_gpu_event_value(events, "hipblaslt_pack_transpose_centered", pack);
    }
    push_gpu_event_value(events, "hipblaslt_int8_i32_matmul", matmul);
    push_gpu_event_value(events, "hipblaslt_i32_to_residue_reduce", reduce);
    push_gpu_event_value(events, "rns_gemm", pack + matmul + reduce);
  }
}

void collect_vector_alu_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* kernel_label = vector_to_rns_chain_kernel_label(args);
  const double status_memset = optional_event_label(samples, "vector_alu_status_memset");
  const double kernel = sum_event_label(events, samples, "rns_gemm", kernel_label);
  const double status_d2h = optional_event_label(samples, "vector_alu_status_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_status_memset", status_memset);
    push_gpu_event_value(events, kernel_label, kernel);
    if (!args.vector_alu_baseline) {
      push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
    }
    push_gpu_event_value(events, "rns_gemm", status_memset + kernel + status_d2h);
  }
}

void collect_ck_deep_gemm_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const auto required_or_zero = [&](const char* label) {
    return all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  };
  const double pack_a = required_or_zero("ck_pack_a_kernel");
  const double pack_b = required_or_zero("ck_pack_b_kernel");
  const double matmul = required_or_zero("ck_wmma_cshuffle_matmul");
  const double copy = optional_event_label(samples, "ck_copy_centered_kernel");
  const double add = optional_event_label(samples, "ck_add_centered_kernel");
  const double zero_fill = result.zero_output_tile_count == 0
                               ? 0.0
                               : sum_event_label(events, samples, "rns_gemm", "ck_zero_output_tile_memset");
  if (events.complete) {
    push_gpu_event_value(events, "ck_pack_a_kernel", pack_a);
    push_gpu_event_value(events, "ck_pack_b_kernel", pack_b);
    push_gpu_event_value(events, "ck_wmma_cshuffle_matmul", matmul);
    push_gpu_event_value(events, "ck_copy_centered_kernel", copy);
    push_gpu_event_value(events, "ck_add_centered_kernel", add);
    if (result.zero_output_tile_count != 0) {
      push_gpu_event_value(events, "ck_zero_output_tile_memset", zero_fill);
    }
    const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
    for (uint32_t index = 0; index < prefix_count; ++index) {
      const std::string pack_a_label = prefix_event_label("ck_prefix_", index, "pack_a");
      const std::string pack_b_label = prefix_event_label("ck_prefix_", index, "pack_b");
      const std::string matmul_label = prefix_event_label("ck_prefix_", index, "matmul");
      push_gpu_event_value(
          events,
          pack_a_label,
          all_tiles_zero ? optional_event_label(samples, pack_a_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", pack_a_label.c_str()));
      push_gpu_event_value(
          events,
          pack_b_label,
          all_tiles_zero ? optional_event_label(samples, pack_b_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", pack_b_label.c_str()));
      push_gpu_event_value(
          events,
          matmul_label,
          all_tiles_zero ? optional_event_label(samples, matmul_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", matmul_label.c_str()));
      const std::string copy_label = prefix_event_label("ck_prefix_", index, "copy_centered");
      const std::string add_label = prefix_event_label("ck_prefix_", index, "add_centered");
      push_gpu_event_value(events, copy_label, optional_event_label(samples, copy_label.c_str()));
      push_gpu_event_value(events, add_label, optional_event_label(samples, add_label.c_str()));
    }
  }
}

void collect_rocwmma_deep_gemm_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* pack_a_label =
      use_prepacked_b_cache ? "rocwmma_pack_a_prepacked_b_kernel" : "rocwmma_pack_a_kernel";
  const char* matmul_label =
      use_prepacked_b_cache ? "rocwmma_matmul_prepacked_b_kernel" : "rocwmma_matmul_kernel";
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const auto required_or_zero = [&](const char* label) {
    return all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  };
  const double pack_a = required_or_zero(pack_a_label);
  const double pack_b = use_prepacked_b_cache ? 0.0 : required_or_zero("rocwmma_pack_b_kernel");
  const double matmul = required_or_zero(matmul_label);
  const double zero_fill = result.zero_output_tile_count == 0
                               ? 0.0
                               : sum_event_label(events, samples, "rns_gemm", "rocwmma_zero_output_tile_memset");
  if (events.complete) {
    push_gpu_event_value(events, pack_a_label, pack_a);
    if (!use_prepacked_b_cache) {
      push_gpu_event_value(events, "rocwmma_pack_b_kernel", pack_b);
    }
    push_gpu_event_value(events, matmul_label, matmul);
    if (result.zero_output_tile_count != 0) {
      push_gpu_event_value(events, "rocwmma_zero_output_tile_memset", zero_fill);
    }
    const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
    for (uint32_t index = 0; index < prefix_count; ++index) {
      if (use_prepacked_b_cache) {
        const std::string pack_a_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b");
        const std::string matmul_prefix = prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b");
        push_gpu_event_value(
            events,
            pack_a_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_a_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_a_prefix.c_str()));
        push_gpu_event_value(
            events,
            matmul_prefix,
            all_tiles_zero ? optional_event_label(samples, matmul_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", matmul_prefix.c_str()));
      } else {
        const std::string pack_a_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_a");
        const std::string pack_b_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_b");
        const std::string matmul_prefix = prefix_event_label("rocwmma_prefix_", index, "matmul");
        push_gpu_event_value(
            events,
            pack_a_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_a_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_a_prefix.c_str()));
        push_gpu_event_value(
            events,
            pack_b_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_b_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_b_prefix.c_str()));
        push_gpu_event_value(
            events,
            matmul_prefix,
            all_tiles_zero ? optional_event_label(samples, matmul_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", matmul_prefix.c_str()));
      }
    }
  }
}

void collect_rns_gemm_gpu_events(
    const Args& args,
    rns8_backend_kind selected_backend,
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache = false) {
  if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
    collect_hipblaslt_gemm_gpu_events(args, events);
  } else if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    collect_vector_alu_gemm_gpu_events(args, events);
  } else if (selected_backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    collect_amdgpu_builtin_gemm_gpu_events(args, result, events);
  } else if (finite_benchmark_semantics(args.semantics) && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    collect_finite_direct_gemm_gpu_events(events);
  } else if (native_to_rns_bridge_path(args, result, selected_backend)) {
    collect_native_to_rns_bridge_gemm_gpu_events(args, events);
  } else {
    collect_gemm_gpu_events(
        result,
        events,
        use_prepacked_b_cache,
        selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0);
    if (finite_benchmark_semantics(args.semantics)) {
      return;
    }
    if (selected_backend == RNS8_BACKEND_CK) {
      collect_ck_deep_gemm_gpu_events(args, result, events);
    } else if (selected_backend == RNS8_BACKEND_ROCWMMA && !args.wrap64_rocwmma_candidate) {
      collect_rocwmma_deep_gemm_gpu_events(args, result, events, use_prepacked_b_cache);
    }
  }
}

void collect_wrap64_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label =
      args.wrap64_rocwmma_candidate ? kWrap64RocwmmaCandidateEventLabel : wrap64_direct_hip_gemm_event_label(args);
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_export_gpu_events(
    const Args& args,
    rns8_backend_kind selected_backend,
    const BenchmarkResult& result,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const auto d2h_samples = event_label_values(samples, "residue_d2h_sync");
    std::size_t d2h_index = 0;
    double status_d2h = 0.0;
    if (args.vector_alu_baseline) {
      const bool has_status_fallback = d2h_index < d2h_samples.size();
      const double status_fallback = has_status_fallback ? d2h_samples[d2h_index] : 0.0;
      ++d2h_index;
      if (!sum_event_label_if_present(samples, "vector_alu_status_d2h", status_d2h) && has_status_fallback) {
        status_d2h = status_fallback;
      }
    }
    const bool has_output_fallback = d2h_index < d2h_samples.size();
    const double output_fallback = has_output_fallback ? d2h_samples[d2h_index] : 0.0;
    double output_d2h = 0.0;
    if (!sum_event_label_if_present(samples, "vector_alu_output_d2h", output_d2h)) {
      if (has_output_fallback) {
        output_d2h = output_fallback;
      } else {
        add_unavailable_reason(
            events, "crt_export missing backend HIP event label vector_alu_output_d2h or residue_d2h_sync");
      }
    }
    if (events.complete) {
      if (args.vector_alu_baseline) {
        push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
      }
      push_gpu_event_value(events, "vector_alu_output_d2h", output_d2h);
      push_gpu_event_value(events, "crt_export", status_d2h + output_d2h);
    }
    return;
  }
  if (finite_benchmark_semantics(args.semantics)) {
    const double kernel = sum_event_label(events, samples, "crt_export", "finite_export_kernel");
    const double d2h = sum_event_label(events, samples, "crt_export", "finite_export_d2h");
    if (events.complete) {
      push_gpu_event_value(events, "finite_export_kernel", kernel);
      push_gpu_event_value(events, "finite_export_d2h", d2h);
      push_gpu_event_value(events, "crt_export", kernel + d2h);
    }
    return;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    const bool requires_status = exact_wide_export_status_check_required(args);
    const double status_memset =
        requires_status ? optional_event_label(samples, "exact_wide_export_status_memset") : 0.0;
    const double kernel = sum_event_label(events, samples, "crt_export", "exact_wide_export_kernel");
    const double status_d2h =
        requires_status ? sum_event_label(events, samples, "crt_export", "exact_wide_export_status_d2h") : 0.0;
    const double d2h = sum_event_label(events, samples, "crt_export", "exact_wide_export_d2h");
    if (events.complete) {
      push_gpu_event_value(events, "exact_wide_export_status_memset", status_memset);
      push_gpu_event_value(events, "exact_wide_export_kernel", kernel);
      push_gpu_event_value(events, "exact_wide_export_status_d2h", status_d2h);
      push_gpu_event_value(events, "exact_wide_export_d2h", d2h);
      push_gpu_event_value(events, "crt_export", status_memset + kernel + status_d2h + d2h);
    }
    return;
  }
  const bool all_zero_direct_hip_scheduled_export =
      selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0 &&
      result.schedule_info_available && result.zero_output_tile_count == result.schedule_info.tile_count;
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double kernel = sum_event_label(events, samples, "crt_export", "crt_export_kernel");
  const double status_d2h = all_zero_direct_hip_scheduled_export
                                ? optional_event_label(samples, "crt_export_status_d2h")
                                : sum_event_label(events, samples, "crt_export", "crt_export_status_d2h");
  const double d2h = sum_event_label(events, samples, "crt_export", "crt_export_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", d2h);
    push_gpu_event_value(events, "crt_export", status_memset + kernel + status_d2h + d2h);
  }
}

void collect_bounded_oneshot_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool resident_fallback = direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend);
  const double pack_h2d =
      resident_fallback ? sum_event_label(events, samples, "oneshot", "pack_h2d")
                        : sum_event_label(events, samples, "oneshot", "residue_h2d_sync");
  const double pack_kernel = resident_fallback ? sum_event_label(events, samples, "oneshot", "pack_kernel") : 0.0;
  const double gemm = sum_event_label(events, samples, "oneshot", "rns_gemm_kernel_group");
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double export_kernel = sum_event_label(events, samples, "oneshot", "crt_export_kernel");
  const double status_d2h = sum_event_label(events, samples, "oneshot", "crt_export_status_d2h");
  const double output_d2h = sum_event_label(events, samples, "oneshot", "crt_export_d2h");
  const double export_total = status_memset + export_kernel + status_d2h + output_d2h;
  if (events.complete) {
    if (resident_fallback) {
      push_gpu_event_value(events, "pack_h2d", pack_h2d);
      push_gpu_event_value(events, "pack_kernel", pack_kernel);
      push_gpu_event_value(events, "pack", pack_h2d + pack_kernel);
    } else {
      push_gpu_event_value(events, "oneshot_native_input_h2d", pack_h2d);
    }
    push_gpu_event_value(events, "rns_gemm_kernel_group", gemm);
    push_gpu_event_value(events, "rns_gemm", gemm);
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", export_kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", output_d2h);
    push_gpu_event_value(events, "crt_export", export_total);
    push_gpu_event_value(events, "oneshot_api_gpu", pack_h2d + pack_kernel + gemm + export_total);
  }
}

void collect_finite_oneshot_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double native_h2d = sum_event_label(events, samples, "oneshot", "residue_h2d_sync");
  const double gemm = sum_event_label(events, samples, "oneshot", "finite_native_gemm_kernel");
  const double export_kernel = sum_event_label(events, samples, "oneshot", "finite_export_kernel");
  const double output_d2h = sum_event_label(events, samples, "oneshot", "finite_export_d2h");
  const double export_total = export_kernel + output_d2h;
  if (events.complete) {
    push_gpu_event_value(events, "oneshot_native_input_h2d", native_h2d);
    push_gpu_event_value(events, "finite_native_gemm_kernel", gemm);
    push_gpu_event_value(events, "rns_gemm", gemm);
    push_gpu_event_value(events, "finite_export_kernel", export_kernel);
    push_gpu_event_value(events, "finite_export_d2h", output_d2h);
    push_gpu_event_value(events, "crt_export", export_total);
    push_gpu_event_value(events, "oneshot_api_gpu", native_h2d + gemm + export_total);
  }
}

void collect_wrap64_export_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "crt_export", "wrap64_export_kernel");
  const double d2h = sum_event_label(events, samples, "crt_export", "wrap64_export_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "wrap64_export_kernel", kernel);
    push_gpu_event_value(events, "wrap64_export_d2h", d2h);
    push_gpu_event_value(events, "crt_export", kernel + d2h);
  }
}

void begin_gpu_event_phase(bool collect) {
  if (!collect) {
    return;
  }
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
}

void end_gpu_event_phase(bool collect) {
  if (!collect) {
    return;
  }
  rns8::detail::hip_direct_timing_set_enabled(false);
}

void record_allocation_after_warmups(BenchmarkResult& result) {
  result.allocation_after_warmups = rns8::detail::hip_direct_allocation_counters_snapshot();
  result.allocation_after_warmups_available = true;
}

void record_allocation_after_repeats(BenchmarkResult& result) {
  result.allocation_after_repeats = rns8::detail::hip_direct_allocation_counters_snapshot();
  result.allocation_after_repeats_available = true;
}

bool gpu_event_timing_available(const Args& args, const BenchmarkResult& result) {
  if (!result.gpu_events.requested || !result.gpu_events.complete) {
    return false;
  }
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const std::size_t repeats = result.samples.pack_us.size();
  const bool use_prepacked_b_cache = uses_runtime_b_prepack_cache(result);
  const auto phase_order = gpu_event_phase_order(args, result, selected_backend, use_prepacked_b_cache);
  if (repeats == 0 || phase_order.empty()) {
    return false;
  }
  for (const auto& label : phase_order) {
    const auto it = result.gpu_events.timings_us.find(label);
    if (it == result.gpu_events.timings_us.end() || it->second.size() != repeats) {
      return false;
    }
  }
  for (const auto& item : result.gpu_events.timings_us) {
    if (std::find(phase_order.begin(), phase_order.end(), item.first) == phase_order.end()) {
      return false;
    }
  }
  return true;
}

