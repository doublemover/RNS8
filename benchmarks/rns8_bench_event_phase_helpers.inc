void append_ck_deep_event_phases(std::vector<std::string>& phases, uint32_t prefix_count, bool zero_output_tiles) {
  phases.push_back("ck_pack_a_kernel");
  phases.push_back("ck_pack_b_kernel");
  phases.push_back("ck_wmma_cshuffle_matmul");
  phases.push_back("ck_copy_centered_kernel");
  phases.push_back("ck_add_centered_kernel");
  if (zero_output_tiles) {
    phases.push_back("ck_zero_output_tile_memset");
  }
  for (uint32_t index = 0; index < prefix_count; ++index) {
    phases.push_back(prefix_event_label("ck_prefix_", index, "pack_a"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "pack_b"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "matmul"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "copy_centered"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "add_centered"));
  }
}

void append_rocwmma_deep_event_phases(
    std::vector<std::string>& phases,
    uint32_t prefix_count,
    bool use_prepacked_b_cache,
    bool zero_output_tiles) {
  phases.push_back(use_prepacked_b_cache ? "rocwmma_pack_a_prepacked_b_kernel" : "rocwmma_pack_a_kernel");
  if (!use_prepacked_b_cache) {
    phases.push_back("rocwmma_pack_b_kernel");
  }
  phases.push_back(use_prepacked_b_cache ? "rocwmma_matmul_prepacked_b_kernel" : "rocwmma_matmul_kernel");
  if (zero_output_tiles) {
    phases.push_back("rocwmma_zero_output_tile_memset");
  }
  for (uint32_t index = 0; index < prefix_count; ++index) {
    if (use_prepacked_b_cache) {
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b"));
    } else {
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_a"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_b"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "matmul"));
    }
  }
}

void append_accelerator_deep_event_phases(
    std::vector<std::string>& phases,
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    bool use_prepacked_b_cache) {
  if (finite_benchmark_semantics(args.semantics)) {
    return;
  }
  const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
  const bool zero_output_tiles = result.zero_output_tile_count != 0;
  if (selected_backend == RNS8_BACKEND_CK) {
    append_ck_deep_event_phases(phases, prefix_count, zero_output_tiles);
  } else if (selected_backend == RNS8_BACKEND_ROCWMMA && !args.wrap64_rocwmma_candidate) {
    append_rocwmma_deep_event_phases(phases, prefix_count, use_prepacked_b_cache, zero_output_tiles);
  }
}

void append_hipblaslt_gemm_event_phases(std::vector<std::string>& phases, const Args& args) {
  if (expects_hipblaslt_pack_transpose_event(args)) {
    phases.push_back("hipblaslt_pack_transpose_centered");
  }
  phases.push_back("hipblaslt_int8_i32_matmul");
  phases.push_back("hipblaslt_i32_to_residue_reduce");
  phases.push_back("rns_gemm");
}

const char* wrap64_direct_hip_gemm_event_label(const Args& args) {
  return rns8::detail::wrap64_hip_gemm_event_label_for_shape(args.m, args.n, args.k);
}

const char* native_to_rns_bridge_event_label(const Args& args) {
  return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_to_rns_kernel" : "native_u64_to_rns_kernel";
}

bool native_to_rns_bridge_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return native_to_rns_bridge_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && result.backend_info_available &&
         result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

const char* vector_to_rns_chain_kernel_label(const Args& args) {
  if (vector_alu_gemv_n1_shape(args)) {
    return args.semantics == BenchSemantics::BoundedI64 ? "vector_alu_i64_gemv_n1_kernel"
                                                        : "vector_alu_u64_gemv_n1_kernel";
  }
  return args.semantics == BenchSemantics::BoundedI64 ? "vector_alu_i64_kernel" : "vector_alu_u64_kernel";
}

bool vector_to_rns_chain_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return vector_to_rns_chain_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && result.backend_info_available &&
         result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

bool vector_to_rns_chain_host_repack_control_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return vector_to_rns_chain_path(args, result, selected_backend) &&
         vector_to_rns_chain_host_repack_control_requested(args);
}

bool direct_hip_bounded_oneshot_resident_fallback_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  const uint32_t requested_prefix = benchmark_prefix(args);
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  return args.oneshot && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && requested_prefix > 0 && selected_prefix > 0 &&
         selected_prefix < requested_prefix;
}

std::vector<std::string> gpu_event_phase_order(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    bool use_prepacked_b_cache) {
  if (result.streaming_overlap_executed) {
    return {
        "pack_h2d",
        "pack_kernel",
        "pack",
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export"};
  }
  if (args.oneshot && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    if (finite_benchmark_semantics(args.semantics)) {
      return {
          "oneshot_native_input_h2d",
          "finite_native_gemm_kernel",
          "rns_gemm",
          "finite_export_kernel",
          "finite_export_d2h",
          "crt_export",
          "oneshot_api_gpu"};
    }
    if (direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend)) {
      return {
          "pack_h2d",
          "pack_kernel",
          "pack",
          "rns_gemm_kernel_group",
          "rns_gemm",
          "crt_export_status_memset",
          "crt_export_kernel",
          "crt_export_status_d2h",
          "crt_export_d2h",
          "crt_export",
          "oneshot_api_gpu"};
    }
    return {
        "oneshot_native_input_h2d",
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
        "oneshot_api_gpu"};
  }
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const char* kernel = vector_to_rns_chain_kernel_label(args);
    return {
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack",
        "vector_alu_status_memset",
        kernel,
        "rns_gemm",
        "vector_alu_status_d2h",
        "vector_alu_output_d2h",
        "crt_export"};
  }
  if (vector_to_rns_chain_host_repack_control_path(args, result, selected_backend)) {
    return {
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack_h2d",
        "pack_kernel",
        "pack",
        "vector_alu_status_memset",
        vector_to_rns_chain_kernel_label(args),
        "vector_alu_status_d2h",
        "vector_alu_output_d2h",
        "vector_to_rns_host_repack_a",
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export"};
  }
  if (vector_to_rns_chain_path(args, result, selected_backend)) {
    return {
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack_h2d",
        "pack_kernel",
        "pack",
        "vector_alu_status_memset",
        vector_to_rns_chain_kernel_label(args),
        "vector_alu_status_d2h",
        native_to_rns_bridge_event_label(args),
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export"};
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    if (args.wrap64_rocwmma_candidate) {
      return {
          "pack_h2d",
          "pack_kernel",
          "pack",
          kWrap64RocwmmaCandidateEventLabel,
          "rns_gemm",
          "wrap64_export_kernel",
          "wrap64_export_d2h",
          "crt_export"};
    }
    return {
        "pack_h2d",
        "pack_kernel",
        "pack",
        wrap64_direct_hip_gemm_event_label(args),
        "rns_gemm",
        "wrap64_export_kernel",
        "wrap64_export_d2h",
        "crt_export"};
  }
  if (finite_benchmark_semantics(args.semantics)) {
    if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
      std::vector<std::string> phases = {
          "finite_pack_h2d",
          "finite_pack_kernel",
          "pack",
      };
      append_hipblaslt_gemm_event_phases(phases, args);
      phases.push_back("finite_export_kernel");
      phases.push_back("finite_export_d2h");
      phases.push_back("crt_export");
      return phases;
    }
    std::vector<std::string> phases = {
        "finite_pack_h2d",
        "finite_pack_kernel",
        "pack",
    };
    if (result.incremental_result_cache_public_contract_available && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      phases.push_back("residue_d2d_sync");
      phases.push_back("finite_resident_gemm_region_kernel");
    } else {
      phases.push_back(
          finite_native_a_reuse_b_path(args, result)
              ? "finite_native_a_gemm_kernel"
              : selected_backend == RNS8_BACKEND_HIP_DIRECT ? "finite_resident_gemm_kernel" : "rns_gemm_kernel_group");
    }
    append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
    phases.push_back("rns_gemm");
    phases.push_back("finite_export_kernel");
    phases.push_back("finite_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
      std::vector<std::string> phases = {
          "pack_h2d",
          "pack_kernel",
          "pack",
      };
      append_hipblaslt_gemm_event_phases(phases, args);
      if (residue_current_output_mode(args)) {
        return phases;
      }
      phases.push_back("exact_wide_export_status_memset");
      phases.push_back("exact_wide_export_kernel");
      phases.push_back("exact_wide_export_status_d2h");
      phases.push_back("exact_wide_export_d2h");
      phases.push_back("crt_export");
      return phases;
    }
    std::vector<std::string> phases = {
        "pack_h2d",
        "pack_kernel",
        "pack",
    };
    if (result.incremental_result_cache_public_contract_available && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      phases.push_back("residue_d2d_sync");
      phases.push_back("rns_gemm_region_kernel_group");
    } else {
      phases.push_back(use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group");
    }
    append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
    phases.push_back("rns_gemm");
    if (residue_current_output_mode(args)) {
      return phases;
    }
    phases.push_back("exact_wide_export_status_memset");
    phases.push_back("exact_wide_export_kernel");
    phases.push_back("exact_wide_export_status_d2h");
    phases.push_back("exact_wide_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
    std::vector<std::string> phases = {
        "pack_h2d",
        "pack_kernel",
        "pack",
    };
    append_hipblaslt_gemm_event_phases(phases, args);
    if (residue_current_output_mode(args)) {
      return phases;
    }
    phases.push_back("crt_export_status_memset");
    phases.push_back("crt_export_kernel");
    phases.push_back("crt_export_status_d2h");
    phases.push_back("crt_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  const char* gemm_phase = use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group";
  if (bounded_native_a_reuse_b_path(args, result)) {
    gemm_phase = bounded_native_a_reuse_b_event_label(args);
  }
  if (bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    gemm_phase = bounded_uniform_small_i8_ab_reuse_a_event_label();
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    gemm_phase = bounded_native_b_reuse_a_u64_large_colpair_event_label();
  }
  if (bounded_direct_hip_skinny_gemv_n1_path(args, result)) {
    gemm_phase = "rns_gemv_n1_kernel_group";
  }
  std::vector<std::string> phases;
  if (bounded_uniform_small_i8_ab_transient_path(args, result)) {
    gemm_phase = bounded_uniform_small_i8_ab_transient_event_label();
    phases = {
        bounded_uniform_small_i8_ab_transient_a_h2d_label(),
        bounded_uniform_small_i8_ab_transient_b_h2d_label(),
        "pack",
        gemm_phase};
  } else {
    phases = {"pack_h2d", "pack_kernel", "pack"};
    if (native_to_rns_bridge_path(args, result, selected_backend)) {
      phases.push_back(native_to_rns_bridge_event_label(args));
    }
    if (result.incremental_result_cache_public_contract_available && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      phases.push_back("residue_d2d_sync");
      phases.push_back("rns_gemm_region_kernel_group");
    } else {
      phases.push_back(gemm_phase);
    }
  }
  if (selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0) {
    phases.push_back("direct_hip_zero_output_tile_memset");
  }
  append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
  phases.push_back("rns_gemm");
  if (residue_current_output_mode(args)) {
    return phases;
  }
  phases.push_back("crt_export_status_memset");
  phases.push_back("crt_export_kernel");
  phases.push_back("crt_export_status_d2h");
  phases.push_back("crt_export_d2h");
  phases.push_back("crt_export");
  return phases;
}

void print_timing_summary(const char* name, const std::vector<uint64_t>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << average(values) << ",\n";
  std::cout << "      \"median\": " << percentile(values, 0.50) << ",\n";
  std::cout << "      \"p95\": " << percentile(values, 0.95) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_single_timing_summary(const char* name, uint64_t value, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << static_cast<double>(value) << ",\n";
  std::cout << "      \"median\": " << static_cast<double>(value) << ",\n";
  std::cout << "      \"p95\": " << static_cast<double>(value) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_gpu_event_summary(const char* name, const std::vector<double>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << average_double(values) << ",\n";
  std::cout << "      \"median\": " << percentile_double(values, 0.50) << ",\n";
  std::cout << "      \"p95\": " << percentile_double(values, 0.95) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_named_gpu_event_array(const char* name, const std::vector<double>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": ";
  print_double_array(values);
  std::cout << (trailing_comma ? "," : "") << "\n";
}

const std::vector<double>& gpu_event_values(const GpuEventSamples& events, const std::string& label) {
  static const std::vector<double> empty;
  const auto it = events.timings_us.find(label);
  return it == events.timings_us.end() ? empty : it->second;
}

void print_gpu_event_timings(const std::vector<std::string>& phase_order, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timings_us\": {\n";
  for (std::size_t index = 0; index < phase_order.size(); ++index) {
    print_named_gpu_event_array(
        phase_order[index].c_str(), gpu_event_values(events, phase_order[index]), index + 1 != phase_order.size());
  }
  std::cout << "  },\n";
}

void print_gpu_event_timing_summary(const std::vector<std::string>& phase_order, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timing_summary_us\": {\n";
  for (std::size_t index = 0; index < phase_order.size(); ++index) {
    print_gpu_event_summary(
        phase_order[index].c_str(), gpu_event_values(events, phase_order[index]), index + 1 != phase_order.size());
  }
  std::cout << "  },\n";
}

uint64_t checksum_i64(const std::vector<int64_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const int64_t value : values) {
    mix_checksum(checksum, static_cast<uint64_t>(value));
  }
  return checksum;
}

uint64_t checksum_u64(const std::vector<uint64_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const uint64_t value : values) {
    mix_checksum(checksum, value);
  }
  return checksum;
}

uint64_t checksum_u8(const std::vector<uint8_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const uint8_t value : values) {
    mix_checksum(checksum, static_cast<uint64_t>(value));
  }
  return checksum;
}

template <typename T>
uint64_t checksum_matrix(const std::vector<T>& values, int64_t rows, int64_t cols, int64_t ld, const char* label) {
  uint64_t checksum = 1469598103934665603ull;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      mix_checksum(checksum, static_cast<uint64_t>(values[row_major_index(row, col, ld, label)]));
    }
  }
  return checksum;
}

uint64_t checksum_limb_matrix(
    const std::vector<uint64_t>& values,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t limb_count,
    const char* label) {
  uint64_t checksum = 1469598103934665603ull;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const std::size_t cell = row_major_index(row, col, ld, label);
      const std::size_t base = cell * static_cast<std::size_t>(limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        mix_checksum(checksum, values[base + static_cast<std::size_t>(limb)]);
      }
    }
  }
  return checksum;
}

template <typename T>
void copy_compact_to_output(
    const std::vector<T>& compact,
    std::vector<T>& output,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    const char* label) {
  if (ld == cols) {
    std::copy(compact.begin(), compact.end(), output.begin());
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    const std::size_t source = row_major_index(row, 0, cols, label);
    const std::size_t destination = row_major_index(row, 0, ld, label);
    std::copy(compact.begin() + static_cast<std::ptrdiff_t>(source),
              compact.begin() + static_cast<std::ptrdiff_t>(source + static_cast<std::size_t>(cols)),
              output.begin() + static_cast<std::ptrdiff_t>(destination));
  }
}

void copy_compact_limb_output(
    const std::vector<uint64_t>& compact,
    std::vector<uint64_t>& output,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t limb_count,
    const char* label) {
  if (ld == cols) {
    std::copy(compact.begin(), compact.end(), output.begin());
    return;
  }
  const auto limbs = static_cast<std::size_t>(limb_count);
  for (int64_t row = 0; row < rows; ++row) {
    const std::size_t source = row_major_index(row, 0, cols, label) * limbs;
    const std::size_t destination = row_major_index(row, 0, ld, label) * limbs;
    const std::size_t count = static_cast<std::size_t>(cols) * limbs;
    std::copy(compact.begin() + static_cast<std::ptrdiff_t>(source),
              compact.begin() + static_cast<std::ptrdiff_t>(source + count),
              output.begin() + static_cast<std::ptrdiff_t>(destination));
  }
}
