bool benchmark_host_matrix_access_valid(int64_t rows, int64_t cols, int64_t ld) {
  return rows > 0 && cols > 0 && ld >= cols;
}

rns8_status pack_exact_wide_limbs_as_rns(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint32_t limb_count,
    bool signed_values,
    uint64_t source_version) {
  if (!ctx || !matrix || !src || limb_count == 0 || limb_count > 32 ||
      !benchmark_host_matrix_access_valid(matrix->desc.rows, matrix->desc.cols, ld) ||
      matrix->prefix == 0 || matrix->prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      matrix->residues.size() !=
          static_cast<std::size_t>(matrix->prefix) * static_cast<std::size_t>(matrix->desc.rows) *
              static_cast<std::size_t>(matrix->desc.cols)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_semantics expected = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
  if (matrix->desc.semantics != expected || ctx->backend != matrix->backend) {
    return RNS8_INVALID_ARGUMENT;
  }

  std::vector<boost::multiprecision::cpp_int> values;
  values.reserve(static_cast<std::size_t>(matrix->desc.rows) * static_cast<std::size_t>(matrix->desc.cols));
  for (int64_t row = 0; row < matrix->desc.rows; ++row) {
    for (int64_t col = 0; col < matrix->desc.cols; ++col) {
      values.push_back(exact_wide_value_from_limbs(
          src + exact_wide_limb_index(row, col, ld, limb_count), limb_count, signed_values));
    }
  }

  for (uint32_t p = 0; p < matrix->prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < matrix->desc.rows; ++row) {
      for (int64_t col = 0; col < matrix->desc.cols; ++col) {
        const std::size_t value_index =
            static_cast<std::size_t>(row * matrix->desc.cols + col);
        matrix->residues[rns8::detail::residue_index(*matrix, p, row, col)] =
            rns8::detail::centered_residue(values[value_index], modulus);
      }
    }
  }

  matrix->source_version = source_version;
  rns8::detail::api::clear_byte_limb_current(*matrix);
  rns8::detail::api::clear_native_current(*matrix);

  if (matrix->hip_residues != nullptr) {
    if (matrix->hip_device_id != ctx->device_id ||
        matrix->hip_residue_bytes != matrix->residues.size() * sizeof(int8_t)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status status = run_timed_status_operation("pack_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(
          ctx->device_id, matrix->hip_residues, matrix->residues.data(), matrix->hip_residue_bytes);
    });
    if (status != RNS8_SUCCESS) {
      rns8::detail::api::clear_device_residues_current(*matrix);
      return status;
    }
    rns8::detail::api::mark_device_residues_current(*matrix);
  } else {
    rns8::detail::api::mark_host_residues_current(*matrix);
  }
  return RNS8_SUCCESS;
}









const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
      return "auto";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_ROCWMMA:
      return "rocwmma";
    case RNS8_BACKEND_AMDGPU_BUILTINS:
      return "amdgpu-builtins";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
  }
  return "unknown";
}

const char* requested_backend_name(const Args& args) {
  if (args.wrap64_rocwmma_candidate) {
    return kWrap64RocwmmaCandidateRequestedBackend;
  }
  return args.vector_alu_baseline ? "hip-vector-alu-int64" : backend_name(args.backend);
}

const char* selected_backend_name(const Args& args, const rns8_device_info& info, const BenchmarkResult* result = nullptr) {
  if (args.vector_alu_baseline) {
    return "hip-vector-alu-int64";
  }
  if (result && result->backend_info_available) {
    return backend_name(result->backend_info.backend);
  }
  return backend_name(info.backend);
}

bool benchmark_gpu_target_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_DIRECT || backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 ||
         backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA ||
         backend == RNS8_BACKEND_AMDGPU_BUILTINS;
}

bool concrete_benchmark_target_id(const char* target) {
  if (!target || target[0] == '\0') {
    return false;
  }
  const std::string value(target);
  return value != "none" && value != "cpu" && value != "unknown";
}

std::string benchmark_target_id_for_context(rns8_context* ctx, rns8_backend_kind backend) {
  if (!benchmark_gpu_target_backend(backend)) {
    return "cpu";
  }
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (rns8_get_device_info(ctx, &info) == RNS8_SUCCESS && concrete_benchmark_target_id(info.gcn_arch)) {
    return info.gcn_arch;
  }
  return "unknown";
}

std::string autotune_key_field(const char* key, const char* field) {
  if (!key || !field || field[0] == '\0') {
    return {};
  }
  const std::string text(key);
  const std::string prefix = std::string(field) + "=";
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(';', start);
    const std::string part = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (part.rfind(prefix, 0) == 0) {
      return part.substr(prefix.size());
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return {};
}

void update_result_target_id_from_key(BenchmarkResult& result) {
  const std::string target = autotune_key_field(result.backend_info.autotune_key, "target_id");
  if (!target.empty()) {
    result.target_id = target;
  } else if (result.backend_info_available && !benchmark_gpu_target_backend(result.backend_info.backend)) {
    result.target_id = "cpu";
  }
}

std::string benchmark_key_target_id(const BenchmarkResult& result) {
  return result.target_id.empty() ? "unknown" : result.target_id;
}

bool bounded_native_a_reuse_b_requested(const Args& args) {
  return !args.oneshot && bounded_benchmark_semantics(args.semantics) && args.reuse_packed_b &&
         !args.reuse_packed_a && !args.streaming_overlap && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1;
}

bool bounded_native_a_reuse_b_uniform_small_a(const Args& args) {
  return args.input_profile == InputProfile::UniformSmall;
}

bool bounded_native_a_reuse_b_u64_large_colpair(const Args& args) {
  return args.semantics == BenchSemantics::BoundedU64 && !bounded_native_a_reuse_b_uniform_small_a(args) &&
         args.m >= 512 && args.n >= 512 && args.k >= 512;
}

bool bounded_uniform_small_i8_ab_reuse_a_requested(const Args& args) {
  return !args.oneshot && bounded_benchmark_semantics(args.semantics) && args.reuse_packed_a &&
         !args.reuse_packed_b && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
}

bool bounded_native_b_reuse_a_u64_large_colpair_requested(const Args& args) {
  return !args.oneshot && args.semantics == BenchSemantics::BoundedU64 && args.reuse_packed_a &&
         !args.reuse_packed_b && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile != InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX) &&
         args.m >= 512 && args.n >= 512 && args.k >= 512;
}

bool bounded_uniform_small_i8_ab_transient_requested(const Args& args) {
  return !args.oneshot && args.transient_uniform_small_inputs && bounded_benchmark_semantics(args.semantics) &&
         !args.reuse_packed_inputs && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
}

bool bounded_residue_channel_fusion_requested(const Args& args) {
  return !args.oneshot && args.residue_channel_fusion && bounded_benchmark_semantics(args.semantics) &&
         !args.reuse_packed_inputs && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
}

const char* input_profile_name(const Args& args) {
  switch (args.input_profile) {
    case InputProfile::UniformSmall:
      return "uniform-small";
    case InputProfile::AdaptiveBands:
      return "adaptive-bands";
    case InputProfile::FiniteBinary:
      return "finite-binary";
    case InputProfile::FiniteSparse:
      return "finite-sparse";
    case InputProfile::FiniteLowHamming:
      return "finite-low-hamming";
    case InputProfile::FiniteSmallCentered:
      return "finite-small-centered";
    case InputProfile::FiniteFullUniform:
      return "finite-full-uniform";
  }
  return "unknown";
}

const char* bound_source_name(const Args& args) {
  return args.bound_source == BoundSource::InputScan ? "input_scan" : "static_profile";
}

const char* bound_discovery_source_name(const Args& args, bool global_bound_scan_available) {
  if (global_bound_scan_available) {
    return "input_row_column_abs_summary";
  }
  if (args.bound_source == BoundSource::InputScan && args.bound_mode == BoundMode::PerTile) {
    return "input_exact_tile_bounds";
  }
  return "static_profile_contract";
}

const char* backend_metadata_source(const Args& args) {
  if (args.wrap64_rocwmma_candidate) {
    return kWrap64RocwmmaCandidateBackendSource;
  }
  if (args.oneshot) {
    return "rns8_bench_public_oneshot_api";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "rns8_bench_residue_channel_fusion_path";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "rns8_bench_uniform_small_i8_ab_transient_path";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "rns8_bench_uniform_small_i8_ab_reuse_b_path";
    }
    return "rns8_bench_native_a_reuse_b_path";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "rns8_bench_uniform_small_i8_ab_reuse_a_path";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args)) {
    return "rns8_bench_native_b_reuse_a_path";
  }
  if (finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
      args.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "rns8_bench_native_a_reuse_b_path";
  }
  return args.vector_alu_baseline ? "rns8_bench_vector_alu_baseline" : "rns8_get_plan_backend_info";
}

bool runtime_vector_alu_backend(const Args& args) {
  return !args.vector_alu_baseline && args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

bool native_to_rns_bridge_requested(const Args& args) {
  return args.native_to_rns_bridge;
}

bool vector_to_rns_chain_requested(const Args& args) {
  return args.vector_to_rns_chain;
}

bool vector_to_rns_chain_host_repack_control_requested(const Args& args) {
  return args.vector_to_rns_chain && args.vector_to_rns_chain_host_repack_control;
}

bool host_api_batch_requested(const Args& args) {
  return args.host_api_batch_size > 1;
}

bool grouped_dispatch_requested(const Args& args) {
  return args.grouped_dispatch_tasks > 1;
}

bool grouped_task_executor_requested(const Args& args) {
  return host_api_batch_requested(args) || grouped_dispatch_requested(args);
}

uint32_t measured_task_count(const Args& args) {
  return grouped_dispatch_requested(args) ? args.grouped_dispatch_tasks : args.host_api_batch_size;
}

bool hip_graph_replay_requested(const Args& args) {
  return args.hip_graph_replay;
}

bool incremental_result_cache_requested(const Args& args) {
  return args.incremental_result_cache != "none";
}

bool public_incremental_result_cache_contract_requested(const Args& args) {
  return args.incremental_result_cache.find("public_contract_v1") != std::string::npos;
}

bool incremental_result_cache_full_recompute_baseline_requested(const Args& args) {
  return args.incremental_result_cache.find("full_recompute_baseline") != std::string::npos ||
         args.incremental_result_cache.find("full-recompute-baseline") != std::string::npos;
}

bool incremental_result_cache_runtime_candidate_requested(const Args& args, rns8_backend_kind selected_backend) {
  return public_incremental_result_cache_contract_requested(args) &&
         !incremental_result_cache_full_recompute_baseline_requested(args) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && !args.reuse_packed_inputs &&
         args.residue_chain_length == 1 && !args.oneshot && !args.streaming_overlap &&
         !args.hip_graph_replay && !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.vector_alu_baseline && !args.wrap64_rocwmma_candidate &&
         !args.transient_uniform_small_inputs && !args.residue_channel_fusion &&
         !args.residue_chain_final_export && !args.residue_chain_independent_final_export;
}

std::string incremental_result_cache_candidate_role(
    const Args& args,
    const BenchmarkResult& result,
    const std::string& selected_backend) {
  if (!incremental_result_cache_requested(args)) {
    return "not_requested";
  }
  if (selected_backend == "cpu-reference" || selected_backend == "cpu") {
    return "cpu_reference_baseline";
  }
  if (incremental_result_cache_full_recompute_baseline_requested(args) && selected_backend == "hip-direct") {
    return "same_backend_full_recompute_baseline";
  }
  if (result.incremental_result_cache_public_contract_available && selected_backend == "hip-direct") {
    return "public_result_cache_candidate";
  }
  return "comparison_candidate";
}

const char* incremental_result_cache_source_identity_policy(const Args& args, bool public_contract_policy) {
  if (!incremental_result_cache_requested(args)) {
    return "none";
  }
  return public_contract_policy ? "matrix_instance_id_exact_match" : "caller_visible_source_identity_required";
}

const char* incremental_result_cache_source_version_policy(const Args& args, bool public_contract_policy) {
  if (!incremental_result_cache_requested(args)) {
    return "none";
  }
  return public_contract_policy ? "nonzero_monotonic_source_versions_required" : "monotonic_source_versions_required";
}

const char* incremental_result_cache_dirty_region_policy(const Args& args, bool public_contract_policy) {
  if (!incremental_result_cache_requested(args)) {
    return "none";
  }
  return public_contract_policy ? "caller_provided_output_rectangles_full_k_recompute"
                                : "explicit_dirty_region_shape_required";
}

const char* incremental_result_cache_result_lifetime_policy(const Args& args, bool public_contract_policy) {
  if (!incremental_result_cache_requested(args)) {
    return "none";
  }
  return public_contract_policy ? "explicit_result_cache_handle_lifetime"
                                : "caller_visible_result_lifetime_required";
}

const char* incremental_result_cache_partial_recompute_policy(const Args& args, bool public_contract_policy) {
  if (!incremental_result_cache_requested(args)) {
    return "none";
  }
  return public_contract_policy ? "restore_cached_output_then_recompute_dirty_rectangles_full_k"
                                : "dirty_region_only_when_version_contract_matches";
}

rns8_dirty_region benchmark_incremental_result_cache_dirty_region(const Args& args) {
  rns8_dirty_region region{};
  region.struct_size = sizeof(region);
  region.abi_version = RNS8_ABI_VERSION;
  region.row_offset = 0;
  region.col_offset = 0;
  const int64_t requested_rows = std::max<int64_t>(1, std::min<int64_t>(args.m, static_cast<int64_t>(args.tile_m)));
  const int64_t requested_cols = std::max<int64_t>(1, std::min<int64_t>(args.n, static_cast<int64_t>(args.tile_n)));
  region.row_extent = requested_rows;
  region.col_extent = requested_cols;
  return region;
}

rns8_result_cache* create_benchmark_result_cache(rns8_context* ctx, const rns8_plan* plan) {
  rns8_result_cache_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.flags = RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
               RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
               RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
               RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  desc.max_dirty_regions = 64;
  rns8_result_cache* cache = nullptr;
  const rns8_status status = rns8_create_result_cache(ctx, plan, &desc, &cache);
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_create_result_cache", status);
  }
  return cache;
}

void capture_incremental_result_cache_info(
    BenchmarkResult& result,
    const rns8_result_cache* cache,
    const std::vector<rns8_dirty_region>& dirty_regions,
    bool public_contract_available) {
  if (!cache) {
    return;
  }
  rns8_result_cache_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const rns8_status status = rns8_get_result_cache_info(cache, &info);
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_result_cache_info", status);
  }
  result.incremental_result_cache_info_available = true;
  result.incremental_result_cache_public_contract_available = public_contract_available;
  result.incremental_result_cache_key_fingerprint = info.result_cache_key_hash;
  result.incremental_result_cache_a_matrix_instance_id = info.a_matrix_instance_id;
  result.incremental_result_cache_b_matrix_instance_id = info.b_matrix_instance_id;
  result.incremental_result_cache_c_matrix_instance_id = info.c_matrix_instance_id;
  result.incremental_result_cache_a_source_version = info.a_source_version;
  result.incremental_result_cache_b_source_version = info.b_source_version;
  result.incremental_result_cache_c_source_version = info.c_source_version;
  result.incremental_result_cache_copied_from_cache_bytes = info.copied_from_cache_bytes;
  result.incremental_result_cache_recomputed_cell_count = info.recomputed_cell_count;
  result.incremental_result_cache_allocation_bytes = info.cache_allocation_bytes;
  result.incremental_result_cache_snapshot_device_bytes = info.snapshot_device_bytes;
  result.incremental_result_cache_dirty_region_count = info.last_dirty_region_count;
  result.incremental_result_cache_recomputed_region_count = info.last_recomputed_region_count;
  result.incremental_result_cache_full_fallback = info.last_full_fallback;
  result.incremental_result_cache_last_cache_hit = info.last_cache_hit;
  result.incremental_result_cache_last_cache_miss = info.last_cache_miss;
  result.incremental_result_cache_last_stale_rejection = info.last_stale_rejection;
  result.incremental_result_cache_stale_reason = info.stale_reason;
  result.incremental_result_cache_fallback_reason = info.fallback_reason;
  result.incremental_result_cache_detail = info.detail;
  result.incremental_result_cache_dirty_regions = dirty_regions;
  if (info.last_full_fallback != 0) {
    result.incremental_result_cache_status = "full_recompute_fallback";
  } else if (info.last_recomputed_region_count != 0) {
    result.incremental_result_cache_status = "partial_recompute";
  } else if (info.last_cache_hit != 0) {
    result.incremental_result_cache_status = "cache_hit";
  } else if (info.last_cache_miss != 0) {
    result.incremental_result_cache_status = "cache_miss";
  } else {
    result.incremental_result_cache_status = "available";
  }
}

void capture_incremental_result_cache_stale_rejection(BenchmarkResult& result, const rns8_result_cache* cache) {
  if (!cache) {
    return;
  }
  rns8_result_cache_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const rns8_status status = rns8_get_result_cache_info(cache, &info);
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_result_cache_info(stale probe)", status);
  }
  result.incremental_result_cache_stale_rejection_covered = info.last_stale_rejection != 0;
  if (info.last_stale_rejection != 0) {
    result.incremental_result_cache_stale_reason = info.stale_reason;
  }
}

bool hip_graph_replay_resident_chain_requested(const Args& args) {
  return hip_graph_replay_requested(args) && !hip_graph_replay_full_bounded_pack_export_requested(args) &&
         !hip_graph_replay_full_finite_pack_export_requested(args) &&
         !hip_graph_replay_full_wrap64_pack_export_requested(args);
}

bool hip_graph_replay_full_bounded_pack_export_used(const BenchmarkResult& result) {
  return result.hip_graph_replay_scope == "direct_hip_bounded_pack_gemm_export";
}

bool oneshot_benchmark_mode(const Args& args) {
  return args.oneshot;
}

bool vector_alu_gemv_n1_shape(const Args& args) {
  return bounded_benchmark_semantics(args.semantics) && args.n == 1 && args.k >= 4096;
}

bool vector_alu_gemv_small_n_shape(const Args& args) {
  return bounded_benchmark_semantics(args.semantics) && args.n > 1 && args.n <= 4 && args.k >= 512;
}

const char* benchmark_execution_mode_name(const Args& args) {
  if (args.oneshot) {
    return "public_oneshot_transient_native_inputs";
  }
  if (args.vector_alu_baseline) {
    if (vector_alu_gemv_n1_shape(args)) {
      return "benchmark_owned_vector_alu_gemv_n1_native_buffers";
    }
    if (vector_alu_gemv_small_n_shape(args)) {
      return "benchmark_owned_vector_alu_gemv_small_n_native_buffers";
    }
    return "benchmark_owned_vector_alu_native_buffers";
  }
  if (runtime_vector_alu_backend(args)) {
    if (vector_alu_gemv_n1_shape(args)) {
      return "public_runtime_vector_alu_gemv_n1_native_buffers";
    }
    if (vector_alu_gemv_small_n_shape(args)) {
      return "public_runtime_vector_alu_gemv_small_n_native_buffers";
    }
    return "public_runtime_vector_alu_native_buffers";
  }
  if (native_to_rns_bridge_requested(args)) {
    return "auto_native_to_rns_bridge";
  }
  if (vector_to_rns_chain_requested(args)) {
    if (vector_to_rns_chain_host_repack_control_requested(args)) {
      return "vector_native_host_export_repack_direct_rns_chain";
    }
    return "vector_native_to_direct_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "benchmark_grouped_dispatch_evidence";
  }
  if (host_api_batch_requested(args)) {
    return "benchmark_host_api_batch";
  }
  if (hip_graph_replay_full_bounded_pack_export_requested(args)) {
    return "hip_graph_replay_bounded_pack_gemm_export";
  }
  if (hip_graph_replay_full_finite_pack_export_requested(args)) {
    return "hip_graph_replay_finite_u8_pack_gemm_export";
  }
  if (hip_graph_replay_full_wrap64_pack_export_requested(args)) {
    return "hip_graph_replay_wrap64_pack_gemm_export";
  }
  if (hip_graph_replay_resident_chain_requested(args)) {
    return "hip_graph_replay_resident_rns_chain";
  }
  if (args.streaming_overlap) {
    return "benchmark_streaming_overlap_resident_b_pipeline";
  }
  if (residue_chain_independent_final_export_requested(args)) {
    return "residue_chain_independent_final_host_export";
  }
  if (residue_chain_final_export_requested(args)) {
    return "residue_chain_final_host_export";
  }
  if (residue_current_output_mode(args)) {
    return "residue_current_rns_chain";
  }
  if (args.wrap64_rocwmma_candidate) {
    return "internal_wrap64_rocwmma_candidate";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "residue_channel_fusion_native_inputs";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "transient_uniform_small_i8_ab_inputs";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "transient_uniform_small_i8_a_resident_i8_b_reuse";
    }
    return "transient_native_a_resident_b_reuse";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "transient_uniform_small_i8_b_resident_i8_a_reuse";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args)) {
    return "transient_native_b_resident_a_reuse";
  }
  if (finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
      args.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "transient_native_a_resident_b_reuse";
  }
  return "persistent_resident_matrices";
}

std::string benchmark_env_value(const char* name) {
#if defined(_MSC_VER)
  char* buffer = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0 || !buffer) {
    return {};
  }
  std::string value(buffer);
  std::free(buffer);
  return value;
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
#endif
}

bool benchmark_env_flag_disabled(const char* name) {
  const std::string value = benchmark_env_value(name);
  return value == "0" || value == "false" || value == "FALSE" || value == "off" ||
         value == "OFF" || value == "no" || value == "NO";
}

bool benchmark_env_flag_enabled(const char* name) {
  const std::string value = benchmark_env_value(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "on" ||
         value == "ON" || value == "yes" || value == "YES";
}

const char* direct_hip_export_staging_policy(const Args& args, rns8_backend_kind selected_backend) {
  if (selected_backend != RNS8_BACKEND_HIP_DIRECT) {
    return "not_applicable";
  }
  if (benchmark_env_flag_disabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return "disabled_by_RNS8_HIP_PINNED_EXPORT_STAGING";
  }
  if (benchmark_env_flag_enabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return "forced_for_large_outputs_by_RNS8_HIP_PINNED_EXPORT_STAGING";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_forced_only_pending_padded_staging_evidence";
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return "exact_wide_signed_forced_only_local_gfx1100_padded_staging_loses";
  }
  return "large_padded_outputs_only_default";
}

constexpr uint64_t kDirectHipPinnedExportStagingThresholdBytes = 64u * 1024u;

const char* pack_mode_name(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "per_repeat_repack";
  }
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "prepacked_reuse";
  }
  return args.reuse_packed_a ? "prepacked_reuse_a" : "prepacked_reuse_b";
}

bool reuses_all_packed_inputs(const Args& args) {
  return args.reuse_packed_a && args.reuse_packed_b;
}

bool all_zero_direct_hip_input_pack_elided(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return selected_backend == RNS8_BACKEND_HIP_DIRECT && bounded_benchmark_semantics(args.semantics) &&
         args.bound_mode == BoundMode::PerTile && !args.oneshot && args.residue_chain_length == 1 &&
         !args.reuse_packed_inputs && result.schedule_info_available && result.schedule_info.tile_count != 0 &&
         result.zero_output_tile_count == result.schedule_info.tile_count;
}

void capture_workspace_arena_info(const rns8_workspace* workspace, BenchmarkResult& result) {
  if (!workspace) {
    return;
  }
  result.workspace_arena_info_available =
      workspace->hip_workspace_arena != nullptr || workspace->hip_workspace_arena_bytes != 0 ||
      workspace->hip_workspace_arena_suballocation_count != 0 || !workspace->hip_workspace_arena_policy.empty();
  result.workspace_arena_size_bytes = static_cast<uint64_t>(workspace->hip_workspace_arena_bytes);
  result.workspace_arena_high_water_mark_bytes =
      static_cast<uint64_t>(workspace->hip_workspace_arena_high_water_mark_bytes);
  result.workspace_arena_suballocation_count = workspace->hip_workspace_arena_suballocation_count;
  result.workspace_arena_target_id = workspace->hip_workspace_arena_target_id;
  result.workspace_arena_policy = workspace->hip_workspace_arena_policy;
}

bool expects_hipblaslt_pack_transpose_event(const Args& args) {
  return !reuses_all_packed_inputs(args);
}

template <typename PackA, typename PackB>
void pack_preused_inputs(const Args& args, uint64_t source_version, const PackA& pack_a, const PackB& pack_b) {
  if (args.reuse_packed_a) {
    pack_a(source_version);
  }
  if (args.reuse_packed_b) {
    pack_b(source_version);
  }
}

struct PackOperandTiming {
  uint64_t a_us = 0;
  uint64_t b_us = 0;
};

template <typename PackA, typename PackB>
PackOperandTiming pack_preused_inputs_timed(
    const Args& args,
    uint64_t source_version,
    const PackA& pack_a,
    const PackB& pack_b) {
  PackOperandTiming timing{};
  if (args.reuse_packed_a) {
    const auto start = std::chrono::steady_clock::now();
    pack_a(source_version);
    const auto end = std::chrono::steady_clock::now();
    timing.a_us = elapsed_us(start, end);
  }
  if (args.reuse_packed_b) {
    const auto start = std::chrono::steady_clock::now();
    pack_b(source_version);
    const auto end = std::chrono::steady_clock::now();
    timing.b_us = elapsed_us(start, end);
  }
  return timing;
}

void record_prepack_setup_timing(
    BenchmarkResult& result,
    uint64_t total_us,
    const PackOperandTiming& pack_timing,
    uint64_t runtime_cache_us) {
  result.prepack_setup_us = total_us;
  result.prepack_setup_available = true;
  result.prepack_setup_pack_a_us = pack_timing.a_us;
  result.prepack_setup_pack_b_us = pack_timing.b_us;
  result.prepack_setup_runtime_cache_us = runtime_cache_us;
  const uint64_t classified = pack_timing.a_us + pack_timing.b_us + runtime_cache_us;
  result.prepack_setup_unclassified_us = total_us > classified ? total_us - classified : 0;
}

template <typename PackA, typename PackB>
void pack_per_repeat_inputs(const Args& args, uint64_t source_version, const PackA& pack_a, const PackB& pack_b) {
  if (!args.reuse_packed_a) {
    pack_a(source_version);
  }
  if (!args.reuse_packed_b) {
    pack_b(source_version);
  }
}

struct OptionalPackOperandTiming {
  bool available = false;
  PackOperandTiming timing{};
};

void accumulate_pack_operand_timing(PackOperandTiming& total, const PackOperandTiming& increment) {
  total.a_us += increment.a_us;
  total.b_us += increment.b_us;
}

template <typename PackA, typename PackB>
PackOperandTiming pack_per_repeat_inputs_timed(
    const Args& args,
    uint64_t source_version,
    const PackA& pack_a,
    const PackB& pack_b) {
  PackOperandTiming timing{};
  if (!args.reuse_packed_a) {
    const auto start = std::chrono::steady_clock::now();
    pack_a(source_version);
    const auto end = std::chrono::steady_clock::now();
    timing.a_us = elapsed_us(start, end);
  }
  if (!args.reuse_packed_b) {
    const auto start = std::chrono::steady_clock::now();
    pack_b(source_version);
    const auto end = std::chrono::steady_clock::now();
    timing.b_us = elapsed_us(start, end);
  }
  return timing;
}

void record_pack_operand_timing(TimingSamples* samples, uint64_t pack_a_us, uint64_t pack_b_us) {
  if (!samples) {
    return;
  }
  samples->pack_a_us.push_back(pack_a_us);
  samples->pack_b_us.push_back(pack_b_us);
}

void record_pack_operand_timing(TimingSamples* samples, const PackOperandTiming& timing) {
  record_pack_operand_timing(samples, timing.a_us, timing.b_us);
}

void record_pack_operand_timing(TimingSamples* samples, const OptionalPackOperandTiming& timing) {
  if (!timing.available) {
    return;
  }
  record_pack_operand_timing(samples, timing.timing);
}

std::string prepack_reuse_operand_text(const Args& args) {
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "A and B";
  }
  if (args.reuse_packed_a) {
    return "A";
  }
  if (args.reuse_packed_b) {
    return "B";
  }
  return "none";
}

std::string per_repeat_pack_operand_text(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "A and B";
  }
  if (reuses_all_packed_inputs(args)) {
    return "none";
  }
  return args.reuse_packed_a ? "B" : "A";
}

std::vector<std::string> prepack_reuse_operands(const Args& args) {
  std::vector<std::string> operands;
  if (args.reuse_packed_a) {
    operands.push_back("A");
  }
  if (args.reuse_packed_b) {
    operands.push_back("B");
  }
  return operands;
}

void force_native_to_rns_bridge_after_pack(const Args& args, rns8_matrix* a_matrix, rns8_matrix* b_matrix) {
  if (!native_to_rns_bridge_requested(args)) {
    return;
  }
  const rns8_status status = rns8::detail::force_native_to_rns_bridge_inputs(a_matrix, b_matrix);
  if (status != RNS8_SUCCESS) {
    fail_status("force_native_to_rns_bridge_inputs", status);
  }
}

const char* prepack_reuse_strategy_name(PrepackReuseStrategy strategy) {
  switch (strategy) {
    case PrepackReuseStrategy::None:
      return "none";
    case PrepackReuseStrategy::PersistentMatrixResidency:
      return "persistent_matrix_residency";
    case PrepackReuseStrategy::RocwmmaReusableBCache:
      return "rocwmma_reusable_b_cache";
  }
  return "unknown";
}

bool uses_runtime_b_prepack_cache(const BenchmarkResult& result) {
  return result.prepack_reuse_strategy == PrepackReuseStrategy::RocwmmaReusableBCache;
}

bool should_probe_reusable_b_prepack_cache(const Args& args) {
  return args.reuse_packed_inputs && args.reuse_packed_b && !args.reuse_packed_a &&
         args.backend != RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

rns8_status run_rns_gemm_with_optional_b_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* a_matrix,
    const rns8_matrix* b_matrix,
    const rns8_prepack_cache* b_cache,
    rns8_matrix* c_matrix,
    rns8_workspace* workspace,
    rns8_result_cache* result_cache = nullptr,
    const rns8_dirty_region* dirty_regions = nullptr,
    uint32_t dirty_region_count = 0) {
  if (result_cache) {
    return rns8_gemm_rns_incremental(
        ctx, plan, a_matrix, b_matrix, c_matrix, workspace, result_cache, dirty_regions, dirty_region_count);
  }
  if (b_cache) {
    return rns8_gemm_rns_prepacked_b(ctx, plan, a_matrix, b_cache, c_matrix, workspace);
  }
  return rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
}

rns8_status run_direct_hip_resident_rns_chain_no_sync(
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    rns8_matrix* scratch_matrix,
    uint32_t chain_length,
    rns8_matrix** latest_output_matrix,
    void* stream = nullptr) {
  if (!plan || !a_matrix || !b_matrix || !c_matrix || chain_length == 0 || !latest_output_matrix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (chain_length > 1 && !scratch_matrix) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_matrix* lhs_matrix = a_matrix;
  rns8_matrix* out_matrix = c_matrix;
  rns8_matrix* final_output_matrix = c_matrix;
  for (uint32_t chain_index = 0; chain_index < chain_length; ++chain_index) {
    const rns8_status status = rns8::detail::hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
        plan, lhs_matrix, b_matrix, out_matrix, stream);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    final_output_matrix = out_matrix;
    lhs_matrix = out_matrix;
    out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
  }
  *latest_output_matrix = final_output_matrix;
  return RNS8_SUCCESS;
}

void run_hip_graph_replay_resident_chain(
    const Args& args,
    BenchmarkResult& result,
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    rns8_matrix* scratch_matrix,
    rns8_matrix*& latest_output_matrix) {
  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_reused_inputs_residue_current_rns_chain";
  result.hip_graph_replay_caveat =
      "captures resident Direct-HIP RNS GEMM launches only; A/B prepack setup and final checksum export are outside "
      "the graph";

  HipGraphReplayState graph_state{};
  std::string graph_error;
  rns8_status status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream) {
        return run_direct_hip_resident_rns_chain_no_sync(
            plan,
            a_matrix,
            b_matrix,
            c_matrix,
            scratch_matrix,
            args.residue_chain_length,
            &latest_output_matrix,
            stream);
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph warmup launch", status);
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph measured launch", status);
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  record_allocation_after_repeats(result);
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
}

bool maybe_create_reusable_b_prepack_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    rns8_matrix* b_matrix,
    rns8_prepack_cache** out,
    BenchmarkResult& result) {
  *out = nullptr;
  const rns8_status cache_status = rns8_create_prepack_cache(ctx, plan, b_matrix, RNS8_OPERAND_B, out);
  if (cache_status == RNS8_UNSUPPORTED_BACKEND) {
    return false;
  }
  if (cache_status != RNS8_SUCCESS) {
    fail_status("rns8_create_prepack_cache(B)", cache_status);
  }
  result.prepack_cache_info = {};
  result.prepack_cache_info.struct_size = sizeof(result.prepack_cache_info);
  result.prepack_cache_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status info_status = rns8_get_prepack_cache_info(*out, &result.prepack_cache_info);
  if (info_status != RNS8_SUCCESS) {
    fail_status("rns8_get_prepack_cache_info(B)", info_status);
  }
  result.prepack_cache_info_available = true;
  return true;
}

void set_backend_text(char* dst, std::size_t dst_size, const char* text) {
  if (dst_size == 0) {
    return;
  }
  std::snprintf(dst, dst_size, "%s", text ? text : "");
  dst[dst_size - 1] = '\0';
}

int64_t output_logical_ld(const Args& args) {
  if (args.output_ld_padding < 0) {
    usage_error("--output-ld-padding must be nonnegative");
  }
  if (args.n > std::numeric_limits<int64_t>::max() - args.output_ld_padding) {
    usage_error("--output-ld-padding makes output leading dimension overflow int64");
  }
  return args.n + args.output_ld_padding;
}

std::size_t output_elements(const Args& args, const char* label) {
  return checked_elements(args.m, output_logical_ld(args), label);
}

std::size_t output_limb_elements(const Args& args, uint32_t limb_count, const char* label) {
  return checked_limb_elements(args.m, output_logical_ld(args), limb_count, label);
}

const char* output_destination_layout(const Args& args) {
  return args.output_ld_padding == 0 ? "contiguous_row_major" : "padded_row_major";
}

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor) {
  const auto unsigned_value = static_cast<uint64_t>(value);
  return (unsigned_value + static_cast<uint64_t>(divisor) - 1u) / static_cast<uint64_t>(divisor);
}

std::size_t row_major_index(int64_t row, int64_t col, int64_t ld, const char* label) {
  const auto u_row = static_cast<uint64_t>(row);
  const auto u_col = static_cast<uint64_t>(col);
  const auto u_ld = static_cast<uint64_t>(ld);
  if (u_ld != 0 && u_row > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) / u_ld) {
    usage_error(std::string("matrix index overflows size_t for ") + label);
  }
  const uint64_t row_offset = u_row * u_ld;
  if (u_col > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) - row_offset) {
    usage_error(std::string("matrix index overflows size_t for ") + label);
  }
  return static_cast<std::size_t>(row_offset + u_col);
}

uint64_t profile_band_for_coordinate(int64_t coordinate, uint32_t tile_size, bool allow_zero) {
  constexpr std::array<uint64_t, 4> zero_based = {0, 1, 4, 16};
  constexpr std::array<uint64_t, 4> nonzero = {1, 2, 8, 16};
  const uint64_t tile = static_cast<uint64_t>(coordinate) / static_cast<uint64_t>(tile_size);
  return (allow_zero ? zero_based : nonzero)[static_cast<std::size_t>(tile % 4u)];
}

int64_t sample_signed_band(std::mt19937_64& rng, uint64_t magnitude) {
  if (magnitude == 0) {
    return 0;
  }
  std::uniform_int_distribution<int64_t> dist(
      -static_cast<int64_t>(magnitude),
      static_cast<int64_t>(magnitude));
  return dist(rng);
}

uint64_t sample_unsigned_band(std::mt19937_64& rng, uint64_t magnitude) {
  if (magnitude == 0) {
    return 0;
  }
  std::uniform_int_distribution<uint64_t> dist(0, magnitude);
  return dist(rng);
}

