bool bounded_native_a_reuse_b_path(const Args& args, const BenchmarkResult& result) {
  return bounded_native_a_reuse_b_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_uniform_small_i8_ab_reuse_a_path(const Args& args, const BenchmarkResult& result) {
  return bounded_uniform_small_i8_ab_reuse_a_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_native_b_reuse_a_u64_large_colpair_path(const Args& args, const BenchmarkResult& result) {
  return bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_direct_hip_skinny_gemv_n1_path(const Args& args, const BenchmarkResult& result) {
  return (args.semantics == BenchSemantics::BoundedI64 ||
          args.semantics == BenchSemantics::BoundedU64) &&
         args.n == 1 && args.residue_chain_length == 1 &&
         args.bound_mode == BoundMode::Global &&
         !args.oneshot && !args.streaming_overlap && !args.hip_graph_replay &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.vector_alu_baseline && !args.wrap64_rocwmma_candidate &&
         !args.transient_uniform_small_inputs && !args.residue_channel_fusion &&
         !args.residue_chain_final_export && !args.residue_chain_independent_final_export &&
         !args.reuse_packed_inputs && !args.reuse_packed_a && !args.reuse_packed_b &&
         !args.incremental_result_cache.empty() && args.incremental_result_cache == "none" &&
         args.tile_shape_variant == "default" &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_direct_hip_skinny_gemv_small_n_path(const Args& args, const BenchmarkResult& result) {
  return (args.semantics == BenchSemantics::BoundedI64 ||
          args.semantics == BenchSemantics::BoundedU64) &&
         args.n > 1 && args.n <= 4 && args.residue_chain_length == 1 &&
         args.bound_mode == BoundMode::Global &&
         !args.oneshot && !args.streaming_overlap && !args.hip_graph_replay &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.vector_alu_baseline && !args.wrap64_rocwmma_candidate &&
         !args.transient_uniform_small_inputs && !args.residue_channel_fusion &&
         !args.residue_chain_final_export && !args.residue_chain_independent_final_export &&
         !args.reuse_packed_inputs && !args.reuse_packed_a && !args.reuse_packed_b &&
         !args.incremental_result_cache.empty() && args.incremental_result_cache == "none" &&
         args.tile_shape_variant == "default" &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_uniform_small_i8_ab_transient_path(const Args& args, const BenchmarkResult& result) {
  return (bounded_uniform_small_i8_ab_transient_requested(args) ||
          bounded_residue_channel_fusion_requested(args)) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool finite_native_a_reuse_b_path(const Args& args, const BenchmarkResult& result) {
  return finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

bool finite_sparse_a_4_to_2_path(const Args& args, const BenchmarkResult& result) {
  return args.sparse_a_4_to_2 && !args.sparse_a_4_to_2_dense_baseline &&
         finite_benchmark_semantics(args.semantics) &&
         result.backend_info_available &&
         (result.backend_info.backend == RNS8_BACKEND_CPU_REFERENCE ||
          result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS);
}

bool rns_sparse_a_4_to_2_path(const Args& args, const BenchmarkResult& result) {
  return args.sparse_a_4_to_2 && !args.sparse_a_4_to_2_dense_baseline &&
         rns_chain_benchmark_semantics(args.semantics) &&
         result.backend_info_available &&
         (result.backend_info.backend == RNS8_BACKEND_CPU_REFERENCE ||
          result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS);
}

const char* benchmark_execution_mode_name(const Args& args, const BenchmarkResult& result) {
  if (rns_sparse_a_4_to_2_path(args, result)) {
    return "resident_sparse_a_explicit_4_to_2_contract_dense_b_rns";
  }
  if (bounded_direct_hip_skinny_gemv_n1_path(args, result)) {
    return "direct_hip_skinny_gemv_n1_resident_rns";
  }
  if (bounded_direct_hip_skinny_gemv_small_n_path(args, result)) {
    return "direct_hip_skinny_gemv_small_n_resident_rns";
  }
  if (bounded_native_a_reuse_b_requested(args) && !bounded_native_a_reuse_b_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args) && !bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
      !bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) && !bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return "persistent_resident_matrices";
  }
  return benchmark_execution_mode_name(args);
}

const char* backend_metadata_source(const Args& args, const BenchmarkResult& result) {
  if (finite_sparse_a_4_to_2_path(args, result) || rns_sparse_a_4_to_2_path(args, result)) {
    return "rns8_bench_explicit_sparse_a_4_to_2_path";
  }
  if (bounded_direct_hip_skinny_gemv_n1_path(args, result)) {
    return "rns8_bench_skinny_gemv_n1_path";
  }
  if (bounded_direct_hip_skinny_gemv_small_n_path(args, result)) {
    return "rns8_bench_skinny_gemv_small_n_path";
  }
  if (bounded_native_a_reuse_b_requested(args) && !bounded_native_a_reuse_b_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args) && !bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
      !bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) && !bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  return backend_metadata_source(args);
}

const char* bounded_direct_hip_skinny_gemv_n1_kernel(const Args& args) {
  return args.semantics == BenchSemantics::BoundedI64
      ? "direct_hip_prefix9_rns_gemv_n1_i64_v1"
      : "direct_hip_prefix9_rns_gemv_n1_u64_v1";
}

const char* bounded_direct_hip_skinny_gemv_small_n_kernel(const Args& args) {
  return args.semantics == BenchSemantics::BoundedI64
      ? "direct_hip_prefix9_rns_gemv_small_n_i64_v1"
      : "direct_hip_prefix9_rns_gemv_small_n_u64_v1";
}

const char* bounded_direct_hip_skinny_gemv_n1_epilogue() {
  return "resident_rns_gemv_n1_centered_residue_then_crt_export";
}

const char* bounded_direct_hip_skinny_gemv_small_n_epilogue() {
  return "resident_rns_gemv_small_n_centered_residue_then_crt_export";
}

const char* bounded_direct_hip_skinny_gemv_n1_workspace() {
  return "resident_rns_inputs_skinny_n1_output";
}

const char* bounded_direct_hip_skinny_gemv_small_n_workspace() {
  return "resident_rns_inputs_skinny_n_le4_output";
}

std::string bounded_direct_hip_skinny_gemv_n1_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue,
    uint64_t bound) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";bound=" << bound
      << ";exact_wide_limb_count="
      << (exact_wide_benchmark_semantics(args.semantics) ? args.exact_wide_limb_count : 0)
      << ";input_profile=" << input_profile_name(args)
      << ";prefix=" << RNS8_DEFAULT_BOUNDED_PREFIX
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=direct_hip_skinny_gemv_n1_resident_rns";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

std::string bounded_direct_hip_skinny_gemv_small_n_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue,
    uint64_t bound) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";bound=" << bound
      << ";exact_wide_limb_count=0"
      << ";input_profile=" << input_profile_name(args)
      << ";prefix=" << RNS8_DEFAULT_BOUNDED_PREFIX
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=direct_hip_skinny_gemv_small_n_resident_rns";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

std::string bounded_native_a_reuse_b_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue,
    uint64_t bound) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";bound=" << bound
      << ";input_profile=" << input_profile_name(args)
      << ";prefix=" << RNS8_DEFAULT_BOUNDED_PREFIX
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=" << benchmark_execution_mode_name(args);
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

std::string finite_native_a_reuse_b_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";finite_modulus=" << args.finite_modulus
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";execution=transient_native_a_resident_b_reuse";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

void apply_bounded_oneshot_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!args.oneshot || !result.backend_info_available) {
    return;
  }
  result.backend_info.performance_validated = 0;
  if (result.backend_info.backend != RNS8_BACKEND_HIP_DIRECT ||
      args.bound_mode != BoundMode::Global ||
      selected_execution_prefix(args, result) != 9) {
    return;
  }
  const char* kernel = bounded_oneshot_kernel(args);
  const char* epilogue = "native_input_centered_residue_then_crt_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_inputs_to_resident_rns_output");
  const std::string key = bounded_oneshot_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_finite_oneshot_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!args.oneshot || !finite_benchmark_semantics(args.semantics) || !result.backend_info_available) {
    return;
  }
  result.backend_info.performance_validated = 0;
  if (result.backend_info.backend != RNS8_BACKEND_HIP_DIRECT) {
    return;
  }
  const char* kernel = "direct_hip_native_finite_u8_gemm_v1";
  if (args.finite_modulus == 256) {
    kernel = "direct_hip_native_finite_u8_gemm_mod256_v1";
  } else if (args.finite_modulus == 255) {
    kernel = "direct_hip_native_finite_u8_gemm_mod255_v1";
  } else if (args.finite_modulus == 251) {
    kernel = "direct_hip_native_finite_u8_gemm_mod251_v1";
  }
  const char* epilogue = "native_u8_centered_residue_then_canonical_u8_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_u8_inputs_to_resident_finite_output");
  const std::string key = finite_oneshot_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_native_a_reuse_b_backend_metadata(const Args& args, BenchmarkResult& result, uint64_t bound) {
  if (!bounded_native_a_reuse_b_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_native_a_reuse_b_kernel(args);
  const char* epilogue = bounded_native_a_reuse_b_epilogue(args);
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_a_resident_rns_b_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_direct_hip_skinny_gemv_n1_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_direct_hip_skinny_gemv_n1_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_direct_hip_skinny_gemv_n1_kernel(args);
  const char* epilogue = bounded_direct_hip_skinny_gemv_n1_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      bounded_direct_hip_skinny_gemv_n1_workspace());
  const std::string key = bounded_direct_hip_skinny_gemv_n1_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_direct_hip_skinny_gemv_small_n_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_direct_hip_skinny_gemv_small_n_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_direct_hip_skinny_gemv_small_n_kernel(args);
  const char* epilogue = bounded_direct_hip_skinny_gemv_small_n_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      bounded_direct_hip_skinny_gemv_small_n_workspace());
  const std::string key =
      bounded_direct_hip_skinny_gemv_small_n_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_uniform_small_i8_ab_reuse_a_kernel();
  const char* epilogue = bounded_uniform_small_i8_ab_reuse_a_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_i8_b_resident_i8_a_rns_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_native_b_reuse_a_backend_metadata(const Args& args, BenchmarkResult& result, uint64_t bound) {
  if (!bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_native_b_reuse_a_u64_large_colpair_kernel();
  const char* epilogue = bounded_native_b_reuse_a_u64_large_colpair_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_b_resident_rns_a_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_uniform_small_i8_ab_transient_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_residue_channel_fusion_requested(args)
      ? bounded_residue_channel_fusion_kernel()
      : bounded_uniform_small_i8_ab_transient_kernel();
  const char* epilogue = bounded_residue_channel_fusion_requested(args)
      ? bounded_residue_channel_fusion_epilogue()
      : bounded_uniform_small_i8_ab_transient_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      bounded_residue_channel_fusion_requested(args)
          ? "width3_residue_fusion_transient_i8_inputs"
          : "transient_i8_a_transient_i8_b_rns_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_finite_native_a_reuse_b_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!finite_native_a_reuse_b_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = finite_native_a_reuse_b_kernel(args.finite_modulus);
  const char* epilogue = "native_a_centered_resident_b_residue_then_canonical_u8_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_u8_a_resident_finite_b_output");
  const std::string key = finite_native_a_reuse_b_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

bool amdgpu_builtin_cdna3_dense_variant_name(const Args& args) {
  return args.tile_shape_variant == "amdgpu-cdna3-mfma-16x16x32" ||
         args.tile_shape_variant == "amdgpu-cdna3-mfma-32x32x16";
}

rns8::detail::amdgpu_builtins_dense_kernel_variant amdgpu_builtin_dense_variant_for_args(const Args& args) {
  if (args.tile_shape_variant == "amdgpu-cdna3-mfma-16x16x32") {
    return rns8::detail::amdgpu_builtins_dense_kernel_variant::cdna3_mfma_16x16x32;
  }
  if (args.tile_shape_variant == "amdgpu-cdna3-mfma-32x32x16") {
    return rns8::detail::amdgpu_builtins_dense_kernel_variant::cdna3_mfma_32x32x16;
  }
  return rns8::detail::amdgpu_builtins_dense_kernel_variant::auto_select;
}

void apply_amdgpu_builtin_dense_variant_backend_metadata(const Args& args, BenchmarkResult& result) {
  rns8::detail::amdgpu_builtins_set_dense_kernel_variant_override(amdgpu_builtin_dense_variant_for_args(args));
  if (!amdgpu_builtin_cdna3_dense_variant_name(args)) {
    return;
  }
  if (result.backend_info.backend != RNS8_BACKEND_AMDGPU_BUILTINS || finite_benchmark_semantics(args.semantics)) {
    usage_error("amdgpu-cdna3 MFMA tile-shape variants require dense RNS --backend amdgpu-builtins");
  }
  if (result.target_id.rfind("gfx942", 0) != 0) {
    usage_error("amdgpu-cdna3 MFMA tile-shape variants require a gfx942 CDNA3 target");
  }
  const char* kernel =
      args.tile_shape_variant == "amdgpu-cdna3-mfma-32x32x16"
          ? "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_centered_epilogue_v1"
          : "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(
      result.backend_info.epilogue_mode,
      sizeof(result.backend_info.epilogue_mode),
      "amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export");
  std::ostringstream out;
  out << "backend=amdgpu-builtins"
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=" << selected_execution_prefix(args, result)
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";tile_shape_variant=" << args.tile_shape_variant;
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export";
  const std::string key = out.str();
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

const char* finite_sparse_a_4_to_2_kernel(const BenchmarkResult& result) {
  if (result.backend_info.backend == RNS8_BACKEND_CPU_REFERENCE) {
    return "cpu_reference_sparse_a_4_to_2_expand_then_finite_u8_gemm_v1";
  }
  if (result.target_id.rfind("gfx120", 0) == 0) {
    return "amdgpu_builtin_rdna4_swmmac_i32_16x16x32_iu8_sparse_a_v1";
  }
  return "amdgpu_builtin_cdna3_smfmac_i32_16x16x64_i8_sparse_a_v1";
}

std::string finite_sparse_a_4_to_2_autotune_key(const Args& args, const BenchmarkResult& result, const char* kernel) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";finite_modulus=" << args.finite_modulus
      << ";prefix=0"
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";sparse_contract=a_4_to_2_structured_k_v1"
      << ";sparse_operand=A"
      << ";sparse_group_size=4"
      << ";sparse_nonzeros_per_group=2"
      << ";sparse_index_layout=canonical_2bit_k_groups_v1"
      << ";sparse_value_signedness=unsigned_u8"
      << ";dense_operand=B";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=finite_u8_centered_residue_then_canonical_u8_export";
  return out.str();
}

void apply_finite_sparse_a_4_to_2_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!finite_sparse_a_4_to_2_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = finite_sparse_a_4_to_2_kernel(result);
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(
      result.backend_info.epilogue_mode,
      sizeof(result.backend_info.epilogue_mode),
      result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS
          ? "amdgpu_builtin_sparse_a_fused_i32_to_centered_residue_then_canonical_u8_export"
          : "finite_u8_centered_residue_then_canonical_u8_export");
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "resident_sparse_a_explicit_4_to_2_contract_dense_b_finite_u8");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS
          ? "amdgpu_builtin_sparse_a_matrix_core_isa_gate_no_divide"
          : "cpu_reference_expand_sparse_a_then_dense_finite_u8");
  const std::string key = finite_sparse_a_4_to_2_autotune_key(args, result, kernel);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

const char* rns_sparse_a_4_to_2_kernel(const BenchmarkResult& result) {
  if (result.backend_info.backend == RNS8_BACKEND_CPU_REFERENCE) {
    return "cpu_reference_sparse_a_4_to_2_expand_then_rns_gemm_v1";
  }
  if (result.target_id.rfind("gfx120", 0) == 0) {
    return "amdgpu_builtin_rdna4_swmmac_i32_16x16x32_iu8_sparse_a_v1";
  }
  return "amdgpu_builtin_cdna3_smfmac_i32_16x16x64_i8_sparse_a_v1";
}

std::string rns_sparse_a_4_to_2_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    uint64_t bound) {
  std::ostringstream out;
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";bound=" << bound
      << ";input_profile=" << input_profile_name(args)
      << ";prefix=" << selected_prefix
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";sparse_contract=a_4_to_2_structured_k_v1"
      << ";sparse_operand=A"
      << ";sparse_group_size=4"
      << ";sparse_nonzeros_per_group=2"
      << ";sparse_index_layout=canonical_2bit_k_groups_v1"
      << ";sparse_value_signedness=signed_i8"
      << ";dense_operand=B";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=sparse_a_centered_residue_rns_output";
  return out.str();
}

void apply_rns_sparse_a_4_to_2_backend_metadata(const Args& args, BenchmarkResult& result, uint64_t bound) {
  if (!rns_sparse_a_4_to_2_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = rns_sparse_a_4_to_2_kernel(result);
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(
      result.backend_info.epilogue_mode,
      sizeof(result.backend_info.epilogue_mode),
      result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS
          ? "amdgpu_builtin_sparse_a_fused_i32_to_centered_residue_rns_output"
          : "centered_residue_rns_output");
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "resident_sparse_a_explicit_4_to_2_contract_dense_b_rns");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      result.backend_info.backend == RNS8_BACKEND_AMDGPU_BUILTINS
          ? "amdgpu_builtin_sparse_a_matrix_core_isa_gate_no_divide"
          : "cpu_reference_expand_sparse_a_then_dense_rns");
  const std::string key = rns_sparse_a_4_to_2_autotune_key(args, result, kernel, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

std::string vector_alu_autotune_key(const Args& args, const BenchmarkResult& result, const char* kernel) {
  std::ostringstream out;
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  out << "backend=hip-vector-alu-int64"
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=" << selected_prefix
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count;
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=direct_int64_export";
  return out.str();
}

void fill_vector_alu_backend_info(const Args& args, BenchmarkResult& result, uint64_t workspace_bytes) {
  const bool signed_semantics = args.semantics == BenchSemantics::BoundedI64;
  const bool gemv_n1 = vector_alu_gemv_n1_shape(args);
  const char* kernel = signed_semantics
      ? (gemv_n1 ? "hip_vector_alu_i64_gemv_n1_exact_192b_v1" : "hip_vector_alu_i64_exact_192b_v1")
      : (gemv_n1 ? "hip_vector_alu_u64_gemv_n1_exact_192b_v1" : "hip_vector_alu_u64_exact_192b_v1");
  result.backend_info = {};
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  result.backend_info.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  result.backend_info.is_accelerator = 0;
  result.backend_info.is_correctness_backend = 1;
  result.backend_info.is_matrix_engine_backend = 0;
  result.backend_info.compiled_kernel_available = 1;
  result.backend_info.exact_differential_validated = 1;
  result.backend_info.performance_validated = 0;
  result.backend_info.workspace_required_bytes = workspace_bytes;
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.accelerator_library, sizeof(result.backend_info.accelerator_library), "HIP runtime");
  set_backend_text(result.backend_info.accelerator_version, sizeof(result.backend_info.accelerator_version), RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  set_backend_text(result.backend_info.capability_status, sizeof(result.backend_info.capability_status), "benchmark_only_vector_alu_baseline");
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), "direct_int64_export");
  set_backend_text(result.backend_info.workspace_mode, sizeof(result.backend_info.workspace_mode), "benchmark_owned_device_buffers");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      "source_level_192bit_limb_accumulator_no_matrix_engine");
  fill_vector_alu_accumulator_info(args, result.backend_info);
  const std::string key = vector_alu_autotune_key(args, result, kernel);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
  result.backend_info_available = true;
}

std::size_t wrap64_compact_limb_bytes(int64_t rows, int64_t cols, const char* label) {
  return checked_bytes(checked_elements(rows, cols, label), sizeof(uint64_t), label);
}

uint64_t wrap64_rocwmma_candidate_workspace_bytes(const Args& args) {
  const std::size_t a_bytes = wrap64_compact_limb_bytes(args.m, args.k, "candidate A byte limbs");
  const std::size_t b_bytes = wrap64_compact_limb_bytes(args.k, args.n, "candidate B byte limbs");
  const std::size_t c_bytes = wrap64_compact_limb_bytes(args.m, args.n, "candidate C byte limbs");
  const std::size_t ab_bytes = checked_add_bytes(a_bytes, b_bytes, "wrap64 candidate A+B byte limbs");
  const std::size_t total = checked_add_bytes(ab_bytes, c_bytes, "wrap64 candidate A+B+C byte limbs");
  return static_cast<uint64_t>(total);
}

void fill_wrap64_rocwmma_candidate_schedule(const Args& args, BenchmarkResult& result) {
  result.schedule_source = kWrap64RocwmmaCandidateScheduleSource;
  result.schedule_info = {};
  result.schedule_info.struct_size = sizeof(result.schedule_info);
  result.schedule_info.abi_version = RNS8_ABI_VERSION;
  result.schedule_info.tile_m = kWrap64RocwmmaCandidateTile;
  result.schedule_info.tile_n = kWrap64RocwmmaCandidateTile;
  result.schedule_info.tile_rows = ceil_div_i64_u32(args.m, kWrap64RocwmmaCandidateTile);
  result.schedule_info.tile_cols = ceil_div_i64_u32(args.n, kWrap64RocwmmaCandidateTile);
  result.schedule_info.tile_count = result.schedule_info.tile_rows * result.schedule_info.tile_cols;
  result.schedule_info.min_required_prefix = 0;
  result.schedule_info.max_required_prefix = 0;
  result.schedule_info.min_selected_prefix = 0;
  result.schedule_info.max_selected_prefix = 0;
  result.schedule_info.prefix_group_count = 0;
  result.schedule_info.adaptive_prefix_active = 0;
  result.schedule_info.adaptive_skip_active = 0;
  result.schedule_info.range_bit_length = 0;
  result.schedule_info_available = true;
}

std::string wrap64_rocwmma_candidate_autotune_key(const Args& args, const BenchmarkResult& result) {
  std::ostringstream out;
  out << "backend=" << kWrap64RocwmmaCandidateRequestedBackend
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=0"
      << ";tile_m=" << result.schedule_info.tile_m
      << ";tile_n=" << result.schedule_info.tile_n
      << ";groups=0;adaptive_prefix=0;adaptive_skip=0";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kWrap64RocwmmaCandidateSelectedKernel
      << ";epilogue=low64_wrap_export";
  return out.str();
}

void fill_wrap64_rocwmma_candidate_backend_info(const Args& args, BenchmarkResult& result, uint64_t workspace_bytes) {
  result.backend_info = {};
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  result.backend_info.backend = RNS8_BACKEND_ROCWMMA;
  result.backend_info.is_accelerator = 1;
  result.backend_info.is_correctness_backend = 0;
  result.backend_info.is_matrix_engine_backend = 1;
  result.backend_info.compiled_kernel_available = 1;
  result.backend_info.exact_differential_validated = 1;
  result.backend_info.performance_validated = 0;
  result.backend_info.workspace_required_bytes = workspace_bytes;
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kWrap64RocwmmaCandidateSelectedKernel);
  set_backend_text(result.backend_info.accelerator_library, sizeof(result.backend_info.accelerator_library), "rocWMMA");
  set_backend_text(
      result.backend_info.accelerator_version,
      sizeof(result.backend_info.accelerator_version),
      "repo-local release/rocm-rel-7.1");
  set_backend_text(
      result.backend_info.capability_status,
      sizeof(result.backend_info.capability_status),
      "internal_wrap64_matrix_engine_candidate");
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), "low64_wrap_export");
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "benchmark_owned_compact_byte_limb_device_buffers");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      "rocwmma_wrap64_byte_gemm36_matrix_isa_gate_no_divide");
  fill_wrap64_rocwmma_candidate_accumulator_info(args, result.backend_info);
  const std::string key = wrap64_rocwmma_candidate_autotune_key(args, result);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
  result.backend_info_available = true;
}

std::string prefix_event_label(const char* prefix, uint32_t index, const char* suffix) {
  char buffer[96];
  std::snprintf(buffer, sizeof(buffer), "%s%02u_%s", prefix, index, suffix);
  return std::string(buffer);
}

uint32_t gpu_event_selected_prefix_count(const Args& args, const BenchmarkResult& result) {
  if (finite_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
      args.vector_alu_baseline || runtime_vector_alu_backend(args)) {
    return 0;
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix > 0) {
    return std::min<uint32_t>(result.schedule_info.max_selected_prefix, RNS8_DEFAULT_MODULUS_COUNT);
  }
  return std::min<uint32_t>(benchmark_prefix(args), RNS8_DEFAULT_MODULUS_COUNT);
}

