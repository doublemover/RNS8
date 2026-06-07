
const char* benchmark_name(const Args& args) {
  if (args.oneshot) {
    return finite_benchmark_semantics(args.semantics) ? "rns8_finite_u8_public_oneshot"
                                                      : "rns8_bounded_gemm_public_oneshot";
  }
  if (args.vector_alu_baseline) {
    return "rns8_bounded_gemm_hip_vector_alu_int64_baseline";
  }
  if (runtime_vector_alu_backend(args)) {
    return "rns8_bounded_gemm_hip_vector_alu_int64_runtime";
  }
  if (native_to_rns_bridge_requested(args)) {
    return "rns8_bounded_gemm_native_to_rns_bridge";
  }
  if (vector_to_rns_chain_requested(args)) {
    return "rns8_bounded_gemm_vector_to_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "rns8_grouped_dispatch_persistent_resident";
  }
  if (host_api_batch_requested(args)) {
    return "rns8_host_api_batch_persistent_resident";
  }
  if (hip_graph_replay_full_bounded_pack_export_requested(args)) {
    return "rns8_hip_graph_replay_bounded_pack_gemm_export";
  }
  if (hip_graph_replay_full_finite_pack_export_requested(args)) {
    return "rns8_hip_graph_replay_finite_u8_pack_gemm_export";
  }
  if (hip_graph_replay_full_wrap64_pack_export_requested(args)) {
    return "rns8_hip_graph_replay_wrap64_pack_gemm_export";
  }
  if (hip_graph_replay_resident_chain_requested(args)) {
    return "rns8_hip_graph_replay_resident_rns_chain";
  }
  if (args.streaming_overlap) {
    return "rns8_streaming_overlap_resident_b_pipeline";
  }
  if (residue_chain_independent_final_export_requested(args)) {
    return "rns8_residue_chain_independent_final_host_export";
  }
  if (residue_chain_final_export_requested(args)) {
    return "rns8_residue_chain_final_host_export";
  }
  if (residue_current_output_mode(args)) {
    return "rns8_residue_current_rns_chain";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "rns8_bounded_gemm_residue_channel_fusion_experiment";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "rns8_bounded_gemm_transient_uniform_small_i8";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "rns8_wrap_u64_persistent_byte_limb";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "rns8_finite_u8_persistent_residue";
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return "rns8_exact_wide_persistent_rns";
  }
  return "rns8_bounded_gemm_persistent_rns";
}

const char* epilogue_type(const Args& args) {
  if (args.vector_alu_baseline || runtime_vector_alu_backend(args)) {
    return "direct_int64_export";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "canonical_u8_export";
  }
  if (residue_current_output_mode(args)) {
    return "residue_current_rns_output";
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return "exact_wide_signed_limb_export";
  }
  if (args.semantics == BenchSemantics::ExactWideUnsigned) {
    return "exact_wide_unsigned_limb_export";
  }
  return args.semantics == BenchSemantics::WrapU64Mod2_64 ? "low64_wrap_export" : "crt_export";
}

const char* residue_output_mode_name(const Args& args) {
  return residue_current_output_mode(args) ? "residue_current_rns" : "host_export";
}

const char* exact_wide_export_status_check_name(const Args& args) {
  if (!exact_wide_benchmark_semantics(args.semantics)) {
    return nullptr;
  }
  return exact_wide_export_status_check_required(args) ? "required_for_range_check"
                                                      : "elided_full_width_device_reconstruction";
}

const char* packed_layout_version(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "byte_limb_v1";
  }
  if (runtime_vector_alu_backend(args)) {
    return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_rowmajor_v1" : "native_u64_rowmajor_v1";
  }
  return nullptr;
}

const char* input_distribution(const Args& args) {
  if (args.input_profile == InputProfile::AdaptiveBands) {
    return args.semantics == BenchSemantics::BoundedI64 ? "signed_adaptive_bands_-16_16"
                                                        : "unsigned_adaptive_bands_0_16";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    switch (args.input_profile) {
      case InputProfile::FiniteBinary:
        return "u8_binary_0_1";
      case InputProfile::FiniteSparse:
        return "u8_sparse_90pct_zero_uniform_nonzero";
      case InputProfile::FiniteLowHamming:
        return "u8_low_hamming_powers_of_two_mod_q";
      case InputProfile::FiniteSmallCentered:
        return "u8_small_centered_minus2_2_mod_q";
      case InputProfile::FiniteFullUniform:
      case InputProfile::UniformSmall:
        return "u8_full_uniform_0_modulus_minus_1";
      case InputProfile::AdaptiveBands:
        return "invalid_finite_adaptive_bands";
    }
  }
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
    case BenchSemantics::ExactWideSigned:
      return "signed_uniform_-16_16";
    case BenchSemantics::BoundedU64:
    case BenchSemantics::ExactWideUnsigned:
      return "unsigned_uniform_0_16";
    case BenchSemantics::WrapU64Mod2_64:
      return "unsigned_rng_u64_full_range";
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "u8_uniform_0_modulus_minus_1";
  }
  return "unknown";
}

const char* tile_bound_pattern(const Args& args) {
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "exact_output_tile_max_abs_v1";
    case BenchSemantics::BoundedU64:
      return "exact_output_tile_max_unsigned_v1";
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "none";
  }
  return "unknown";
}

bool adaptive_execution_applied(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  const rns8_backend_kind selected_backend =
      result.backend_info_available ? result.backend_info.backend : info.backend;
  return args.bound_mode == BoundMode::PerTile &&
         (selected_backend == RNS8_BACKEND_CPU_REFERENCE || selected_backend == RNS8_BACKEND_HIP_DIRECT ||
          selected_backend == RNS8_BACKEND_CK || selected_backend == RNS8_BACKEND_ROCWMMA) &&
         schedule_uses_adaptive_work(result);
}

const char* selected_kernel_name(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  if (result.backend_info_available && result.backend_info.selected_kernel[0] != '\0') {
    return result.backend_info.selected_kernel;
  }
  if (adaptive_execution_applied(args, info, result)) {
    if (result.zero_output_tile_count != 0) {
      return "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3";
    }
    return "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return rns8::detail::wrap64_hip_selected_kernel_for_shape(args.m, args.n, args.k);
  }
  return nullptr;
}

int64_t benchmark_k_block_size(const Args& args, const BenchmarkResult& result) {
  if (result.backend_info_available && result.backend_info.accumulator_k_block_size > 0 &&
      result.backend_info.accumulator_k_block_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return static_cast<int64_t>(result.backend_info.accumulator_k_block_size);
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return args.k;
  }
  return std::min<int64_t>(args.k, RNS8_SAFE_INT32_K_BLOCK);
}

std::string schedule_hash_string(const Args& args, const BenchmarkResult& result) {
  std::ostringstream out;
  out << "tile_rows=" << result.schedule_info.tile_rows << ";tile_cols=" << result.schedule_info.tile_cols
      << ";selected_prefix=" << selected_execution_prefix(args, result)
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";tile_bound_hash=" << (args.bound_mode == BoundMode::PerTile ? result.tile_bound_hash : 0);
  return out.str();
}

void print_nullable_std_string(const std::string& value) {
  if (value.empty()) {
    std::cout << "null";
  } else {
    std::cout << "\"" << json_escape(value) << "\"";
  }
}

std::string resolved_next_op_hint(const Args& args, const BenchmarkResult& result) {
  if (args.next_op_hint != NextOpHint::Auto) {
    return next_op_hint_name(args.next_op_hint);
  }
  if (residue_chain_final_export_requested(args)) {
    return "final-export";
  }
  if (residue_current_output_mode(args)) {
    return "rns-gemm";
  }
  if (args.reuse_packed_b && !args.reuse_packed_a) {
    return "reuse-b";
  }
  if (runtime_vector_alu_backend(args) || args.vector_alu_baseline) {
    return "native-gemm";
  }
  if (result.lowering_info_available && result.lowering_info.native_to_rns_available) {
    return "native-to-rns";
  }
  return "final-export";
}

const char* next_op_hint_source(const Args& args) {
  return args.next_op_hint == NextOpHint::Auto ? "benchmark_default" : "cli";
}

const char* output_status_handling(const Args& args) {
  if (residue_current_output_mode(args) || (exact_wide_benchmark_semantics(args.semantics) &&
                                            !exact_wide_export_status_check_required(args))) {
    return "structurally_elided";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return "not_applicable";
  }
  return "required";
}

const char* output_status_event_policy(const Args& args) {
  const char* handling = output_status_handling(args);
  if (std::string(handling) == "required") {
    return "status_memset_and_status_d2h_labels_required_when_gpu_events_available";
  }
  if (std::string(handling) == "structurally_elided") {
    return "status_labels_zero_filled_or_absent_because_no_per_repeat_status_export_launches";
  }
  return "no_range_status_for_semantic";
}

std::string target_namespace_for_id(const std::string& target_id) {
  if (target_id == "cpu") {
    return "cpu";
  }
  if (target_id == "gfx1100") {
    return "gfx1100";
  }
  if (target_id.rfind("gfx11", 0) == 0) {
    return "gfx11xx";
  }
  if (target_id.rfind("gfx12", 0) == 0) {
    return "gfx12xx";
  }
  if (target_id.rfind("gfx94", 0) == 0 || target_id.rfind("gfx9", 0) == 0) {
    return "gfx9xx_gfx94x";
  }
  return "unknown";
}

std::string target_visibility_key() {
  for (const char* name : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
    const std::string value = trim_ascii_whitespace(environment_value(name));
    if (!value.empty()) {
      return std::string(name) + "=" + value;
    }
  }
  return "runtime-default-visible-devices";
}

std::string target_cache_key(const rns8_device_info& info, const BenchmarkResult& result) {
  std::ostringstream out;
  out << "arch=" << benchmark_key_target_id(result)
      << ";device_name=" << info.name
      << ";hip_sdk_or_rocm=" << RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION
      << ";hip_runtime=" << info.hip_runtime_version;
  return out.str();
}

std::string target_instance_id(const rns8_device_info& info, const BenchmarkResult& result) {
  std::ostringstream out;
  out << target_cache_key(info, result)
      << ";device_index=" << info.device_id
      << ";visibility=" << target_visibility_key();
  return out.str();
}

std::string target_review_group_key(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const std::string target_id = benchmark_key_target_id(result);
  std::ostringstream out;
  out << target_namespace_for_id(target_id)
      << "/target=" << target_id
      << "/backend=" << selected_backend
      << "/semantics=" << semantics_name(args.semantics)
      << "/device_index=" << info.device_id
      << "/device_name=" << info.name
      << "/target_instance=" << target_instance_id(info, result)
      << "/visibility=" << target_visibility_key()
      << "/configured=" << RNS8_CONFIGURED_AMDGPU_TARGETS
      << "/runtime=" << info.hip_runtime_version;
  return out.str();
}

std::string benchmark_pack_layout(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_byte_limb_planes";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "finite_u8_centered_residue";
  }
  if (runtime_vector_alu_backend(args) || args.vector_alu_baseline) {
    return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_row_major" : "native_u64_row_major";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "native_i8_row_major_residue_channel_width3";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) ||
      (bounded_native_a_reuse_b_requested(args) && bounded_native_a_reuse_b_uniform_small_a(args)) ||
      bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "native_i8_row_major_uniform_small";
  }
  if (result.packing_info_available && result.packing_info.uses_matrix_engine_pack_layout) {
    return "matrix_engine_transient_pack_layout";
  }
  if (result.packing_info_available && result.packing_info.uses_transient_pack_workspace) {
    return "transient_backend_pack_layout";
  }
  return "resident_rns_residue_planes";
}

const char* benchmark_fusion_mode(const Args& args) {
  return bounded_residue_channel_fusion_requested(args)
      ? "residue_channel_width3_experimental_benchmark_only"
      : "none";
}

uint32_t benchmark_residue_group_width(const Args& args) {
  return bounded_residue_channel_fusion_requested(args) ? 3u : 1u;
}

const char* benchmark_residue_group_layout(const Args& args) {
  return bounded_residue_channel_fusion_requested(args)
      ? "first_prefix9_moduli_contiguous_width3_groups"
      : "one_modulus_per_residue_plane";
}

std::string generated_reducer_identity(const Args& args, const BenchmarkResult& result) {
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  if (bounded_benchmark_semantics(args.semantics) &&
      selected_prefix > 0 && selected_prefix <= RNS8_DEFAULT_BOUNDED_PREFIX &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_fixed_prefix_" + std::to_string(selected_prefix) + "_generated_reducer_v1";
  }
  if (exact_wide_benchmark_semantics(args.semantics) &&
      selected_prefix == RNS8_MAX_SUPPORTED_PREFIX &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_fixed_prefix_20_generated_reducer_v1";
  }
  if (finite_benchmark_semantics(args.semantics) &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_finite_modulus_" + std::to_string(args.finite_modulus) + "_fixed_reducer_v1";
  }
