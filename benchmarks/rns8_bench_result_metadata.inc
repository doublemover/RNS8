void print_u64_array(const std::vector<uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_double_array(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_string_array(const std::vector<std::string>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "\"" << json_escape(values[i]) << "\"";
  }
  std::cout << "]";
}

std::vector<std::string> required_speedup_baselines(const Args& args, const char* selected_backend) {
  std::vector<std::string> baselines;
  const std::string selected = selected_backend ? selected_backend : "";
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
    case BenchSemantics::BoundedU64:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-vector-alu-int64") {
        baselines.push_back("same_contract_direct_hip_vector_alu_int64");
      }
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      if (args.oneshot) {
        baselines.push_back("same_contract_direct_hip_persistent_rns");
      }
      break;
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      break;
    case BenchSemantics::WrapU64Mod2_64:
      baselines.push_back("same_contract_cpu_wrap64_byte_limb_reference");
      baselines.push_back("same_contract_direct_hip_wrap64_byte_gemm36");
      break;
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      if (args.oneshot) {
        baselines.push_back("same_contract_direct_hip_persistent_finite_u8");
      }
      break;
  }
  return baselines;
}

void print_comparison_baseline(const Args& args, const rns8_device_info& info, const BenchmarkResult& result) {
  const bool performance_validated = result.backend_info.performance_validated != 0;
  const char* selected = selected_backend_name(args, info, &result);
  std::cout << "  \"comparison_baseline\": {\n";
  std::cout << "    \"status\": \""
            << (performance_validated ? "reviewed_release_same_contract_baseline"
                                      : "required_not_recorded")
            << "\",\n";
  std::cout << "    \"speedup_claimed\": false,\n";
  std::cout << "    \"selected_reference\": null,\n";
  std::cout << "    \"required_before_speedup_claim\": ";
  print_string_array(required_speedup_baselines(args, selected));
  std::cout << ",\n";
  std::cout << "    \"reason\": \"";
  if (performance_validated) {
    std::cout << "performance_validated=true from an exact reviewed release autotune cache hit for this plan key";
  } else {
    std::cout << "performance_validated=false; raw capture has not been promoted against same-contract CPU and GPU "
                 "baseline evidence";
  }
  std::cout << "\"\n";
  std::cout << "  },\n";
}

void print_single_u64_array(uint64_t value) {
  std::cout << "[" << value << "]";
}

void capture_schedule_info(rns8_plan* plan, BenchmarkResult& result) {
  result.schedule_info.struct_size = sizeof(result.schedule_info);
  result.schedule_info.abi_version = RNS8_ABI_VERSION;
  const auto start = std::chrono::steady_clock::now();
  rns8_status status = rns8_get_plan_schedule_info(plan, &result.schedule_info);
  if (status == RNS8_SUCCESS && result.schedule_info.tile_count != 0) {
    if (result.schedule_info.tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      usage_error("plan tile schedule is too large to inspect for skip counters");
    }
    std::vector<rns8_plan_tile_schedule_entry> entries(
        static_cast<std::size_t>(result.schedule_info.tile_count));
    uint64_t written = 0;
    status = rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written);
    if (status == RNS8_SUCCESS && written != static_cast<uint64_t>(entries.size())) {
      usage_error("plan tile schedule query returned an unexpected entry count");
    }
    if (status == RNS8_SUCCESS) {
      for (const auto& entry : entries) {
        if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
          ++result.zero_output_tile_count;
          result.zero_output_selected_residue_plane_count += entry.selected_prefix;
        }
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_schedule_info/rns8_get_plan_tile_schedule", status);
  }
  result.schedule_query_us = elapsed_us(start, end);
  result.schedule_info_available = true;
}

void capture_backend_info(rns8_plan* plan, BenchmarkResult& result) {
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status status = rns8_get_plan_backend_info(plan, &result.backend_info);
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_backend_info", status);
  }
  result.backend_info_available = true;
  update_result_target_id_from_key(result);
}

void capture_export_plan_info(const Args& args, rns8_plan* plan, BenchmarkResult& result) {
  const uint32_t limb_count = exact_wide_benchmark_semantics(args.semantics) ? args.exact_wide_limb_count : 0;
  const rns8::detail::api::export_reconstruction_plan export_plan =
      rns8::detail::api::make_export_plan(*plan, c_semantics(args.semantics), limb_count);
  result.export_plan_info_available = true;
  result.export_plan_source = "rns8_internal_export_plan";
  result.export_output_layout = rns8::detail::api::export_output_layout_name(export_plan.output_layout);
  result.export_selector_status_policy =
      rns8::detail::api::export_status_policy_name(export_plan.status_policy);
  result.export_d2h_policy = rns8::detail::api::export_d2h_policy_name(export_plan.d2h_policy);
  result.export_selected_kernel = export_plan.selected_export_kernel ? export_plan.selected_export_kernel : "";
  result.export_selector_key = export_plan.selector_key;
  result.export_selector_policy = export_plan.selector_policy;
  result.export_semantic_contract = export_plan.semantic_contract;
  result.export_backend = export_plan.backend;
  result.export_target_id = export_plan.target_id;
  result.export_prefix_contract = export_plan.prefix_contract;
  result.export_signedness = export_plan.signedness;
  result.export_final_output_mode = export_plan.final_output_mode;
  result.export_cache_visibility = export_plan.cache_visibility;
  result.export_stale_entry_reason = export_plan.stale_entry_reason;
  result.export_status_elision_reason = export_plan.status_elision_reason;
  result.export_limb_count = export_plan.limb_count;
  result.export_requires_tile_metadata = export_plan.requires_hip_tile_metadata;
  result.export_all_zero_tiled_output = export_plan.all_zero_tiled_output;
}

void capture_packing_and_lowering_info(const Args& args, rns8_plan* plan, BenchmarkResult& result) {
  result.packing_info.struct_size = sizeof(result.packing_info);
  result.packing_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status packing_status = rns8_get_plan_packing_info(plan, &result.packing_info);
  if (packing_status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_packing_info", packing_status);
  }
  result.packing_info_available = true;
  if (result.backend_info_available && result.schedule_info_available) {
    result.lowering_info =
        rns8::detail::describe_plan_lowering(result.backend_info, result.packing_info, result.schedule_info);
    result.lowering_info_available = true;
  }
  capture_export_plan_info(args, plan, result);
}

void append_accumulator_key_fields(std::ostringstream& out, const rns8_plan_backend_info& info) {
  out << ";accumulator_type=" << info.accumulator_type
      << ";accumulator_signedness=" << info.accumulator_signedness
      << ";accumulator_modulus_policy=" << info.accumulator_modulus_policy
      << ";k_block_size=" << info.accumulator_k_block_size
      << ";k_block_cap=" << info.accumulator_k_block_cap;
}

void fill_vector_alu_accumulator_info(const Args& args, rns8_plan_backend_info& info) {
  info.accumulator_k_block_size = args.k > 0 ? static_cast<uint64_t>(args.k) : 0u;
  info.accumulator_k_block_cap = 0;
  info.accumulator_modulus = 0;
  info.accumulator_max_lhs_abs = 0;
  info.accumulator_max_rhs_abs = 0;
  info.accumulator_max_product = 0;
  info.accumulator_uses_int32_inner_product = 0;
  info.accumulator_safe_for_k_block = 1;
  set_backend_text(
      info.accumulator_input_domain,
      sizeof(info.accumulator_input_domain),
      args.semantics == BenchSemantics::BoundedI64 ? "native_i64_values" : "native_u64_values");
  set_backend_text(
      info.accumulator_signedness,
      sizeof(info.accumulator_signedness),
      args.semantics == BenchSemantics::BoundedI64 ? "signed_i64x_signed_i64" : "unsigned_u64x_unsigned_u64");
  set_backend_text(info.accumulator_type, sizeof(info.accumulator_type), "software_192bit_limb");
  set_backend_text(
      info.accumulator_modulus_policy,
      sizeof(info.accumulator_modulus_policy),
      "native_exact_integer_output");
  set_backend_text(
      info.accumulator_safety_status,
      sizeof(info.accumulator_safety_status),
      "exact_192bit_limb_no_int32_k_cap");
}

void fill_wrap64_rocwmma_candidate_accumulator_info(const Args& args, rns8_plan_backend_info& info) {
  info.accumulator_k_block_size = args.k > 0 ? static_cast<uint64_t>(args.k) : 0u;
  info.accumulator_k_block_cap = 32768u;
  info.accumulator_modulus = 0;
  info.accumulator_max_lhs_abs = 255u;
  info.accumulator_max_rhs_abs = 255u;
  info.accumulator_max_product = 255u * 255u;
  info.accumulator_uses_int32_inner_product = 1;
  info.accumulator_safe_for_k_block =
      info.accumulator_k_block_size > 0 && info.accumulator_k_block_size <= info.accumulator_k_block_cap &&
              info.accumulator_max_product <=
                  static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / info.accumulator_k_block_size
          ? 1u
          : 0u;
  set_backend_text(info.accumulator_input_domain, sizeof(info.accumulator_input_domain), "compact_u8_byte_limb_pairs");
  set_backend_text(info.accumulator_signedness, sizeof(info.accumulator_signedness), "unsigned_u8x_unsigned_u8");
  set_backend_text(info.accumulator_type, sizeof(info.accumulator_type), "int32_then_int64_diagonal");
  set_backend_text(
      info.accumulator_modulus_policy,
      sizeof(info.accumulator_modulus_policy),
      "mod_2_64_wraparound_byte_limb");
  set_backend_text(
      info.accumulator_safety_status,
      sizeof(info.accumulator_safety_status),
      info.accumulator_safe_for_k_block ? "safe_int32_byte_limb_gemm36_k_block"
                                        : "unsafe_int32_byte_limb_gemm36_k_block");
}

std::string bounded_oneshot_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue) {
  std::ostringstream out;
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  out << "backend=" << backend_name(result.backend_info.backend)
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
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=public_oneshot_transient_native_inputs";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

const char* bounded_oneshot_kernel(const Args& args) {
  return bounded_benchmark_semantics(args.semantics) && args.m >= 512 && args.n >= 512 && args.k >= 512
      ? "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
      : "direct_hip_prefix9_native_input_grouped_rns_gemm_v1";
}

std::string finite_oneshot_autotune_key(
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
      << ";execution=public_oneshot_transient_native_inputs";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

const char* finite_native_a_reuse_b_kernel(uint16_t modulus) {
  if (modulus == 256) {
    return "direct_hip_native_a_finite_u8_gemm_mod256_v1";
  }
  if (modulus == 255) {
    return "direct_hip_native_a_finite_u8_gemm_mod255_v1";
  }
  if (modulus == 251) {
    return "direct_hip_native_a_finite_u8_gemm_mod251_v1";
  }
  return "direct_hip_native_a_finite_u8_gemm_v1";
}

const char* bounded_native_a_reuse_b_kernel(const Args& args) {
  if (bounded_native_a_reuse_b_uniform_small_a(args)) {
    return "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2";
  }
  if (bounded_native_a_reuse_b_u64_large_colpair(args)) {
    return "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2";
  }
  return args.semantics == BenchSemantics::BoundedI64
      ? "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
      : "direct_hip_native_a_u64_prefix9_reuse_b_grouped_rns_gemm_v1";
}

const char* bounded_native_a_reuse_b_epilogue(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "uniform_small_i8_ab_resident_b_residue_then_crt_export"
      : "native_a_centered_resident_b_residue_then_crt_export";
}

const char* bounded_native_a_reuse_b_event_label(const Args& args) {
  if (bounded_native_a_reuse_b_uniform_small_a(args)) {
    return "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group";
  }
  if (bounded_native_a_reuse_b_u64_large_colpair(args)) {
    return "bounded_native_a_colpair_reuse_b_gemm_kernel_group";
  }
  return "bounded_native_a_reuse_b_gemm_kernel_group";
}

const char* bounded_native_a_reuse_b_pack_h2d_label(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "bounded_uniform_small_i8_a_h2d"
      : "bounded_native_a_h2d";
}

const char* bounded_native_a_reuse_b_b_h2d_label(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "bounded_uniform_small_i8_b_h2d"
      : "bounded_native_b_h2d";
}

const char* bounded_uniform_small_i8_ab_reuse_a_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1";
}

const char* bounded_uniform_small_i8_ab_reuse_a_epilogue() {
  return "uniform_small_i8_ab_resident_a_residue_then_crt_export";
}

const char* bounded_uniform_small_i8_ab_reuse_a_event_label() {
  return "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group";
}

const char* bounded_uniform_small_i8_ab_reuse_a_pack_h2d_label() {
  return "bounded_uniform_small_i8_b_h2d";
}

const char* bounded_uniform_small_i8_ab_transient_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1";
}

const char* bounded_residue_channel_fusion_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_residue_channel_width3_experimental_v0";
}

const char* bounded_uniform_small_i8_ab_transient_epilogue() {
  return "uniform_small_i8_ab_transient_residue_then_crt_export";
}

const char* bounded_residue_channel_fusion_epilogue() {
  return "width3_residue_fusion_transient_then_crt_export";
}

const char* bounded_uniform_small_i8_ab_transient_event_label() {
  return "bounded_uniform_small_i8_ab_transient_gemm_kernel_group";
}

const char* bounded_uniform_small_i8_ab_transient_a_h2d_label() {
  return "bounded_uniform_small_i8_a_h2d";
}

const char* bounded_uniform_small_i8_ab_transient_b_h2d_label() {
  return "bounded_uniform_small_i8_b_h2d";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_kernel() {
  return "direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_epilogue() {
  return "resident_a_native_b_centered_residue_then_crt_export";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_event_label() {
  return "bounded_native_b_colpair_reuse_a_gemm_kernel_group";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label() {
  return "bounded_native_b_h2d";
}

