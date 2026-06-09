const char* semantics_name_for_key(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
}

const char* bound_kind_name_for_key(rns8_bound_kind bound_kind) {
  switch (bound_kind) {
    case RNS8_BOUND_NONE:
      return "none";
    case RNS8_BOUND_GLOBAL_MAX_ABS:
      return "global_max_abs";
    case RNS8_BOUND_GLOBAL_MAX_UNSIGNED:
      return "global_max_unsigned";
    case RNS8_BOUND_PER_TILE_MAX_ABS:
      return "per_tile_max_abs";
    case RNS8_BOUND_PER_TILE_MAX_UNSIGNED:
      return "per_tile_max_unsigned";
    case RNS8_BOUND_INPUT_RANGE_AND_K:
      return "input_range_and_k";
  }
  return "unknown";
}

const char* bound_contract_name_for_plan(const rns8_plan& plan) {
  if (plan.desc.bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K) {
    return "input_range_and_k_derived_output_bound";
  }
  if (is_per_tile_bound_kind(plan.desc.bound_kind)) {
    return "per_tile_output_bounds";
  }
  if (plan.desc.bound_kind == RNS8_BOUND_GLOBAL_MAX_ABS ||
      plan.desc.bound_kind == RNS8_BOUND_GLOBAL_MAX_UNSIGNED) {
    return "global_output_bound";
  }
  return "not_applicable";
}

bool direct_hip_bounded_skinny_gemv_n1_plan(const rns8_plan& plan) {
  return plan.backend == RNS8_BACKEND_HIP_DIRECT &&
         plan.tile_schedule.empty() &&
         plan.prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         plan.desc.n == 1 &&
         (plan.desc.semantics == RNS8_BOUNDED_I64 || plan.desc.semantics == RNS8_BOUNDED_U64);
}

std::string selected_kernel_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_CPU_REFERENCE) {
    return uses_finite_storage(plan.desc.semantics) ? "cpu_reference_finite_u8_gemm_v1"
                                                    : "cpu_reference_scalar_rns_gemm_v1";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "cpu_wrap64_byte_limb_reference_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return rns8::detail::wrap64_hip_selected_kernel_for_shape(plan.desc.m, plan.desc.n, plan.desc.k);
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      if (plan.desc.finite_modulus == 256) {
        return "direct_hip_tiled_finite_u8_gemm_mod256_v1";
      }
      if (plan.desc.finite_modulus == 255) {
        return "direct_hip_tiled_finite_u8_gemm_mod255_v1";
      }
      if (plan.desc.finite_modulus == 251) {
        return "direct_hip_tiled_finite_u8_gemm_mod251_v1";
      }
      return "direct_hip_tiled_finite_u8_gemm_v1";
    }
    if (direct_hip_bounded_skinny_gemv_n1_plan(plan)) {
      return plan.desc.semantics == RNS8_BOUNDED_I64 ? "direct_hip_prefix9_rns_gemv_n1_i64_v1"
                                                     : "direct_hip_prefix9_rns_gemv_n1_u64_v1";
    }
    if (plan.tile_schedule.empty() && plan.prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
        (plan.desc.semantics == RNS8_BOUNDED_I64 || plan.desc.semantics == RNS8_BOUNDED_U64)) {
      return "direct_hip_prefix9_grouped_rns_gemm_v1";
    }
    if (plan.tile_schedule.empty() && plan.prefix == RNS8_MAX_SUPPORTED_PREFIX &&
        (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED)) {
      return "direct_hip_prefix20_grouped_rns_gemm_v1";
    }
    if ((plan.schedule_flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
      if ((plan.schedule_flags & RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0) {
        return "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1";
      }
      return "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3";
    }
    if ((plan.schedule_flags & RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0) {
      return "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1";
    }
    return "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    if (plan.desc.n == 1 && plan.desc.k >= 4096) {
      return plan.desc.semantics == RNS8_BOUNDED_I64 ? "hip_vector_alu_i64_gemv_n1_exact_192b_v1"
                                                     : "hip_vector_alu_u64_gemv_n1_exact_192b_v1";
    }
    if (plan.desc.n > 1 && plan.desc.n <= 4 && plan.desc.k >= 512) {
      return plan.desc.semantics == RNS8_BOUNDED_I64 ? "hip_vector_alu_i64_gemv_small_n_exact_192b_v1"
                                                     : "hip_vector_alu_u64_gemv_small_n_exact_192b_v1";
    }
    return plan.desc.semantics == RNS8_BOUNDED_I64 ? "hip_vector_alu_i64_exact_192b_v1"
                                                   : "hip_vector_alu_u64_exact_192b_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (!plan.tile_schedule.empty()) {
      return "ck_wmma_cshuffle_tiled_i8_i32_default_moduli_static_centered_epilogue_v3";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return ck_finite_selected_kernel(plan.desc.finite_modulus);
    }
    return "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    if (!plan.tile_schedule.empty()) {
      return "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return rocwmma_finite_selected_kernel(plan.desc.finite_modulus);
    }
    return "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2";
  }
  if (plan.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    if (uses_finite_storage(plan.desc.semantics)) {
      if (plan.backend_target_id.rfind("gfx110", 0) == 0) {
        return "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1";
      }
      if (plan.backend_target_id == "gfx1200" || plan.backend_target_id == "gfx1201") {
        return "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1";
      }
      if (plan.backend_target_id.rfind("gfx942", 0) == 0 &&
          rns8::detail::amdgpu_builtins_use_cdna3_mfma_32x32x16(plan.desc.m, plan.desc.n, plan.desc.k)) {
        return "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_finite_u8_epilogue_v1";
      }
      return "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_finite_u8_epilogue_v1";
    }
    if (plan.backend_target_id.rfind("gfx110", 0) == 0) {
      return "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_centered_epilogue_v1";
    }
    if (plan.backend_target_id == "gfx1200" || plan.backend_target_id == "gfx1201") {
      return "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_centered_epilogue_v1";
    }
    if (plan.backend_target_id.rfind("gfx942", 0) == 0 &&
        rns8::detail::amdgpu_builtins_use_cdna3_mfma_32x32x16(plan.desc.m, plan.desc.n, plan.desc.k)) {
      return "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_centered_epilogue_v1";
    }
    return "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1";
  }
  return "not_implemented";
}

std::string epilogue_mode_for_plan(const rns8_plan& plan) {
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "low64_wrap_export";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "direct_int64_export";
  }
  if (direct_hip_bounded_skinny_gemv_n1_plan(plan)) {
    return "resident_rns_gemv_n1_centered_residue_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "separate_i32_scratch_reduce_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "separate_i32_scratch_reduce_rns_output";
    }
    return "separate_i32_scratch_reduce_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "ck_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "ck_fused_i32_to_centered_residue_rns_output";
    }
    return "ck_fused_i32_to_centered_residue_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "rocwmma_fused_i32_to_centered_residue_rns_output";
    }
    return "rocwmma_fused_i32_to_centered_residue_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "amdgpu_builtin_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "amdgpu_builtin_fused_i32_to_centered_residue_rns_output";
    }
    return "amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export";
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return "fused_centered_residue_then_canonical_u8_export";
  }
  if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    return "fused_centered_residue_rns_output";
  }
  return "fused_centered_residue_then_crt_export";
}

std::string workspace_mode_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (direct_hip_bounded_skinny_gemv_n1_plan(plan)) {
      return "resident_rns_inputs_skinny_n1_output";
    }
    return plan.tile_schedule.empty() ? "resident_device_buffers"
                                      : "resident_device_buffers_with_active_prefix_tiled_schedule";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "native_device_i64_u64_buffers";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "resident_device_buffers_with_hipblaslt_scratch";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
#if RNS8_CK_USE_XDL
    return "resident_device_buffers_with_ck_centered_pack_workspace";
#else
    return "resident_device_buffers_with_ck_canonical_pack_workspace";
#endif
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    return "resident_device_buffers_with_rocwmma_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    return "resident_device_buffers_direct_amdgpu_builtin_matrix_core_no_dense_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "host_byte_limb_reference_workspace";
  }
  return "host_reference_workspace";
}

rns8_output_domain output_domain_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB;
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return RNS8_OUTPUT_DOMAIN_FINITE_U8;
  }
  return RNS8_OUTPUT_DOMAIN_RNS_RESIDUE;
}

const char* output_domain_name(rns8_output_domain domain) {
  switch (domain) {
    case RNS8_OUTPUT_DOMAIN_RNS_RESIDUE:
      return "rns_residue_current";
    case RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64:
      return "native_i64_u64_current";
    case RNS8_OUTPUT_DOMAIN_FINITE_U8:
      return "finite_u8_current";
    case RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB:
      return "wrap64_byte_limb_current";
  }
  return "unknown";
}

bool plan_output_device_current(const rns8_plan& plan) {
  return hip_device_backend(plan.backend) ? true : false;
}

bool concrete_device_target_id(const std::string& target) {
  return !target.empty() && target != "none" && target != "cpu" && target != "unknown";
}

std::string target_id_for_plan_context(const rns8_plan& plan, const rns8_context* ctx) {
  if (!hip_device_backend(plan.backend)) {
    return "cpu";
  }
  if (ctx && concrete_device_target_id(ctx->device_info.gcn_arch)) {
    return ctx->device_info.gcn_arch;
  }
  return "unknown";
}

std::string target_id_for_plan_key(const rns8_plan& plan) {
  if (!plan.backend_target_id.empty()) {
    return plan.backend_target_id;
  }
  return hip_device_backend(plan.backend) ? "unknown" : "cpu";
}

uint32_t next_op_flags_for_plan(const rns8_plan& plan) {
  uint32_t flags = RNS8_NEXT_OP_FINAL_EXPORT;
  const rns8_output_domain domain = output_domain_for_plan(plan);
  if (domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE) {
    flags |= RNS8_NEXT_OP_RNS_GEMM;
  }
  if (domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64) {
    flags |= RNS8_NEXT_OP_NATIVE_GEMM;
    if (plan.desc.semantics == RNS8_BOUNDED_I64 || plan.desc.semantics == RNS8_BOUNDED_U64) {
      flags |= RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE;
    }
  }
  if (rocwmma_b_prepack_cache_supported(plan)) {
    flags |= RNS8_NEXT_OP_REUSABLE_B_PREPACK;
  }
  return flags;
}

const char* next_op_hint_for_plan(const rns8_plan& plan) {
  const rns8_output_domain domain = output_domain_for_plan(plan);
  if (domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64) {
    return "produces native bounded device output; next native GEMM can consume it directly, and native-to-RNS conversion is available for mixed-storage AUTO chains";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE) {
    return "produces residue-current RNS output; next same-contract RNS GEMM can consume it without logical export";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_FINITE_U8) {
    return "produces finite-current centered residues; export converts to canonical uint8 output";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB) {
    return "produces wrap64 byte-limb-current output; export reconstructs low 64-bit words";
  }
  return "unknown output domain";
}

std::string isa_evidence_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine";
    }
    if (uses_finite_storage(plan.desc.semantics) &&
        (plan.desc.finite_modulus == 251 || plan.desc.finite_modulus == 255 || plan.desc.finite_modulus == 256)) {
      return "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide";
    }
    return "rns8_hip_direct_reciprocal_isa_gate";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "source_level_192bit_limb_accumulator_no_matrix_engine";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_library_int8_matmul_specialized_reduce_251_255_256";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return "ck_cshuffle_int8_matrix_isa_gate_no_divide";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    return "rocwmma_i8_matrix_isa_gate_no_divide";
  }
  if (plan.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    return "amdgpu_builtin_matrix_isa_gate_no_divide";
  }
  return "not_applicable_cpu";
}

bool accumulator_uses_int32_inner_product_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 ||
      plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB ||
      plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return false;
  }
  return uses_rns_storage(plan.desc.semantics) || uses_finite_storage(plan.desc.semantics);
}

uint64_t accumulator_k_block_cap_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
      rns8::detail::wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
    return static_cast<uint64_t>(rns8::detail::kWrap64HipU32AccumulatorMaxK);
  }
  if (!accumulator_uses_int32_inner_product_for_plan(plan)) {
    return 0;
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return 32768u;
  }
  return static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK);
}

uint64_t accumulator_k_block_size_for_plan(const rns8_plan& plan) {
  if (plan.desc.k <= 0) {
    return 0;
  }
  const uint64_t k = static_cast<uint64_t>(plan.desc.k);
  const uint64_t cap = accumulator_k_block_cap_for_plan(plan);
  if (cap == 0) {
    return k;
  }
  return k < cap ? k : cap;
}

uint64_t accumulator_modulus_for_plan(const rns8_plan& plan) {
  return uses_finite_storage(plan.desc.semantics) ? plan.desc.finite_modulus : 0u;
}

uint64_t accumulator_max_abs_input_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
      rns8::detail::wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
    return 255u;
  }
  return accumulator_uses_int32_inner_product_for_plan(plan) ? 128u : 0u;
}

uint64_t accumulator_max_product_for_plan(const rns8_plan& plan) {
  const uint64_t lhs = accumulator_max_abs_input_for_plan(plan);
  const uint64_t rhs = accumulator_max_abs_input_for_plan(plan);
  return lhs * rhs;
}

uint32_t accumulator_safe_for_k_block_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
      rns8::detail::wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
    return 1u;
  }
  if (!accumulator_uses_int32_inner_product_for_plan(plan)) {
    return 1u;
  }
  const uint64_t k_block = accumulator_k_block_size_for_plan(plan);
  const uint64_t product = accumulator_max_product_for_plan(plan);
  if (k_block == 0 || product == 0) {
    return 0u;
  }
  return product <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / k_block ? 1u : 0u;
}

std::string accumulator_input_domain_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return plan.desc.semantics == RNS8_BOUNDED_I64 ? "native_i64_values" : "native_u64_values";
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "uint8_byte_limb_pairs";
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return "centered_i8_finite_u8_residues";
  }
  return "centered_i8_rns_residue_planes";
}

std::string accumulator_signedness_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return plan.desc.semantics == RNS8_BOUNDED_I64 ? "signed_i64x_signed_i64" : "unsigned_u64x_unsigned_u64";
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "unsigned_byte_limb";
  }
  if (accumulator_uses_int32_inner_product_for_plan(plan)) {
    return "signed_i8x_signed_i8";
  }
  return "not_applicable";
}

std::string accumulator_type_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "software_192bit_limb";
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
      rns8::detail::wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
    return "uint32_low_diagonal_then_uint64_carry";
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "uint64_wraparound_byte_limb";
  }
  if (accumulator_uses_int32_inner_product_for_plan(plan)) {
    return "int32";
  }
  return "not_applicable";
}

std::string accumulator_modulus_policy_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "native_exact_integer_output";
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "mod_2_64_wraparound_byte_limb";
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return "finite_u8_modulus";
  }
  return "selected_rns_modulus_ladder";
}

std::string accumulator_safety_status_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "exact_192bit_limb_no_int32_k_cap";
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
      rns8::detail::wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
    return "safe_uint32_byte_limb_gemm36_k_block";
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "exact_mod_2_64_byte_limb_no_int32_k_cap";
  }
  return accumulator_safe_for_k_block_for_plan(plan) ? "safe_int32_k_block_split" : "unsafe_int32_k_block";
}

