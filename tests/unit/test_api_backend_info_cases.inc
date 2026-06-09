TEST_CASE("public accelerator backend context kinds fail fast") {
  std::vector<rns8_backend_kind> backends;
#if !defined(RNS8_ENABLE_ROCWMMA) || !RNS8_ENABLE_ROCWMMA
  backends.push_back(RNS8_BACKEND_ROCWMMA);
#endif
#if !defined(RNS8_ENABLE_CK) || !RNS8_ENABLE_CK
  backends.push_back(RNS8_BACKEND_CK);
#endif
#if !defined(RNS8_ENABLE_HIPBLASLT) || !RNS8_ENABLE_HIPBLASLT
  backends.push_back(RNS8_BACKEND_HIPBLASLT);
#endif
#if !defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) || !RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  backends.push_back(RNS8_BACKEND_AMDGPU_BUILTINS);
#endif
  for (const rns8_backend_kind backend : backends) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);
  }
}

TEST_CASE("AMDGPU builtins CDNA MFMA route selection is conservative") {
  CHECK_FALSE(rns8::detail::amdgpu_builtins_use_cdna3_mfma_32x32x16(64, 64, 64));
  CHECK_FALSE(rns8::detail::amdgpu_builtins_use_cdna3_mfma_32x32x16(128, 64, 128));
  CHECK(rns8::detail::amdgpu_builtins_use_cdna3_mfma_32x32x16(128, 128, 128));
}

TEST_CASE("AMDGPU builtins plan kernel identity follows target and shape") {
  rns8_plan plan{};
  plan.backend = RNS8_BACKEND_AMDGPU_BUILTINS;
  plan.desc.semantics = RNS8_BOUNDED_I64;
  plan.desc.m = 64;
  plan.desc.n = 64;
  plan.desc.k = 64;
  plan.backend_target_id = "gfx942";
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1");

  plan.desc.m = 128;
  plan.desc.n = 128;
  plan.desc.k = 128;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_centered_epilogue_v1");

  plan.backend_target_id = "gfx1100";
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_centered_epilogue_v1");

  plan.backend_target_id = "gfx1201";
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_centered_epilogue_v1");

  plan.desc.semantics = RNS8_FINITE_FIELD_U8;
  plan.desc.finite_modulus = 251;
  plan.backend_target_id = "gfx942";
  plan.desc.m = 64;
  plan.desc.n = 64;
  plan.desc.k = 64;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_finite_u8_epilogue_v1");

  plan.desc.m = 128;
  plan.desc.n = 128;
  plan.desc.k = 128;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_finite_u8_epilogue_v1");

  plan.backend_target_id = "gfx1201";
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1");
}

TEST_CASE("Direct HIP bounded skinny plans report GEMV route metadata") {
  rns8_plan plan{};
  plan.backend = RNS8_BACKEND_HIP_DIRECT;
  plan.prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  plan.desc.semantics = RNS8_BOUNDED_I64;
  plan.desc.m = 512;
  plan.desc.n = 1;
  plan.desc.k = 512;

  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_rns_gemv_n1_i64_v1");
  CHECK(
      rns8::detail::api::epilogue_mode_for_plan(plan) ==
      "resident_rns_gemv_n1_centered_residue_then_crt_export");
  CHECK(
      rns8::detail::api::workspace_mode_for_plan(plan) ==
      "resident_rns_inputs_skinny_n1_output");

  plan.desc.semantics = RNS8_BOUNDED_U64;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_rns_gemv_n1_u64_v1");

  plan.desc.n = 2;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_rns_gemv_small_n_u64_v1");
  CHECK(
      rns8::detail::api::epilogue_mode_for_plan(plan) ==
      "resident_rns_gemv_small_n_centered_residue_then_crt_export");
  CHECK(
      rns8::detail::api::workspace_mode_for_plan(plan) ==
      "resident_rns_inputs_skinny_n_le8_output");

  plan.desc.semantics = RNS8_BOUNDED_I64;
  plan.desc.n = 4;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_rns_gemv_small_n_i64_v1");

  plan.desc.n = 8;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_rns_gemv_small_n_i64_v1");
  CHECK(
      rns8::detail::api::workspace_mode_for_plan(plan) ==
      "resident_rns_inputs_skinny_n_le8_output");

  plan.desc.n = 9;
  CHECK(
      rns8::detail::api::selected_kernel_for_plan(plan) ==
      "direct_hip_prefix9_grouped_rns_gemm_v1");
  CHECK(rns8::detail::api::workspace_mode_for_plan(plan) == "resident_device_buffers");
}

TEST_CASE("malformed accelerator plan descriptors fail before unsupported routing") {
  rns8_context* ctx = create_cpu_context();
  const rns8_backend_kind backends[] = {RNS8_BACKEND_CK, RNS8_BACKEND_ROCWMMA, RNS8_BACKEND_AMDGPU_BUILTINS};

  for (const rns8_backend_kind backend : backends) {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = backend;
    desc.m = 1;
    desc.n = 1;
    desc.k = 1;
    desc.bound = 1;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    auto malformed = desc;
    malformed.flags = 0x80000000u;
    CHECK(rns8_create_plan(ctx, &malformed, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    malformed = desc;
    malformed.m = 0;
    CHECK(rns8_create_plan(ctx, &malformed, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public backend capability info separates correctness and accelerator candidates") {
  rns8_backend_capability_info cpu{};
  cpu.struct_size = sizeof(cpu);
  cpu.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_backend_capability_info(RNS8_BACKEND_CPU_REFERENCE, &cpu) == RNS8_SUCCESS);
  CHECK(cpu.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(cpu.is_accelerator == 0);
  CHECK(cpu.is_correctness_backend == 1);
  CHECK(cpu.compiled_kernel_available == 1);
  CHECK(cpu.exact_differential_validated == 1);
  CHECK(cpu.performance_validated == 0);
  CHECK(std::string(cpu.status) == "implemented_correctness_backend");
  CHECK(std::string(cpu.selected_kernel) == "cpu_reference_scalar_rns_gemm_v1");

  rns8_backend_capability_info wrap{};
  wrap.struct_size = sizeof(wrap);
  wrap.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_backend_capability_info(RNS8_BACKEND_WRAP64_BYTE_LIMB, &wrap) == RNS8_SUCCESS);
  CHECK(wrap.supports_wrap64 == 1);
  CHECK(std::string(wrap.selected_kernel) == "cpu_wrap64_byte_limb_reference_v1");

  const rns8_backend_kind accelerators[] = {
      RNS8_BACKEND_HIPBLASLT,
      RNS8_BACKEND_CK,
      RNS8_BACKEND_ROCWMMA,
      RNS8_BACKEND_AMDGPU_BUILTINS,
  };
  for (const rns8_backend_kind backend : accelerators) {
    rns8_backend_capability_info capability{};
    capability.struct_size = sizeof(capability);
    capability.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_backend_capability_info(backend, &capability) == RNS8_SUCCESS);
    CHECK(capability.is_accelerator == 1);
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    if (backend == RNS8_BACKEND_HIPBLASLT) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_baseline_backend");
      CHECK(
          std::string(capability.selected_kernel) ==
          "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2");
      CHECK(std::string(capability.epilogue_mode) == "separate_i32_scratch_residue_reduce");
      CHECK(
          std::string(capability.isa_evidence) ==
          "hipblaslt_library_int8_matmul_specialized_reduce_251_255_256");
      continue;
    }
#endif
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
    if (backend == RNS8_BACKEND_CK) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_opt_in_ck_backend");
      CHECK(
          std::string(capability.selected_kernel) ==
          "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3");
      CHECK(std::string(capability.epilogue_mode) == "ck_fused_i32_to_centered_residue");
#if RNS8_CK_USE_XDL
      CHECK(std::string(capability.workspace_mode) ==
            "resident_device_buffers_with_ck_centered_pack_workspace");
#else
      CHECK(std::string(capability.workspace_mode) ==
            "resident_device_buffers_with_ck_canonical_pack_workspace");
#endif
      CHECK(std::string(capability.isa_evidence) == "ck_cshuffle_int8_matrix_isa_gate_no_divide");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
      continue;
    }
#endif
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
    if (backend == RNS8_BACKEND_ROCWMMA) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_opt_in_rocwmma_backend");
      CHECK(
          std::string(capability.selected_kernel) ==
          "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2");
      CHECK(std::string(capability.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_rocwmma_pack_workspace");
      CHECK(std::string(capability.isa_evidence) == "rocwmma_i8_matrix_isa_gate_no_divide");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
      continue;
    }
#endif
#if defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
    if (backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_opt_in_amdgpu_builtin_backend");
      CHECK(
          std::string(capability.selected_kernel) ==
          "amdgpu_builtin_matrix_core_target_family_runtime_dispatch_v1");
      CHECK(
          std::string(capability.epilogue_mode) ==
          "amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_direct_matrix_core_no_dense_pack_workspace");
      CHECK(std::string(capability.isa_evidence) == "amdgpu_builtin_matrix_isa_gate_no_divide");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
      continue;
    }
#endif
    CHECK(capability.is_correctness_backend == 0);
    CHECK(capability.requires_feature_detection == 1);
    CHECK(capability.enable_flag_fail_fast == 1);
    CHECK(capability.candidate_evidence_only == 1);
    CHECK(capability.compiled_kernel_available == 0);
    CHECK(capability.exact_differential_validated == 0);
    CHECK(capability.performance_validated == 0);
    CHECK(std::string(capability.isa_evidence) == "not_validated");
    if (backend == RNS8_BACKEND_CK) {
      CHECK(std::string(capability.status) == "not_enabled_in_this_build");
      CHECK(
          std::string(capability.selected_kernel) ==
          "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3_disabled");
      CHECK(std::string(capability.epilogue_mode) == "ck_fused_i32_to_centered_residue_disabled");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_ck_pack_workspace_disabled");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
    } else if (backend == RNS8_BACKEND_ROCWMMA) {
      CHECK(std::string(capability.status) == "not_enabled_in_this_build");
      CHECK(
          std::string(capability.selected_kernel) ==
          "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2_disabled");
      CHECK(std::string(capability.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue_disabled");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_rocwmma_pack_workspace_disabled");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
    } else if (backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      CHECK(std::string(capability.status) == "not_compiled_in_this_build");
      CHECK(
          std::string(capability.selected_kernel) ==
          "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1_disabled");
      CHECK(std::string(capability.epilogue_mode) == "amdgpu_builtin_fused_i32_to_centered_residue_disabled");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_direct_matrix_core_no_dense_pack_workspace_disabled");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
    } else {
      CHECK(std::string(capability.status) == "not_implemented_evidence_only");
      CHECK(std::string(capability.epilogue_mode) == "not_implemented");
    }
  }

  rns8_backend_capability_info bad_abi{};
  bad_abi.struct_size = sizeof(bad_abi) - 1;
  bad_abi.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_backend_capability_info(RNS8_BACKEND_CPU_REFERENCE, &bad_abi) == RNS8_INVALID_ARGUMENT);

  rns8_backend_capability_info unknown{};
  unknown.struct_size = sizeof(unknown);
  unknown.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_backend_capability_info(static_cast<rns8_backend_kind>(999), &unknown) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("public plan backend info exposes selected kernel and autotune contract") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_U64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 2;
    desc.n = 3;
    desc.k = 4;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.is_accelerator == 0);
    CHECK(info.is_correctness_backend == 1);
    CHECK(info.is_matrix_engine_backend == 0);
    CHECK(info.compiled_kernel_available == 1);
    CHECK(info.exact_differential_validated == 1);
    CHECK(info.performance_validated == 0);
    CHECK(info.workspace_required_bytes == 0);
    CHECK(std::string(info.selected_kernel) == "cpu_reference_scalar_rns_gemm_v1");
    CHECK(std::string(info.capability_status) == "implemented_correctness_backend");
    CHECK(std::string(info.epilogue_mode) == "fused_centered_residue_then_crt_export");
    CHECK(std::string(info.workspace_mode) == "host_reference_workspace");
    CHECK(std::string(info.isa_evidence) == "not_applicable_cpu");
    CHECK(info.accumulator_uses_int32_inner_product == 1);
    CHECK(info.accumulator_safe_for_k_block == 1);
    CHECK(info.accumulator_k_block_size == 4);
    CHECK(info.accumulator_k_block_cap == 65536);
    CHECK(info.accumulator_modulus == 0);
    CHECK(info.accumulator_max_lhs_abs == 128);
    CHECK(info.accumulator_max_rhs_abs == 128);
    CHECK(info.accumulator_max_product == 16384);
    CHECK(std::string(info.accumulator_input_domain) == "centered_i8_rns_residue_planes");
    CHECK(std::string(info.accumulator_signedness) == "signed_i8x_signed_i8");
    CHECK(std::string(info.accumulator_type) == "int32");
    CHECK(std::string(info.accumulator_modulus_policy) == "selected_rns_modulus_ladder");
    CHECK(std::string(info.accumulator_safety_status) == "safe_int32_k_block_split");
    CHECK(std::string(info.autotune_key).find("backend=cpu-reference;target_id=cpu;semantics=bounded_u64") == 0);
    CHECK(std::string(info.autotune_key).find(";accumulator_type=int32;") != std::string::npos);
    CHECK(std::string(info.autotune_key).find(";k_block_size=4;") != std::string::npos);
    CHECK(std::string(info.autotune_key).find(";k_block_cap=65536;") != std::string::npos);
    CHECK(std::string(info.autotune_key).find(";kernel=cpu_reference_scalar_rns_gemm_v1;") != std::string::npos);

    rns8_plan_backend_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_plan_backend_info(plan, &bad_abi) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap_ctx = create_wrap_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_WRAP_U64_MOD_2_64;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    desc.m = 2;
    desc.n = 2;
    desc.k = 2;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(wrap_ctx, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(std::string(info.selected_kernel) == "cpu_wrap64_byte_limb_reference_v1");
    CHECK(std::string(info.epilogue_mode) == "low64_wrap_export");
    CHECK(std::string(info.workspace_mode) == "host_byte_limb_reference_workspace");
    CHECK(info.accumulator_uses_int32_inner_product == 0);
    CHECK(info.accumulator_safe_for_k_block == 1);
    CHECK(info.accumulator_k_block_size == 2);
    CHECK(info.accumulator_k_block_cap == 0);
    CHECK(std::string(info.accumulator_input_domain) == "uint8_byte_limb_pairs");
    CHECK(std::string(info.accumulator_signedness) == "unsigned_byte_limb");
    CHECK(std::string(info.accumulator_type) == "uint64_wraparound_byte_limb");
    CHECK(std::string(info.accumulator_modulus_policy) == "mod_2_64_wraparound_byte_limb");
    CHECK(std::string(info.accumulator_safety_status) == "exact_mod_2_64_byte_limb_no_int32_k_cap");
    CHECK(std::string(info.autotune_key).find("semantics=wrap_u64_mod_2_64") != std::string::npos);
    CHECK(std::string(info.autotune_key).find(";accumulator_type=uint64_wraparound_byte_limb;") != std::string::npos);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_backend_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_plan_backend_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("public grouped dispatch contract info reports explicit grouped requirements") {
  rns8_context* cpu = create_cpu_context();
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_U64, 4, 4, 4);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);

  rns8_grouped_dispatch_contract_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_grouped_dispatch_contract_info(plan, 32, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(info.semantics == RNS8_BOUNDED_U64);
  CHECK(info.task_count == 32);
  CHECK(info.descriptor_contract_supported == 0);
  CHECK(info.public_execution_available == 0);
  CHECK(info.same_shape_required == 1);
  CHECK(info.same_semantics_required == 1);
  CHECK(info.compact_row_major_required == 1);
  CHECK(info.unique_matrix_handles_required == 1);
  CHECK(info.unique_workspace_handles_required == 1);
  CHECK(info.same_plan_fingerprint_required == 1);
  CHECK(info.same_device_required == 1);
  CHECK(info.device_resident_inputs_required == 1);
  CHECK(info.per_task_source_versions_required == 1);
  CHECK(info.produces_device_current_output == 1);
  CHECK(info.final_export_required_for_host_output == 1);
  CHECK(info.per_task_status_required == 1);
  CHECK(info.checksum_policy_required == 1);
  CHECK(info.descriptor_reuse_validated == 1);
  CHECK((info.flags & RNS8_GROUPED_DISPATCH_CONTRACT_BENCHMARK_ONLY_EXECUTION) != 0);
  CHECK((info.flags & RNS8_GROUPED_DISPATCH_CONTRACT_PUBLIC_PACK_EXPLICIT) != 0);
  CHECK((info.flags & RNS8_GROUPED_DISPATCH_CONTRACT_PUBLIC_EXPORT_EXPLICIT) != 0);
  CHECK((info.flags & RNS8_GROUPED_DISPATCH_CONTRACT_GENERIC_BUCKETING_UNAVAILABLE) != 0);
  CHECK((info.flags & RNS8_GROUPED_DISPATCH_CONTRACT_AUTO_ROUTING_DISABLED) != 0);
  CHECK(std::string(info.descriptor_layout) == "same_shape_resident_task_triplets_v1");
  CHECK(std::string(info.bucket_policy) == "single_same_shape_bucket");
  CHECK(std::string(info.source_version_policy) == "per_task_monotonic_source_version_repack");
  CHECK(std::string(info.workspace_policy) == "one_workspace_per_task_shared_plan");
  CHECK(std::string(info.matrix_ownership_policy) == "caller_owns_unique_task_triplets_until_grouped_call_returns");
  CHECK(std::string(info.descriptor_reuse_policy) == "reuse_after_shape_workspace_source_validation");
  CHECK(std::string(info.stride_policy) == "matrix_ld_matches_logical_shape_host_output_ld_explicit");
  CHECK(std::string(info.output_currentness_policy) ==
        "device_residue_current_after_grouped_gemm_host_output_after_export");
  CHECK(std::string(info.lifetime_policy) == "task_matrices_and_workspaces_valid_until_grouped_call_returns");
  CHECK(std::string(info.checksum_policy) == "combined_per_task_checksum_u64");
  CHECK(std::string(info.status_policy) == "fail_fast_per_task_operation_status");
  CHECK(std::string(info.device_descriptor_policy) == "device_pointer_tables_and_compact_slabs");
  CHECK(std::string(info.output_domain) == "native_i64_u64_host");
  CHECK(std::string(info.unsupported_reason) == "backend_not_hip_direct");
  CHECK(rns8_gemm_rns_grouped(cpu, plan, nullptr, 32) == RNS8_INVALID_ARGUMENT);

  rns8_grouped_dispatch_contract_info bad_abi{};
  bad_abi.struct_size = sizeof(bad_abi) - 1;
  bad_abi.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_grouped_dispatch_contract_info(plan, 32, &bad_abi) == RNS8_INVALID_ARGUMENT);

  rns8_grouped_dispatch_contract_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_grouped_dispatch_contract_info(nullptr, 32, &null_info) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_plan(plan);
  rns8_destroy_context(cpu);
}
