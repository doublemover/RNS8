TEST_CASE("public plan packing info exposes resident and transient layout contracts") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 2;
    desc.n = 3;
    desc.k = 4;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_packing_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.semantics == RNS8_BOUNDED_I64);
    CHECK(info.uses_resident_matrix_inputs == 1);
    CHECK(info.uses_transient_pack_workspace == 0);
    CHECK(info.uses_matrix_engine_pack_layout == 0);
    CHECK(info.reusable_prepack_cache_available == 0);
    CHECK(info.production_prepack_cache_available == 0);
    CHECK(info.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
    CHECK(info.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
    CHECK(info.output_host_current == 1);
    CHECK(info.output_device_current == 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_FINAL_EXPORT) != 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) == 0);
    CHECK(std::string(info.input_domain_name) == "rns_residue_current");
    CHECK(std::string(info.output_domain_name) == "rns_residue_current");
    CHECK(std::string(info.next_op_hint).find("without logical export") != std::string::npos);
    CHECK(info.a_pack_workspace_bytes == 0);
    CHECK(info.b_pack_workspace_bytes == 0);
    CHECK(info.accumulator_workspace_bytes == 0);
    CHECK(info.library_workspace_bytes == 0);
    CHECK(info.total_transient_workspace_bytes == 0);
    CHECK(std::string(info.a_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.b_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.output_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.prepack_cache_scope) == "host_resident_no_prepack_cache");

    rns8_plan_packing_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_plan_packing_info(plan, &bad_abi) == RNS8_INVALID_ARGUMENT);

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

    rns8_plan_packing_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(info.semantics == RNS8_WRAP_U64_MOD_2_64);
    CHECK(info.uses_resident_matrix_inputs == 1);
    CHECK(info.uses_transient_pack_workspace == 0);
    CHECK(info.uses_matrix_engine_pack_layout == 0);
    CHECK(info.input_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB);
    CHECK(info.output_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB);
    CHECK(info.output_host_current == 1);
    CHECK(info.output_device_current == 0);
    CHECK(info.next_op_flags == RNS8_NEXT_OP_FINAL_EXPORT);
    CHECK(std::string(info.output_domain_name) == "wrap64_byte_limb_current");
    CHECK(std::string(info.a_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.b_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.output_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.prepack_cache_scope) == "host_byte_limb_no_prepack_cache");

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_packing_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_plan_packing_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("internal plan lowering description classifies domains and continuation choices") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 4;
    desc.n = 5;
    desc.k = 6;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
    const auto lowering = lowering_for_plan(plan);
    CHECK(lowering.operation == "MatMul");
    CHECK(lowering.semantic_contract == "bounded_i64");
    CHECK(lowering.backend_family == "cpu_reference");
    CHECK(lowering.input_domain == "rns_residue_current");
    CHECK(lowering.output_domain == "rns_residue_current");
    CHECK(lowering.desired_output == "final_export_or_rns_chain");
    CHECK(lowering.schedule_strategy == "minimum_proven_uniform_prefix_schedule");
    CHECK(lowering.packing_strategy == "resident_matrix_inputs");
    CHECK(lowering.reuse_strategy == "resident_inputs_no_prepack");
    CHECK(lowering.conversion_strategy == "no_conversion_needed_for_rns_chain");
    CHECK(lowering.lowering_path.find("RnsResidueCurrent") != std::string::npos);
    CHECK(lowering.final_export_available);
    CHECK(lowering.rns_continuation_available);
    CHECK_FALSE(lowering.native_continuation_available);
    CHECK_FALSE(lowering.native_to_rns_available);

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
    const auto lowering = lowering_for_plan(plan);
    CHECK(lowering.semantic_contract == "wrap_u64_mod_2_64");
    CHECK(lowering.backend_family == "wrap64_reference");
    CHECK(lowering.input_domain == "wrap64_byte_limb_current");
    CHECK(lowering.output_domain == "wrap64_byte_limb_current");
    CHECK(lowering.desired_output == "final_export");
    CHECK(lowering.schedule_strategy == "semantic_specific_no_rns_prefix_schedule");
    CHECK(lowering.conversion_strategy == "wrap64_byte_limb_final_export_or_same_semantic_reuse");
    CHECK(lowering.lowering_path.find("Low64Export") != std::string::npos);
    CHECK(lowering.final_export_available);
    CHECK_FALSE(lowering.rns_continuation_available);
    CHECK_FALSE(lowering.native_continuation_available);
    CHECK_FALSE(lowering.native_to_rns_available);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_backend_info vector_backend{};
  vector_backend.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  rns8_plan_packing_info vector_packing{};
  vector_packing.semantics = RNS8_BOUNDED_I64;
  vector_packing.output_domain = RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64;
  vector_packing.next_op_flags =
      RNS8_NEXT_OP_FINAL_EXPORT | RNS8_NEXT_OP_NATIVE_GEMM | RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE;
  std::snprintf(vector_packing.input_domain_name, sizeof(vector_packing.input_domain_name), "%s", "native_i64_u64_current");
  std::snprintf(
      vector_packing.output_domain_name,
      sizeof(vector_packing.output_domain_name),
      "%s",
      "native_i64_u64_current");
  rns8_plan_schedule_info vector_schedule{};
  vector_schedule.max_selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  const auto vector_lowering =
      rns8::detail::describe_plan_lowering(vector_backend, vector_packing, vector_schedule);
  CHECK(vector_lowering.backend_family == "native_vector_alu");
  CHECK(vector_lowering.desired_output == "final_export_or_native_chain");
  CHECK(vector_lowering.conversion_strategy == "native_to_rns_available_for_mixed_storage_auto");
  CHECK(vector_lowering.lowering_path.find("NativeToRns") != std::string::npos);
  CHECK(vector_lowering.native_continuation_available);
  CHECK(vector_lowering.native_to_rns_available);
  CHECK_FALSE(vector_lowering.rns_continuation_available);
}

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
TEST_CASE("hipBLASLt plan packing info reports transient pack and scratch bytes") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIPBLASLT;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_HIPBLASLT;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_FINAL_EXPORT) != 0);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == static_cast<uint64_t>(m * n * sizeof(int32_t)));
  CHECK(packing.library_workspace_bytes > 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "hipblaslt_a_transposed_centered_i8_mk16_v1");
  CHECK(std::string(packing.prepack_cache_scope) == "transient_per_dispatch_workspace");
  CHECK(packing.production_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
TEST_CASE("CK plan packing info reports canonical transient pack workspace") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CK;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CK;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == static_cast<uint64_t>(m * n));
  CHECK(packing.library_workspace_bytes == 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1");
  CHECK(std::string(packing.b_layout_version) == "ck_b_canonical_colmajor_i8_n64_kblock32768_v1");
  CHECK(packing.reusable_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
TEST_CASE("rocWMMA plan packing info reports transient workspace and reusable B cache") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_ROCWMMA;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_ROCWMMA;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_REUSABLE_B_PREPACK) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == 0);
  CHECK(packing.library_workspace_bytes == 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "rocwmma_a_rowmajor_i8_m16_kblock65536_v1");
  CHECK(std::string(packing.b_layout_version) == "rns_i8_tile_swizzled_b_v1");
  CHECK(packing.reusable_prepack_cache_available == 1);
  CHECK(std::string(packing.prepack_cache_scope) == "reusable_b_prepack_cache");
  CHECK(packing.production_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

