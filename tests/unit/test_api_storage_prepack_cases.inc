TEST_CASE("public matrix storage info exposes source-versioned resident layouts") {
  rns8_context* cpu = create_cpu_context();
  {
    auto desc = bounded_matrix_desc(2, 3, RNS8_BOUNDED_I64);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.semantics == RNS8_BOUNDED_I64);
    CHECK(info.logical_layout == RNS8_LAYOUT_ROW_MAJOR);
    CHECK(info.bound_kind == RNS8_BOUND_GLOBAL_MAX_ABS);
    CHECK(info.rows == 2);
    CHECK(info.cols == 3);
    CHECK(info.logical_ld == 3);
    CHECK(info.max_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.source_version == 0);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_byte_limbs_current == 0);
    CHECK(info.device_byte_limbs_current == 0);
    CHECK(info.uses_residue_storage == 1);
    CHECK(info.uses_byte_limb_storage == 0);
    CHECK(info.hip_device_id == -1);
    CHECK(info.host_residue_bytes == 2 * 3 * RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.device_residue_bytes == 0);
    CHECK(info.host_byte_limb_bytes == 0);
    CHECK(info.device_byte_limb_bytes == 0);
    CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.storage_scope) == "host_resident_storage");

    const int64_t values[] = {1, -2, 3, 4, -5, 6};
    REQUIRE(rns8_pack_i64(cpu, matrix, values, 3, 42) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.source_version == 42);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);
    CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");

    rns8_matrix_storage_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_matrix_storage_info(matrix, &bad_abi) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(matrix);
  }
  {
    auto desc = finite_matrix_desc(2, 2, RNS8_FINITE_RING_U8);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.semantics == RNS8_FINITE_RING_U8);
    CHECK(info.max_prefix == 0);
    CHECK(info.finite_modulus == 0);
    CHECK(info.host_residues_current == 0);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_residue_bytes == 4);
    CHECK(std::string(info.layout_version) == "finite_u8_centered_residue_v1");
    CHECK(std::string(info.storage_scope) == "host_resident_storage");

    const uint8_t values[] = {1, 2, 3, 4};
    REQUIRE(rns8_pack_finite_u8(cpu, matrix, 251, values, 2, 7) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.finite_modulus == 251);
    CHECK(info.source_version == 7);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);

    rns8_destroy_matrix(matrix);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap = create_wrap_context();
  {
    auto desc = wrap_matrix_desc(2, 3);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(wrap, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(info.semantics == RNS8_WRAP_U64_MOD_2_64);
    CHECK(info.max_prefix == 0);
    CHECK(info.uses_residue_storage == 0);
    CHECK(info.uses_byte_limb_storage == 1);
    CHECK(info.host_residues_current == 0);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_byte_limbs_current == 1);
    CHECK(info.device_byte_limbs_current == 0);
    CHECK(info.host_residue_bytes == 0);
    CHECK(info.host_byte_limb_bytes == 2 * 3 * 8);
    CHECK(std::string(info.layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.storage_scope) == "host_byte_limb_storage");

    const uint64_t values[] = {0, 1, 2, 3, 4, 5};
    REQUIRE(rns8_pack_u64(wrap, matrix, values, 3, 99) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.source_version == 99);
    CHECK(info.host_byte_limbs_current == 1);
    CHECK(info.device_byte_limbs_current == 0);

    rns8_destroy_matrix(matrix);
  }
  rns8_destroy_context(wrap);

  rns8_matrix_storage_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_matrix_storage_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("C++ matrix wrapper exposes storage info") {
  rns8::Context context(-1, RNS8_BACKEND_CPU_REFERENCE);
  rns8::Matrix matrix(context, bounded_matrix_desc(1, 2, RNS8_BOUNDED_U64));

  const auto info = matrix.storage_info();
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(info.semantics == RNS8_BOUNDED_U64);
  CHECK(info.rows == 1);
  CHECK(info.cols == 2);
  CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");
}

TEST_CASE("public prepack cache key info validates operand role and source version") {
  rns8_context* cpu = create_cpu_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 2;
  desc.n = 3;
  desc.k = 4;
  desc.bound = 64;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  auto a_desc = bounded_matrix_desc(desc.m, desc.k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(desc.k, desc.n, RNS8_BOUNDED_I64);
  REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &b) == RNS8_SUCCESS);

  const int64_t a_values[] = {1, -2, 3, 4, -5, 6, 7, -8};
  const int64_t b_values[] = {1, 2, -3, 4, -5, 6, 7, 8, -9, 10, -11, 12};
  REQUIRE(rns8_pack_i64(cpu, a, a_values, desc.k, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, b, b_values, desc.n, 22) == RNS8_SUCCESS);

  auto a_key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(a_key.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(a_key.semantics == RNS8_BOUNDED_I64);
  CHECK(a_key.operand_role == RNS8_OPERAND_A);
  CHECK(a_key.cache_key_valid == 1);
  CHECK(a_key.reusable_prepack_cache_available == 0);
  CHECK(a_key.production_prepack_cache_available == 0);
  CHECK(a_key.hip_device_id == -1);
  CHECK(a_key.reserved0 == 0);
  CHECK(a_key.matrix_rows == desc.m);
  CHECK(a_key.matrix_cols == desc.k);
  CHECK(a_key.max_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(a_key.source_version == 11);
  CHECK(a_key.plan_fingerprint != 0);
  CHECK(a_key.cache_key_hash != 0);
  CHECK(std::string(a_key.matrix_layout_version) == "rns_centered_residue_planes_v1");
  CHECK(std::string(a_key.operand_layout_version) == "rns_centered_residue_planes_v1");
  CHECK(std::string(a_key.cache_scope) == "validated_key_no_production_cache");
  CHECK(std::string(a_key.cache_key).find("operand=A") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("target_id=cpu") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("kernel=cpu_reference_scalar_rns_gemm_v1") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("prefix_schedule_hash=") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("k_block_size=0") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("source_version=11") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("hip_device_id=-1") != std::string::npos);

  const auto b_key = prepack_cache_key_info(plan, b, RNS8_OPERAND_B);
  CHECK(b_key.operand_role == RNS8_OPERAND_B);
  CHECK(b_key.matrix_rows == desc.k);
  CHECK(b_key.matrix_cols == desc.n);
  CHECK(b_key.source_version == 22);
  CHECK(b_key.plan_fingerprint == a_key.plan_fingerprint);
  CHECK(b_key.cache_key_hash != a_key.cache_key_hash);
  CHECK(std::string(b_key.cache_key).find("operand=B") != std::string::npos);

  const uint64_t old_a_hash = a_key.cache_key_hash;
  REQUIRE(rns8_pack_i64(cpu, a, a_values, desc.k, 12) == RNS8_SUCCESS);
  a_key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(a_key.source_version == 12);
  CHECK(a_key.cache_key_hash != old_a_hash);
  CHECK(std::string(a_key.cache_key).find("source_version=12") != std::string::npos);

  rns8_prepack_cache_key_info bad_abi{};
  bad_abi.struct_size = sizeof(bad_abi) - 1;
  bad_abi.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &bad_abi) == RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(nullptr, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, nullptr, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, static_cast<rns8_operand_role>(99), &info) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_B, &info) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(b);
  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(cpu);
}

TEST_CASE("public prepack cache key info rejects finite modulus mismatch") {
  rns8_context* cpu = create_cpu_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_RING_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.finite_modulus = 251;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  auto a_desc = finite_matrix_desc(desc.m, desc.k, RNS8_FINITE_RING_U8);
  REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &a) == RNS8_SUCCESS);

  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);

  const uint8_t values[] = {1, 2, 3, 4};
  REQUIRE(rns8_pack_finite_u8(cpu, a, 251, values, desc.k, 33) == RNS8_SUCCESS);
  const auto key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.finite_modulus == 251);
  CHECK(key.source_version == 33);
  CHECK(std::string(key.matrix_layout_version) == "finite_u8_centered_residue_v1");

  REQUIRE(rns8_pack_finite_u8(cpu, a, 255, values, desc.k, 34) == RNS8_SUCCESS);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(cpu);
}

TEST_CASE("public prepack cache key info covers wrap64 byte-limb operands") {
  rns8_context* wrap = create_wrap_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = 2;
  desc.n = 2;
  desc.k = 3;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  auto a_desc = wrap_matrix_desc(desc.m, desc.k);
  REQUIRE(rns8_create_plan(wrap, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap, &a_desc, &a) == RNS8_SUCCESS);

  const uint64_t values[] = {0, 1, 2, 3, 4, 5};
  REQUIRE(rns8_pack_u64(wrap, a, values, desc.k, 55) == RNS8_SUCCESS);
  const auto key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
  CHECK(key.semantics == RNS8_WRAP_U64_MOD_2_64);
  CHECK(key.operand_role == RNS8_OPERAND_A);
  CHECK(key.matrix_rows == desc.m);
  CHECK(key.matrix_cols == desc.k);
  CHECK(key.max_prefix == 0);
  CHECK(key.source_version == 55);
  CHECK(key.hip_device_id == -1);
  CHECK(std::string(key.matrix_layout_version) == "wrap64_byte_limb_v1");
  CHECK(std::string(key.operand_layout_version) == "wrap64_byte_limb_v1");
  CHECK(std::string(key.cache_key).find("semantics=wrap_u64_mod_2_64") != std::string::npos);

  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(wrap);
}

TEST_CASE("C++ prepack cache key helper exposes validated key material") {
  rns8::Context context(-1, RNS8_BACKEND_CPU_REFERENCE);
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 2;
  desc.bound = 8;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8::Plan plan(context, desc);
  rns8::Matrix a(context, bounded_matrix_desc(desc.m, desc.k, RNS8_BOUNDED_U64));
  const uint64_t values[] = {1, 2};
  REQUIRE(rns8_pack_u64(context.get(), a.get(), values, desc.k, 77) == RNS8_SUCCESS);

  const auto key = rns8::prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.operand_role == RNS8_OPERAND_A);
  CHECK(key.source_version == 77);
  CHECK(key.hip_device_id == -1);
  CHECK(key.cache_key_hash != 0);
}

