TEST_CASE("rocWMMA fused bounded RNS backend matches CPU and direct HIP") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);

  constexpr int64_t m = 65;
  constexpr int64_t n = 79;
  constexpr int64_t k = 33;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = (row + col) % 5 == 0 ? -9 : (row - 2 * col) % 11;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = (row * 3 + col * 5) % 17 - 8;
    }
  }
  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> rocwmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_HIP_DIRECT);
  auto rocwmma_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_ROCWMMA);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(rocwmma, &rocwmma_desc, A.data(), k, B.data(), n, rocwmma_out.data(), n) == RNS8_SUCCESS);
  require_same_i64(cpu_out, hip_out);
  require_same_i64(cpu_out, rocwmma_out);

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA reusable B prepack cache matches normal GEMM and CPU") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);

  constexpr int64_t m = 37;
  constexpr int64_t n = 53;
  constexpr int64_t k = 35;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = ((row * 7 + col * 3) % 19) - 9;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = ((row * 5 - col * 11) % 23) - 11;
    }
  }

  auto cpu_desc = bounded_i64_desc(m, n, k, 4000000, RNS8_BACKEND_CPU_REFERENCE);
  auto rocwmma_desc = bounded_i64_desc(m, n, k, 4000000, RNS8_BACKEND_ROCWMMA);
  cpu_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  rocwmma_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* rocwmma_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* rocwmma_workspace = nullptr;
  rns8_workspace* cached_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* rocwmma_a = nullptr;
  rns8_matrix* rocwmma_b = nullptr;
  rns8_matrix* rocwmma_c = nullptr;
  rns8_matrix* cached_c = nullptr;
  rns8_prepack_cache* b_cache = nullptr;

  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(rocwmma, &rocwmma_desc, &rocwmma_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &rocwmma_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &cached_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, RNS8_DEFAULT_BOUNDED_PREFIX);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(rocwmma, &a_desc, &rocwmma_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(rocwmma, &b_desc, &rocwmma_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &rocwmma_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &cached_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_b, B.data(), n, 202) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(rocwmma, rocwmma_a, A.data(), k, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(rocwmma, rocwmma_b, B.data(), n, 202) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(rocwmma_plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.reusable_prepack_cache_available == 1);
  CHECK(packing.production_prepack_cache_available == 1);
  CHECK(std::string(packing.prepack_cache_scope) == "production_reusable_b_prepack_cache");

  rns8_plan_schedule_info schedule{};
  schedule.struct_size = sizeof(schedule);
  schedule.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(rocwmma_plan, &schedule) == RNS8_SUCCESS);
  CHECK(schedule.max_selected_prefix < RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(schedule.adaptive_skip_active == 1);

  rns8_prepack_cache_key_info key{};
  key.struct_size = sizeof(key);
  key.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_prepack_cache_key_info(rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &key) == RNS8_SUCCESS);
  CHECK(key.reusable_prepack_cache_available == 1);
  CHECK(key.production_prepack_cache_available == 1);
  CHECK(std::string(key.cache_scope) == "validated_production_b_prepack_cache_key");
  CHECK(key.hip_device_id == default_visible_hip_test_device());
  CHECK(key.max_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(std::string(key.operand_layout_version) == "rns_i8_tile_swizzled_b_v1");
  CHECK(std::string(key.cache_key).find("prepack-v2") == 0);
  CHECK(std::string(key.cache_key).find("target_id=") != std::string::npos);
  CHECK(
      std::string(key.cache_key).find("kernel=rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2") !=
      std::string::npos);
  CHECK(std::string(key.cache_key).find("prepack_kernel=rocwmma_rns_i8_tile_swizzled_b_prepack_v1") !=
        std::string::npos);
  CHECK(std::string(key.cache_key).find("prefix_schedule_hash=") != std::string::npos);
  CHECK(std::string(key.cache_key).find("tile_m=128") != std::string::npos);
  CHECK(std::string(key.cache_key).find("tile_n=128") != std::string::npos);
  CHECK(std::string(key.cache_key).find("operand_tile_m=16") != std::string::npos);
  CHECK(std::string(key.cache_key).find("operand_tile_n=16") != std::string::npos);
  CHECK(std::string(key.cache_key).find("k_block_size=48") != std::string::npos);
  CHECK(std::string(key.cache_key).find("k_block_cap=65536") != std::string::npos);
  CHECK(std::string(key.cache_key).find("source_version=202") != std::string::npos);
  const std::string expected_device_key = "hip_device_id=" + std::to_string(default_visible_hip_test_device());
  CHECK(std::string(key.cache_key).find(expected_device_key) != std::string::npos);

  CHECK(rns8_create_prepack_cache(cpu, cpu_plan, cpu_b, RNS8_OPERAND_B, &b_cache) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(b_cache == nullptr);
  CHECK(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_a, RNS8_OPERAND_A, &b_cache) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(b_cache == nullptr);
  rns8::test::set_device_residues_current(*rocwmma_b, false);
  CHECK(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &b_cache) == RNS8_INVALID_ARGUMENT);
  CHECK(b_cache == nullptr);
  rns8::test::set_device_residues_current(*rocwmma_b, true);
  REQUIRE(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &b_cache) == RNS8_SUCCESS);

  rns8_prepack_cache_info cache_info{};
  cache_info.struct_size = sizeof(cache_info);
  cache_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_prepack_cache_info(b_cache, &cache_info) == RNS8_SUCCESS);
  CHECK(cache_info.backend == RNS8_BACKEND_ROCWMMA);
  CHECK(cache_info.semantics == RNS8_BOUNDED_I64);
  CHECK(cache_info.operand_role == RNS8_OPERAND_B);
  CHECK(cache_info.cache_key_valid == 1);
  CHECK(cache_info.reusable_prepack_cache_available == 1);
  CHECK(cache_info.production_prepack_cache_available == 1);
  CHECK(cache_info.hip_device_id == 0);
  CHECK(cache_info.reserved0 == 0);
  CHECK(cache_info.matrix_rows == k);
  CHECK(cache_info.matrix_cols == n);
  CHECK(cache_info.k == k);
  CHECK(cache_info.max_prefix == schedule.max_selected_prefix);
  CHECK(cache_info.finite_modulus == 0);
  CHECK(cache_info.source_version == key.source_version);
  CHECK(cache_info.plan_fingerprint == key.plan_fingerprint);
  CHECK(cache_info.cache_key_hash == key.cache_key_hash);
  CHECK(cache_info.device_bytes > 0);
  CHECK(cache_info.operand_pack_bytes > 0);
  CHECK(std::string(cache_info.matrix_layout_version) == std::string(key.matrix_layout_version));
  CHECK(std::string(cache_info.operand_layout_version) == std::string(key.operand_layout_version));
  CHECK(std::string(cache_info.cache_scope) == "runtime_production_b_prepack_cache");
  CHECK(std::string(cache_info.cache_key) == std::string(key.cache_key));

  rns8_prepack_cache_info bad_cache_info{};
  bad_cache_info.struct_size = sizeof(bad_cache_info) - 1;
  bad_cache_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_info(b_cache, &bad_cache_info) == RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache_info null_cache_info{};
  null_cache_info.struct_size = sizeof(null_cache_info);
  null_cache_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_info(nullptr, &null_cache_info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_info(b_cache, nullptr) == RNS8_INVALID_ARGUMENT);

  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(rocwmma, rocwmma_plan, rocwmma_a, rocwmma_b, rocwmma_c, rocwmma_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, b_cache, cached_c, cached_workspace) == RNS8_SUCCESS);

  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> rocwmma_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> cached_out(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_export_i64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(rocwmma, rocwmma_plan, rocwmma_c, rocwmma_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(rocwmma, rocwmma_plan, cached_c, cached_out.data(), n) == RNS8_SUCCESS);
  require_same_i64(cpu_out, rocwmma_out);
  require_same_i64(cpu_out, cached_out);

  rns8_prepack_cache role_mismatch = *b_cache;
  role_mismatch.operand_role = RNS8_OPERAND_A;
  CHECK(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, &role_mismatch, cached_c, cached_workspace) ==
        RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache target_mismatch = *b_cache;
  target_mismatch.target_id = "gfx0000";
  CHECK(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, &target_mismatch, cached_c, cached_workspace) ==
        RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache device_mismatch = *b_cache;
  device_mismatch.hip_device_id = 999;
  CHECK(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, &device_mismatch, cached_c, cached_workspace) ==
        RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache source_version_mismatch = *b_cache;
  ++source_version_mismatch.source_version;
  CHECK(rns8_gemm_rns_prepacked_b(
            rocwmma, rocwmma_plan, rocwmma_a, &source_version_mismatch, cached_c, cached_workspace) ==
        RNS8_INVALID_ARGUMENT);

  rns8_destroy_prepack_cache(b_cache);
  rns8_destroy_matrix(cached_c);
  rns8_destroy_matrix(rocwmma_c);
  rns8_destroy_matrix(rocwmma_b);
  rns8_destroy_matrix(rocwmma_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(cached_workspace);
  rns8_destroy_workspace(rocwmma_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(rocwmma_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(rocwmma);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA reusable B prepack cache covers unsigned and exact-wide RNS semantics") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);

  SECTION("bounded u64") {
    constexpr int64_t m = 23;
    constexpr int64_t n = 29;
    constexpr int64_t k = 31;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * k + col)] = static_cast<uint64_t>((row * 5 + col * 7) % 17);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>((row * 11 + col * 13 + 3) % 19);
      }
    }

    auto cpu_desc = bounded_u64_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto rocwmma_desc = bounded_u64_desc(m, n, k, 1000000, RNS8_BACKEND_ROCWMMA);
    cpu_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    rocwmma_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* rocwmma_plan = nullptr;
    rns8_workspace* cpu_workspace = nullptr;
    rns8_workspace* rocwmma_workspace = nullptr;
    rns8_workspace* cached_workspace = nullptr;
    rns8_matrix* cpu_a = nullptr;
    rns8_matrix* cpu_b = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* rocwmma_a = nullptr;
    rns8_matrix* rocwmma_b = nullptr;
    rns8_matrix* rocwmma_c = nullptr;
    rns8_matrix* cached_c = nullptr;
    rns8_prepack_cache* b_cache = nullptr;

    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(rocwmma, &rocwmma_desc, &rocwmma_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &cached_workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &a_desc, &rocwmma_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &b_desc, &rocwmma_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &rocwmma_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &cached_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 302) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(rocwmma, rocwmma_a, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(rocwmma, rocwmma_b, B.data(), n, 302) == RNS8_SUCCESS);

    rns8_prepack_cache_key_info key{};
    key.struct_size = sizeof(key);
    key.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_key_info(rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &key) == RNS8_SUCCESS);
    CHECK(key.reusable_prepack_cache_available == 1);
    CHECK(key.production_prepack_cache_available == 1);
    CHECK(std::string(key.cache_scope) == "validated_production_b_prepack_cache_key");
    REQUIRE(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &b_cache) == RNS8_SUCCESS);
    rns8_prepack_cache_info cache_info{};
    cache_info.struct_size = sizeof(cache_info);
    cache_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_info(b_cache, &cache_info) == RNS8_SUCCESS);
    CHECK(cache_info.reusable_prepack_cache_available == 1);
    CHECK(cache_info.production_prepack_cache_available == 1);
    CHECK(std::string(cache_info.cache_scope) == "runtime_production_b_prepack_cache");
    REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(rocwmma, rocwmma_plan, rocwmma_a, rocwmma_b, rocwmma_c, rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, b_cache, cached_c, cached_workspace) == RNS8_SUCCESS);

    std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> rocwmma_out(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> cached_out(static_cast<std::size_t>(m * n), 0);
    REQUIRE(rns8_export_u64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(rocwmma, rocwmma_plan, rocwmma_c, rocwmma_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(rocwmma, rocwmma_plan, cached_c, cached_out.data(), n) == RNS8_SUCCESS);
    require_same_u64(cpu_out, rocwmma_out);
    require_same_u64(cpu_out, cached_out);

    rns8_destroy_prepack_cache(b_cache);
    rns8_destroy_matrix(cached_c);
    rns8_destroy_matrix(rocwmma_c);
    rns8_destroy_matrix(rocwmma_b);
    rns8_destroy_matrix(rocwmma_a);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_matrix(cpu_b);
    rns8_destroy_matrix(cpu_a);
    rns8_destroy_workspace(cached_workspace);
    rns8_destroy_workspace(rocwmma_workspace);
    rns8_destroy_workspace(cpu_workspace);
    rns8_destroy_plan(rocwmma_plan);
    rns8_destroy_plan(cpu_plan);
  }

  SECTION("exact-wide signed") {
    constexpr int64_t m = 17;
    constexpr int64_t n = 19;
    constexpr int64_t k = 23;
    constexpr uint32_t limb_count = 2;
    std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * k + col)] = ((row * 3 + col * 5) % 29) - 14;
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] = ((row * 7 - col * 2) % 31) - 15;
      }
    }

    auto cpu_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto rocwmma_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_ROCWMMA);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* rocwmma_plan = nullptr;
    rns8_workspace* cpu_workspace = nullptr;
    rns8_workspace* rocwmma_workspace = nullptr;
    rns8_workspace* cached_workspace = nullptr;
    rns8_matrix* cpu_a = nullptr;
    rns8_matrix* cpu_b = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* rocwmma_a = nullptr;
    rns8_matrix* rocwmma_b = nullptr;
    rns8_matrix* rocwmma_c = nullptr;
    rns8_matrix* cached_c = nullptr;
    rns8_prepack_cache* b_cache = nullptr;

    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(rocwmma, &rocwmma_desc, &rocwmma_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &cached_workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &a_desc, &rocwmma_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &b_desc, &rocwmma_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &rocwmma_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &cached_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 401) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(cpu, cpu_b, B.data(), n, 402) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(rocwmma, rocwmma_a, A.data(), k, 401) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(rocwmma, rocwmma_b, B.data(), n, 402) == RNS8_SUCCESS);

    rns8_prepack_cache_key_info key{};
    key.struct_size = sizeof(key);
    key.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_key_info(rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &key) == RNS8_SUCCESS);
    CHECK(key.reusable_prepack_cache_available == 1);
    CHECK(key.production_prepack_cache_available == 0);
    CHECK(std::string(key.cache_scope) == "validated_reusable_b_prepack_cache_key_evidence_only");
    REQUIRE(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &b_cache) == RNS8_SUCCESS);
    rns8_prepack_cache_info cache_info{};
    cache_info.struct_size = sizeof(cache_info);
    cache_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_info(b_cache, &cache_info) == RNS8_SUCCESS);
    CHECK(cache_info.reusable_prepack_cache_available == 1);
    CHECK(cache_info.production_prepack_cache_available == 0);
    CHECK(std::string(cache_info.cache_scope) == "runtime_reusable_b_prepack_cache");
    REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(rocwmma, rocwmma_plan, rocwmma_a, rocwmma_b, rocwmma_c, rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, b_cache, cached_c, cached_workspace) == RNS8_SUCCESS);

    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> rocwmma_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> cached_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(rocwmma, rocwmma_plan, rocwmma_c, rocwmma_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(rocwmma, rocwmma_plan, cached_c, cached_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    require_same_u64(cpu_limbs, rocwmma_limbs);
    require_same_u64(cpu_limbs, cached_limbs);

    rns8_destroy_prepack_cache(b_cache);
    rns8_destroy_matrix(cached_c);
    rns8_destroy_matrix(rocwmma_c);
    rns8_destroy_matrix(rocwmma_b);
    rns8_destroy_matrix(rocwmma_a);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_matrix(cpu_b);
    rns8_destroy_matrix(cpu_a);
    rns8_destroy_workspace(cached_workspace);
    rns8_destroy_workspace(rocwmma_workspace);
    rns8_destroy_workspace(cpu_workspace);
    rns8_destroy_plan(rocwmma_plan);
    rns8_destroy_plan(cpu_plan);
  }

  SECTION("exact-wide unsigned") {
    constexpr int64_t m = 13;
    constexpr int64_t n = 17;
    constexpr int64_t k = 19;
    constexpr uint32_t limb_count = 2;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * k + col)] = static_cast<uint64_t>((row * 17 + col * 3 + 1) % 251);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>((row * 5 + col * 19 + 2) % 241);
      }
    }

    auto cpu_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto rocwmma_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_ROCWMMA);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* rocwmma_plan = nullptr;
    rns8_workspace* cpu_workspace = nullptr;
    rns8_workspace* rocwmma_workspace = nullptr;
    rns8_workspace* cached_workspace = nullptr;
    rns8_matrix* cpu_a = nullptr;
    rns8_matrix* cpu_b = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* rocwmma_a = nullptr;
    rns8_matrix* rocwmma_b = nullptr;
    rns8_matrix* rocwmma_c = nullptr;
    rns8_matrix* cached_c = nullptr;
    rns8_prepack_cache* b_cache = nullptr;

    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(rocwmma, &rocwmma_desc, &rocwmma_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(rocwmma, rocwmma_plan, &cached_workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &a_desc, &rocwmma_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &b_desc, &rocwmma_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &rocwmma_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(rocwmma, &c_desc, &cached_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 501) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 502) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(rocwmma, rocwmma_a, A.data(), k, 501) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(rocwmma, rocwmma_b, B.data(), n, 502) == RNS8_SUCCESS);

    rns8_prepack_cache_key_info key{};
    key.struct_size = sizeof(key);
    key.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_key_info(rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &key) == RNS8_SUCCESS);
    CHECK(key.reusable_prepack_cache_available == 1);
    CHECK(key.production_prepack_cache_available == 0);
    CHECK(std::string(key.cache_scope) == "validated_reusable_b_prepack_cache_key_evidence_only");
    REQUIRE(rns8_create_prepack_cache(rocwmma, rocwmma_plan, rocwmma_b, RNS8_OPERAND_B, &b_cache) == RNS8_SUCCESS);
    rns8_prepack_cache_info cache_info{};
    cache_info.struct_size = sizeof(cache_info);
    cache_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepack_cache_info(b_cache, &cache_info) == RNS8_SUCCESS);
    CHECK(cache_info.reusable_prepack_cache_available == 1);
    CHECK(cache_info.production_prepack_cache_available == 0);
    CHECK(std::string(cache_info.cache_scope) == "runtime_reusable_b_prepack_cache");
    REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(rocwmma, rocwmma_plan, rocwmma_a, rocwmma_b, rocwmma_c, rocwmma_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_prepacked_b(rocwmma, rocwmma_plan, rocwmma_a, b_cache, cached_c, cached_workspace) == RNS8_SUCCESS);

    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> rocwmma_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> cached_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(rocwmma, rocwmma_plan, rocwmma_c, rocwmma_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(rocwmma, rocwmma_plan, cached_c, cached_limbs.data(), n, limb_count) ==
            RNS8_SUCCESS);
    require_same_u64(cpu_limbs, rocwmma_limbs);
    require_same_u64(cpu_limbs, cached_limbs);

    rns8_destroy_prepack_cache(b_cache);
    rns8_destroy_matrix(cached_c);
    rns8_destroy_matrix(rocwmma_c);
    rns8_destroy_matrix(rocwmma_b);
    rns8_destroy_matrix(rocwmma_a);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_matrix(cpu_b);
    rns8_destroy_matrix(cpu_a);
    rns8_destroy_workspace(cached_workspace);
    rns8_destroy_workspace(rocwmma_workspace);
    rns8_destroy_workspace(cpu_workspace);
    rns8_destroy_plan(rocwmma_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(cpu);
}

