TEST_CASE("Direct-HIP result cache recomputes dirty bounded output rectangles", "[hip][result-cache]") {
  if (!hip_available()) {
    SKIP("HIP direct backend unavailable");
  }

  constexpr int64_t m = 4;
  constexpr int64_t n = 5;
  constexpr int64_t k = 3;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_gemm_desc cpu_desc = signed_desc(m, n, k, 512, RNS8_BACKEND_CPU_REFERENCE);
  rns8_gemm_desc hip_desc = signed_desc(m, n, k, 512, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  rns8_matrix_desc a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  rns8_matrix_desc b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  rns8_matrix_desc c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  std::vector<int64_t> A = {1, -2, 3, 4, 5, -6, -1, 2, -3, 6, -4, 1};
  std::vector<int64_t> B = {2, 0, -1, 3, 1, -2, 4, 1, 0, -3, 5, -1, 2, 1, 4};
  REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_b, B.data(), n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_b, B.data(), n, 2) == RNS8_SUCCESS);

  rns8_result_cache_desc cache_desc{};
  cache_desc.struct_size = sizeof(cache_desc);
  cache_desc.abi_version = RNS8_ABI_VERSION;
  cache_desc.flags = RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
                     RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
                     RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
                     RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  cache_desc.max_dirty_regions = 4;
  rns8_result_cache* cache = nullptr;
  REQUIRE(rns8_create_result_cache(hip, hip_plan, &cache_desc, &cache) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_rns_incremental(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace, cache, nullptr, 0) ==
          RNS8_SUCCESS);

  A[static_cast<std::size_t>(1 * k + 0)] = -7;
  A[static_cast<std::size_t>(1 * k + 1)] = 8;
  A[static_cast<std::size_t>(1 * k + 2)] = 2;
  REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 3) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_a, A.data(), k, 3) == RNS8_SUCCESS);

  rns8_workspace* cpu_workspace = nullptr;
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  std::vector<int64_t> expected(static_cast<std::size_t>(m * n));
  REQUIRE(rns8_export_i64(cpu, cpu_plan, cpu_c, expected.data(), n) == RNS8_SUCCESS);

  rns8_dirty_region dirty{};
  dirty.struct_size = sizeof(dirty);
  dirty.abi_version = RNS8_ABI_VERSION;
  dirty.row_offset = 1;
  dirty.col_offset = 0;
  dirty.row_extent = 1;
  dirty.col_extent = n;
  REQUIRE(rns8_gemm_rns_incremental(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace, cache, &dirty, 1) ==
          RNS8_SUCCESS);
  std::vector<int64_t> actual(static_cast<std::size_t>(m * n));
  REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, actual.data(), n) == RNS8_SUCCESS);
  CHECK(actual == expected);

  rns8_result_cache_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_result_cache_info(cache, &info) == RNS8_SUCCESS);
  CHECK(info.initialized == 1);
  CHECK(info.last_recomputed_region_count == 1);
  CHECK(info.recomputed_cell_count == static_cast<uint64_t>(n));
  CHECK(info.a_matrix_instance_id != 0);
  CHECK(info.b_matrix_instance_id != 0);

  REQUIRE(rns8_pack_i64(hip, hip_a, A.data(), k, 4) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns_incremental(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace, cache, nullptr, 0) ==
        RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_get_result_cache_info(cache, &info) == RNS8_SUCCESS);
  CHECK(info.last_stale_rejection == 1);

  REQUIRE(rns8_destroy_result_cache(cache) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(cpu) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(hip) == RNS8_SUCCESS);
}

TEST_CASE("Direct-HIP result cache recomputes dirty finite-u8 output rectangles", "[hip][result-cache]") {
  if (!hip_available()) {
    SKIP("HIP direct backend unavailable");
  }

  constexpr int64_t m = 3;
  constexpr int64_t n = 4;
  constexpr int64_t k = 3;
  constexpr uint16_t modulus = 251;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_gemm_desc cpu_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, modulus);
  rns8_gemm_desc hip_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, modulus);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  rns8_matrix_desc a_desc = matrix_desc(m, k, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  rns8_matrix_desc b_desc = matrix_desc(k, n, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  rns8_matrix_desc c_desc = matrix_desc(m, n, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  std::vector<uint8_t> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<uint8_t> B = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, modulus, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(cpu, cpu_b, modulus, B.data(), n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(hip, hip_a, modulus, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(hip, hip_b, modulus, B.data(), n, 2) == RNS8_SUCCESS);

  rns8_result_cache_desc cache_desc{};
  cache_desc.struct_size = sizeof(cache_desc);
  cache_desc.abi_version = RNS8_ABI_VERSION;
  cache_desc.flags = RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
                     RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
                     RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
                     RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  cache_desc.max_dirty_regions = 4;
  rns8_result_cache* cache = nullptr;
  REQUIRE(rns8_create_result_cache(hip, hip_plan, &cache_desc, &cache) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8_incremental(
              hip, hip_plan, modulus, hip_a, hip_b, hip_c, hip_workspace, cache, nullptr, 0) == RNS8_SUCCESS);

  A[static_cast<std::size_t>(2 * k + 0)] = 41;
  A[static_cast<std::size_t>(2 * k + 1)] = 42;
  A[static_cast<std::size_t>(2 * k + 2)] = 43;
  REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, modulus, A.data(), k, 3) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(hip, hip_a, modulus, A.data(), k, 3) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(cpu, cpu_plan, modulus, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  std::vector<uint8_t> expected(static_cast<std::size_t>(m * n));
  REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, modulus, cpu_c, expected.data(), n) == RNS8_SUCCESS);

  rns8_dirty_region dirty{};
  dirty.struct_size = sizeof(dirty);
  dirty.abi_version = RNS8_ABI_VERSION;
  dirty.row_offset = 2;
  dirty.col_offset = 0;
  dirty.row_extent = 1;
  dirty.col_extent = n;
  REQUIRE(rns8_gemm_finite_u8_incremental(
              hip, hip_plan, modulus, hip_a, hip_b, hip_c, hip_workspace, cache, &dirty, 1) == RNS8_SUCCESS);
  std::vector<uint8_t> actual(static_cast<std::size_t>(m * n));
  REQUIRE(rns8_export_finite_u8(hip, hip_plan, modulus, hip_c, actual.data(), n) == RNS8_SUCCESS);
  CHECK(actual == expected);

  REQUIRE(rns8_destroy_result_cache(cache) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(cpu) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(hip) == RNS8_SUCCESS);
}
