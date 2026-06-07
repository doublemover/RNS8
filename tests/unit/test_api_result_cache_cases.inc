TEST_CASE("matrix storage info exposes stable instance identity", "[api][result-cache]") {
  rns8_context* ctx = create_cpu_context();
  rns8_matrix_desc desc = bounded_matrix_desc(2, 2, RNS8_BOUNDED_I64);
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &desc, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &desc, &b) == RNS8_SUCCESS);

  rns8_matrix_storage_info a_info{};
  a_info.struct_size = sizeof(a_info);
  a_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_storage_info(a, &a_info) == RNS8_SUCCESS);
  rns8_matrix_storage_info b_info{};
  b_info.struct_size = sizeof(b_info);
  b_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_storage_info(b, &b_info) == RNS8_SUCCESS);

  CHECK(a_info.matrix_instance_id != 0);
  CHECK(b_info.matrix_instance_id != 0);
  CHECK(a_info.matrix_instance_id != b_info.matrix_instance_id);

  REQUIRE(rns8_destroy_matrix(a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(b) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("result cache create validates ABI before backend support", "[api][result-cache]") {
  rns8_context* ctx = create_cpu_context();
  rns8_gemm_desc desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 2, 2, 2);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_result_cache_desc cache_desc{};
  cache_desc.struct_size = sizeof(cache_desc) - 1;
  cache_desc.abi_version = RNS8_ABI_VERSION;
  cache_desc.flags = RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
                     RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
                     RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
                     RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  rns8_result_cache* cache = nullptr;
  CHECK(rns8_create_result_cache(ctx, plan, &cache_desc, &cache) == RNS8_INVALID_ARGUMENT);
  CHECK(cache == nullptr);

  cache_desc.struct_size = sizeof(cache_desc);
  CHECK(rns8_create_result_cache(ctx, plan, &cache_desc, &cache) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(cache == nullptr);

  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("result cache info rejects invalid ABI structs", "[api][result-cache]") {
  rns8_result_cache_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_result_cache_info(nullptr, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_invalidate_result_cache(nullptr) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_destroy_result_cache(nullptr) == RNS8_SUCCESS);
}
