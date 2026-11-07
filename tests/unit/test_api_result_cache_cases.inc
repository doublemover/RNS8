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
  cache_desc.flags =
      RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES | RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
      RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE | RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  rns8_result_cache* cache = nullptr;
  CHECK(rns8_create_result_cache(ctx, plan, &cache_desc, &cache) == RNS8_INVALID_ARGUMENT);
  CHECK(cache == nullptr);

  cache_desc.struct_size = sizeof(cache_desc);
  CHECK(rns8_create_result_cache(ctx, plan, &cache_desc, &cache) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(cache == nullptr);

  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("incremental dirty descriptors reject before conversion or cache state changes",
          "[api][abi][result-cache][host]") {
  for (const bool finite : {false, true}) {
    auto* ctx = create_cpu_context();
    const auto desc = finite ? finite_gemm_desc(RNS8_FINITE_RING_U8, 1, 1, 1, 6)
                             : bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    const auto md = finite ? finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8)
                           : bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
    rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
    for (auto** out : {&a, &b, &c}) {
      REQUIRE(rns8_create_matrix(ctx, &md, out) == RNS8_SUCCESS);
      const uint8_t byte = 1;
      const int64_t word = 1;
      REQUIRE((finite ? rns8_pack_finite_u8(ctx, *out, 6, &byte, 1, 1)
                      : rns8_pack_i64(ctx, *out, &word, 1, 1)) == RNS8_SUCCESS);
    }
    // CPU has no result-cache backend. This host-only state object checks real
    // API preflight and its unsupported positive control, not HIP execution.
    rns8_result_cache cache{};
    cache.max_dirty_regions = 1;
    cache.last_cache_hit = true;
    cache.last_dirty_region_count = 71;
    cache.detail = "preflight canary";
    auto call = [&](const rns8_dirty_region* regions, uint32_t count) {
      return finite
                 ? rns8_gemm_finite_u8_incremental(ctx, plan, 6, a, b, c, workspace, &cache,
                                                   regions, count)
                 : rns8_gemm_rns_incremental(ctx, plan, a, b, c, workspace, &cache, regions, count);
    };
    const auto generation = c->content_generation, references = ctx->prepared_references;
    const auto residues = c->residues;
    const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    rns8_dirty_region valid{};
    valid.struct_size = sizeof(valid);
    valid.abi_version = RNS8_ABI_VERSION;
    valid.row_extent = valid.col_extent = 1;
    for (uint32_t bits : {1u, UINT32_C(0x80000000), UINT32_MAX}) {
      auto invalid = valid;
      invalid.flags = bits;
      CHECK(call(&invalid, 1) == RNS8_INVALID_ARGUMENT);
      CHECK(invalid.flags == bits);
    }
    for (unsigned mode = 0; mode < 5; ++mode) {
      auto invalid = valid;
      if (mode == 0)
        --invalid.struct_size;
      else if (mode == 1)
        ++invalid.abi_version;
      else if (mode == 2)
        invalid.row_offset = -1;
      else if (mode == 3)
        invalid.col_extent = 2;
      else
        invalid.row_extent = 0;
      CHECK(call(&invalid, 1) == RNS8_INVALID_ARGUMENT);
    }
    alignas(rns8_dirty_region) uint64_t short_header = sizeof(short_header);
    CHECK(call(reinterpret_cast<const rns8_dirty_region*>(&short_header), 1) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(short_header == sizeof(short_header));
    CHECK(call(&valid, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(call(nullptr, 1) == RNS8_INVALID_ARGUMENT);
    CHECK(call(&valid, 2) == RNS8_INVALID_ARGUMENT); // Reject count before a second read.
    CHECK(call(&valid, 1) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(call(nullptr, 0) == RNS8_UNSUPPORTED_BACKEND);
    struct ExtendedDirtyRegion {
      rns8_dirty_region value;
      uint64_t canary;
    } extended{valid, UINT64_C(0xfedcba9876543210)};
    extended.value.struct_size = sizeof(extended);
    CHECK(call(&extended.value, 1) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(extended.canary == UINT64_C(0xfedcba9876543210));
    CHECK(cache.last_cache_hit);
    CHECK(cache.last_dirty_region_count == 71);
    CHECK(cache.detail == "preflight canary");
    CHECK(cache.in_flight_uses == 0);
    CHECK(c->residues == residues);
    CHECK(c->content_generation == generation);
    CHECK(ctx->prepared_references == references);
    CHECK_FALSE(workspace->in_flight);
    for (const auto* matrix : {a, b, c}) {
      CHECK_FALSE(matrix->in_flight_writer);
      CHECK(matrix->in_flight_readers == 0);
    }
    const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(after.allocate_calls == allocations.allocate_calls);
    CHECK(after.free_calls == allocations.free_calls);
    CHECK(after.h2d_calls == allocations.h2d_calls);
    CHECK(after.d2h_calls == allocations.d2h_calls);
    for (auto* matrix : {a, b, c})
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}

TEST_CASE("result cache info rejects invalid ABI structs", "[api][result-cache]") {
  rns8_result_cache_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_result_cache_info(nullptr, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_invalidate_result_cache(nullptr) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_destroy_result_cache(nullptr) == RNS8_SUCCESS);
}

TEST_CASE("result cache failed snapshot preparation cannot remain initialized",
          "[api][result-cache]") {
  rns8_context ctx{};
  rns8_result_cache cache{};
  rns8_matrix output{};
  cache.initialized = true;
  // Zero-sized preparation is rejected before any backend operation. This
  // verifies the host publication invariant without claiming GPU fault coverage.
  CHECK(rns8::detail::api::snapshot_result_cache_from_output(ctx, cache, output) ==
        RNS8_INVALID_ARGUMENT);
  CHECK_FALSE(cache.initialized);
  cache.initialized = true;
  CHECK(rns8::detail::api::free_result_cache_snapshot(cache) == RNS8_SUCCESS);
  CHECK_FALSE(cache.initialized);
  CHECK(cache.hip_snapshot_residues == nullptr);
  CHECK(cache.cache_allocation_bytes == 0);
}

TEST_CASE("generated HIP values invalidate downstream result cache after upstream mutation",
          "[api][result-cache]") {
  if (!rns8::detail::api::direct_hip_compiled())
    SKIP("requires actual HIP execution");
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 2, 2, 2);
  desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  desc.bound = 10000;
  rns8_plan* plan = nullptr;
  rns8_workspace* ws = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &ws) == RNS8_SUCCESS);
  auto md = bounded_matrix_desc(2, 2, RNS8_BOUNDED_I64);
  rns8_matrix *a = nullptr, *alternate = nullptr, *b = nullptr, *intermediate = nullptr,
              *out = nullptr;
  for (auto** m : {&a, &alternate, &b, &intermediate, &out})
    REQUIRE(rns8_create_matrix(ctx, &md, m) == RNS8_SUCCESS);
  const int64_t identity[] = {1, 0, 0, 1}, first[] = {1, 2, 3, 4}, second[] = {5, 6, 7, 8};
  REQUIRE(rns8_pack_i64(ctx, b, identity, 2, 1) == RNS8_SUCCESS);
  rns8_result_cache_desc cd{};
  cd.struct_size = sizeof(cd);
  cd.abi_version = RNS8_ABI_VERSION;
  cd.flags =
      RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES | RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
      RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE | RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  rns8_result_cache* cache = nullptr;
  REQUIRE(rns8_create_result_cache(ctx, plan, &cd, &cache) == RNS8_SUCCESS);
  auto cached = [&]() {
    return rns8_gemm_rns_incremental(ctx, plan, intermediate, b, out, ws, cache, nullptr, 0);
  };
  for (bool replace_handle : {false, true}) {
    REQUIRE(rns8_invalidate_result_cache(cache) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, first, 2, replace_handle ? 7 : 0) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, a, b, intermediate, ws) == RNS8_SUCCESS);
    REQUIRE(cached() == RNS8_SUCCESS);
    REQUIRE(cached() == RNS8_SUCCESS);
    CHECK(cache->last_cache_hit);
    int64_t result[4]{};
    REQUIRE(rns8_export_i64(ctx, plan, out, result, 2) == RNS8_SUCCESS);
    for (int i = 0; i < 4; ++i)
      CHECK(result[i] == first[i]);
    auto* changed = replace_handle ? alternate : a;
    REQUIRE(rns8_pack_i64(ctx, changed, second, 2, replace_handle ? 7 : 0) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, changed, b, intermediate, ws) == RNS8_SUCCESS);
    CHECK(cached() == RNS8_INVALID_ARGUMENT);
    CHECK(cache->last_stale_rejection);
    CHECK_FALSE(cache->last_cache_hit);
    REQUIRE(rns8_invalidate_result_cache(cache) == RNS8_SUCCESS);
    REQUIRE(cached() == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(ctx, plan, out, result, 2) == RNS8_SUCCESS);
    for (int i = 0; i < 4; ++i)
      CHECK(result[i] == second[i]);
  }
  rns8_destroy_result_cache(cache);
  for (auto* m : {a, alternate, b, intermediate, out})
    rns8_destroy_matrix(m);
  rns8_destroy_workspace(ws);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
