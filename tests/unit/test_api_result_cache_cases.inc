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

TEST_CASE("result cache failed snapshot preparation cannot remain initialized", "[api][result-cache]") {
  rns8_context ctx{};
  rns8_result_cache cache{};
  rns8_matrix output{};
  cache.initialized = true;
  // Zero-sized preparation is rejected before any backend operation. This
  // verifies the host publication invariant without claiming GPU fault coverage.
  CHECK(rns8::detail::api::snapshot_result_cache_from_output(ctx, cache, output) == RNS8_INVALID_ARGUMENT);
  CHECK_FALSE(cache.initialized);
  cache.initialized = true;
  CHECK(rns8::detail::api::free_result_cache_snapshot(cache) == RNS8_SUCCESS);
  CHECK_FALSE(cache.initialized);
  CHECK(cache.hip_snapshot_residues == nullptr);
  CHECK(cache.cache_allocation_bytes == 0);
}

TEST_CASE("generated HIP values invalidate downstream result cache after upstream mutation", "[api][result-cache]") {
  if (!rns8::detail::api::direct_hip_compiled()) SKIP("requires actual HIP execution");
  rns8_context_options options{};
  options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;
  REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  auto desc=bounded_gemm_desc(RNS8_BOUNDED_I64,2,2,2);
  desc.requested_backend=RNS8_BACKEND_HIP_DIRECT; desc.bound=10000;
  rns8_plan* plan=nullptr; rns8_workspace* ws=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
  auto md=bounded_matrix_desc(2,2,RNS8_BOUNDED_I64);
  rns8_matrix *a=nullptr,*alternate=nullptr,*b=nullptr,*intermediate=nullptr,*out=nullptr;
  for(auto** m:{&a,&alternate,&b,&intermediate,&out})
    REQUIRE(rns8_create_matrix(ctx,&md,m)==RNS8_SUCCESS);
  const int64_t identity[]={1,0,0,1}, first[]={1,2,3,4}, second[]={5,6,7,8};
  REQUIRE(rns8_pack_i64(ctx,b,identity,2,1)==RNS8_SUCCESS);
  rns8_result_cache_desc cd{};
  cd.struct_size=sizeof(cd); cd.abi_version=RNS8_ABI_VERSION;
  cd.flags=RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
           RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
           RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
           RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  rns8_result_cache* cache=nullptr;
  REQUIRE(rns8_create_result_cache(ctx,plan,&cd,&cache)==RNS8_SUCCESS);
  auto cached=[&]() {return rns8_gemm_rns_incremental(ctx,plan,intermediate,b,out,ws,cache,nullptr,0);};
  for(bool replace_handle:{false,true}) {
    REQUIRE(rns8_invalidate_result_cache(cache)==RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx,a,first,2,replace_handle?7:0)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx,plan,a,b,intermediate,ws)==RNS8_SUCCESS);
    REQUIRE(cached()==RNS8_SUCCESS);
    REQUIRE(cached()==RNS8_SUCCESS);
    CHECK(cache->last_cache_hit);
    int64_t result[4]{};
    REQUIRE(rns8_export_i64(ctx,plan,out,result,2)==RNS8_SUCCESS);
    for(int i=0;i<4;++i) CHECK(result[i]==first[i]);
    auto* changed=replace_handle?alternate:a;
    REQUIRE(rns8_pack_i64(ctx,changed,second,2,replace_handle?7:0)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx,plan,changed,b,intermediate,ws)==RNS8_SUCCESS);
    CHECK(cached()==RNS8_INVALID_ARGUMENT);
    CHECK(cache->last_stale_rejection); CHECK_FALSE(cache->last_cache_hit);
    REQUIRE(rns8_invalidate_result_cache(cache)==RNS8_SUCCESS);
    REQUIRE(cached()==RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(ctx,plan,out,result,2)==RNS8_SUCCESS);
    for(int i=0;i<4;++i) CHECK(result[i]==second[i]);
  }
  rns8_destroy_result_cache(cache);
  for(auto* m:{a,alternate,b,intermediate,out}) rns8_destroy_matrix(m);
  rns8_destroy_workspace(ws); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
}
