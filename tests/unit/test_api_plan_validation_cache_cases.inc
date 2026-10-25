TEST_CASE("immutable schedule validation is amortized without hash-only equivalence", "[api][schedule]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  auto desc = exact_desc(RNS8_BOUNDED_U64, 16384, 16384, 1);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc.tile_m = desc.tile_n = 128;
  desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  std::vector<uint64_t> bounds(128 * 128, 100000);
  std::vector<uint8_t> zero_rows(16384), zero_cols(16384);
  desc.tile_bounds = bounds.data(); desc.tile_bounds_count = bounds.size();
  desc.zero_a_rows = zero_rows.data(); desc.zero_a_rows_count = zero_rows.size();
  desc.zero_b_cols = zero_cols.data(); desc.zero_b_cols_count = zero_cols.size();
  rns8_plan *plan = nullptr, *equivalent = nullptr, *different = nullptr;
  plan_validation_work = {};
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan_validation_work.exhaustive_validations == 1);
  REQUIRE(plan_validation_work.fingerprint_traversals == 1);
  REQUIRE(plan->validated_contract);
  const auto& launch = plan->validated_contract->launch;
  CHECK(launch.entry_count == 128 * 128);
  CHECK(launch.max_row_blocks == 8);
  CHECK(launch.max_col_blocks == 8);
  CHECK(launch.max_elements == 128 * 128);
  CHECK(launch.i32_launch_qualified);
  CHECK_FALSE(launch.zero_output_tiles);
  CHECK(launch.uniform_zero_selected_prefix == 0);
  CHECK(plan_validation_work.backend_schedule_seals == 1);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits == 128 * 128);
  REQUIRE(rns8_create_plan(ctx, &desc, &equivalent) == RNS8_SUCCESS);
  REQUIRE(plan->validated_contract != equivalent->validated_contract);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  plan_validation_work = {};
  for (int iteration = 0; iteration < 32; ++iteration) {
    REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_SUCCESS);
    REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
    REQUIRE(plan_workspace_fingerprint(*plan) == workspace->schedule_fingerprint);
  }
  REQUIRE(plan_validation_work.exhaustive_validations == 0);
  REQUIRE(plan_validation_work.fingerprint_traversals == 0);
  REQUIRE(plan_validation_work.equivalent_comparisons == 1);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits == 0);
  // Caller-owned arrays are copied. Mutation cannot alter an existing seal.
  bounds[0] = 99999;
  REQUIRE(rns8_create_plan(ctx, &desc, &different) == RNS8_SUCCESS);
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  // Force an artificial hash collision: structural equality must still reject.
  auto collision = std::make_shared<rns8_validated_plan_contract>(*different->validated_contract);
  collision->fingerprint = plan->validated_contract->fingerprint;
  different->validated_contract = collision;
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  const auto fingerprint = workspace->schedule_fingerprint;
  workspace->schedule_fingerprint ^= 1;
  REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_INVALID_ARGUMENT);
  workspace->schedule_fingerprint = fingerprint;
  // Diagnostics check coverage and proof fields even on a sealed plan. They
  // never silently reseal changed state or certify the previous fingerprint.
  const auto entry = plan->tile_schedule[0];
  plan->tile_schedule[0].row_extent -= 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->tile_schedule[0].required_prefix += 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->zero_a_rows[0] = 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->zero_a_rows[0] = 0;
  REQUIRE(revalidate_plan_schedule_contract(*plan));
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  // Workspace owns the immutable contract, not a dangling original-plan pointer.
  REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(equivalent) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(different) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("ordinary HIP tiled GEMM consumes the immutable launch seal", "[api][schedule][hip]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail::api;
  rns8_context_options options{};
  options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;
  REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  auto desc=bounded_gemm_desc(RNS8_BOUNDED_U64,128,128,1);
  desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  desc.bound_kind=RNS8_BOUND_PER_TILE_MAX_UNSIGNED;desc.bound=0;desc.tile_m=desc.tile_n=64;
  uint64_t bounds[]{1,1000,1000,1};desc.tile_bounds=bounds;desc.tile_bounds_count=4;
  rns8_plan *plan=nullptr,*equivalent=nullptr;
  rns8_workspace* workspace=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx,&desc,&equivalent)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  auto ad=bounded_matrix_desc(128,1,RNS8_BOUNDED_U64);
  auto bd=bounded_matrix_desc(1,128,RNS8_BOUNDED_U64);
  auto cd=bounded_matrix_desc(128,128,RNS8_BOUNDED_U64);
  REQUIRE(rns8_create_matrix(ctx,&ad,&a)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&bd,&b)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&cd,&c)==RNS8_SUCCESS);
  std::vector<uint64_t> ones(128,1),actual(128*130,UINT64_MAX);
  REQUIRE(rns8_pack_u64(ctx,a,ones.data(),1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx,b,ones.data(),128,0)==RNS8_SUCCESS);
  plan_validation_work={};
  for(int iteration=0;iteration<8;++iteration) {
    auto* selected=iteration%2?plan:equivalent;
    REQUIRE(rns8_gemm(ctx,selected,a,b,c,workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(ctx,selected,c,actual.data(),130)==RNS8_SUCCESS);
    for(int row=0;row<128;++row) {
      for(int col=0;col<128;++col) REQUIRE(actual[row*130+col]==1);
      CHECK(actual[row*130+128]==UINT64_MAX);CHECK(actual[row*130+129]==UINT64_MAX);
    }
  }
  CHECK(plan_validation_work.backend_schedule_fast_submissions==8);
  CHECK(plan_validation_work.backend_schedule_raw_submissions==0);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits==0);
  CHECK(plan_validation_work.exhaustive_validations==0);
  CHECK(plan_validation_work.fingerprint_traversals==0);
  CHECK(plan_validation_work.equivalent_comparisons==1);
  REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(equivalent)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#endif
}
