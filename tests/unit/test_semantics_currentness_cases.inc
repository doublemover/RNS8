TEST_CASE("persistent bounded RNS GEMM rejects same-shape stale schedule workspaces and matrix tiles") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  const uint64_t bounds_a[4] = {0, 0, 7000000, 1000000000};
  const uint64_t bounds_b[4] = {0, 0, 7000000, 999999999};
  const uint64_t bounds_tile[1] = {1000000000};
  std::vector<uint64_t> A(static_cast<std::size_t>(m), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(n), 7);
  A.back() = 1000000;
  B.back() = 1000;

  auto desc_a = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  desc_a.m = m;
  desc_a.n = n;
  desc_a.k = k;
  desc_a.bound = 0;
  desc_a.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc_a.tile_m = 64;
  desc_a.tile_n = 64;
  desc_a.tile_bounds = bounds_a;
  desc_a.tile_bounds_count = 4;

  auto desc_b = desc_a;
  desc_b.tile_bounds = bounds_b;

  auto desc_tile = desc_a;
  desc_tile.tile_m = 128;
  desc_tile.tile_n = 128;
  desc_tile.tile_bounds = bounds_tile;
  desc_tile.tile_bounds_count = 1;

  rns8_plan* plan_a = nullptr;
  rns8_plan* plan_b = nullptr;
  rns8_plan* plan_tile = nullptr;
  rns8_workspace* workspace_a = nullptr;
  rns8_workspace* workspace_b = nullptr;
  rns8_workspace* workspace_tile = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* wrong_a_tile = nullptr;
  rns8_matrix* wrong_c_tile = nullptr;

  REQUIRE(rns8_create_plan(ctx, &desc_a, &plan_a) == RNS8_SUCCESS);
  REQUIRE_FALSE(plan_a->tile_schedule.empty());
  CHECK(plan_a->tile_schedule[0].range_bit_length == 0);
  REQUIRE(rns8_create_plan(ctx, &desc_b, &plan_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &desc_tile, &plan_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_a, &workspace_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_b, &workspace_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_tile, &workspace_tile) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), n, 1) == RNS8_SUCCESS);

  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_a) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_b) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_tile) == RNS8_INVALID_ARGUMENT);

  auto wrong_a_desc = a_desc;
  wrong_a_desc.tile_m = 128;
  wrong_a_desc.tile_n = 128;
  auto wrong_c_desc = c_desc;
  wrong_c_desc.tile_m = 128;
  wrong_c_desc.tile_n = 128;
  REQUIRE(rns8_create_matrix(ctx, &wrong_a_desc, &wrong_a_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_desc, &wrong_c_tile) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan_a, wrong_a_tile, b_matrix, c_matrix, workspace_a) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, wrong_c_tile, workspace_a) == RNS8_INVALID_ARGUMENT);
  uint64_t out[4] = {};
  CHECK(rns8_export_u64(ctx, plan_a, wrong_c_tile, out, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_c_tile);
  rns8_destroy_matrix(wrong_a_tile);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace_tile);
  rns8_destroy_workspace(workspace_b);
  rns8_destroy_workspace(workspace_a);
  rns8_destroy_plan(plan_tile);
  rns8_destroy_plan(plan_b);
  rns8_destroy_plan(plan_a);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS APIs reject stale plan schedule metadata before dispatch") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 1;
  desc.n = 1;
  desc.k = 1;
  desc.bound = 100000;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);

  plan->schedule_prefix_group_count = 0;
  CHECK(rns8_get_plan_schedule_info(plan, &info) == RNS8_INVALID_ARGUMENT);
  uint64_t written = 0;
  CHECK(rns8_get_plan_tile_schedule(plan, nullptr, 0, &written) == RNS8_INVALID_ARGUMENT);
  rns8_workspace* workspace = nullptr;
  CHECK(rns8_create_workspace(ctx, plan, &workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(workspace == nullptr);
  plan->schedule_prefix_group_count = 1;

  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  const uint64_t a = 3;
  const uint64_t b = 5;
  uint64_t c = 0;
  REQUIRE(rns8_pack_u64(ctx, a_matrix, &a, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, &b, 1, 1) == RNS8_SUCCESS);

  const uint32_t saved_min_selected = plan->schedule_min_selected_prefix;
  plan->schedule_min_selected_prefix = 0;
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, &c, 1) == RNS8_INVALID_ARGUMENT);
  plan->schedule_min_selected_prefix = saved_min_selected;

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded export rejects output matrices outside the plan contract") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 1;
  desc.bound = 100000;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  uint64_t dst[4] = {};
  rns8_matrix* c_matrix = nullptr;
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, dst, 2) == RNS8_SUCCESS);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 1);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_prefix, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_bound_kind = nullptr;
  auto wrong_c_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_bound_kind_desc, &wrong_c_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_bound_kind, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_c_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("matrix creation rejects descriptor sizes that overflow owned storage") {
  {
    rns8_context* ctx = create_cpu();
    auto desc = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }

  {
    rns8_context* ctx = create_wrap64();
    auto desc = matrix_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = 0;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }
}
