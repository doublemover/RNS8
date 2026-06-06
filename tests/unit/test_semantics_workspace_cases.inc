TEST_CASE("tile dimensions are powers of two from 64 to 512") {
  rns8_context* ctx = create_cpu();
  auto valid = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  valid.tile_m = 256;
  valid.tile_n = 512;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &valid, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);

  for (const uint32_t bad_tile : {32u, 96u, 513u}) {
    auto bad_plan = valid;
    bad_plan.tile_m = bad_tile;
    plan = nullptr;
    CHECK(rns8_create_plan(ctx, &bad_plan, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto bad_matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    bad_matrix.tile_n = bad_tile;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &bad_matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects incompatible matrix and workspace contracts") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 100000;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  auto smaller_desc = desc;
  smaller_desc.k = 1;
  rns8_plan* smaller_plan = nullptr;
  rns8_workspace* wrong_workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &smaller_desc, &smaller_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, smaller_plan, &wrong_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, wrong_workspace) == RNS8_WORKSPACE_TOO_SMALL);

  rns8_matrix* wrong_b_shape = nullptr;
  auto wrong_b_shape_desc = matrix_desc(3, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_b_shape_desc, &wrong_b_shape) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, wrong_b_shape, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 1);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, wrong_c_prefix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_a_bound_kind = nullptr;
  auto wrong_a_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_a_bound_kind_desc, &wrong_a_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, wrong_a_bound_kind, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_a_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(wrong_b_shape);
  rns8_destroy_workspace(wrong_workspace);
  rns8_destroy_plan(smaller_plan);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded RNS GEMM stamps output source versions from packed inputs") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 1000;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_workspace* wrong_workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto smaller_desc = desc;
  smaller_desc.k = 1;
  rns8_plan* smaller_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &smaller_desc, &smaller_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, smaller_plan, &wrong_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  const uint64_t A0[] = {1, 2, 3, 4};
  const uint64_t B0[] = {5, 6, 7, 8};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A0, 2, 17) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B0, 2, 23) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);

  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(c_matrix->source_version == first_output_version);

  const uint64_t B1[] = {8, 7, 6, 5};
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B1, 2, 24) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  const uint64_t second_output_version = c_matrix->source_version;
  CHECK(second_output_version != 0);
  CHECK(second_output_version != first_output_version);

  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, wrong_workspace) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c_matrix->source_version == second_output_version);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(wrong_workspace);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(smaller_plan);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects same-shape workspaces from different semantic contracts") {
  rns8_context* ctx = create_cpu();

  auto bounded = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  bounded.m = 2;
  bounded.n = 2;
  bounded.k = 1;
  bounded.bound = 100;
  bounded.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.m = bounded.m;
  exact.n = bounded.n;
  exact.k = bounded.k;
  exact.bound = 0;
  exact.max_prefix = bounded.max_prefix;

  const uint64_t tile_bound = 100;
  auto per_tile = bounded;
  per_tile.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  per_tile.bound = 0;
  per_tile.tile_m = 64;
  per_tile.tile_n = 64;
  per_tile.tile_bounds = &tile_bound;
  per_tile.tile_bounds_count = 1;

  rns8_plan* bounded_plan = nullptr;
  rns8_plan* exact_plan = nullptr;
  rns8_plan* per_tile_plan = nullptr;
  rns8_workspace* bounded_workspace = nullptr;
  rns8_workspace* exact_workspace = nullptr;
  rns8_workspace* per_tile_workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  REQUIRE(rns8_create_plan(ctx, &bounded, &bounded_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &exact, &exact_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &per_tile, &per_tile_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, exact_plan, &exact_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, per_tile_plan, &per_tile_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(2, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto b_desc = matrix_desc(1, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, bounded_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, exact_workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, per_tile_workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(per_tile_workspace);
  rns8_destroy_workspace(exact_workspace);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(per_tile_plan);
  rns8_destroy_plan(exact_plan);
  rns8_destroy_plan(bounded_plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded CPU matrices preserve source versions and currentness across reuse") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 1000;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  const uint64_t A0[] = {1, 2, 3, 4};
  const uint64_t B0[] = {5, 6, 7, 8};
  uint64_t C[4] = {};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A0, 2, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B0, 2, 202) == RNS8_SUCCESS);
  CHECK(a_matrix->source_version == 101);
  CHECK(b_matrix->source_version == 202);
  CHECK(a_matrix->host_residues_current);
  CHECK_FALSE(a_matrix->device_residues_current);
  CHECK(b_matrix->host_residues_current);
  CHECK_FALSE(b_matrix->device_residues_current);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == 19);
  CHECK(C[1] == 22);
  CHECK(C[2] == 43);
  CHECK(C[3] == 50);
  CHECK(a_matrix->source_version == 101);
  CHECK(b_matrix->source_version == 202);
  CHECK(c_matrix->host_residues_current);
  CHECK_FALSE(c_matrix->device_residues_current);

  const uint64_t A1[] = {9, 1, 2, 3};
  const uint64_t B1[] = {4, 5, 6, 7};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A1, 2, 303) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B1, 2, 404) == RNS8_SUCCESS);
  CHECK(a_matrix->source_version == 303);
  CHECK(b_matrix->source_version == 404);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == 42);
  CHECK(C[1] == 52);
  CHECK(C[2] == 26);
  CHECK(C[3] == 31);
  rns8::test::set_host_residues_current(*c_matrix, false);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

