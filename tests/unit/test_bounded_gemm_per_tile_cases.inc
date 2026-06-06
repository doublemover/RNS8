TEST_CASE("bounded CPU export reports range errors for too-small valid per-tile bounds") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int64_t ldc = 67;
  {
    const std::vector<uint64_t> bounds = {5, 10, 10, 10};
    std::vector<int64_t> A(m * k, 2);
    std::vector<int64_t> B(k * n, 3);
    std::vector<int64_t> C(m * ldc, -99);
    auto desc = i64_desc(m, n, k, 0);
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
    desc.tile_m = 64;
    desc.tile_n = 64;
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = bounds.size();
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), ldc) == RNS8_RANGE_ERROR);
    check_all_equal<int64_t>(C, -99);
  }
  {
    const std::vector<uint64_t> bounds = {5, 10, 10, 10};
    std::vector<uint64_t> A(m * k, 2);
    std::vector<uint64_t> B(k * n, 3);
    std::vector<uint64_t> C(m * ldc, 123);
    auto desc = u64_desc(m, n, k, 0);
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
    desc.tile_m = 64;
    desc.tile_n = 64;
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = bounds.size();
    CHECK(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), ldc) == RNS8_RANGE_ERROR);
    check_all_equal<uint64_t>(C, 123);
  }
  {
    const std::vector<uint64_t> bounds = {0, 10, 10, 10};
    std::vector<uint64_t> A(m * k, 2);
    std::vector<uint64_t> B(k * n, 3);
    std::vector<uint64_t> C(m * ldc, 123);
    auto desc = u64_desc(m, n, k, 0);
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
    desc.tile_m = 64;
    desc.tile_n = 64;
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = bounds.size();
    CHECK(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), ldc) == RNS8_RANGE_ERROR);
    check_all_equal<uint64_t>(C, 123);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU oneshot materializes proven zero per-tile outputs") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int64_t ldc = n + 1;
  std::vector<uint64_t> A(m * k);
  std::vector<uint64_t> B(k * n);
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
  for (int64_t row = 0; row < m; ++row) {
    A[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[static_cast<std::size_t>(col)] = col < 64 ? 0 : 7;
  }

  const std::vector<uint64_t> bounds = {0, 7, 0, 7000};
  auto desc = u64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] == A[static_cast<std::size_t>(row)] * B[static_cast<std::size_t>(col)]);
    }
    CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xdeadbeefdeadbeefull);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU oneshot executes and exports per-tile unsigned prefixes") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  std::vector<uint64_t> A(m * k);
  std::vector<uint64_t> B(k * n);
  std::vector<uint64_t> C(m * n, 0);
  for (int64_t row = 0; row < m; ++row) {
    A[row] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[col] = col < 64 ? 7 : 1000;
  }

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto desc = u64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[row * n + col] == A[row] * B[col]);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU oneshot executes and exports per-tile signed prefixes") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  std::vector<int64_t> A(m * k);
  std::vector<int64_t> B(k * n);
  std::vector<int64_t> C(m * n, 0);
  for (int64_t row = 0; row < m; ++row) {
    A[row] = row < 64 ? -2 : -1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[col] = col < 64 ? 3 : -1000;
  }

  const std::vector<uint64_t> bounds = {6, 2000, 3000000, 1000000000};
  auto desc = i64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[row * n + col] == A[row] * B[col]);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU persistent GEMM accepts compact per-tile signed storage") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  std::vector<int64_t> A(m * k);
  std::vector<int64_t> B(k * n);
  std::vector<int64_t> C(m * n, 0);
  for (int64_t row = 0; row < m; ++row) {
    A[row] = row < 64 ? -2 : -1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[col] = col < 64 ? 3 : -1000;
  }

  const std::vector<uint64_t> bounds = {6, 2000, 3000000, 1000000000};
  auto desc = i64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  REQUIRE(info.max_selected_prefix > 0);
  REQUIRE(info.max_selected_prefix < plan->prefix);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_PER_TILE_MAX_ABS, info.max_selected_prefix);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_PER_TILE_MAX_ABS, info.max_selected_prefix);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_PER_TILE_MAX_ABS, info.max_selected_prefix);
  a_desc.tile_m = desc.tile_m;
  a_desc.tile_n = desc.tile_n;
  b_desc.tile_m = desc.tile_m;
  b_desc.tile_n = desc.tile_n;
  c_desc.tile_m = desc.tile_m;
  c_desc.tile_n = desc.tile_n;

  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A.data(), k, 1000000) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B.data(), n, 1000) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(ctx, plan, c_matrix, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[row * n + col] == A[row] * B[col]);
    }
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded per-tile bound contracts reject missing or inconsistent bounds") {
  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(65, 65, 1, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> too_few = {7, 1000, 7000000};
  desc.tile_bounds = too_few.data();
  desc.tile_bounds_count = too_few.size();
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  desc.bound = 1;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto signed_desc = i64_desc(1, 1, 1, 0);
  signed_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  const std::vector<uint64_t> too_large = {std::numeric_limits<uint64_t>::max()};
  signed_desc.tile_bounds = too_large.data();
  signed_desc.tile_bounds_count = too_large.size();
  CHECK(rns8_create_plan(ctx, &signed_desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  rns8_destroy_context(ctx);
}

TEST_CASE("wrap64 plan schedule reports non-RNS byte-limb scheduling") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = 65;
  desc.n = 64;
  desc.k = 3;
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_rows == 2);
  CHECK(info.tile_cols == 1);
  CHECK(info.tile_count == 2);
  CHECK(info.min_required_prefix == 0);
  CHECK(info.max_required_prefix == 0);
  CHECK(info.min_selected_prefix == 0);
  CHECK(info.max_selected_prefix == 0);
  CHECK(info.prefix_group_count == 0);
  CHECK(info.range_bit_length == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded i64 contract rejects magnitudes beyond int64 min") {
  rns8_context* ctx = create_cpu();
  auto desc = i64_desc(1, 1, 1, (1ull << 63u) + 1u);
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}
