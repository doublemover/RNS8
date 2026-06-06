TEST_CASE("bounded i64 oneshot handles signs and cancellation") {
  rns8_context* ctx = create_cpu();
  const int64_t A[] = {7, -3, 5, -11, 13, 17};
  const int64_t B[] = {2, -5, 19, 23, -29, 31};
  int64_t C[4] = {};
  auto desc = i64_desc(2, 2, 3, 100000);
  CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 3, B, 2, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == -188);
  CHECK(C[1] == 51);
  CHECK(C[2] == -268);
  CHECK(C[3] == 881);

  const int64_t A_cancel[] = {123456789, 123456789};
  const int64_t B_cancel[] = {1, -1};
  int64_t C_cancel[] = {99};
  auto cancel_desc = i64_desc(1, 1, 2, 1);
  CHECK(rns8_gemm_i64_oneshot(ctx, &cancel_desc, A_cancel, 2, B_cancel, 1, C_cancel, 1) == RNS8_SUCCESS);
  CHECK(C_cancel[0] == 0);

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded i64 and u64 oneshot handle full boundary outputs") {
  rns8_context* ctx = create_cpu();
  {
    const int64_t A[] = {std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<int64_t>::max());
  }
  {
    const int64_t A[] = {-std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == -std::numeric_limits<int64_t>::max());
  }
  {
    const int64_t A[] = {std::numeric_limits<int64_t>::min()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, 1ull << 63u);
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<int64_t>::min());
  }
  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t C[] = {0};
    auto desc = u64_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    CHECK(rns8_gemm_u64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<uint64_t>::max());
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU full-width outputs preserve padded leading dimensions") {
  rns8_context* ctx = create_cpu();
  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 2;
    constexpr int64_t lda = 3;
    constexpr int64_t ldb = 3;
    constexpr int64_t ldc = 4;
    constexpr int64_t sentinel = INT64_C(0x1020304050607);
    std::vector<int64_t> A(static_cast<std::size_t>(m * lda), sentinel);
    std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), -sentinel);
    std::vector<int64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
    A[0] = std::numeric_limits<int64_t>::min();
    A[1] = 0;
    A[lda] = std::numeric_limits<int64_t>::max();
    A[lda + 1] = -std::numeric_limits<int64_t>::max();
    B[0] = 1;
    B[1] = 0;
    B[ldb] = 0;
    B[ldb + 1] = 1;

    auto desc = i64_desc(m, n, k, uint64_t{1} << 63u);
    REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), ldc) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<int64_t>::min());
    CHECK(C[1] == 0);
    CHECK(C[2] == sentinel);
    CHECK(C[3] == sentinel);
    CHECK(C[ldc] == std::numeric_limits<int64_t>::max());
    CHECK(C[ldc + 1] == -std::numeric_limits<int64_t>::max());
    CHECK(C[ldc + 2] == sentinel);
    CHECK(C[ldc + 3] == sentinel);
  }
  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 2;
    constexpr int64_t lda = 3;
    constexpr int64_t ldb = 3;
    constexpr int64_t ldc = 4;
    constexpr uint64_t sentinel = 0xaaaaaaaaaaaaaaaaull;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
    std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
    A[0] = std::numeric_limits<uint64_t>::max();
    A[1] = std::numeric_limits<uint64_t>::max() - 1u;
    A[lda] = 0;
    A[lda + 1] = 1;
    B[0] = 1;
    B[1] = 0;
    B[ldb] = 0;
    B[ldb + 1] = 1;

    auto desc = u64_desc(m, n, k, std::numeric_limits<uint64_t>::max());
    REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), ldc) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<uint64_t>::max());
    CHECK(C[1] == std::numeric_limits<uint64_t>::max() - 1u);
    CHECK(C[2] == sentinel);
    CHECK(C[3] == sentinel);
    CHECK(C[ldc] == 0);
    CHECK(C[ldc + 1] == 1);
    CHECK(C[ldc + 2] == sentinel);
    CHECK(C[ldc + 3] == sentinel);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded defaults select the minimum proven global prefix") {
  CHECK(RNS8_DEFAULT_BOUNDED_PREFIX == 9u);

  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(2, 2, 3, 1000);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.min_required_prefix == 2u);
  CHECK(info.max_required_prefix == 2u);
  CHECK(info.min_selected_prefix == 2u);
  CHECK(info.max_selected_prefix == 2u);
  CHECK(info.adaptive_skip_active == 1u);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded fixed-prefix flag preserves the requested global prefix") {
  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(2, 2, 3, 1000);
  desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.min_required_prefix == 2u);
  CHECK(info.max_required_prefix == 2u);
  CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.adaptive_skip_active == 0u);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded input-range descriptors derive output bounds and run persistent CPU GEMM") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 3;
  auto desc = i64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
  desc.lhs_bound = 7;
  desc.rhs_bound = 11;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  CHECK(plan->desc.bound == static_cast<uint64_t>(k) * desc.lhs_bound * desc.rhs_bound);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K);
  CHECK(info.effective_bound == static_cast<uint64_t>(k) * desc.lhs_bound * desc.rhs_bound);
  CHECK(info.lhs_bound == desc.lhs_bound);
  CHECK(info.rhs_bound == desc.rhs_bound);
  CHECK(std::strcmp(info.bound_contract, "input_range_and_k_derived_output_bound") == 0);
  CHECK(info.min_required_prefix == rns8::detail::required_prefix_for_range(
                                      rns8::detail::cpp_int(2) *
                                      rns8::detail::cpp_int(info.effective_bound)));
  CHECK(info.max_selected_prefix == info.min_required_prefix);

  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, plan->prefix);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, plan->prefix);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, plan->prefix);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  const int64_t A[] = {7, -2, 3, -1, 4, -5};
  const int64_t B[] = {2, -3, 6, 1, -4, 5};
  int64_t C[4] = {};
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 10) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(ctx, plan, c_matrix, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == -10);
  CHECK(C[1] == -8);
  CHECK(C[2] == 42);
  CHECK(C[3] == -18);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded input-range descriptors run through one-shot APIs") {
  rns8_context* ctx = create_cpu();
  {
    auto desc = i64_desc(1, 2, 2, 0);
    desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
    desc.lhs_bound = 5;
    desc.rhs_bound = 6;
    const int64_t A[] = {5, -4};
    const int64_t B[] = {6, -3, 2, 1};
    int64_t C[] = {0, 0};
    REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A, 2, B, 2, C, 2) == RNS8_SUCCESS);
    CHECK(C[0] == 22);
    CHECK(C[1] == -19);
  }
  {
    auto desc = u64_desc(2, 1, 2, 0);
    desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
    desc.lhs_bound = 9;
    desc.rhs_bound = 8;
    const uint64_t A[] = {9, 1, 2, 3};
    const uint64_t B[] = {8, 7};
    uint64_t C[] = {0, 0};
    REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A, 2, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == 79);
    CHECK(C[1] == 37);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded input-range descriptors reject stale fields and overflowing products") {
  rns8_context* ctx = create_cpu();
  {
    auto desc = u64_desc(1, 1, 4, 0);
    desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
    desc.bound = 1;
    desc.lhs_bound = 1;
    desc.rhs_bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = u64_desc(1, 1, 4, 0);
    desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
    desc.lhs_bound = std::numeric_limits<uint64_t>::max();
    desc.rhs_bound = 2;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
    CHECK(plan == nullptr);
  }
  {
    auto desc = i64_desc(1, 1, 4, 0);
    desc.bound_kind = RNS8_BOUND_INPUT_RANGE_AND_K;
    desc.lhs_bound = uint64_t{1} << 63u;
    desc.rhs_bound = 2;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
    CHECK(plan == nullptr);
  }
  {
    auto desc = u64_desc(1, 1, 1, 1);
    desc.lhs_bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_FINITE_RING_U8;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 1;
    desc.n = 1;
    desc.k = 1;
    desc.finite_modulus = 255;
    desc.lhs_bound = 1;
    const uint8_t A[] = {3};
    const uint8_t B[] = {5};
    uint8_t C[] = {0};
    CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &desc, 255, A, 1, B, 1, C, 1) ==
          RNS8_INVALID_ARGUMENT);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU export reports range errors for too-small valid global bounds") {
  rns8_context* ctx = create_cpu();
  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    constexpr int64_t ldb = 3;
    constexpr int64_t ldc = 4;
    constexpr int64_t sentinel = INT64_C(-0x123456789);
    const int64_t A[] = {6, -6};
    const int64_t B[] = {7, -7, 999};
    std::vector<int64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
    auto desc = i64_desc(m, n, k, 41);
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, k, B, ldb, C.data(), ldc) == RNS8_RANGE_ERROR);
    check_all_equal(C, sentinel);
  }
  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    constexpr int64_t ldb = 3;
    constexpr int64_t ldc = 4;
    constexpr uint64_t sentinel = 0xbadc0ffee0ddf00dull;
    const uint64_t A[] = {6, 8};
    const uint64_t B[] = {7, 1, 999};
    std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
    auto desc = u64_desc(m, n, k, 41);
    CHECK(rns8_gemm_u64_oneshot(ctx, &desc, A, k, B, ldb, C.data(), ldc) == RNS8_RANGE_ERROR);
    check_all_equal(C, sentinel);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded plan creation rejects insufficient prefix range") {
  rns8_context* ctx = create_cpu();
  auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  desc.max_prefix = 8;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded prefix-9 is the first full-width i64 and u64 contract") {
  CHECK(RNS8_DEFAULT_BOUNDED_PREFIX == 9u);
  rns8_context* ctx = create_cpu();

  {
    auto desc = i64_desc(1, 1, 1, uint64_t{1} << 63u);
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX - 1u;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
    CHECK(plan == nullptr);

    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_schedule_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.min_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.max_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = u64_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX - 1u;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
    CHECK(plan == nullptr);

    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_schedule_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.min_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.max_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

