TEST_CASE("bounded semantics reject bound none explicitly") {
  rns8_context* ctx = create_cpu();
  {
    auto desc = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("reserved public ABI flags are hard rejected") {
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    options.flags = 1;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(-1, &options, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIPBLASLT;
    options.flags = 1;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }

  rns8_context* ctx = create_cpu();
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.flags = 0x80000000u;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.flags = 1;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded one-shot APIs validate ABI and leading dimensions before dispatch") {
  rns8_context* ctx = create_cpu();
  int64_t signed_a[4] = {1, 2, 3, 4};
  int64_t signed_b[4] = {5, 6, 7, 8};
  int64_t signed_c[4] = {};
  uint64_t unsigned_a[4] = {1, 2, 3, 4};
  uint64_t unsigned_b[4] = {5, 6, 7, 8};
  uint64_t unsigned_c[4] = {};

  auto signed_desc = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  signed_desc.m = 2;
  signed_desc.n = 2;
  signed_desc.k = 2;
  signed_desc.bound = 100;
  auto invalid_signed_abi = signed_desc;
  invalid_signed_abi.struct_size = 0;
  CHECK(rns8_gemm_i64_oneshot(ctx, &invalid_signed_abi, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 1, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 2, signed_b, 1, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 2, signed_b, 2, signed_c, 1) ==
        RNS8_INVALID_ARGUMENT);
  auto malformed_future_signed = signed_desc;
  malformed_future_signed.requested_backend = RNS8_BACKEND_HIPBLASLT;
  malformed_future_signed.bound_kind = static_cast<rns8_bound_kind>(0x7fffu);
  CHECK(rns8_gemm_i64_oneshot(ctx, &malformed_future_signed, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  auto valid_future_signed = signed_desc;
  valid_future_signed.requested_backend = RNS8_BACKEND_HIPBLASLT;
  CHECK(rns8_gemm_i64_oneshot(ctx, &valid_future_signed, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_UNSUPPORTED_BACKEND);

  auto unsigned_desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  unsigned_desc.m = 2;
  unsigned_desc.n = 2;
  unsigned_desc.k = 2;
  unsigned_desc.bound = 100;
  auto invalid_unsigned_abi = unsigned_desc;
  invalid_unsigned_abi.abi_version = 0;
  CHECK(rns8_gemm_u64_oneshot(ctx, &invalid_unsigned_abi, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 1, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 2, unsigned_b, 1, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 2, unsigned_b, 2, unsigned_c, 1) ==
        RNS8_INVALID_ARGUMENT);
  auto malformed_future_unsigned = unsigned_desc;
  malformed_future_unsigned.requested_backend = RNS8_BACKEND_CK;
  malformed_future_unsigned.bound_kind = static_cast<rns8_bound_kind>(0x7fffu);
  CHECK(rns8_gemm_u64_oneshot(ctx, &malformed_future_unsigned, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  auto valid_future_unsigned = unsigned_desc;
  valid_future_unsigned.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_gemm_u64_oneshot(ctx, &valid_future_unsigned, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_UNSUPPORTED_BACKEND);

  rns8_destroy_context(ctx);
}

TEST_CASE("unsupported semantic contracts do not fall through to bounded CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_WRAP_U64_MOD_2_64}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("public finite ring and field u8 oneshot use explicit modulus contracts") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 4;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 5;
  const std::vector<uint8_t> A = {
      254, 128, 7, 3, 0xaa, 0xaa,
      5, 250, 251, 1, 0xaa, 0xaa,
  };
  const std::vector<uint8_t> B = {
      2, 3, 4, 0xbb, 0xbb,
      250, 11, 9, 0xbb, 0xbb,
      13, 17, 19, 0xbb, 0xbb,
      23, 29, 31, 0xbb, 0xbb,
  };

  {
    auto desc = finite_desc(RNS8_FINITE_RING_U8, m, n, k);
    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_gemm_finite_ring_u8_oneshot(ctx, &desc, 255, A.data(), lda, B.data(), ldb, C.data(), ldc) ==
            RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == exact_finite_cell(A, B, lda, ldb, k, row, col, 255));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xcc);
    }
  }

  {
    auto desc = finite_desc(RNS8_FINITE_FIELD_U8, m, n, k);
    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xdd);
    REQUIRE(rns8_gemm_finite_field_u8_oneshot(ctx, &desc, 251, A.data(), lda, B.data(), ldb, C.data(), ldc) ==
            RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == exact_finite_cell(A, B, lda, ldb, k, row, col, 251));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xdd);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public finite ring and field u8 persistent matrices use explicit modulus contracts") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 4;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 5;
  const std::vector<uint8_t> A = {
      254, 128, 7, 3, 0xaa, 0xaa,
      5, 250, 251, 1, 0xaa, 0xaa,
  };
  const std::vector<uint8_t> B = {
      2, 3, 4, 0xbb, 0xbb,
      250, 11, 9, 0xbb, 0xbb,
      13, 17, 19, 0xbb, 0xbb,
      23, 29, 31, 0xbb, 0xbb,
  };

  for (const auto item : {std::pair{RNS8_FINITE_RING_U8, uint16_t{255}},
                          std::pair{RNS8_FINITE_FIELD_U8, uint16_t{251}}}) {
    auto desc = finite_desc(item.first, m, n, k);
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
    CHECK(info.min_required_prefix == 0);
    CHECK(info.max_selected_prefix == 0);
    CHECK(info.prefix_group_count == 0);

    auto a_desc = matrix_desc(m, k, item.first, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, item.first, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, item.first, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(ctx, a_matrix, item.second, A.data(), lda, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(ctx, b_matrix, item.second, B.data(), ldb, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(ctx, plan, item.second, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_export_finite_u8(ctx, plan, item.second, c_matrix, C.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
              exact_finite_cell(A, B, lda, ldb, k, row, col, item.second));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xcc);
    }

    std::vector<uint64_t> wrong_export(static_cast<std::size_t>(m * ldc), 0);
    CHECK(rns8_export_u64(ctx, plan, c_matrix, wrong_export.data(), ldc) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("finite u8 oneshot descriptors reject stale CRT metadata and invalid fields") {
  rns8_context* ctx = create_cpu();
  const uint8_t A[1] = {3};
  const uint8_t B[1] = {5};
  uint8_t C[1] = {0};

  auto ring = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 1, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 251, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 257, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto field = finite_desc(RNS8_FINITE_FIELD_U8, 1, 1, 1);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 241, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 251, A, 0, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto stale_prefix = ring;
  stale_prefix.max_prefix = 1;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &stale_prefix, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto stale_bound = ring;
  stale_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  stale_bound.bound = 15;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &stale_bound, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &ring, 251, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  ring.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 255, A, 1, B, 1, C, 1) == RNS8_UNSUPPORTED_BACKEND);

  rns8_destroy_context(ctx);
}

TEST_CASE("finite u8 persistent descriptors reject stale CRT metadata and modulus mismatches") {
  rns8_context* ctx = create_cpu();
  const uint8_t A[1] = {3};
  const uint8_t B[1] = {5};
  uint8_t C[1] = {0};

  auto stale_prefix = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  stale_prefix.max_prefix = 1;
  rns8_plan* rejected_plan = nullptr;
  CHECK(rns8_create_plan(ctx, &stale_prefix, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto missing_modulus = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  missing_modulus.finite_modulus = 0;
  CHECK(rns8_create_plan(ctx, &missing_modulus, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto stale_nonfinite_modulus = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  stale_nonfinite_modulus.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  stale_nonfinite_modulus.finite_modulus = 251;
  CHECK(rns8_create_plan(ctx, &stale_nonfinite_modulus, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto stale_matrix_desc = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  stale_matrix_desc.max_prefix = 1;
  rns8_matrix* rejected_matrix = nullptr;
  CHECK(rns8_create_matrix(ctx, &stale_matrix_desc, &rejected_matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_matrix == nullptr);

  auto accelerator = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  accelerator.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_create_plan(ctx, &accelerator, &rejected_plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(rejected_plan == nullptr);

  auto desc = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto matrix = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &c_matrix) == RNS8_SUCCESS);

  CHECK(rns8_export_finite_u8(ctx, plan, 255, c_matrix, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_pack_finite_u8(ctx, a_matrix, 1, A, 1, 1) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_pack_finite_u8(ctx, a_matrix, 255, A, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(ctx, b_matrix, 255, B, 1, 1) == RNS8_SUCCESS);
  CHECK(rns8_gemm_finite_u8(ctx, plan, 251, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_gemm_finite_u8(ctx, plan, 255, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(rns8_export_finite_u8(ctx, plan, 251, c_matrix, C, 1) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_export_finite_u8(ctx, plan, 255, c_matrix, C, 1) == RNS8_SUCCESS);
  CHECK(C[0] == 15);

  auto field_desc = finite_desc(RNS8_FINITE_FIELD_U8, 1, 1, 1);
  rns8_plan* field_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &field_desc, &field_plan) == RNS8_SUCCESS);
  auto field_matrix_desc = matrix_desc(1, 1, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE);
  rns8_matrix* field_matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &field_matrix_desc, &field_matrix) == RNS8_SUCCESS);
  CHECK(rns8_pack_finite_u8(ctx, field_matrix, 255, A, 1, 1) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(field_matrix);
  rns8_destroy_plan(field_plan);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("unknown public enum values are invalid before backend routing") {
  rns8_context* ctx = create_cpu();
  const auto unknown_semantics = invalid_public_enum_value<rns8_semantics>();
  const auto unknown_bound_kind = invalid_public_enum_value<rns8_bound_kind>();
  const auto unknown_layout = invalid_public_enum_value<rns8_layout>();
  const auto unknown_backend = invalid_public_enum_value<rns8_backend_kind>();

  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, unknown_bound_kind);
    desc.requested_backend = RNS8_BACKEND_HIPBLASLT;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(unknown_semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    desc.requested_backend = RNS8_BACKEND_CK;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.requested_backend = unknown_backend;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, unknown_bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto matrix = matrix_desc(unknown_semantics, RNS8_BOUND_NONE);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.logical_layout = unknown_layout;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_FINITE_RING_U8, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.requested_backend = RNS8_BACKEND_ROCWMMA;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.logical_layout = RNS8_LAYOUT_COLUMN_MAJOR;
    matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = unknown_backend;
    rns8_context* unknown_ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &unknown_ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(unknown_ctx == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("known but unimplemented descriptor contracts report unsupported status") {
  rns8_context* ctx = create_cpu();
  {
    auto column_major = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    column_major.logical_layout = RNS8_LAYOUT_COLUMN_MAJOR;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &column_major, &storage) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

