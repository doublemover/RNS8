TEST_CASE("exact-wide signed CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 2;
  const int64_t A[] = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 17,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 31};
  const int64_t B[] = {
      std::numeric_limits<int64_t>::max() - 5,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 11,
      std::numeric_limits<int64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  c_desc.max_prefix = plan->prefix;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_signed_residues(*c_matrix, A, k, B, n, m, n, k);
  constexpr uint32_t limb_count = 3;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_i64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = signed_twos_complement_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * n + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
  }
  constexpr uint64_t range_sentinel = 0x123456789abcdef0ull;
  std::vector<uint64_t> too_few(static_cast<std::size_t>(m * n), range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, too_few.data(), n, 1) == RNS8_RANGE_ERROR);
  for (const uint64_t limb : too_few) {
    CHECK(limb == range_sentinel);
  }
  CHECK(rns8_export_i64(ctx, plan, c_matrix, reinterpret_cast<int64_t*>(limbs.data()), n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide signed CPU export rejects bounded unsigned and wrap interpretations") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = 1;
  const int64_t A[] = {-7};
  const int64_t B[] = {11};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  uint64_t limbs[2] = {0xddddddddddddddddull, 0xeeeeeeeeeeeeeeeeull};
  CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(limbs[0] == 0xddddddddddddddddull);
  CHECK(limbs[1] == 0xeeeeeeeeeeeeeeeeull);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide signed CPU limb export preserves padded destination cells") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 1;
  const int64_t ld = 3;
  const uint64_t sentinel = 0xdecafbaddecafbadull;
  const int64_t A[] = {5, -7};
  const int64_t B[] = {11, -13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  constexpr uint32_t limb_count = 2;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_i64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = signed_twos_complement_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
    const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
    CHECK(limbs[padding] == sentinel);
    CHECK(limbs[padding + 1] == sentinel);
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 3};
  const uint64_t B[] = {
      std::numeric_limits<uint64_t>::max() - 5,
      std::numeric_limits<uint64_t>::max() - 7,
      std::numeric_limits<uint64_t>::max() - 11,
      std::numeric_limits<uint64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  c_desc.max_prefix = plan->prefix;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_unsigned_residues(*c_matrix, A, k, B, n, m, n, k);
  constexpr uint32_t limb_count = 3;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_u64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = unsigned_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * n + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
  }
  constexpr uint64_t range_sentinel = 0x0fedcba987654321ull;
  std::vector<uint64_t> too_few(static_cast<std::size_t>(m * n), range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, too_few.data(), n, 1) == RNS8_RANGE_ERROR);
  for (const uint64_t limb : too_few) {
    CHECK(limb == range_sentinel);
  }
  CHECK(rns8_export_u64(ctx, plan, c_matrix, limbs.data(), n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU export rejects signed bounded and wrap interpretations") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = 1;
  const uint64_t A[] = {17};
  const uint64_t B[] = {19};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  uint64_t limbs[2] = {0xccccccccccccccccull, 0xbbbbbbbbbbbbbbbbull};
  CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(limbs[0] == 0xccccccccccccccccull);
  CHECK(limbs[1] == 0xbbbbbbbbbbbbbbbbull);
  CHECK(rns8_export_i64(ctx, plan, c_matrix, reinterpret_cast<int64_t*>(limbs), n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

