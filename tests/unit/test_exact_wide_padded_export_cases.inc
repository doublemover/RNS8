TEST_CASE("exact-wide unsigned CPU limb export preserves padded destination cells") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 1;
  const int64_t ld = 4;
  const uint64_t sentinel = 0xf00df00df00df00dull;
  const uint64_t A[] = {17, 19};
  const uint64_t B[] = {23, 29};

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

  constexpr uint32_t limb_count = 2;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_u64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = unsigned_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
    for (int64_t col = n; col < ld; ++col) {
      const std::size_t padding = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(limbs[padding] == sentinel);
      CHECK(limbs[padding + 1] == sentinel);
    }
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU max-width public limb export preserves padded element stride") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;
  constexpr int64_t ld = 3;
  constexpr uint32_t limb_count = 32;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(-1),
        -(boost::multiprecision::cpp_int(1) << 63u),
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1),
    };
    fill_exact_residue_matrix(c_matrix, values);

    constexpr uint64_t sentinel = 0xa5a5a5a5a5a5a5a5ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto expected = signed_twos_complement_limbs(values[static_cast<std::size_t>(row * n + col)], limb_count);
        const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
        CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
      }
      const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(limbs[padding + limb] == sentinel);
      }
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1) << 63u,
        boost::multiprecision::cpp_int(1) << 127u,
        boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()),
    };
    fill_exact_residue_matrix(c_matrix, values);

    constexpr uint64_t sentinel = 0x5a5a5a5a5a5a5a5aull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto expected = unsigned_limbs(values[static_cast<std::size_t>(row * n + col)], limb_count);
        const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
        CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
      }
      const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(limbs[padding + limb] == sentinel);
      }
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

