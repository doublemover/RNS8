TEST_CASE("exact-wide CPU range errors preserve every destination cell") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(1) << 63u});

    constexpr uint64_t sentinel = 0x3131313131313131ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n), sentinel);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 1) == RNS8_RANGE_ERROR);
    CHECK(limbs == std::vector<uint64_t>(static_cast<std::size_t>(m * n), sentinel));

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
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(1) << 64u});

    constexpr uint64_t sentinel = 0x4141414141414141ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n), sentinel);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 1) == RNS8_RANGE_ERROR);
    CHECK(limbs == std::vector<uint64_t>(static_cast<std::size_t>(m * n), sentinel));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU export rejects stale or non-RNS matrix state") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(-1)});

    uint64_t limbs[2] = {0x5151515151515151ull, 0x5252525252525252ull};
    rns8::test::set_host_residues_current(*c_matrix, false);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x5151515151515151ull);
    CHECK(limbs[1] == 0x5252525252525252ull);
    rns8::test::set_host_residues_current(*c_matrix, true);

    c_matrix->byte_limbs.push_back(0);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x5151515151515151ull);
    CHECK(limbs[1] == 0x5252525252525252ull);
    c_matrix->byte_limbs.clear();

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
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(1)});

    uint64_t limbs[2] = {0x6161616161616161ull, 0x6262626262626262ull};
    rns8::test::set_host_residues_current(*c_matrix, false);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x6161616161616161ull);
    CHECK(limbs[1] == 0x6262626262626262ull);
    rns8::test::set_host_residues_current(*c_matrix, true);

    rns8::test::set_host_byte_limbs_current(*c_matrix, true);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x6161616161616161ull);
    CHECK(limbs[1] == 0x6262626262626262ull);
    rns8::test::set_host_byte_limbs_current(*c_matrix, false);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU descriptors reject bounded metadata") {
  rns8_context* ctx = create_cpu();

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    rns8_matrix* out = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &out) == RNS8_INVALID_ARGUMENT);
    CHECK(out == nullptr);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    rns8_matrix* out = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &out) == RNS8_INVALID_ARGUMENT);
    CHECK(out == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU export rejects invalid fixed-width ABI parameters") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 1;
  constexpr uint64_t sentinel = 0x4d4d4d4d4d4d4d4dull;
  std::vector<uint64_t> limbs(66, sentinel);

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n - 1, 2) == RNS8_INVALID_ARGUMENT);
    for (const uint64_t limb : limbs) {
      CHECK(limb == sentinel);
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

    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n - 1, 2) ==
          RNS8_INVALID_ARGUMENT);
    for (const uint64_t limb : limbs) {
      CHECK(limb == sentinel);
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}
