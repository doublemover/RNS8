TEST_CASE("exact-wide axis summaries admit a cancellation-safe wide chain", "[exact-axis]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_SIGNED, 2);
  const int64_t high = INT64_MAX;
  const int64_t a[] = {high, high, high, high};
  const int64_t b[] = {high, -high, -high, high};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 2, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], b, 2, 1) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_max_magnitude == 0);
  REQUIRE(f.multiply(2, 0, 3) == RNS8_SUCCESS);
  std::vector<uint64_t> limbs(12, 999);
  REQUIRE(rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[3], limbs.data(), 2, 3) ==
          RNS8_SUCCESS);
  CHECK(limbs == std::vector<uint64_t>(12, 0));

  const int64_t positive[] = {high, high, high, high};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], positive, 2, 2) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_max_magnitude == boost::multiprecision::cpp_int(2) * high * high);
  const auto before = f.matrices[3]->residues;
  const auto version = f.matrices[3]->source_version;
  const auto row_sum = f.matrices[3]->exact_axis_ranges.rows[0].absolute_sum;
  CHECK(f.multiply(2, 0, 3) == RNS8_RANGE_ERROR);
  CHECK(f.matrices[3]->residues == before);
  CHECK(f.matrices[3]->source_version == version);
  CHECK(f.matrices[3]->exact_axis_ranges.rows[0].absolute_sum == row_sum);
}

TEST_CASE("exact-wide axis summaries use wide sums and ignore logical padding", "[exact-axis]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics, 2);
    const int64_t s[] = {INT64_MIN, INT64_MIN, INT64_MAX, 1, -1, INT64_MIN};
    const uint64_t u[] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, 1, 0, UINT64_MAX};
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[0], s, 3, 1)
                 : rns8_pack_u64(f.ctx, f.matrices[0], u, 3, 1)) == RNS8_SUCCESS);
    const auto& axes = f.matrices[0]->exact_axis_ranges;
    REQUIRE(axes.rows.size() == 2);
    REQUIRE(axes.cols.size() == 2);
    const boost::multiprecision::cpp_int maximum = semantics == RNS8_EXACT_WIDE_SIGNED
                                                       ? boost::multiprecision::cpp_int(1) << 63
                                                       : boost::multiprecision::cpp_int(UINT64_MAX);
    CHECK(axes.rows[0].absolute_sum == 2 * maximum);
    CHECK(axes.rows[0].sum == (semantics == RNS8_EXACT_WIDE_SIGNED ? -2 * maximum : 2 * maximum));
    CHECK(axes.rows[1].absolute_sum == (semantics == RNS8_EXACT_WIDE_SIGNED ? 2 : 1));
    CHECK(axes.rows[1].sum == (semantics == RNS8_EXACT_WIDE_SIGNED ? 0 : 1));
    CHECK(axes.rows[0].sum_known);
  }
}

TEST_CASE("exact-wide axis bounds propagate through a heterogeneous unsigned chain", "[exact-axis]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_UNSIGNED, 3);
  const uint64_t a[] = {100, 0, 0, 0, 7, 0, 0, 0, 1};
  const uint64_t identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  REQUIRE(rns8_pack_u64(f.ctx, f.matrices[0], a, 3, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(f.ctx, f.matrices[1], identity, 3, 1) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_max_magnitude == 100);
  for (const auto& axis : f.matrices[2]->exact_axis_ranges.rows) CHECK_FALSE(axis.sum_known);
  REQUIRE(f.multiply(2, 1, 3) == RNS8_SUCCESS);
  CHECK(f.matrices[3]->exact_max_magnitude <= 300);
  std::vector<uint64_t> limbs(18);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(f.ctx, f.plan, f.matrices[3], limbs.data(), 3, 2) ==
          RNS8_SUCCESS);
  for (unsigned cell = 0; cell < 9; ++cell) {
    CHECK(limbs[2 * cell] == a[cell]);
    CHECK(limbs[2 * cell + 1] == 0);
  }
}

TEST_CASE("exact-wide missing axis proofs retain scalar rejection and repacking restores them",
          "[exact-axis]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_SIGNED, 2);
  const int64_t a[] = {INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX};
  const int64_t b[] = {INT64_MAX, -INT64_MAX, -INT64_MAX, INT64_MAX};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 2, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], b, 2, 1) == RNS8_SUCCESS);
  f.matrices[0]->exact_axis_ranges.rows.clear();
  f.matrices[1]->exact_axis_ranges.cols.clear();
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_axis_ranges.rows.empty());
  CHECK(f.matrices[2]->exact_max_magnitude == boost::multiprecision::cpp_int(2) * INT64_MAX * INT64_MAX);
  const auto before = f.matrices[3]->residues;
  CHECK(f.multiply(2, 1, 3) == RNS8_RANGE_ERROR);
  CHECK(f.matrices[3]->residues == before);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 2, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], b, 2, 2) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  REQUIRE(f.multiply(2, 1, 3) == RNS8_SUCCESS);
  CHECK(f.matrices[3]->exact_max_magnitude == 0);
}

TEST_CASE("exact-wide rejected repack preserves axis proof and scalar commits clear stale summaries",
          "[exact-axis]") {
  auto* ctx = create_cpu();
  auto md = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
  md.max_prefix = 1;
  rns8_matrix* matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
  const int64_t seven = 7, outside = 128;
  REQUIRE(rns8_pack_i64(ctx, matrix, &seven, 1, 1) == RNS8_SUCCESS);
  const auto bytes = matrix->residues;
  CHECK(rns8_pack_i64(ctx, matrix, &outside, 1, 2) == RNS8_RANGE_ERROR);
  CHECK(matrix->residues == bytes);
  CHECK(matrix->source_version == 1);
  CHECK(matrix->exact_axis_ranges.rows[0].sum == 7);
  CHECK(matrix->exact_axis_ranges.cols[0].sum == 7);
  rns8_plan plan;
  plan.desc.semantics = md.semantics;
  plan.prefix = 1;
  boost::multiprecision::cpp_int bound = 11;
  rns8::detail::commit_exact_range(*matrix, plan, bound);
  CHECK(matrix->exact_axis_ranges.rows.empty());
  CHECK(matrix->exact_axis_ranges.cols.empty());
  CHECK(matrix->exact_max_magnitude == 11);
  rns8_destroy_matrix(matrix);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide axis bounds cover deterministic signed and unsigned differential products",
          "[exact-axis]") {
  uint64_t seed = 20261008;
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics, 3);
    for (unsigned trial = 0; trial < 64; ++trial) {
      INFO("semantics=" << semantics << " trial=" << trial);
      int64_t a[9], b[9];
      uint64_t au[9], bu[9];
      for (unsigned i = 0; i < 9; ++i) {
        seed = seed * 6364136223846793005ULL + 1;
        au[i] = (seed >> 32) % 23;
        a[i] = static_cast<int64_t>(au[i]) - 11;
        seed = seed * 6364136223846793005ULL + 1;
        bu[i] = (seed >> 32) % 23;
        b[i] = static_cast<int64_t>(bu[i]) - 11;
      }
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_pack_i64(f.ctx, f.matrices[0], a, 3, trial + 1)
                   : rns8_pack_u64(f.ctx, f.matrices[0], au, 3, trial + 1)) == RNS8_SUCCESS);
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_pack_i64(f.ctx, f.matrices[1], b, 3, trial + 1)
                   : rns8_pack_u64(f.ctx, f.matrices[1], bu, 3, trial + 1)) == RNS8_SUCCESS);
      REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
      std::vector<uint64_t> limbs(18);
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[2], limbs.data(), 3, 2)
                   : rns8_export_exact_wide_unsigned_limbs(f.ctx, f.plan, f.matrices[2], limbs.data(), 3,
                                                           2)) == RNS8_SUCCESS);
      for (int64_t row = 0; row < 3; ++row) {
        boost::multiprecision::cpp_int row_sum = 0;
        for (int64_t col = 0; col < 3; ++col) {
          const auto exact = semantics == RNS8_EXACT_WIDE_SIGNED
                                 ? rns8::detail::exact_i64_gemm_cell(a, 3, b, 3, row, col, 3)
                                 : rns8::detail::exact_u64_gemm_cell(au, 3, bu, 3, row, col, 3);
          const boost::multiprecision::cpp_int magnitude = boost::multiprecision::abs(exact);
          CHECK(magnitude <= f.matrices[2]->exact_max_magnitude);
          CHECK(magnitude <= f.matrices[2]->exact_axis_ranges.rows[row].max_magnitude);
          CHECK(magnitude <= f.matrices[2]->exact_axis_ranges.cols[col].max_magnitude);
          row_sum += magnitude;
          const auto expected = semantics == RNS8_EXACT_WIDE_SIGNED ? signed_twos_complement_limbs(exact, 2)
                                                                    : unsigned_limbs(exact, 2);
          CHECK(limbs[2 * (row * 3 + col)] == expected[0]);
          CHECK(limbs[2 * (row * 3 + col) + 1] == expected[1]);
        }
        CHECK(row_sum <= f.matrices[2]->exact_axis_ranges.rows[row].absolute_sum);
      }
    }
  }
}

TEST_CASE("exact-wide rectangular axis admission preserves padded exports", "[exact-axis]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics, 2, 3, 4, 20);
    const int64_t a[] = {1, -2, 3, -4, INT64_MAX, 0, 5, -6, 7, INT64_MIN};
    const int64_t b[] = {2, -3, 4,  INT64_MIN, -5,  6,  -7,  INT64_MAX,
                         8, -9, 10, INT64_MIN, -11, 12, -13, INT64_MAX};
    const uint64_t au[] = {1, 2, 3, 4, UINT64_MAX, 0, 5, 6, 7, UINT64_MAX};
    const uint64_t bu[] = {2, 3, 4,  UINT64_MAX, 5,  6,  7,  UINT64_MAX,
                           8, 9, 10, UINT64_MAX, 11, 12, 13, UINT64_MAX};
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[0], a, 5, 1)
                 : rns8_pack_u64(f.ctx, f.matrices[0], au, 5, 1)) == RNS8_SUCCESS);
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[1], b, 4, 1)
                 : rns8_pack_u64(f.ctx, f.matrices[1], bu, 4, 1)) == RNS8_SUCCESS);
    CHECK(f.matrices[0]->exact_max_magnitude == 7);
    CHECK(f.matrices[1]->exact_max_magnitude == 13);
    REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
    CHECK(f.matrices[2]->exact_axis_ranges.rows.size() == 2);
    CHECK(f.matrices[2]->exact_axis_ranges.cols.size() == 3);
    std::vector<uint64_t> limbs(16, 101);
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[2], limbs.data(), 4, 2)
                 : rns8_export_exact_wide_unsigned_limbs(f.ctx, f.plan, f.matrices[2], limbs.data(), 4, 2)) ==
            RNS8_SUCCESS);
    for (int64_t col = 0; col < 3; ++col) {
      boost::multiprecision::cpp_int column_sum = 0;
      for (int64_t row = 0; row < 2; ++row) {
        const auto value = semantics == RNS8_EXACT_WIDE_SIGNED
                               ? rns8::detail::exact_i64_gemm_cell(a, 5, b, 4, row, col, 4)
                               : rns8::detail::exact_u64_gemm_cell(au, 5, bu, 4, row, col, 4);
        const auto expected = semantics == RNS8_EXACT_WIDE_SIGNED ? signed_twos_complement_limbs(value, 2)
                                                                  : unsigned_limbs(value, 2);
        CHECK(limbs[2 * (row * 4 + col)] == expected[0]);
        CHECK(limbs[2 * (row * 4 + col) + 1] == expected[1]);
        column_sum += boost::multiprecision::abs(value);
      }
      CHECK(column_sum <= f.matrices[2]->exact_axis_ranges.cols[col].absolute_sum);
    }
    CHECK(limbs[6] == 101);
    CHECK(limbs[7] == 101);
    CHECK(limbs[14] == 101);
    CHECK(limbs[15] == 101);
  }
}
