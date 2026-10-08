namespace {

void check_lift_metadata(const rns8_matrix& matrix, const rns8_matrix& before) {
  CHECK(matrix.prefix == before.prefix);
  CHECK(matrix.desc.max_prefix == before.desc.max_prefix);
  CHECK(matrix.matrix_instance_id == before.matrix_instance_id);
  CHECK(matrix.backend == before.backend);
  CHECK(matrix.source_version == before.source_version);
  CHECK(matrix.exact_max_magnitude == before.exact_max_magnitude);
  CHECK(matrix.host_residues_current == before.host_residues_current);
  CHECK(matrix.device_residues_current == before.device_residues_current);
  CHECK(matrix.host_native_current == before.host_native_current);
  CHECK(matrix.device_native_current == before.device_native_current);
  CHECK(matrix.host_byte_limbs_current == before.host_byte_limbs_current);
  CHECK(matrix.device_byte_limbs_current == before.device_byte_limbs_current);
  CHECK(matrix.native_i64 == before.native_i64);
  CHECK(matrix.native_u64 == before.native_u64);
  REQUIRE(matrix.exact_axis_ranges.rows.size() == before.exact_axis_ranges.rows.size());
  REQUIRE(matrix.exact_axis_ranges.cols.size() == before.exact_axis_ranges.cols.size());
  const auto check_axis = [](const auto& after, const auto& saved) {
    CHECK(after.max_magnitude == saved.max_magnitude);
    CHECK(after.absolute_sum == saved.absolute_sum);
    CHECK(after.sum == saved.sum);
    CHECK(after.minimum == saved.minimum);
    CHECK(after.maximum == saved.maximum);
    CHECK(after.deviation_sum == saved.deviation_sum);
    CHECK(after.sum_known == saved.sum_known);
  };
  for (std::size_t i = 0; i < before.exact_axis_ranges.rows.size(); ++i)
    check_axis(matrix.exact_axis_ranges.rows[i], before.exact_axis_ranges.rows[i]);
  for (std::size_t i = 0; i < before.exact_axis_ranges.cols.size(); ++i)
    check_axis(matrix.exact_axis_ranges.cols[i], before.exact_axis_ranges.cols[i]);
}

void check_lift_unchanged(const rns8_matrix& matrix, const rns8_matrix& before) {
  CHECK(matrix.residues == before.residues);
  CHECK(matrix.exact_range_prefix == before.exact_range_prefix);
  check_lift_metadata(matrix, before);
}

void check_lift_planes(const rns8_matrix& matrix, const std::vector<boost::multiprecision::cpp_int>& values,
                       uint32_t prefix) {
  for (uint32_t p = 0; p < prefix; ++p) {
    for (std::size_t cell = 0; cell < values.size(); ++cell) {
      CHECK(matrix.residues[values.size() * p + cell] ==
            rns8::detail::centered_residue(values[cell], rns8::detail::kDefaultModuli[p]));
    }
  }
}

// Internal oracle fixture for source-prefix boundaries that native-width plans
// cannot produce. This proof is constructed from known full integers, never
// inferred from their possibly aliased CRT representatives.
void proven_lift_fixture(rns8_matrix& matrix, const std::vector<boost::multiprecision::cpp_int>& values,
                         uint32_t prefix) {
  fill_exact_residue_matrix(&matrix, values);
  matrix.exact_max_magnitude = 0;
  for (const auto& value : values)
    matrix.exact_max_magnitude = std::max(matrix.exact_max_magnitude,
                                          boost::multiprecision::cpp_int(boost::multiprecision::abs(value)));
  matrix.exact_axis_ranges = {};
  matrix.exact_range_prefix = prefix;
  std::fill(matrix.residues.begin() + values.size() * prefix, matrix.residues.end(), int8_t(99));
}

struct ExactLiftPlan {
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  ExactLiftPlan(rns8_context* ctx, rns8_semantics semantics, int64_t m, int64_t n, int64_t k,
                uint32_t prefix) {
    auto desc = exact_desc(semantics, m, n, k);
    desc.max_prefix = prefix;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  }
  ~ExactLiftPlan() {
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }
};

}  // namespace

TEST_CASE("exact-wide CPU lifting enables wider signed and unsigned resident chains", "[exact-lift]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics, 1, 18);
    const int64_t sa = INT64_MIN, sb = INT64_MAX;
    const uint64_t ua = UINT64_MAX, ub = UINT64_MAX - 1;
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[0], &sa, 1, 11)
                 : rns8_pack_u64(f.ctx, f.matrices[0], &ua, 1, 11)) == RNS8_SUCCESS);
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[1], &sb, 1, 12)
                 : rns8_pack_u64(f.ctx, f.matrices[1], &ub, 1, 12)) == RNS8_SUCCESS);
    REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
    auto& result = *f.matrices[2];
    const boost::multiprecision::cpp_int expected = semantics == RNS8_EXACT_WIDE_SIGNED
                                                        ? boost::multiprecision::cpp_int(sa) * sb
                                                        : boost::multiprecision::cpp_int(ua) * ub;
    CHECK(result.exact_range_prefix == 18);
    result.residues[18] = result.residues[19] = 99;  // Unwritten planes must be replaced.
    const auto before = result;
    auto* storage = result.residues.data();
    auto* row_proofs = result.exact_axis_ranges.rows.data();
    ExactLiftPlan larger(f.ctx, semantics, 1, 1, 1, 20);
    uint64_t output[] = {777, 777, 777};
    const auto export_wide = [&]() {
      return semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_export_exact_wide_signed_limbs(f.ctx, larger.plan, &result, output, 1, 3)
                 : rns8_export_exact_wide_unsigned_limbs(f.ctx, larger.plan, &result, output, 1, 3);
    };
    CHECK(export_wide() == RNS8_RANGE_ERROR);
    CHECK(output[0] == 777);
    REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &result, 19, 1) == RNS8_SUCCESS);
    CHECK(result.exact_range_prefix == 19);
    CHECK(result.residues[19] == 99);
    CHECK(export_wide() == RNS8_RANGE_ERROR);
    REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &result, 20, 1) == RNS8_SUCCESS);
    CHECK(result.exact_range_prefix == 20);
    CHECK(result.residues.data() == storage);
    CHECK(result.exact_axis_ranges.rows.data() == row_proofs);
    check_lift_metadata(result, before);
    check_lift_planes(result, {expected}, 20);
    REQUIRE(export_wide() == RNS8_SUCCESS);
    CHECK(std::vector<uint64_t>(output, output + 3) == (semantics == RNS8_EXACT_WIDE_SIGNED
                                                            ? signed_twos_complement_limbs(expected, 3)
                                                            : unsigned_limbs(expected, 3)));
    const int64_t sone = 1;
    const uint64_t uone = 1;
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[1], &sone, 1, 13)
                 : rns8_pack_u64(f.ctx, f.matrices[1], &uone, 1, 13)) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(f.ctx, larger.plan, &result, f.matrices[1], f.matrices[3], larger.workspace) ==
            RNS8_SUCCESS);
    check_lift_planes(*f.matrices[3], {expected}, 20);
  }
}

TEST_CASE("exact-wide CPU lifting covers every source-prefix strict boundary", "[exact-lift]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics);
    auto& matrix = *f.matrices[0];
    for (uint32_t prefix = 1; prefix < 20; ++prefix) {
      INFO("prefix=" << prefix << " semantics=" << semantics);
      const auto product = rns8::detail::modulus_product(prefix);
      const boost::multiprecision::cpp_int limit = semantics == RNS8_EXACT_WIDE_SIGNED
                                                       ? boost::multiprecision::cpp_int((product - 1) / 2)
                                                       : boost::multiprecision::cpp_int(product - 1);
      const std::vector<boost::multiprecision::cpp_int> values =
          semantics == RNS8_EXACT_WIDE_SIGNED
              ? std::vector<boost::multiprecision::cpp_int>{0, 1, -1, limit, -limit}
              : std::vector<boost::multiprecision::cpp_int>{0, 1, limit};
      for (const auto& value : values) {
        proven_lift_fixture(matrix, {value}, prefix);
        const auto before = matrix;
        REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &matrix, prefix + 1, 1) == RNS8_SUCCESS);
        CHECK(matrix.exact_range_prefix == prefix + 1);
        CHECK(std::equal(before.residues.begin(), before.residues.begin() + prefix, matrix.residues.begin()));
        check_lift_planes(matrix, {value}, prefix + 1);
        check_lift_metadata(matrix, before);
      }
      proven_lift_fixture(matrix, {0}, prefix);
      matrix.exact_max_magnitude =
          semantics == RNS8_EXACT_WIDE_SIGNED ? boost::multiprecision::cpp_int(product / 2) : product;
      const auto ambiguous = matrix;
      CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, UINT64_MAX) == RNS8_RANGE_ERROR);
      check_lift_unchanged(matrix, ambiguous);
    }
  }
}

TEST_CASE("exact-wide CPU lifting obeys suffix budgets and preserves failed stages", "[exact-lift]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_SIGNED, 2, 18);
  const int64_t a[] = {-7, 2, 3, -4}, identity[] = {1, 0, 0, 1};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 2, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], identity, 2, 2) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  auto& matrix = *f.matrices[2];
  const auto before = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 7) == RNS8_WORKSPACE_TOO_SMALL);
  check_lift_unchanged(matrix, before);
  REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 19, 4) == RNS8_SUCCESS);
  const auto partial = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 3) == RNS8_WORKSPACE_TOO_SMALL);
  check_lift_unchanged(matrix, partial);

  // First three cells stage successfully; the last source cell is malformed.
  const auto last_source = matrix.residues[4 + 3];
  matrix.residues[4 + 3] = -128;  // Outside modulus 255's centered interval.
  const auto malformed = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 4) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(matrix, malformed);
  matrix.residues[4 + 3] = last_source;

  // A valid centered byte can still contradict the retained integer proof.
  const auto last_first = matrix.residues[3];
  matrix.residues[3] = static_cast<int8_t>(last_first + 1);
  const auto inconsistent = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 4) == RNS8_RANGE_ERROR);
  check_lift_unchanged(matrix, inconsistent);
  matrix.residues[3] = last_first;
  REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 4) == RNS8_SUCCESS);
  check_lift_planes(matrix, {-7, 2, 3, -4}, 20);
  check_lift_metadata(matrix, before);
  const auto lifted = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 0) == RNS8_SUCCESS);
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 18, 0) == RNS8_SUCCESS);
  check_lift_unchanged(matrix, lifted);
}

TEST_CASE("exact-wide CPU lifting preserves cancellation and rejects later overflow", "[exact-lift]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_SIGNED, 2, 18);
  const int64_t h = INT64_MAX;
  const int64_t a[] = {h, h, h, h}, balanced[] = {h, -h, -h, h}, positive[] = {h, h, h, h};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 2, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], balanced, 2, 2) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  auto& matrix = *f.matrices[2];
  CHECK(matrix.exact_max_magnitude == 0);
  const auto zero = matrix;
  REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 8) == RNS8_SUCCESS);
  check_lift_metadata(matrix, zero);
  check_lift_planes(matrix, {0, 0, 0, 0}, 20);
  ExactLiftPlan larger(f.ctx, RNS8_EXACT_WIDE_SIGNED, 2, 2, 2, 20);
  REQUIRE(rns8_gemm_rns(f.ctx, larger.plan, &matrix, f.matrices[0], f.matrices[3], larger.workspace) ==
          RNS8_SUCCESS);
  check_lift_planes(*f.matrices[3], {0, 0, 0, 0}, 20);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], positive, 2, 3) == RNS8_SUCCESS);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 8) == RNS8_SUCCESS);
  const auto destination = *f.matrices[3];
  CHECK(rns8_gemm_rns(f.ctx, larger.plan, &matrix, f.matrices[0], f.matrices[3], larger.workspace) ==
        RNS8_RANGE_ERROR);
  check_lift_unchanged(*f.matrices[3], destination);
}

TEST_CASE("exact-wide CPU lifting rejects absent proofs and invalid ownership or capacity", "[exact-lift]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_UNSIGNED);
  auto& matrix = *f.matrices[0];
  const uint64_t value = 7;
  REQUIRE(rns8_pack_u64(f.ctx, &matrix, &value, 1, 31) == RNS8_SUCCESS);
  const auto packed = matrix;
  CHECK(rns8_lift_exact_wide_cpu(nullptr, &matrix, 20, 100) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, nullptr, 20, 100) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 0, 100) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 21, 100) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(matrix, packed);
  for (uint32_t proof : {0u, 21u}) {
    matrix.exact_range_prefix = proof;
    const auto before = matrix;
    CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_RANGE_ERROR);
    check_lift_unchanged(matrix, before);
  }
  matrix = packed;
  matrix.exact_max_magnitude = -1;
  const auto negative = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_RANGE_ERROR);
  check_lift_unchanged(matrix, negative);
  matrix = packed;
  matrix.host_residues_current = false;
  const auto stale = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(matrix, stale);
  matrix = packed;
  matrix.backend = RNS8_BACKEND_HIP_DIRECT;
  const auto foreign = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(matrix, foreign);
  f.ctx->backend = RNS8_BACKEND_HIP_DIRECT;  // Validate rejection only; no device calls.
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_UNSUPPORTED_BACKEND);
  check_lift_unchanged(matrix, foreign);
  f.ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
  matrix = packed;
  matrix.desc.semantics = RNS8_BOUNDED_U64;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_INVALID_ARGUMENT);
  matrix = packed;
  matrix.desc.rows = matrix.desc.cols = matrix.desc.logical_ld = INT64_MAX;
  const auto overflow = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, UINT64_MAX) == RNS8_RANGE_ERROR);
  check_lift_unchanged(matrix, overflow);
  matrix = packed;
  matrix.residues.pop_back();
  const auto truncated = matrix;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, &matrix, 20, 100) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(matrix, truncated);
  matrix = packed;

  auto desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
  desc.max_prefix = 1;
  rns8_matrix* small = nullptr;
  REQUIRE(rns8_create_matrix(f.ctx, &desc, &small) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(f.ctx, small, &value, 1, 1) == RNS8_SUCCESS);
  const auto limited = *small;
  CHECK(rns8_lift_exact_wide_cpu(f.ctx, small, 2, 100) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(*small, limited);
  rns8_destroy_matrix(small);
}
