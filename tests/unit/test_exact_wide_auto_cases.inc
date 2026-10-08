namespace {

struct ExactAutoFixture : ExactChainFixture {
  explicit ExactAutoFixture(rns8_semantics semantics, int64_t m = 1, int64_t n = 1, int64_t k = 1,
                            uint32_t ceiling = 20)
      : ExactChainFixture(semantics, m, n, k, ceiling) {
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    plan = nullptr;
    workspace = nullptr;
    auto desc = exact_desc(semantics, m, n, k);
    desc.max_prefix = ceiling;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  }
  rns8_status automatic(unsigned a, unsigned b, unsigned c, uint64_t budget, uint32_t& selected) {
    return rns8_gemm_exact_wide_cpu_auto(ctx, plan, matrices[a], matrices[b], matrices[c], workspace, budget,
                                         &selected);
  }
};

void check_auto_export(ExactAutoFixture& f, unsigned c, uint32_t selected,
                       const std::vector<boost::multiprecision::cpp_int>& expected) {
  ExactLiftPlan export_plan(f.ctx, f.plan->desc.semantics, f.plan->desc.m, f.plan->desc.n, f.plan->desc.k,
                            selected);
  std::vector<uint64_t> limbs(expected.size() * 3);
  const bool signed_values = f.plan->desc.semantics == RNS8_EXACT_WIDE_SIGNED;
  REQUIRE((signed_values
               ? rns8_export_exact_wide_signed_limbs(f.ctx, export_plan.plan, f.matrices[c], limbs.data(),
                                                     f.plan->desc.n, 3)
               : rns8_export_exact_wide_unsigned_limbs(f.ctx, export_plan.plan, f.matrices[c], limbs.data(),
                                                       f.plan->desc.n, 3)) == RNS8_SUCCESS);
  for (std::size_t cell = 0; cell < expected.size(); ++cell) {
    const auto reference =
        signed_values ? signed_twos_complement_limbs(expected[cell], 3) : unsigned_limbs(expected[cell], 3);
    CHECK(std::vector<uint64_t>(limbs.begin() + 3 * cell, limbs.begin() + 3 * cell + 3) == reference);
  }
}

}  // namespace

TEST_CASE("exact-wide opt-in CPU selection widens public signed and unsigned chains", "[exact-auto]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactAutoFixture f(semantics);
    const int64_t signed_high = INT64_MIN, signed_scale = 1024;
    const uint64_t unsigned_high = UINT64_MAX, unsigned_scale = 1024;
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[0], &signed_high, 1, 11)
                 : rns8_pack_u64(f.ctx, f.matrices[0], &unsigned_high, 1, 11)) == RNS8_SUCCESS);
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[1], &signed_scale, 1, 12)
                 : rns8_pack_u64(f.ctx, f.matrices[1], &unsigned_scale, 1, 12)) == RNS8_SUCCESS);
    REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
    REQUIRE(f.multiply(0, 1, 3) == RNS8_SUCCESS);
    const boost::multiprecision::cpp_int value =
        semantics == RNS8_EXACT_WIDE_SIGNED ? boost::multiprecision::cpp_int(signed_high) * signed_scale
                                            : boost::multiprecision::cpp_int(unsigned_high) * unsigned_scale;
    const boost::multiprecision::cpp_int expected = value * value;
    const boost::multiprecision::cpp_int range =
        semantics == RNS8_EXACT_WIDE_SIGNED ? 2 * expected : expected;
    const uint32_t required = rns8::detail::required_prefix_for_range(range);
    REQUIRE(required > f.plan->prefix);
    const auto before_a = *f.matrices[2], before_b = *f.matrices[3], before_c = *f.matrices[0];
    const auto base = f.plan->prefix;
    const auto workspace_prefix = f.workspace->prefix;
    const auto workspace_fingerprint = f.workspace->schedule_fingerprint;
    CHECK(f.multiply(2, 3, 0) == RNS8_RANGE_ERROR);  // Existing call remains conservative.
    check_lift_unchanged(*f.matrices[0], before_c);
    uint32_t selected = 777;
    const uint64_t budget = 2 * (required - base) + required;
    CHECK(f.automatic(2, 3, 0, budget - 1, selected) == RNS8_WORKSPACE_TOO_SMALL);
    CHECK(selected == 777);
    check_lift_unchanged(*f.matrices[2], before_a);
    check_lift_unchanged(*f.matrices[3], before_b);
    check_lift_unchanged(*f.matrices[0], before_c);
    REQUIRE(f.automatic(2, 3, 0, budget, selected) == RNS8_SUCCESS);
    CHECK(selected == required);
    CHECK(f.matrices[2]->exact_range_prefix == required);
    CHECK(f.matrices[3]->exact_range_prefix == required);
    CHECK(f.matrices[0]->exact_range_prefix == required);
    check_lift_metadata(*f.matrices[2], before_a);
    check_lift_metadata(*f.matrices[3], before_b);
    check_lift_planes(*f.matrices[2], {value}, required);
    check_lift_planes(*f.matrices[3], {value}, required);
    check_lift_planes(*f.matrices[0], {expected}, required);
    CHECK(f.plan->prefix == base);
    CHECK(f.plan->desc.max_prefix == 20);
    CHECK(f.workspace->prefix == workspace_prefix);
    CHECK(f.workspace->schedule_fingerprint == workspace_fingerprint);
    uint64_t inadequate[3] = {999, 999, 999};
    CHECK((semantics == RNS8_EXACT_WIDE_SIGNED
               ? rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[0], inadequate, 1, 3)
               : rns8_export_exact_wide_unsigned_limbs(f.ctx, f.plan, f.matrices[0], inadequate, 1, 3)) ==
          RNS8_RANGE_ERROR);
    CHECK(inadequate[0] == 999);
    CHECK(inadequate[1] == 999);
    CHECK(inadequate[2] == 999);
    check_auto_export(f, 0, selected, {expected});
  }
}

TEST_CASE("exact-wide opt-in CPU selection handles asymmetric initialized ranges", "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_SIGNED);
  const boost::multiprecision::cpp_int a = -rns8::detail::modulus_product(19) / 4;
  proven_lift_fixture(*f.matrices[0], {a}, 19);
  proven_lift_fixture(*f.matrices[1], {3}, 18);
  const auto before_a = *f.matrices[0], before_b = *f.matrices[1], before_c = *f.matrices[2];
  uint32_t selected = 777;
  CHECK(f.automatic(0, 1, 2, 22, selected) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[1], before_b);
  check_lift_unchanged(*f.matrices[2], before_c);
  REQUIRE(f.automatic(0, 1, 2, 23, selected) == RNS8_SUCCESS);
  CHECK(selected == 20);  // A needs one plane, B two, output twenty.
  check_lift_metadata(*f.matrices[0], before_a);
  check_lift_metadata(*f.matrices[1], before_b);
  check_lift_planes(*f.matrices[0], {a}, 20);
  check_lift_planes(*f.matrices[1], {3}, 20);
  check_auto_export(f, 2, selected, {3 * a});
}

TEST_CASE("exact-wide opt-in CPU selection honors strict output product boundaries", "[exact-auto]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactAutoFixture f(semantics);
    const uint32_t base = f.plan->prefix;
    const auto product = rns8::detail::modulus_product(base);
    for (int adjustment : {-1, 0, 1}) {
      INFO("semantics=" << semantics << " adjustment=" << adjustment);
      boost::multiprecision::cpp_int a = product / (semantics == RNS8_EXACT_WIDE_SIGNED ? 4 : 2);
      a += adjustment;
      if (semantics == RNS8_EXACT_WIDE_SIGNED) a = -a;
      proven_lift_fixture(*f.matrices[0], {a}, base);
      proven_lift_fixture(*f.matrices[1], {2}, 1);
      uint32_t selected = 777;
      const uint32_t expected_prefix = adjustment < 0 ? base : base + 1;
      const uint64_t budget = (expected_prefix - base) + (expected_prefix - 1) + expected_prefix;
      REQUIRE(f.automatic(0, 1, 2, budget, selected) == RNS8_SUCCESS);
      CHECK(selected == expected_prefix);
      check_lift_planes(*f.matrices[2], {2 * a}, selected);
      check_auto_export(f, 2, selected, {2 * a});
    }
  }
}

TEST_CASE("exact-wide opt-in CPU selection rolls back both operands after late staging failure",
          "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_SIGNED, 2, 3, 2);
  const boost::multiprecision::cpp_int h = rns8::detail::modulus_product(18) / 4;
  proven_lift_fixture(*f.matrices[0], {-h, 0, 0, h}, 18);
  proven_lift_fixture(*f.matrices[1], {3, 0, -3, 0, 3, 0}, 18);
  const auto a_before = *f.matrices[0], b_before = *f.matrices[1], c_before = *f.matrices[2];
  auto* a_storage = f.matrices[0]->residues.data();
  auto* b_storage = f.matrices[1]->residues.data();
  auto* c_storage = f.matrices[2]->residues.data();
  uint32_t selected = 777;
  const uint64_t budget = 4 + 6 + 6 * 19;
  CHECK(f.automatic(0, 1, 2, budget - 1, selected) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], a_before);
  check_lift_unchanged(*f.matrices[1], b_before);
  check_lift_unchanged(*f.matrices[2], c_before);

  f.matrices[1]->residues[6 + 5] = -128;  // Last B cell, after A fully stages.
  const auto malformed_b = *f.matrices[1];
  CHECK(f.automatic(0, 1, 2, budget, selected) == RNS8_INVALID_ARGUMENT);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], a_before);
  check_lift_unchanged(*f.matrices[1], malformed_b);
  check_lift_unchanged(*f.matrices[2], c_before);
  f.matrices[1]->residues[6 + 5] = b_before.residues[6 + 5];
  f.matrices[1]->residues[5] = 1;  // Valid byte, reconstructed value contradicts proof.
  const auto inconsistent_b = *f.matrices[1];
  CHECK(f.automatic(0, 1, 2, budget, selected) == RNS8_RANGE_ERROR);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], a_before);
  check_lift_unchanged(*f.matrices[1], inconsistent_b);
  check_lift_unchanged(*f.matrices[2], c_before);
  f.matrices[1]->residues[5] = b_before.residues[5];
  REQUIRE(f.automatic(0, 1, 2, budget, selected) == RNS8_SUCCESS);
  CHECK(selected == 19);
  CHECK(f.matrices[0]->residues.data() == a_storage);
  CHECK(f.matrices[1]->residues.data() == b_storage);
  CHECK(f.matrices[2]->residues.data() == c_storage);
  CHECK(std::equal(c_before.residues.begin() + 6 * 19, c_before.residues.end(),
                   f.matrices[2]->residues.begin() + 6 * 19));
  check_lift_metadata(*f.matrices[0], a_before);
  check_lift_metadata(*f.matrices[1], b_before);
  check_auto_export(f, 2, selected, {-3 * h, 0, 3 * h, 0, 3 * h, 0});
}

TEST_CASE("exact-wide opt-in CPU selection deduplicates shared inputs and honors fixed plans",
          "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_SIGNED);
  proven_lift_fixture(*f.matrices[0], {3}, 1);
  const auto before_a = *f.matrices[0], before_c = *f.matrices[2];
  uint32_t selected = 777;
  const uint32_t base = f.plan->prefix;
  const uint64_t budget = base - 1 + base;
  CHECK(f.automatic(0, 0, 2, budget - 1, selected) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[2], before_c);
  REQUIRE(f.automatic(0, 0, 2, budget, selected) == RNS8_SUCCESS);
  CHECK(selected == base);
  check_lift_planes(*f.matrices[2], {9}, selected);
  check_lift_metadata(*f.matrices[0], before_a);
  const auto lifted = *f.matrices[0];
  CHECK(f.automatic(0, 0, 0, UINT64_MAX, selected) == RNS8_INVALID_ARGUMENT);
  check_lift_unchanged(*f.matrices[0], lifted);

  ExactLiftPlan fixed(f.ctx, RNS8_EXACT_WIDE_SIGNED, 1, 1, 1, 20);
  proven_lift_fixture(*f.matrices[0], {3}, 1);
  REQUIRE(rns8_gemm_exact_wide_cpu_auto(f.ctx, fixed.plan, f.matrices[0], f.matrices[0], f.matrices[2],
                                        fixed.workspace, 39, &selected) == RNS8_SUCCESS);
  CHECK(selected == 20);
  check_lift_planes(*f.matrices[2], {9}, 20);
}

TEST_CASE("exact-wide opt-in CPU selection rejects inadequate ceilings and absent proofs", "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_UNSIGNED);
  const boost::multiprecision::cpp_int a = rns8::detail::modulus_product(19) - 1;
  proven_lift_fixture(*f.matrices[0], {a}, 19);
  proven_lift_fixture(*f.matrices[1], {2}, 1);
  uint32_t selected = 777;
  for (unsigned restricted : {0u, 1u, 2u}) {
    const auto saved = *f.matrices[restricted];
    f.matrices[restricted]->prefix = f.matrices[restricted]->desc.max_prefix = 19;
    f.matrices[restricted]->residues.resize(19);
    if (restricted == 2) f.matrices[restricted]->exact_range_prefix = 19;
    const auto before_a = *f.matrices[0], before_b = *f.matrices[1], before_c = *f.matrices[2];
    CHECK(f.automatic(0, 1, 2, UINT64_MAX, selected) == RNS8_RANGE_ERROR);
    CHECK(selected == 777);
    check_lift_unchanged(*f.matrices[0], before_a);
    check_lift_unchanged(*f.matrices[1], before_b);
    check_lift_unchanged(*f.matrices[2], before_c);
    *f.matrices[restricted] = saved;
  }
  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
  desc.max_prefix = 19;
  rns8_plan* limited = nullptr;
  rns8_workspace* limited_workspace = nullptr;
  REQUIRE(rns8_create_plan(f.ctx, &desc, &limited) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(f.ctx, limited, &limited_workspace) == RNS8_SUCCESS);
  const auto before_a = *f.matrices[0], before_b = *f.matrices[1], before_c = *f.matrices[2];
  CHECK(rns8_gemm_exact_wide_cpu_auto(f.ctx, limited, f.matrices[0], f.matrices[1], f.matrices[2],
                                      limited_workspace, UINT64_MAX, &selected) == RNS8_RANGE_ERROR);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[1], before_b);
  check_lift_unchanged(*f.matrices[2], before_c);
  rns8_destroy_workspace(limited_workspace);
  rns8_destroy_plan(limited);

  ExactLiftPlan incompatible(f.ctx, RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1, 20);
  CHECK(rns8_gemm_exact_wide_cpu_auto(f.ctx, f.plan, f.matrices[0], f.matrices[1], f.matrices[2],
                                      incompatible.workspace, UINT64_MAX,
                                      &selected) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[1], before_b);
  check_lift_unchanged(*f.matrices[2], before_c);
  for (auto bound : {boost::multiprecision::cpp_int(-1), rns8::detail::modulus_product(19)}) {
    f.matrices[0]->exact_max_magnitude = bound;
    const auto invalid_proof = *f.matrices[0];
    CHECK(f.automatic(0, 1, 2, UINT64_MAX, selected) == RNS8_RANGE_ERROR);
    CHECK(selected == 777);
    check_lift_unchanged(*f.matrices[0], invalid_proof);
    check_lift_unchanged(*f.matrices[1], before_b);
    check_lift_unchanged(*f.matrices[2], before_c);
  }
  *f.matrices[0] = before_a;
  f.matrices[0]->exact_range_prefix = 0;
  proven_lift_fixture(*f.matrices[1], {0}, 1);  // Even a zero output cannot authorize an unknown input.
  const auto unknown = *f.matrices[0], zero = *f.matrices[1];
  CHECK(f.automatic(0, 1, 2, UINT64_MAX, selected) == RNS8_RANGE_ERROR);
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], unknown);
  check_lift_unchanged(*f.matrices[1], zero);
  check_lift_unchanged(*f.matrices[2], before_c);
}

TEST_CASE("exact-wide opt-in CPU selection retains cancellation proofs and fails conservatively",
          "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_SIGNED, 2, 2, 2);
  const boost::multiprecision::cpp_int h = rns8::detail::modulus_product(20) / 4;
  const std::vector<boost::multiprecision::cpp_int> a = {h, h, h, h}, b = {h, -h, -h, h};
  proven_lift_fixture(*f.matrices[0], a, 20);
  proven_lift_fixture(*f.matrices[1], b, 20);
  const auto before_a = *f.matrices[0], before_b = *f.matrices[1], before_c = *f.matrices[2];
  uint32_t selected = 777;
  CHECK(f.automatic(0, 1, 2, UINT64_MAX, selected) == RNS8_RANGE_ERROR);
  CHECK(selected == 777);  // No proof from observing a reference GEMM's cancellation.
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[1], before_b);
  check_lift_unchanged(*f.matrices[2], before_c);
  f.matrices[0]->exact_axis_ranges =
      rns8::detail::native_exact_ranges(a.data(), 2, 2, 2, f.matrices[0]->exact_max_magnitude);
  f.matrices[1]->exact_axis_ranges =
      rns8::detail::native_exact_ranges(b.data(), 2, 2, 2, f.matrices[1]->exact_max_magnitude);
  const auto proved_a = *f.matrices[0], proved_b = *f.matrices[1];
  const uint64_t budget = 4 * f.plan->prefix;  // Inputs need no extra planes.
  REQUIRE(f.automatic(0, 1, 2, budget, selected) == RNS8_SUCCESS);
  CHECK(selected == f.plan->prefix);
  CHECK(f.matrices[2]->exact_max_magnitude == 0);
  CHECK(f.matrices[0]->exact_range_prefix == 20);
  CHECK(f.matrices[1]->exact_range_prefix == 20);
  check_lift_unchanged(*f.matrices[0], proved_a);
  check_lift_unchanged(*f.matrices[1], proved_b);
  check_auto_export(f, 2, selected, {0, 0, 0, 0});
}

TEST_CASE("exact-wide opt-in CPU selection rejects invalid calls without mutation", "[exact-auto]") {
  ExactAutoFixture f(RNS8_EXACT_WIDE_SIGNED);
  const int64_t value = -7;
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], &value, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], &value, 1, 2) == RNS8_SUCCESS);
  const auto before_a = *f.matrices[0], before_b = *f.matrices[1], before_c = *f.matrices[2];
  uint32_t selected = 777;
  CHECK(rns8_gemm_exact_wide_cpu_auto(nullptr, f.plan, f.matrices[0], f.matrices[1], f.matrices[2],
                                      f.workspace, 100, &selected) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_exact_wide_cpu_auto(f.ctx, f.plan, f.matrices[0], f.matrices[1], f.matrices[2], f.workspace,
                                      100, nullptr) == RNS8_INVALID_ARGUMENT);
  CHECK(f.automatic(0, 1, 0, 100, selected) == RNS8_INVALID_ARGUMENT);
  CHECK(f.automatic(0, 1, 1, 100, selected) == RNS8_INVALID_ARGUMENT);
  f.matrices[0]->host_residues_current = false;
  CHECK(f.automatic(0, 1, 2, 100, selected) == RNS8_INVALID_ARGUMENT);
  f.matrices[0]->host_residues_current = true;
  f.matrices[1]->desc.cols = 2;
  CHECK(f.automatic(0, 1, 2, 100, selected) == RNS8_INVALID_ARGUMENT);
  f.matrices[1]->desc.cols = 1;
  const auto backend = f.plan->backend;
  f.plan->backend = f.ctx->backend = RNS8_BACKEND_HIP_DIRECT;
  CHECK(f.automatic(0, 1, 2, 100, selected) == RNS8_UNSUPPORTED_BACKEND);
  f.plan->backend = f.ctx->backend = backend;
  CHECK(selected == 777);
  check_lift_unchanged(*f.matrices[0], before_a);
  check_lift_unchanged(*f.matrices[1], before_b);
  check_lift_unchanged(*f.matrices[2], before_c);
}
