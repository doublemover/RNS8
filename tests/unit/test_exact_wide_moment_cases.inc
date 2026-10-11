TEST_CASE("exact-wide moment bounds cover every small signed dot product", "[exact-moment]") {
  using namespace rns8::detail;
  // Exhaustive three-element axes catch cancellation, zero variance, and rounding
  // toward zero of negative midpoint centers without depending on RNS GEMM.
  for (unsigned ai = 0; ai < 125; ++ai) {
    for (unsigned bi = 0; bi < 125; ++bi) {
      int64_t a[3], b[3];
      unsigned av = ai, bv = bi;
      for (unsigned i = 0; i < 3; ++i) {
        a[i] = static_cast<int64_t>(av % 5) - 2;
        b[i] = static_cast<int64_t>(bv % 5) - 2;
        av /= 5;
        bv /= 5;
      }
      cpp_int maximum;
      auto ar = native_exact_ranges(a, 1, 3, 3, maximum).rows[0];
      auto br = native_exact_ranges(b, 3, 1, 1, maximum).cols[0];
      const cpp_int exact = exact_i64_gemm_cell(a, 3, b, 1, 0, 0, 3);
      const cpp_int magnitude = boost::multiprecision::abs(exact);
      const auto bound = exact_axis_product_bound(ar, br, 3);
      REQUIRE(bound >= magnitude);
      // Conservative output moments may exceed the actual squared norm, and
      // do not carry exact sums. Both properties must remain safe downstream.
      ar.sum_known = br.sum_known = false;
      ar.squared_sum += 7;
      br.squared_sum += 11;
      REQUIRE(exact_axis_product_bound(ar, br, 3) >= magnitude);
      ar.squared_sum_known = br.squared_sum_known = false;
      REQUIRE(bound <= exact_axis_product_bound(ar, br, 3));
    }
  }
}

TEST_CASE("exact-wide covariance and norm bounds improve distinct inequalities", "[exact-moment]") {
  using namespace rns8::detail;
  cpp_int maximum;
  const int64_t a[] = {3, 3, 2}, b[] = {3, 0, -3};
  auto ar = native_exact_ranges(a, 1, 3, 3, maximum).rows[0];
  auto br = native_exact_ranges(b, 3, 1, 1, maximum).cols[0];
  CHECK(exact_axis_product_bound(ar, br, 3) == 3);
  ar.squared_sum_known = false;
  CHECK(exact_axis_product_bound(ar, br, 3) == 6);

  const int64_t uneven[] = {10, 1, 1};
  ar = native_exact_ranges(uneven, 1, 3, 3, maximum).rows[0];
  br = ar;
  ar.sum_known = br.sum_known = false;
  CHECK(exact_axis_product_bound(ar, br, 3) == 102);
  ar.squared_sum_known = false;
  CHECK(exact_axis_product_bound(ar, br, 3) == 120);

  const int64_t nonsquare[] = {1, 1, 0};
  br = native_exact_ranges(nonsquare, 3, 1, 1, maximum).cols[0];
  ar = native_exact_ranges(a, 1, 3, 3, maximum).rows[0];
  // floor(sqrt(22*2)) = 6; the exact dot product is also 6.
  ar.sum_known = br.sum_known = false;
  CHECK(exact_axis_product_bound(ar, br, 3) == 6);
}

TEST_CASE("exact-wide native moments handle full widths padding and K block boundaries",
          "[exact-moment]") {
  using namespace rns8::detail;
  for (int64_t k : {1, 65535, 65536, 65537}) {
    std::vector<int64_t> a(static_cast<std::size_t>(k) + 1, INT64_MIN);
    std::vector<uint64_t> b(static_cast<std::size_t>(k) + 1, UINT64_MAX);
    a.back() = 17;
    b.back() = 17;
    cpp_int maximum;
    const auto ar = native_exact_ranges(a.data(), 1, k, k + 1, maximum).rows[0];
    const auto br = native_exact_ranges(b.data(), 1, k, k + 1, maximum).rows[0];
    const cpp_int signed_square = cpp_int(1) << 126;
    const cpp_int unsigned_square = cpp_int(UINT64_MAX) * UINT64_MAX;
    REQUIRE(ar.squared_sum_known);
    CHECK(ar.squared_sum == k * signed_square);
    CHECK(br.squared_sum == k * unsigned_square);
    CHECK(exact_axis_product_bound(ar, ar, k) == k * signed_square);
    CHECK(exact_axis_product_bound(br, br, k) == k * unsigned_square);
  }
}

void check_exact_moment_chain(rns8_backend_kind backend) {
  using namespace rns8::detail;
  for (auto sem : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture first(sem, 3, 1, 1, 20, backend);
    ExactChainFixture next(sem, 1, 1, 3, 17, backend);
    const cpp_int product = modulus_product(17);
    const unsigned factor = sem == RNS8_EXACT_WIDE_SIGNED ? 2 : 1;
    const int64_t t = int64_t(1) << 59;
    // Same safe signed native scale in both semantics; unsigned uses twice
    // the last operand to place its bound near its larger uniqueness limit.
    const int64_t h = (product / (cpp_int(2200) * t)).convert_to<int64_t>();
    const int64_t a[] = {10 * t, t, t};
    const int64_t diagonal[] = {h};
    const int64_t scale = 2 / factor;
    const int64_t b[] = {100 * scale, 10 * scale, 10 * scale};
    auto pack = [&](rns8_context* ctx, rns8_matrix* matrix, const int64_t* values, std::size_t count,
                    int64_t ld) {
      if (sem == RNS8_EXACT_WIDE_SIGNED) return rns8_pack_i64(ctx, matrix, values, ld, 1);
      std::vector<uint64_t> unsigned_values(values, values + count);
      return rns8_pack_u64(ctx, matrix, unsigned_values.data(), ld, 1);
    };
    REQUIRE(pack(first.ctx, first.matrices[0], a, 3, 1) == RNS8_SUCCESS);
    REQUIRE(pack(first.ctx, first.matrices[1], diagonal, 1, 1) == RNS8_SUCCESS);
    REQUIRE(first.multiply(0, 1, 2) == RNS8_SUCCESS);
    auto* resident = first.matrices[2];
    const cpp_int unit = cpp_int(t) * h;
    CHECK(resident->exact_axis_ranges.cols[0].squared_sum == 102 * unit * unit);
    CHECK_FALSE(resident->exact_axis_ranges.cols[0].sum_known);
    REQUIRE(pack(next.ctx, next.matrices[0], b, 3, 3) == RNS8_SUCCESS);
    const auto saved_axes = resident->exact_axis_ranges;
    resident->exact_axis_ranges.cols[0].squared_sum_known = false;
    CHECK(rns8_gemm_rns(next.ctx, next.plan, next.matrices[0], resident, next.matrices[2],
                        next.workspace) == RNS8_RANGE_ERROR);
    resident->exact_axis_ranges = saved_axes;
    REQUIRE(rns8_gemm_rns(next.ctx, next.plan, next.matrices[0], resident, next.matrices[2],
                          next.workspace) == RNS8_SUCCESS);
    CHECK(next.matrices[2]->exact_max_magnitude == 1020 * scale * unit);
    if (backend == RNS8_BACKEND_CPU_REFERENCE) {
      uint32_t selected = 99;
      const auto before = next.matrices[2]->residues;
      CHECK(rns8_gemm_exact_wide_cpu_auto(next.ctx, next.plan, next.matrices[0], resident,
                                          next.matrices[2], next.workspace, 16,
                                          &selected) == RNS8_WORKSPACE_TOO_SMALL);
      CHECK(selected == 99);
      CHECK(next.matrices[2]->residues == before);
      REQUIRE(rns8_gemm_exact_wide_cpu_auto(next.ctx, next.plan, next.matrices[0], resident,
                                            next.matrices[2], next.workspace, 17,
                                            &selected) == RNS8_SUCCESS);
      CHECK(selected == 17);
    }
    uint64_t limbs[3]{};
    REQUIRE(
        (sem == RNS8_EXACT_WIDE_SIGNED
             ? rns8_export_exact_wide_signed_limbs(next.ctx, next.plan, next.matrices[2], limbs, 1, 3)
             : rns8_export_exact_wide_unsigned_limbs(next.ctx, next.plan, next.matrices[2], limbs, 1,
                                                     3)) == RNS8_SUCCESS);
    const auto expected = unsigned_limbs(1020 * scale * unit, 3);
    for (unsigned i = 0; i < 3; ++i) CHECK(limbs[i] == expected[i]);
  }
}

TEST_CASE("exact-wide propagated moments admit a narrower resident chain and auto transaction",
          "[exact-moment]") {
  check_exact_moment_chain(RNS8_BACKEND_CPU_REFERENCE);
}

TEST_CASE("exact-wide moment admission exports exact resident HIP chains", "[exact-moment][hip]") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* context = nullptr;
  const auto status = rns8_create_context(0, &options, &context);
  if (status == RNS8_UNSUPPORTED_BACKEND)
    SKIP("Direct HIP device required for moment admission qualification");
  REQUIRE(status == RNS8_SUCCESS);
  rns8_destroy_context(context);
  check_exact_moment_chain(RNS8_BACKEND_HIP_DIRECT);
}
