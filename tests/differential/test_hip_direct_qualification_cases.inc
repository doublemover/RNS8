TEST_CASE("qualification pack covers dispatch thresholds and plane tails", "[qualification][hip]") {
  if (!hip_available()) SKIP("HIP device required for qualification");
  auto* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  auto* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t boundaries[] = {0, 1, -1, 127, 128, -128, -129, 251, -251,
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
  for (const int64_t cols : {255, 256, 257, 258, 259, 4095, 4096, 4097}) {
    for (const int64_t padding : {0, 3}) {
      for (const uint32_t prefix : {1u, 8u, 9u, 20u}) {
        INFO("cols=" << cols << " padding=" << padding << " prefix=" << prefix);
        auto md = matrix_desc(1, cols, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
        md.max_prefix = prefix;
        std::vector<int64_t> values(static_cast<std::size_t>(cols + padding), 999);
        for (int64_t c = 0; c < cols; ++c) values[c] = boundaries[c % 11];
        rns8_matrix* cm = nullptr;
        rns8_matrix* hm = nullptr;
        REQUIRE(rns8_create_matrix(cpu, &md, &cm) == RNS8_SUCCESS);
        REQUIRE(rns8_create_matrix(hip, &md, &hm) == RNS8_SUCCESS);
        REQUIRE(rns8_pack_i64(cpu, cm, values.data(), cols + padding, 1) == RNS8_SUCCESS);
        REQUIRE(rns8_pack_i64(hip, hm, values.data(), cols + padding, 1) == RNS8_SUCCESS);
        REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
            hip->device_id, hm->residues.data(), hm->hip_residues, hm->hip_residue_bytes) == RNS8_SUCCESS);
        CHECK(hm->residues == cm->residues);
        rns8_destroy_matrix(hm);
        rns8_destroy_matrix(cm);
      }
    }
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("qualification bounded export checks every prefix and preserves output", "[qualification][hip]") {
  if (!hip_available()) SKIP("HIP device required for qualification");
  auto* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  auto* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t cols = 65;
  for (const uint32_t prefix : {1u, 7u, 8u, 9u, 20u}) {
    for (const bool is_signed : {false, true}) {
      auto cd = is_signed ? signed_desc(1, cols, 1, 10, RNS8_BACKEND_CPU_REFERENCE)
                          : unsigned_desc(1, cols, 1, 10, RNS8_BACKEND_CPU_REFERENCE);
      cd.max_prefix = prefix;
      cd.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
      auto hd = cd;
      hd.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      auto md = matrix_desc(1, cols, cd.semantics, cd.bound_kind);
      md.max_prefix = prefix;
      rns8_plan *cp = nullptr, *hp = nullptr;
      rns8_matrix *cm = nullptr, *hm = nullptr;
      REQUIRE(rns8_create_plan(cpu, &cd, &cp) == RNS8_SUCCESS);
      REQUIRE(rns8_create_plan(hip, &hd, &hp) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(cpu, &md, &cm) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(hip, &md, &hm) == RNS8_SUCCESS);
      // Success includes the prefix-8 overflow counterexample x=2. Failure in
      // each cell catches broken lane-status aggregation, including wave tails.
      for (int64_t bad_cell = -1; bad_cell < cols; ++bad_cell) {
        INFO("prefix=" << prefix << " signed=" << is_signed << " bad_cell=" << bad_cell);
        std::vector<boost::multiprecision::cpp_int> exact(cols, 2);
        if (bad_cell >= 0) exact[bad_cell] = is_signed ? -11 : 11;
        fill_exact_residue_matrix(cm, exact);
        fill_exact_residue_matrix(hm, exact);
        upload_exact_residues_to_hip(hm);
        const auto expected = bad_cell < 0 ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
        if (is_signed) {
          std::vector<int64_t> co(cols + 3, -777), ho = co;
          CHECK(rns8_export_i64(cpu, cp, cm, co.data(), cols + 3) == expected);
          CHECK(rns8_export_i64(hip, hp, hm, ho.data(), cols + 3) == expected);
          CHECK(ho == co);
        } else {
          std::vector<uint64_t> co(cols + 3, 777), ho = co;
          CHECK(rns8_export_u64(cpu, cp, cm, co.data(), cols + 3) == expected);
          CHECK(rns8_export_u64(hip, hp, hm, ho.data(), cols + 3) == expected);
          CHECK(ho == co);
        }
      }
      rns8_destroy_matrix(hm); rns8_destroy_matrix(cm);
      rns8_destroy_plan(hp); rns8_destroy_plan(cp);
    }
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("qualification persistent GEMM spans multiple safe K blocks", "[qualification][hip]") {
  if (!hip_available()) SKIP("HIP device required for qualification");
  auto* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  auto* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  for (const int64_t k : {65535, 65536, 65537, 131073, 262145}) {
    INFO("k=" << k);
    auto cd = signed_desc(1, 1, k, static_cast<uint64_t>(k) * 127 * 127, RNS8_BACKEND_CPU_REFERENCE);
    auto hd = cd; hd.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan *cp = nullptr, *hp = nullptr;
    rns8_workspace *cw = nullptr, *hw = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cd, &cp) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hd, &hp) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cp, &cw) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hp, &hw) == RNS8_SUCCESS);
    auto ad = matrix_desc(1, k, cd.semantics, cd.bound_kind);
    auto bd = matrix_desc(k, 1, cd.semantics, cd.bound_kind);
    auto od = matrix_desc(1, 1, cd.semantics, cd.bound_kind);
    rns8_matrix *ca = nullptr, *cb = nullptr, *cc = nullptr, *ha = nullptr, *hb = nullptr, *hc = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &ad, &ca) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &bd, &cb) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &od, &cc) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &ad, &ha) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &bd, &hb) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &od, &hc) == RNS8_SUCCESS);
    std::vector<int64_t> a(k, -127), b(k, -127);
    REQUIRE(rns8_pack_i64(cpu, ca, a.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(cpu, cb, b.data(), 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(hip, ha, a.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(hip, hb, b.data(), 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(cpu, cp, ca, cb, cc, cw) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(hip, hp, ha, hb, hc, hw) == RNS8_SUCCESS);
    int64_t co = 0, ho = 0;
    REQUIRE(rns8_export_i64(cpu, cp, cc, &co, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(hip, hp, hc, &ho, 1) == RNS8_SUCCESS);
    CHECK(ho == co);
    CHECK(ho == k * 127 * 127);
    for (auto* matrix : {ha, hb, hc, ca, cb, cc}) rns8_destroy_matrix(matrix);
    rns8_destroy_workspace(hw); rns8_destroy_workspace(cw);
    rns8_destroy_plan(hp); rns8_destroy_plan(cp);
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("qualification tiled GEMM never infers tile zero from host mirrors", "[qualification][hip]") {
  if (!hip_available()) SKIP("HIP device required for qualification");
  auto* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  auto* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  // One tile, with either a genuinely zero row/column or values whose first
  // residue is zero but whose integer value is nonzero. One-sided host mirrors
  // also cover the old unscanned operand's accidentally all-zero mask.
  for (const bool multiple_of_256 : {false, true}) {
    for (const unsigned host_mirrors : {0u, 1u, 2u, 3u}) {
      INFO("multiple_of_256=" << multiple_of_256 << " mirrors=" << host_mirrors);
      const std::vector<int64_t> a = multiple_of_256 ? std::vector<int64_t>{256, 512}
                                                   : std::vector<int64_t>{0, 7};
      const std::vector<int64_t> b = multiple_of_256 ? std::vector<int64_t>{3, 5}
                                                   : std::vector<int64_t>{0, 11};
      const std::vector<uint64_t> bounds = {2560};
      auto cd = per_tile_signed_desc(2, 2, 1, bounds, RNS8_BACKEND_CPU_REFERENCE);
      auto hd = per_tile_signed_desc(2, 2, 1, bounds, RNS8_BACKEND_HIP_DIRECT);
      rns8_plan* hp = nullptr;
      rns8_workspace* hw = nullptr;
      rns8_matrix *ha = nullptr, *hb = nullptr, *hc = nullptr;
      REQUIRE(rns8_create_plan(hip, &hd, &hp) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(hip, hp, &hw) == RNS8_SUCCESS);
      auto ad = matrix_desc(2, 1, hd.semantics, hd.bound_kind);
      auto bd = matrix_desc(1, 2, hd.semantics, hd.bound_kind);
      auto od = matrix_desc(2, 2, hd.semantics, hd.bound_kind);
      ad.tile_m = bd.tile_m = od.tile_m = hd.tile_m;
      ad.tile_n = bd.tile_n = od.tile_n = hd.tile_n;
      REQUIRE(rns8_create_matrix(hip, &ad, &ha) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(hip, &bd, &hb) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(hip, &od, &hc) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_i64(hip, ha, a.data(), 1, 1) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_i64(hip, hb, b.data(), 2, 1) == RNS8_SUCCESS);
      for (unsigned operand = 0; operand < 2; ++operand) {
        if ((host_mirrors & (1u << operand)) == 0) continue;
        auto* matrix = operand == 0 ? ha : hb;
        REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                    hip->device_id, matrix->residues.data(), matrix->hip_residues,
                    matrix->hip_residue_bytes) == RNS8_SUCCESS);
        matrix->host_residues_current = true;  // Coherent optional host mirror.
      }
      std::vector<int64_t> expected(4, -777), actual(4, -777);
      REQUIRE(rns8_gemm_i64_oneshot(cpu, &cd, a.data(), 1, b.data(), 2,
                                  expected.data(), 2) == RNS8_SUCCESS);
      REQUIRE(rns8_gemm_rns(hip, hp, ha, hb, hc, hw) == RNS8_SUCCESS);
      REQUIRE(rns8_export_i64(hip, hp, hc, actual.data(), 2) == RNS8_SUCCESS);
      CHECK(actual == expected);
      CHECK(actual[3] == a[1] * b[1]);
      CHECK(actual[3] != 0);
      for (auto* matrix : {ha, hb, hc}) rns8_destroy_matrix(matrix);
      rns8_destroy_workspace(hw);
      rns8_destroy_plan(hp);
    }
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
