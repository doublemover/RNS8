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
