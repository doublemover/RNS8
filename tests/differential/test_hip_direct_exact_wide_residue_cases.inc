TEST_CASE("direct HIP exact-wide RNS output matches CPU residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const int64_t A[] = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max() - 19};
  const int64_t B[] = {
      std::numeric_limits<int64_t>::max() - 3,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 7,
      std::numeric_limits<int64_t>::max() - 11};

  rns8_plan* cpu_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  auto cpu_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);

  rns8_plan* hip_plan = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  auto hip_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              hip_c->hip_device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);
  CHECK(hip_c->residues == cpu_c->residues);
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xaaaaaaaaaaaaaaaaull);
  REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  rns8::test::set_host_residues_current(*hip_c, false);
  REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK_FALSE(hip_c->host_residues_current);
  CHECK(hip_c->hip_export_buffer != nullptr);
  CHECK(hip_c->hip_status_buffer == nullptr);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 2) * limb_count)] == 0xaaaaaaaaaaaaaaaaull);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 2)] == UINT64_MAX);
  constexpr uint64_t signed_range_sentinel = 0x1212121212121212ull;
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), signed_range_sentinel);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), signed_range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(hip_c->hip_status_buffer != nullptr);
  CHECK(too_few_cpu == std::vector<uint64_t>(static_cast<std::size_t>(m * n), signed_range_sentinel));
  CHECK(too_few_hip == std::vector<uint64_t>(static_cast<std::size_t>(m * n), signed_range_sentinel));
  CHECK_FALSE(hip_c->host_residues_current);
  std::vector<uint64_t> invalid_limb_layout(66, signed_range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide GEMM rejects host-current stale device inputs") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide stale-input GEMM rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 2;
  constexpr int8_t c_sentinel = -41;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(-3), boost::multiprecision::cpp_int(5)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(7), boost::multiprecision::cpp_int(-11)});
    upload_exact_residues_to_hip(b_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    rns8::test::set_host_residues_current(*c_matrix, false);
    rns8::test::set_device_residues_current(*c_matrix, true);

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(a_matrix->host_residues_current);
    CHECK_FALSE(a_matrix->device_residues_current);
    CHECK_FALSE(b_matrix->host_residues_current);
    CHECK(b_matrix->device_residues_current);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(13), boost::multiprecision::cpp_int(17)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(19), boost::multiprecision::cpp_int(23)});
    upload_exact_residues_to_hip(a_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    rns8::test::set_host_residues_current(*c_matrix, false);
    rns8::test::set_device_residues_current(*c_matrix, true);

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK_FALSE(a_matrix->host_residues_current);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->host_residues_current);
    CHECK_FALSE(b_matrix->device_residues_current);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP exact-wide signed export matches CPU fixed-width boundary ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide signed export boundary smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  auto cpu_desc = exact_signed_desc(m, n, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_signed_desc(m, n, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(cpu_plan->prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  fill_exact_residue_matrix(cpu_c, {half, boost::multiprecision::cpp_int(-1)});
  fill_exact_residue_matrix(hip_c, {half, boost::multiprecision::cpp_int(-1)});
  upload_exact_residues_to_hip(hip_c);

  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0x9a9a9a9a9a9a9a9aull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0x9a9a9a9a9a9a9a9aull);
  REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + n) * limb_count)] == 0x9a9a9a9a9a9a9a9aull);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count)] ==
        std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 1)] ==
        std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 2)] ==
        std::numeric_limits<uint64_t>::max());

  std::vector<uint64_t> cpu_one_limb(static_cast<std::size_t>(m * n), 0x1010101010101010ull);
  std::vector<uint64_t> hip_one_limb(static_cast<std::size_t>(m * n), 0x1010101010101010ull);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_one_limb.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_one_limb.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one_limb == std::vector<uint64_t>(static_cast<std::size_t>(m * n), 0x1010101010101010ull));
  CHECK(hip_one_limb == std::vector<uint64_t>(static_cast<std::size_t>(m * n), 0x1010101010101010ull));

  std::vector<uint64_t> wrong_export(static_cast<std::size_t>(m * n * 2), 0x2020202020202020ull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, wrong_export.data(), n, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, wrong_export.data(), n, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(wrong_export == std::vector<uint64_t>(static_cast<std::size_t>(m * n * 2), 0x2020202020202020ull));

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide fixed limb export widths match CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide fixed limb-width smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(-1)});
  fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(-1)});
  upload_exact_residues_to_hip(hip_c);

  for (const uint32_t limb_count : {1u, 2u, 3u, 4u, 8u, 16u, 32u}) {
    std::vector<uint64_t> cpu_limbs(limb_count, 0);
    std::vector<uint64_t> hip_limbs(limb_count, 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), 1, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), 1, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    for (uint64_t limb : hip_limbs) {
      CHECK(limb == std::numeric_limits<uint64_t>::max());
    }
  }

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);

  auto cpu_unsigned_desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_unsigned_desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_unsigned_plan = nullptr;
  rns8_plan* hip_unsigned_plan = nullptr;
  rns8_matrix* cpu_unsigned_c = nullptr;
  rns8_matrix* hip_unsigned_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_unsigned_desc, &cpu_unsigned_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_unsigned_desc, &hip_unsigned_plan) == RNS8_SUCCESS);
  auto unsigned_c_desc = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &unsigned_c_desc, &cpu_unsigned_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &unsigned_c_desc, &hip_unsigned_c) == RNS8_SUCCESS);
  fill_exact_residue_matrix(cpu_unsigned_c, {boost::multiprecision::cpp_int(1)});
  fill_exact_residue_matrix(hip_unsigned_c, {boost::multiprecision::cpp_int(1)});
  upload_exact_residues_to_hip(hip_unsigned_c);

  for (const uint32_t limb_count : {1u, 2u, 3u, 4u, 8u, 16u, 32u}) {
    std::vector<uint64_t> cpu_limbs(limb_count, 0x4444444444444444ull);
    std::vector<uint64_t> hip_limbs(limb_count, 0x5555555555555555ull);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(
                cpu, cpu_unsigned_plan, cpu_unsigned_c, cpu_limbs.data(), 1, limb_count) == RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(
                hip, hip_unsigned_plan, hip_unsigned_c, hip_limbs.data(), 1, limb_count) == RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK(hip_limbs[0] == 1);
    for (uint32_t limb = 1; limb < limb_count; ++limb) {
      CHECK(hip_limbs[limb] == 0);
    }
  }

  rns8_destroy_matrix(hip_unsigned_c);
  rns8_destroy_matrix(cpu_unsigned_c);
  rns8_destroy_plan(hip_unsigned_plan);
  rns8_destroy_plan(cpu_unsigned_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide max-width padded export matches CPU ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide max-width padded export smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;
  constexpr int64_t limb_ld = 3;
  constexpr uint32_t limb_count = 32;

  {
    auto cpu_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(cpu_plan->prefix);
    const boost::multiprecision::cpp_int half = product / 2;
    fill_exact_residue_matrix(
        cpu_c,
        {half,
         boost::multiprecision::cpp_int(-1),
         -(boost::multiprecision::cpp_int(1) << 63u),
         boost::multiprecision::cpp_int(1)});
    fill_exact_residue_matrix(
        hip_c,
        {half,
         boost::multiprecision::cpp_int(-1),
         -(boost::multiprecision::cpp_int(1) << 63u),
         boost::multiprecision::cpp_int(1)});
    upload_exact_residues_to_hip(hip_c);

    constexpr uint64_t sentinel = 0x1919191919191919ull;
    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK_FALSE(hip_c->host_residues_current);
    for (int64_t row = 0; row < m; ++row) {
      const std::size_t padding = static_cast<std::size_t>((row * limb_ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(hip_limbs[padding + limb] == sentinel);
      }
    }
    const std::size_t minus_one = static_cast<std::size_t>((0 * limb_ld + 1) * limb_count);
    for (uint32_t limb = 0; limb < limb_count; ++limb) {
      CHECK(hip_limbs[minus_one + limb] == std::numeric_limits<uint64_t>::max());
    }

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  {
    auto cpu_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(
        cpu_c,
        {boost::multiprecision::cpp_int(0),
         boost::multiprecision::cpp_int(1) << 63u,
         boost::multiprecision::cpp_int(1) << 127u,
         boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max())});
    fill_exact_residue_matrix(
        hip_c,
        {boost::multiprecision::cpp_int(0),
         boost::multiprecision::cpp_int(1) << 63u,
         boost::multiprecision::cpp_int(1) << 127u,
         boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max())});
    upload_exact_residues_to_hip(hip_c);

    constexpr uint64_t sentinel = 0x2929292929292929ull;
    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK_FALSE(hip_c->host_residues_current);
    for (int64_t row = 0; row < m; ++row) {
      const std::size_t padding = static_cast<std::size_t>((row * limb_ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(hip_limbs[padding + limb] == sentinel);
      }
    }
    const std::size_t high_bit = static_cast<std::size_t>((1 * limb_ld + 0) * limb_count);
    CHECK(hip_limbs[high_bit] == 0);
    CHECK(hip_limbs[high_bit + 1] == (uint64_t{1} << 63u));
    for (uint32_t limb = 2; limb < limb_count; ++limb) {
      CHECK(hip_limbs[high_bit + limb] == 0);
    }

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

