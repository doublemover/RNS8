TEST_CASE("direct HIP exact-wide large padded export staging preserves ABI and elides status") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide large padded export staging smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 56;
  constexpr int64_t n = 56;
  constexpr int64_t k = 1;
  constexpr int64_t limb_ld = 61;
  constexpr uint32_t limb_count = 3;
  static_assert(m * n * limb_count * static_cast<int64_t>(sizeof(uint64_t)) > 64 * 1024);

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

  std::vector<boost::multiprecision::cpp_int> values;
  values.reserve(static_cast<std::size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int value =
          ((row - col) * 257) + static_cast<int64_t>((row * col) % 97);
      if ((row + col) % 19 == 0) {
        value = -value - 1;
      }
      values.push_back(value);
    }
  }
  fill_exact_residue_matrix(cpu_c, values);
  fill_exact_residue_matrix(hip_c, values);
  upload_exact_residues_to_hip(hip_c);

  constexpr uint64_t sentinel = 0x4848484848484848ull;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
  REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);

  {
    ScopedEnvVar disable_staging("RNS8_HIP_PINNED_EXPORT_STAGING", "0");
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    const auto export_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);

    CHECK(has_timing_label(export_events, "exact_wide_export_kernel"));
    CHECK(has_timing_label(export_events, "exact_wide_export_d2h"));
    CHECK_FALSE(has_timing_label(export_events, "export_host_staging_copy"));
    CHECK_FALSE(has_timing_label(export_events, "exact_wide_export_status_memset"));
    CHECK_FALSE(has_timing_label(export_events, "exact_wide_export_status_d2h"));
    CHECK(hip_limbs == cpu_limbs);
  }

  {
    ScopedEnvVar force_staging("RNS8_HIP_PINNED_EXPORT_STAGING", "1");
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    const auto export_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);

    CHECK(has_timing_label(export_events, "exact_wide_export_kernel"));
    CHECK(has_timing_label(export_events, "exact_wide_export_d2h"));
    CHECK(has_timing_label(export_events, "export_host_staging_copy"));
    CHECK_FALSE(has_timing_label(export_events, "exact_wide_export_status_memset"));
    CHECK_FALSE(has_timing_label(export_events, "exact_wide_export_status_d2h"));
    CHECK(hip_c->hip_status_buffer == nullptr);
    CHECK(hip_limbs == cpu_limbs);
    for (int64_t row = 0; row < m; ++row) {
      const std::size_t padding = static_cast<std::size_t>((row * limb_ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(hip_limbs[padding + limb] == sentinel);
      }
    }
  }

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide export requires device-current resident residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide device-current export smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = exact_signed_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(-1), boost::multiprecision::cpp_int(1)});

    constexpr uint64_t sentinel = 0x6a6a6a6a6a6a6a6aull;
    std::vector<uint64_t> limbs(4, sentinel);
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs == std::vector<uint64_t>(4, sentinel));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    upload_exact_residues_to_hip(c_matrix);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_SUCCESS);
    CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());
    CHECK(limbs[1] == std::numeric_limits<uint64_t>::max());
    CHECK(limbs[2] == 1);
    CHECK(limbs[3] == 0);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});

    constexpr uint64_t sentinel = 0x7b7b7b7b7b7b7b7bull;
    std::vector<uint64_t> limbs(4, sentinel);
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs == std::vector<uint64_t>(4, sentinel));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    upload_exact_residues_to_hip(c_matrix);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_SUCCESS);
    CHECK(limbs[0] == 1);
    CHECK(limbs[1] == 0);
    CHECK(limbs[2] == 0);
    CHECK(limbs[3] == 0);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP exact-wide unsigned export matches CPU fixed-width boundary ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned export boundary smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int two_to_64 = boost::multiprecision::cpp_int(1) << 64;
  const boost::multiprecision::cpp_int max_u64 = two_to_64 - 1;
  fill_exact_residue_matrix(cpu_c, {two_to_64, max_u64});
  fill_exact_residue_matrix(hip_c, {two_to_64, max_u64});
  upload_exact_residues_to_hip(hip_c);

  constexpr uint32_t limb_count = 2;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(limb_ld * limb_count), 0x4b4b4b4b4b4b4b4bull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(limb_ld * limb_count), 0x4b4b4b4b4b4b4b4bull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[0] == 0);
  CHECK(hip_limbs[1] == 1);
  CHECK(hip_limbs[2] == std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[3] == 0);
  CHECK(hip_limbs[static_cast<std::size_t>(2 * limb_count)] == 0x4b4b4b4b4b4b4b4bull);
  CHECK(hip_limbs[static_cast<std::size_t>(2 * limb_count + 1)] == 0x4b4b4b4b4b4b4b4bull);

  std::vector<uint64_t> cpu_one_limb(2, 0x5151515151515151ull);
  std::vector<uint64_t> hip_one_limb(2, 0x5151515151515151ull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_one_limb.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_one_limb.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one_limb == std::vector<uint64_t>(2, 0x5151515151515151ull));
  CHECK(hip_one_limb == std::vector<uint64_t>(2, 0x5151515151515151ull));

  std::vector<uint64_t> cpu_wide(64, 0x6262626262626262ull);
  std::vector<uint64_t> hip_wide(64, 0x6262626262626262ull);
  fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});
  fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});
  upload_exact_residues_to_hip(hip_c);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_wide.data(), 2, 32) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_wide.data(), 2, 32) == RNS8_SUCCESS);
  CHECK(hip_wide == cpu_wide);
  CHECK(hip_wide[0] == 1);
  for (std::size_t limb = 1; limb < 32; ++limb) {
    CHECK(hip_wide[limb] == 0);
  }

  std::vector<uint64_t> wrong_export(4, 0x7373737373737373ull);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, wrong_export.data(), 2, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, wrong_export.data(), 2, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(wrong_export == std::vector<uint64_t>(4, 0x7373737373737373ull));

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide unsigned high-bit magnitude matches CPU fixed-width ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned high-bit smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int high_bit = boost::multiprecision::cpp_int(1) << 127u;
  fill_exact_residue_matrix(cpu_c, {high_bit, boost::multiprecision::cpp_int(0)});
  fill_exact_residue_matrix(hip_c, {high_bit, boost::multiprecision::cpp_int(0)});
  upload_exact_residues_to_hip(hip_c);

  std::vector<uint64_t> cpu_two(4, 0x8c8c8c8c8c8c8c8cull);
  std::vector<uint64_t> hip_two(4, 0x8c8c8c8c8c8c8c8cull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_two.data(), 2, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_two.data(), 2, 2) == RNS8_SUCCESS);
  CHECK(hip_two == cpu_two);
  CHECK(hip_two[0] == 0);
  CHECK(hip_two[1] == (uint64_t{1} << 63u));
  CHECK(hip_two[2] == 0);
  CHECK(hip_two[3] == 0);

  std::vector<uint64_t> cpu_one(2, 0x9d9d9d9d9d9d9d9dull);
  std::vector<uint64_t> hip_one(2, 0x9d9d9d9d9d9d9d9dull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_one.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_one.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one == std::vector<uint64_t>(2, 0x9d9d9d9d9d9d9d9dull));
  CHECK(hip_one == std::vector<uint64_t>(2, 0x9d9d9d9d9d9d9d9dull));

  std::vector<uint64_t> cpu_wide(64, 0xaeaeaeaeaeaeaeaeull);
  std::vector<uint64_t> hip_wide(64, 0xaeaeaeaeaeaeaeaeull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_wide.data(), 2, 32) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_wide.data(), 2, 32) == RNS8_SUCCESS);
  CHECK(hip_wide == cpu_wide);
  CHECK(hip_wide[0] == 0);
  CHECK(hip_wide[1] == (uint64_t{1} << 63u));
  for (std::size_t limb = 2; limb < 32; ++limb) {
    CHECK(hip_wide[limb] == 0);
  }

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide descriptors reject stale bounded metadata") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide descriptor rejection smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr uint64_t tile_bound = 1;

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.tile_bounds = &tile_bound;
    desc.tile_bounds_count = 1;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.tile_bounds = &tile_bound;
    desc.tile_bounds_count = 1;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* storage = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(hip, &matrix, &storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xcdcdcdcdcdcdcdcdull, 0xcecececececececeull};
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcdcdcdcdcdcdcdcdull);
    CHECK(limbs[1] == 0xcecececececececeull);
    rns8_destroy_matrix(storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* storage = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(hip, &matrix, &storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xdfdfdfdfdfdfdfdfull, 0xe0e0e0e0e0e0e0e0ull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xdfdfdfdfdfdfdfdfull);
    CHECK(limbs[1] == 0xe0e0e0e0e0e0e0e0ull);
    rns8_destroy_matrix(storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);

    auto bounded = matrix_desc(1, 1, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* bounded_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &bounded, &bounded_storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xababababababababull, 0xbcbcbcbcbcbcbcbcull};
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, bounded_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xababababababababull);
    CHECK(limbs[1] == 0xbcbcbcbcbcbcbcbcull);

    auto wrap = matrix_desc(1, 1, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    rns8_matrix* wrap_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &wrap, &wrap_storage) == RNS8_SUCCESS);
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, wrap_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xababababababababull);
    CHECK(limbs[1] == 0xbcbcbcbcbcbcbcbcull);

    rns8_destroy_matrix(wrap_storage);
    rns8_destroy_matrix(bounded_storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);

    auto bounded = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* bounded_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &bounded, &bounded_storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xcbcbcbcbcbcbcbcbull, 0xdcdcdcdcdcdcdcdcull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, bounded_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcbcbcbcbcbcbcbcbull);
    CHECK(limbs[1] == 0xdcdcdcdcdcdcdcdcull);

    auto wrap = matrix_desc(1, 1, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    rns8_matrix* wrap_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &wrap, &wrap_storage) == RNS8_SUCCESS);
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, wrap_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcbcbcbcbcbcbcbcbull);
    CHECK(limbs[1] == 0xdcdcdcdcdcdcdcdcull);

    rns8_destroy_matrix(wrap_storage);
    rns8_destroy_matrix(bounded_storage);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP exact-wide unsigned RNS output matches CPU residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 19};
  const uint64_t B[] = {
      std::numeric_limits<uint64_t>::max() - 3,
      std::numeric_limits<uint64_t>::max() - 5,
      0x8080808080808080ull,
      0x0102030405060708ull};

  rns8_plan* cpu_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  auto cpu_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);

  rns8_plan* hip_plan = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  auto hip_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              hip_c->hip_device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);
  CHECK(hip_c->residues == cpu_c->residues);
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xbbbbbbbbbbbbbbbbull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  rns8::test::set_host_residues_current(*hip_c, false);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK_FALSE(hip_c->host_residues_current);
  CHECK(hip_c->hip_export_buffer != nullptr);
  CHECK(hip_c->hip_status_buffer == nullptr);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 2) * limb_count)] == 0xbbbbbbbbbbbbbbbbull);
  constexpr uint64_t unsigned_range_sentinel = 0x3434343434343434ull;
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(hip_c->hip_status_buffer != nullptr);
  CHECK(too_few_cpu == std::vector<uint64_t>(static_cast<std::size_t>(m * n), unsigned_range_sentinel));
  CHECK(too_few_hip == std::vector<uint64_t>(static_cast<std::size_t>(m * n), unsigned_range_sentinel));
  CHECK_FALSE(hip_c->host_residues_current);
  std::vector<uint64_t> invalid_limb_layout(66, unsigned_range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), n - 1, limb_count) ==
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

