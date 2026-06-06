TEST_CASE("direct HIP wrap64 colpair path matches CPU for odd column tail") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP colpair smoke");
  }

  constexpr int64_t m = 256;
  constexpr int64_t n = 257;
  constexpr int64_t k = 5;
  constexpr int64_t lda = 7;
  constexpr int64_t ldb = 259;
  constexpr int64_t ldc = 260;
  constexpr uint64_t c_sentinel = 0xcdcdcdcdcdcdcdcdull;
  const std::string expected_kernel = "direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5";
  ScopedEnvVar enable_colpair("RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT", "1");

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          (static_cast<uint64_t>(row + 1) << 48u) ^ (static_cast<uint64_t>(col + 3) * 0x0101010101010101ull);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          (static_cast<uint64_t>(col + 5) << 40u) ^ (static_cast<uint64_t>(row + 7) * 0x1111111111111111ull);
    }
  }
  A[0] = std::numeric_limits<uint64_t>::max();
  A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = 0x8080808080808080ull;
  B[0] = 0x0102030405060708ull;
  B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = std::numeric_limits<uint64_t>::max();

  rns8_context* cpu = create_context(RNS8_BACKEND_WRAP64_BYTE_LIMB);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = wrap_desc(m, n, k, RNS8_BACKEND_WRAP64_BYTE_LIMB);
  auto hip_desc = wrap_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_out = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_out = nullptr;

  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info hip_info{};
  hip_info.struct_size = sizeof(hip_info);
  hip_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(hip_plan, &hip_info) == RNS8_SUCCESS);
  CHECK(std::string(hip_info.selected_kernel) == expected_kernel);
  CHECK(std::string(hip_info.autotune_key).find(std::string("kernel=") + expected_kernel + ";") !=
        std::string::npos);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 501) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 502) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 501) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 502) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);

  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  auto hip_gemm_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_gemm_events, "wrap64_byte_gemm36_colpair_2d_kernel"));

  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  for (int64_t row = 0; row < m; ++row) {
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n - 1)] ==
          rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, n - 1, k));
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
  }

  rns8_destroy_matrix(hip_out);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_out);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP public wrap64 path matches CPU for carry-heavy tiled export") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP carry smoke");
  }

  constexpr int64_t m = 17;
  constexpr int64_t n = 17;
  constexpr int64_t k = 19;
  constexpr int64_t lda = 23;
  constexpr int64_t ldb = 21;
  constexpr int64_t ldc = 20;
  constexpr uint64_t c_sentinel = 0x6464646464646464ull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  fill_wrap64_carry_heavy_inputs(A, m, k, lda, B, n, ldb);

  rns8_context* cpu = create_context(RNS8_BACKEND_WRAP64_BYTE_LIMB);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = wrap_desc(m, n, k, RNS8_BACKEND_WRAP64_BYTE_LIMB);
  auto hip_desc = wrap_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_out = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_out = nullptr;

  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 301) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 302) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 301) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 302) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_oneshot == cpu_oneshot);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(hip_c[out_index] == expected);
      CHECK(hip_oneshot[out_index] == expected);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      CHECK(hip_oneshot[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
  }

  rns8_destroy_matrix(hip_out);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_out);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP wrap64 rejects CRT-style descriptors") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP wrap64 descriptor rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {2};
  uint64_t C[] = {0};
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc = wrap_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  const auto rejected_start = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto check_reject_allocations_unchanged = [&] {
    const auto rejected_now = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(rejected_now.allocate_calls == rejected_start.allocate_calls);
    CHECK(rejected_now.free_calls == rejected_start.free_calls);
    CHECK(rejected_now.allocated_bytes == rejected_start.allocated_bytes);
  };
  auto check_invalid_wrap_descriptor = [&](const rns8_gemm_desc& bad_desc) {
    rns8_plan* rejected_plan = nullptr;
    CHECK(rns8_gemm_wrap_u64_oneshot(hip, &bad_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_create_plan(hip, &bad_desc, &rejected_plan) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected_plan == nullptr);
    if (rejected_plan) {
      rns8_destroy_plan(rejected_plan);
    }
    check_reject_allocations_unchanged();
  };
  auto check_invalid_wrap_access = [&](int64_t bad_lda, int64_t bad_ldb, int64_t bad_ldc) {
    C[0] = 0x123456789abcdef0ull;
    CHECK(rns8_gemm_wrap_u64_oneshot(hip, &desc, A, bad_lda, B, bad_ldb, C, bad_ldc) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(C[0] == 0x123456789abcdef0ull);
    check_reject_allocations_unchanged();
  };
  check_invalid_wrap_access(0, n, n);
  check_invalid_wrap_access(k, 0, n);
  check_invalid_wrap_access(k, n, 0);

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  check_invalid_wrap_descriptor(bounded_looking);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  check_invalid_wrap_descriptor(prefixed);

  auto bound_only = desc;
  bound_only.bound = 1;
  check_invalid_wrap_descriptor(bound_only);

  auto flagged = desc;
  flagged.flags = 1;
  check_invalid_wrap_descriptor(flagged);

  auto bad_tile_m = desc;
  bad_tile_m.tile_m = 32;
  check_invalid_wrap_descriptor(bad_tile_m);

  auto bad_tile_n = desc;
  bad_tile_n.tile_n = 96;
  check_invalid_wrap_descriptor(bad_tile_n);

  const uint64_t tile_bound = 1;
  auto tile_bounded = desc;
  tile_bounded.tile_bounds = &tile_bound;
  tile_bounded.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(tile_bounded);

  auto tile_count_only = desc;
  tile_count_only.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(tile_count_only);

  auto per_tile_wrap = desc;
  per_tile_wrap.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  per_tile_wrap.tile_m = 64;
  per_tile_wrap.tile_n = 64;
  per_tile_wrap.tile_bounds = &tile_bound;
  per_tile_wrap.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(per_tile_wrap);

  auto matrix = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto check_invalid_wrap_matrix_descriptor = [&](const rns8_matrix_desc& bad_desc) {
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &bad_desc, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
    if (storage) {
      rns8_destroy_matrix(storage);
    }
    check_reject_allocations_unchanged();
  };
  auto bounded_matrix_descriptor = matrix;
  bounded_matrix_descriptor.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  check_invalid_wrap_matrix_descriptor(bounded_matrix_descriptor);
  auto prefixed_matrix_descriptor = matrix;
  prefixed_matrix_descriptor.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  check_invalid_wrap_matrix_descriptor(prefixed_matrix_descriptor);
  auto flagged_matrix_descriptor = matrix;
  flagged_matrix_descriptor.flags = 1;
  check_invalid_wrap_matrix_descriptor(flagged_matrix_descriptor);
  auto bad_ld_matrix_descriptor = matrix;
  bad_ld_matrix_descriptor.logical_ld = -1;
  check_invalid_wrap_matrix_descriptor(bad_ld_matrix_descriptor);
  auto bad_tile_matrix_descriptor = matrix;
  bad_tile_matrix_descriptor.tile_m = 32;
  check_invalid_wrap_matrix_descriptor(bad_tile_matrix_descriptor);

  rns8_plan* valid_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &desc, &valid_plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(hip, valid_plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* wrap_a = nullptr;
  rns8_matrix* wrap_b = nullptr;
  rns8_matrix* wrap_c = nullptr;
  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &wrap_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &wrap_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &wrap_c) == RNS8_SUCCESS);
  CHECK_FALSE(wrap_a->host_byte_limbs_current);
  CHECK_FALSE(wrap_b->host_byte_limbs_current);
  CHECK_FALSE(wrap_c->host_byte_limbs_current);
  CHECK_FALSE(wrap_a->device_byte_limbs_current);
  CHECK_FALSE(wrap_b->device_byte_limbs_current);
  CHECK_FALSE(wrap_c->device_byte_limbs_current);
  const int64_t signed_A[] = {-1};
  uint64_t limbs[] = {0};
  int64_t signed_out[] = {0};
  CHECK(rns8_pack_i64(hip, wrap_a, signed_A, k, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_i64(hip, valid_plan, wrap_c, signed_out, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, valid_plan, wrap_c, limbs, n, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, valid_plan, wrap_c, limbs, n, 1) == RNS8_INVALID_ARGUMENT);

  REQUIRE(rns8_pack_u64(hip, wrap_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, wrap_b, B, n, 2) == RNS8_SUCCESS);

  rns8::test::set_host_byte_limbs_current(*wrap_a, true);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*wrap_a, false);

  auto bounded_workspace_desc = unsigned_desc(m, n, k, 2, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* bounded_plan = nullptr;
  rns8_workspace* bounded_workspace = nullptr;
  REQUIRE(rns8_create_plan(hip, &bounded_workspace_desc, &bounded_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, bounded_workspace) == RNS8_INVALID_ARGUMENT);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(bounded_plan);

  rns8::test::set_host_residues_current(*wrap_a, true);
  CHECK(rns8_pack_u64(hip, wrap_a, A, k, 3) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_residues_current(*wrap_a, false);

  rns8::test::set_device_residues_current(*wrap_b, true);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_device_residues_current(*wrap_b, false);

  rns8::test::set_host_byte_limbs_current(*wrap_a, true);
  rns8::test::set_device_byte_limbs_current(*wrap_a, false);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*wrap_a, false);
  rns8::test::set_device_byte_limbs_current(*wrap_a, true);

  REQUIRE(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, 0, 0, k));

  rns8_plan_tile_schedule_entry stale_entry{};
  stale_entry.struct_size = sizeof(stale_entry);
  stale_entry.abi_version = RNS8_ABI_VERSION;
  stale_entry.required_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  stale_entry.selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  const uint32_t original_tile_m = valid_plan->desc.tile_m;
  const uint32_t original_tile_n = valid_plan->desc.tile_n;
  auto check_rejects_plan_mutation = [&](auto mutate, auto restore) {
    mutate();
    CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
    restore();
  };
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED; },
      [&] { valid_plan->desc.bound_kind = RNS8_BOUND_NONE; });
  check_rejects_plan_mutation([&] { valid_plan->desc.bound = 1; }, [&] { valid_plan->desc.bound = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX; },
      [&] { valid_plan->desc.max_prefix = 0; });
  check_rejects_plan_mutation([&] { valid_plan->desc.flags = 1; }, [&] { valid_plan->desc.flags = 0; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_m = 64; }, [&] { valid_plan->desc.tile_m = original_tile_m; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_n = 64; }, [&] { valid_plan->desc.tile_n = original_tile_n; });
  check_rejects_plan_mutation([&] { valid_plan->prefix = 1; }, [&] { valid_plan->prefix = 0; });
  check_rejects_plan_mutation([&] { valid_plan->modulus_product = 1; }, [&] { valid_plan->modulus_product = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.tile_bounds = &tile_bound; },
      [&] { valid_plan->desc.tile_bounds = nullptr; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_bounds_count = 1; }, [&] { valid_plan->desc.tile_bounds_count = 0; });
  check_rejects_plan_mutation([&] { valid_plan->tile_bounds.push_back(1); }, [&] { valid_plan->tile_bounds.clear(); });
  check_rejects_plan_mutation(
      [&] { valid_plan->tile_schedule.push_back(stale_entry); },
      [&] { valid_plan->tile_schedule.clear(); });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_rows; }, [&] { --valid_plan->schedule_tile_rows; });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_cols; }, [&] { --valid_plan->schedule_tile_cols; });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_count; }, [&] { --valid_plan->schedule_tile_count; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_min_required_prefix = 1; },
      [&] { valid_plan->schedule_min_required_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_max_required_prefix = 1; },
      [&] { valid_plan->schedule_max_required_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_min_selected_prefix = 1; },
      [&] { valid_plan->schedule_min_selected_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_max_selected_prefix = 1; },
      [&] { valid_plan->schedule_max_selected_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_prefix_group_count = 1; },
      [&] { valid_plan->schedule_prefix_group_count = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_range_bit_length = 64; },
      [&] { valid_plan->schedule_range_bit_length = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_adaptive_prefix_active = 1; },
      [&] { valid_plan->schedule_adaptive_prefix_active = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_adaptive_skip_active = 1; },
      [&] { valid_plan->schedule_adaptive_skip_active = 0; });
  check_rejects_plan_mutation([&] { valid_plan->schedule_flags = 1; }, [&] { valid_plan->schedule_flags = 0; });

  REQUIRE(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, 0, 0, k));

  rns8::test::set_host_residues_current(*wrap_c, true);
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_residues_current(*wrap_c, false);

  rns8::test::set_host_byte_limbs_current(*wrap_c, true);
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*wrap_c, false);

  rns8::test::set_host_byte_limbs_current(*wrap_c, true);
  rns8::test::set_device_byte_limbs_current(*wrap_c, false);
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*wrap_c, false);
  rns8::test::set_device_byte_limbs_current(*wrap_c, true);

  rns8_matrix* residue_a = nullptr;
  rns8_matrix* residue_b = nullptr;
  rns8_matrix* residue_c = nullptr;
  auto residue_a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto residue_b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto residue_c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(hip, &residue_a_desc, &residue_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &residue_b_desc, &residue_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &residue_c_desc, &residue_c) == RNS8_SUCCESS);
  CHECK(residue_a->hip_residues != nullptr);
  CHECK(residue_a->hip_byte_limbs == nullptr);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, residue_a, residue_b, residue_c, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(hip, valid_plan, residue_c, C, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(residue_c);
  rns8_destroy_matrix(residue_b);
  rns8_destroy_matrix(residue_a);
  rns8_destroy_matrix(wrap_c);
  rns8_destroy_matrix(wrap_b);
  rns8_destroy_matrix(wrap_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(valid_plan);

  rns8_destroy_context(hip);
}

