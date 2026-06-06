TEST_CASE("public wrap64 CPU resident path covers large padded tile-tail full-width data") {
  constexpr int64_t m = 17;
  constexpr int64_t n = 19;
  constexpr int64_t k = 33;
  constexpr int64_t lda = 41;
  constexpr int64_t ldb = 29;
  constexpr int64_t ldc = 23;
  constexpr uint64_t a_padding = 0xaaaaaaaaaaaaaaaaull;
  constexpr uint64_t b_padding = 0xbbbbbbbbbbbbbbbbull;
  constexpr uint64_t c_sentinel = 0xcdcdcdcdcdcdcdcdull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), a_padding);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), b_padding);
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);

  auto fill_inputs = [&](uint64_t salt) {
    std::mt19937_64 rng(0x7772617036345f6cull ^ salt);
    std::fill(A.begin(), A.end(), a_padding ^ salt);
    std::fill(B.begin(), B.end(), b_padding ^ (salt << 1u));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        uint64_t value = rng() ^ (salt + static_cast<uint64_t>(row * 131 + col * 17));
        if ((row + col) % 13 == 0) {
          value = std::numeric_limits<uint64_t>::max();
        } else if ((row + col) % 13 == 1) {
          value = 0;
        } else if ((row + col) % 13 == 2) {
          value = 0x8080808080808080ull;
        } else if ((row + col) % 13 == 3) {
          value = 0x7f807f807f807f80ull;
        }
        A[static_cast<std::size_t>(row * lda + col)] = value;
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        uint64_t value = rng() ^ (salt + static_cast<uint64_t>(row * 257 + col * 29));
        if ((row * 3 + col) % 17 == 0) {
          value = std::numeric_limits<uint64_t>::max() - 1;
        } else if ((row * 3 + col) % 17 == 1) {
          value = 1;
        } else if ((row * 3 + col) % 17 == 2) {
          value = 0xfefdfcfbfaf9f8f7ull;
        } else if ((row * 3 + col) % 17 == 3) {
          value = 0x0102030405060708ull;
        }
        B[static_cast<std::size_t>(row * ldb + col)] = value;
      }
    }
  };

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  const auto* a_bytes = a_matrix->byte_limbs.data();
  const auto* b_bytes = b_matrix->byte_limbs.data();
  const auto* c_bytes = c_matrix->byte_limbs.data();
  REQUIRE(a_matrix->byte_limbs.size() == static_cast<std::size_t>(m * k * 8));
  REQUIRE(b_matrix->byte_limbs.size() == static_cast<std::size_t>(k * n * 8));
  REQUIRE(c_matrix->byte_limbs.size() == static_cast<std::size_t>(m * n * 8));
  CHECK(a_matrix->residues.empty());
  CHECK(b_matrix->residues.empty());
  CHECK(c_matrix->residues.empty());

  auto run_and_check = [&](uint64_t a_version, uint64_t b_version) {
    std::fill(C.begin(), C.end(), c_sentinel);
    std::fill(oneshot.begin(), oneshot.end(), c_sentinel);
    REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), ldb, b_version) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, oneshot.data(), ldc) ==
            RNS8_SUCCESS);
    CHECK(a_matrix->byte_limbs.data() == a_bytes);
    CHECK(b_matrix->byte_limbs.data() == b_bytes);
    CHECK(c_matrix->byte_limbs.data() == c_bytes);
    CHECK(a_matrix->source_version == a_version);
    CHECK(b_matrix->source_version == b_version);
    CHECK(C == oneshot);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t kk = 0; kk < k; ++kk) {
        CHECK(load_compact_u64_limbs(a_matrix->byte_limbs, row * k + kk) ==
              A[static_cast<std::size_t>(row * lda + kk)]);
      }
    }
    for (int64_t kk = 0; kk < k; ++kk) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(load_compact_u64_limbs(b_matrix->byte_limbs, kk * n + col) ==
              B[static_cast<std::size_t>(kk * ldb + col)]);
      }
    }
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
        const uint64_t expected = expected_wrap_cell(A, lda, B, ldb, row, col, k);
        CHECK(C[out_index] == expected);
        CHECK(
            C[out_index] ==
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
        CHECK(C[out_index] ==
              rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  };

  fill_inputs(0x1234);
  run_and_check(301, 302);
  fill_inputs(0x5678);
  run_and_check(401, 402);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 CPU path rejects residue-backed and stale byte-limb matrices") {
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {3};

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  c_matrix->residues.assign(1, 0);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  c_matrix->residues.clear();

  rns8::test::set_host_residues_current(*a_matrix, true);
  CHECK(rns8_pack_u64(ctx, a_matrix, A, k, 3) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_residues_current(*a_matrix, false);

  rns8::test::set_device_residues_current(*b_matrix, true);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_device_residues_current(*b_matrix, false);

  rns8::test::set_device_byte_limbs_current(*c_matrix, true);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_device_byte_limbs_current(*c_matrix, false);

  rns8::test::set_host_residues_current(*c_matrix, true);
  uint64_t C[] = {0};
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_residues_current(*c_matrix, false);

  c_matrix->byte_limbs.clear();
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  c_matrix->byte_limbs.assign(static_cast<std::size_t>(m * n * 8), 0);

  rns8::test::set_host_byte_limbs_current(*a_matrix, false);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*a_matrix, true);

  rns8::test::set_host_byte_limbs_current(*c_matrix, false);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_host_byte_limbs_current(*c_matrix, true);

  a_matrix->hip_upload_buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  a_matrix->hip_upload_bytes = 8;
  CHECK(rns8_pack_u64(ctx, a_matrix, A, k, 4) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  a_matrix->hip_upload_buffer = nullptr;
  a_matrix->hip_upload_bytes = 0;

  c_matrix->hip_export_buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  c_matrix->hip_export_bytes = 8;
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  c_matrix->hip_export_buffer = nullptr;
  c_matrix->hip_export_bytes = 0;

  c_matrix->hip_status_buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  c_matrix->hip_status_bytes = 4;
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  c_matrix->hip_status_buffer = nullptr;
  c_matrix->hip_status_bytes = 0;

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 CPU path rejects stale bounded schedule metadata") {
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {3};

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  const uint32_t original_tile_m = plan->desc.tile_m;
  const uint32_t original_tile_n = plan->desc.tile_n;
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 2) == RNS8_SUCCESS);

  rns8_plan_tile_schedule_entry stale_entry{};
  stale_entry.struct_size = sizeof(stale_entry);
  stale_entry.abi_version = RNS8_ABI_VERSION;
  stale_entry.required_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  stale_entry.selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  auto check_gemm_rejects_plan_mutation = [&](auto mutate, auto restore) {
    mutate();
    CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    restore();
  };
  check_gemm_rejects_plan_mutation(
      [&] { plan->desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED; },
      [&] { plan->desc.bound_kind = RNS8_BOUND_NONE; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.bound = 1; }, [&] { plan->desc.bound = 0; });
  check_gemm_rejects_plan_mutation(
      [&] { plan->desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX; },
      [&] { plan->desc.max_prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.flags = 1; }, [&] { plan->desc.flags = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.tile_m = 64; }, [&] { plan->desc.tile_m = original_tile_m; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.tile_n = 64; }, [&] { plan->desc.tile_n = original_tile_n; });
  check_gemm_rejects_plan_mutation([&] { plan->prefix = 1; }, [&] { plan->prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->modulus_product = 1; }, [&] { plan->modulus_product = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.tile_bounds = A; }, [&] { plan->desc.tile_bounds = nullptr; });
  check_gemm_rejects_plan_mutation([&] { plan->desc.tile_bounds_count = 1; }, [&] { plan->desc.tile_bounds_count = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->tile_bounds.push_back(1); }, [&] { plan->tile_bounds.clear(); });
  check_gemm_rejects_plan_mutation(
      [&] { plan->tile_schedule.push_back(stale_entry); },
      [&] { plan->tile_schedule.clear(); });
  check_gemm_rejects_plan_mutation([&] { ++plan->schedule_tile_rows; }, [&] { --plan->schedule_tile_rows; });
  check_gemm_rejects_plan_mutation([&] { ++plan->schedule_tile_cols; }, [&] { --plan->schedule_tile_cols; });
  check_gemm_rejects_plan_mutation([&] { ++plan->schedule_tile_count; }, [&] { --plan->schedule_tile_count; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_min_required_prefix = 1; }, [&] { plan->schedule_min_required_prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_max_required_prefix = 1; }, [&] { plan->schedule_max_required_prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_min_selected_prefix = 1; }, [&] { plan->schedule_min_selected_prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_max_selected_prefix = 1; }, [&] { plan->schedule_max_selected_prefix = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_prefix_group_count = 1; }, [&] { plan->schedule_prefix_group_count = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_range_bit_length = 64; }, [&] { plan->schedule_range_bit_length = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_adaptive_prefix_active = 1; }, [&] { plan->schedule_adaptive_prefix_active = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_adaptive_skip_active = 1; }, [&] { plan->schedule_adaptive_skip_active = 0; });
  check_gemm_rejects_plan_mutation([&] { plan->schedule_flags = 1; }, [&] { plan->schedule_flags = 0; });

  REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  uint64_t C[] = {0};

  auto check_export_rejects_plan_mutation = [&](auto mutate, auto restore) {
    mutate();
    CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
    restore();
  };
  check_export_rejects_plan_mutation(
      [&] { plan->desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED; },
      [&] { plan->desc.bound_kind = RNS8_BOUND_NONE; });
  check_export_rejects_plan_mutation([&] { plan->desc.bound = 1; }, [&] { plan->desc.bound = 0; });
  check_export_rejects_plan_mutation(
      [&] { plan->desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX; },
      [&] { plan->desc.max_prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->desc.flags = 1; }, [&] { plan->desc.flags = 0; });
  check_export_rejects_plan_mutation([&] { plan->desc.tile_m = 64; }, [&] { plan->desc.tile_m = original_tile_m; });
  check_export_rejects_plan_mutation([&] { plan->desc.tile_n = 64; }, [&] { plan->desc.tile_n = original_tile_n; });
  check_export_rejects_plan_mutation([&] { plan->prefix = 1; }, [&] { plan->prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->modulus_product = 1; }, [&] { plan->modulus_product = 0; });
  check_export_rejects_plan_mutation([&] { plan->desc.tile_bounds = A; }, [&] { plan->desc.tile_bounds = nullptr; });
  check_export_rejects_plan_mutation([&] { plan->desc.tile_bounds_count = 1; }, [&] { plan->desc.tile_bounds_count = 0; });
  check_export_rejects_plan_mutation([&] { plan->tile_bounds.push_back(1); }, [&] { plan->tile_bounds.clear(); });
  check_export_rejects_plan_mutation(
      [&] { plan->tile_schedule.push_back(stale_entry); },
      [&] { plan->tile_schedule.clear(); });
  check_export_rejects_plan_mutation([&] { ++plan->schedule_tile_rows; }, [&] { --plan->schedule_tile_rows; });
  check_export_rejects_plan_mutation([&] { ++plan->schedule_tile_cols; }, [&] { --plan->schedule_tile_cols; });
  check_export_rejects_plan_mutation([&] { ++plan->schedule_tile_count; }, [&] { --plan->schedule_tile_count; });
  check_export_rejects_plan_mutation([&] { plan->schedule_min_required_prefix = 1; }, [&] { plan->schedule_min_required_prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_max_required_prefix = 1; }, [&] { plan->schedule_max_required_prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_min_selected_prefix = 1; }, [&] { plan->schedule_min_selected_prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_max_selected_prefix = 1; }, [&] { plan->schedule_max_selected_prefix = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_prefix_group_count = 1; }, [&] { plan->schedule_prefix_group_count = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_range_bit_length = 64; }, [&] { plan->schedule_range_bit_length = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_adaptive_prefix_active = 1; }, [&] { plan->schedule_adaptive_prefix_active = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_adaptive_skip_active = 1; }, [&] { plan->schedule_adaptive_skip_active = 0; });
  check_export_rejects_plan_mutation([&] { plan->schedule_flags = 1; }, [&] { plan->schedule_flags = 0; });

  REQUIRE(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_SUCCESS);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 path rejects CRT metadata and RNS APIs") {
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {2};
  uint64_t C[] = {0};

  rns8_context* wrap_ctx = create_wrap64();
  rns8_context* cpu_ctx = create_cpu();
  auto desc = wrap_desc(m, n, k);

  CHECK(rns8_gemm_wrap_u64_oneshot(cpu_ctx, &desc, A, k, B, n, C, n) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_gemm_u64_oneshot(wrap_ctx, &desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  const int64_t signed_A[] = {-1};
  const int64_t signed_B[] = {2};
  int64_t signed_C[] = {0};
  CHECK(rns8_gemm_i64_oneshot(wrap_ctx, &desc, signed_A, k, signed_B, n, signed_C, n) == RNS8_INVALID_ARGUMENT);

  rns8_plan* plan = nullptr;
  auto check_invalid_wrap_descriptor = [&](const rns8_gemm_desc& bad_desc) {
    rns8_plan* rejected_plan = nullptr;
    CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bad_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_wrap_u64_oneshot(cpu_ctx, &bad_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_create_plan(wrap_ctx, &bad_desc, &rejected_plan) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected_plan == nullptr);
    if (rejected_plan) {
      rns8_destroy_plan(rejected_plan);
      rejected_plan = nullptr;
    }
    CHECK(rns8_create_plan(cpu_ctx, &bad_desc, &rejected_plan) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected_plan == nullptr);
    if (rejected_plan) {
      rns8_destroy_plan(rejected_plan);
    }
  };

  auto check_invalid_wrap_access = [&](int64_t bad_lda, int64_t bad_ldb, int64_t bad_ldc) {
    C[0] = 0x123456789abcdef0ull;
    CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &desc, A, bad_lda, B, bad_ldb, C, bad_ldc) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(C[0] == 0x123456789abcdef0ull);
    CHECK(rns8_gemm_wrap_u64_oneshot(cpu_ctx, &desc, A, bad_lda, B, bad_ldb, C, bad_ldc) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(C[0] == 0x123456789abcdef0ull);
  };
  check_invalid_wrap_access(0, n, n);
  check_invalid_wrap_access(k, 0, n);
  check_invalid_wrap_access(k, n, 0);

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  check_invalid_wrap_descriptor(bounded_looking);

  auto malformed_with_future_backend = bounded_looking;
  malformed_with_future_backend.requested_backend = RNS8_BACKEND_HIPBLASLT;
  check_invalid_wrap_descriptor(malformed_with_future_backend);

  auto valid_with_future_backend = desc;
  valid_with_future_backend.requested_backend = RNS8_BACKEND_HIPBLASLT;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &valid_with_future_backend, A, k, B, n, C, n) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_create_plan(wrap_ctx, &valid_with_future_backend, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto auto_backend = desc;
  auto_backend.requested_backend = RNS8_BACKEND_AUTO;
  CHECK(rns8_gemm_wrap_u64_oneshot(cpu_ctx, &auto_backend, A, k, B, n, C, n) == RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &auto_backend, A, k, B, n, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == std::numeric_limits<uint64_t>::max() - 1u);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  check_invalid_wrap_descriptor(prefixed);

  CHECK(rns8_create_plan(cpu_ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

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

  rns8_gemm_desc bounded_desc{};
  bounded_desc.struct_size = sizeof(bounded_desc);
  bounded_desc.abi_version = RNS8_ABI_VERSION;
  bounded_desc.semantics = RNS8_BOUNDED_U64;
  bounded_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  bounded_desc.m = m;
  bounded_desc.n = n;
  bounded_desc.k = k;
  bounded_desc.bound = 2;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bounded_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_create_plan(wrap_ctx, &bounded_desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto cpu_bounded_desc = bounded_desc;
  cpu_bounded_desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_plan* bounded_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu_ctx, &cpu_bounded_desc, &bounded_plan) == RNS8_SUCCESS);
  rns8_workspace* bounded_workspace = nullptr;
  REQUIRE(rns8_create_workspace(cpu_ctx, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);

  rns8_matrix_desc matrix{};
  matrix.struct_size = sizeof(matrix);
  matrix.abi_version = RNS8_ABI_VERSION;
  matrix.rows = 1;
  matrix.cols = 1;
  matrix.logical_ld = 1;
  matrix.semantics = RNS8_WRAP_U64_MOD_2_64;
  matrix.bound_kind = RNS8_BOUND_NONE;
  matrix.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  rns8_matrix* storage = nullptr;
  auto check_invalid_wrap_matrix_descriptor = [&](const rns8_matrix_desc& bad_desc) {
    CHECK(rns8_create_matrix(wrap_ctx, &bad_desc, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
    if (storage) {
      rns8_destroy_matrix(storage);
      storage = nullptr;
    }
    CHECK(rns8_create_matrix(cpu_ctx, &bad_desc, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
    if (storage) {
      rns8_destroy_matrix(storage);
      storage = nullptr;
    }
  };
  CHECK(rns8_create_matrix(cpu_ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);
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
  REQUIRE(rns8_create_plan(wrap_ctx, &desc, &valid_plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(wrap_ctx, valid_plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  REQUIRE(rns8_create_matrix(wrap_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  uint64_t limbs[] = {0};
  CHECK(rns8_pack_i64(wrap_ctx, a_matrix, signed_A, k, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(wrap_ctx, valid_plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  int64_t signed_out[] = {0};
  CHECK(rns8_export_i64(wrap_ctx, valid_plan, c_matrix, signed_out, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(wrap_ctx, valid_plan, c_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(wrap_ctx, valid_plan, c_matrix, limbs, n, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(wrap_ctx, valid_plan, c_matrix, limbs, n, 1) == RNS8_INVALID_ARGUMENT);

  auto bounded_matrix = wrap_matrix_desc(m, n);
  bounded_matrix.semantics = RNS8_BOUNDED_U64;
  bounded_matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_create_matrix(wrap_ctx, &bounded_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);

  rns8_matrix* cpu_bounded_matrix = nullptr;
  REQUIRE(rns8_create_matrix(cpu_ctx, &bounded_matrix, &cpu_bounded_matrix) == RNS8_SUCCESS);
  CHECK(cpu_bounded_matrix->residues.size() == static_cast<std::size_t>(RNS8_DEFAULT_BOUNDED_PREFIX));
  CHECK(cpu_bounded_matrix->byte_limbs.empty());
  CHECK(rns8_gemm_wrap_u64(wrap_ctx, valid_plan, a_matrix, b_matrix, c_matrix, bounded_workspace) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(wrap_ctx, valid_plan, cpu_bounded_matrix, C, n) == RNS8_INVALID_ARGUMENT);
  rns8_destroy_matrix(cpu_bounded_matrix);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(valid_plan);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(bounded_plan);

  rns8_destroy_context(cpu_ctx);
  rns8_destroy_context(wrap_ctx);
}
