TEST_CASE("direct HIP bounded per-tile workspace and matrix schedule metadata are part of residency contract") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded per-tile schedule contract smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int64_t lda = 1;
  constexpr int64_t ldb = n;
  constexpr int64_t ldc = n;
  constexpr uint64_t sentinel = 0xdadadadadadadadaull;
  const std::vector<uint64_t> bounds_a = {0, 0, 7000000, 1000000000};
  const std::vector<uint64_t> bounds_b = {0, 0, 7000000, 999999999};
  const std::vector<uint64_t> bounds_tile = {1000000000};

  auto make_desc = [&](const std::vector<uint64_t>& bounds, uint32_t tile_m, uint32_t tile_n, rns8_backend_kind backend) {
    auto desc = unsigned_desc(m, n, k, 0, backend);
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
    desc.tile_m = tile_m;
    desc.tile_n = tile_n;
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = static_cast<uint64_t>(bounds.size());
    return desc;
  };

  std::vector<uint64_t> A(static_cast<std::size_t>(m), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(n), 7);
  A.back() = 1000000;
  B.back() = 1000;
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = make_desc(bounds_a, 64, 64, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = make_desc(bounds_a, 64, 64, RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc_b = make_desc(bounds_b, 64, 64, RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc_tile = make_desc(bounds_tile, 128, 128, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_plan* plan_b = nullptr;
  rns8_plan* plan_tile = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_workspace* workspace_b = nullptr;
  rns8_workspace* workspace_tile = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* wrong_a_tile = nullptr;
  rns8_matrix* wrong_c_tile = nullptr;

  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(plan->tile_schedule.empty());
  CHECK(plan->tile_schedule[0].range_bit_length == 0);
  REQUIRE(rns8_create_plan(hip, &hip_desc_b, &plan_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc_tile, &plan_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan_b, &workspace_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan_tile, &workspace_tile) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  auto wrong_a_desc = a_desc;
  wrong_a_desc.tile_m = 128;
  wrong_a_desc.tile_n = 128;
  auto wrong_c_desc = c_desc;
  wrong_c_desc.tile_m = 128;
  wrong_c_desc.tile_n = 128;
  REQUIRE(rns8_create_matrix(hip, &wrong_a_desc, &wrong_a_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &wrong_c_desc, &wrong_c_tile) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, wrong_a_tile, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);

  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  const uint64_t warmed_output_version = c_matrix->source_version;
  CHECK(warmed_output_version != 0);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  CHECK(workspace->schedule_tile_count == bounds_a.size());
  CHECK(workspace->schedule_prefix_group_count >= 1);
  CHECK(workspace->schedule_fingerprint != workspace_b->schedule_fingerprint);
  CHECK(workspace->schedule_fingerprint != workspace_tile->schedule_fingerprint);

  CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace_b) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace_tile) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, plan, wrong_a_tile, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(hip, plan, wrong_c_tile, hip_c.data(), ldc) == RNS8_INVALID_ARGUMENT);
  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(c_matrix->source_version == warmed_output_version);

  rns8_destroy_matrix(wrong_c_tile);
  rns8_destroy_matrix(wrong_a_tile);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace_tile);
  rns8_destroy_workspace(workspace_b);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan_tile);
  rns8_destroy_plan(plan_b);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent per-tile bounded u64 K-split reuses resident buffers across mixed prefixes") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent per-tile direct HIP bounded K-split reuse smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 2;
  constexpr uint64_t sentinel = 0xa5a5a5a5a5a5a5a5ull;
  const uint64_t ku = static_cast<uint64_t>(k);
  const std::vector<uint64_t> bounds = {2u * ku * 1000u, ku * 1000000u * 1000u};

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t row = 0; row < m; ++row) {
      const uint64_t value = row < 64 ? (variant == 0 ? 1u : 2u) : (variant == 0 ? 1000000u : 500000u);
      for (int64_t kk = 0; kk < k; ++kk) {
        A[static_cast<std::size_t>(row * lda + kk)] = value;
      }
    }
    for (int64_t kk = 0; kk < k; ++kk) {
      B[static_cast<std::size_t>(kk * ldb)] = variant == 0 ? 1000u : 997u;
      B[static_cast<std::size_t>(kk * ldb + 1)] = variant == 0 ? 999u : 991u;
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t pad = n; pad < ldc; ++pad) {
        CHECK(cpu_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
      }
    }
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_count == bounds.size());
  CHECK(info.adaptive_prefix_active == 1);
  CHECK(info.adaptive_skip_active == 1);
  CHECK(info.min_selected_prefix >= 1);
  CHECK(info.min_selected_prefix < info.max_selected_prefix);
  CHECK(info.max_selected_prefix < RNS8_DEFAULT_BOUNDED_PREFIX);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);
  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  REQUIRE(warmed_snapshot.a_upload != nullptr);
  REQUIRE(warmed_snapshot.b_upload != nullptr);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  REQUIRE(warmed_snapshot.c_export != nullptr);
  REQUIRE(warmed_snapshot.c_status != nullptr);
  REQUIRE(warmed_snapshot.allocations.allocate_calls > 0);
  REQUIRE(warmed_snapshot.allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  CHECK(c_matrix->source_version != 0);
  CHECK(c_matrix->source_version != first_output_version);
  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(a_matrix->device_residues_current);
  CHECK_FALSE(a_matrix->host_residues_current);
  CHECK(b_matrix->device_residues_current);
  CHECK_FALSE(b_matrix->host_residues_current);
  CHECK(c_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent bounded i64 K-split reuses resident storage with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP bounded K-split smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 3;
  constexpr int64_t ldb = n + 2;
  constexpr int64_t ldc = n + 2;
  constexpr int64_t sentinel = INT64_C(-0x123456789abc);
  const uint64_t bound = static_cast<uint64_t>(k) * 127u * 127u;

  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(kk)] = variant == 0 ? 127 : (kk % 2 == 0 ? 127 : -127);
      B[static_cast<std::size_t>(kk * ldb)] = 127;
      B[static_cast<std::size_t>(kk * ldb + 1)] = -127;
    }
  };

  auto compare_outputs = [&](int64_t expected_col0, int64_t expected_col1) {
    CHECK(cpu_c[0] == expected_col0);
    CHECK(cpu_c[1] == expected_col1);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(cpu_c[2] == sentinel);
    CHECK(cpu_c[3] == sentinel);
    CHECK(hip_c[2] == sentinel);
    CHECK(hip_c[3] == sentinel);
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);
  const uint32_t expected_prefix =
      rns8::detail::required_prefix_for_range(boost::multiprecision::cpp_int(2) * boost::multiprecision::cpp_int(bound));

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == expected_prefix);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.min_selected_prefix == expected_prefix);
  CHECK(info.max_selected_prefix == expected_prefix);
  CHECK(info.adaptive_prefix_active == 0);
  CHECK(info.adaptive_skip_active == 1);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;
  REQUIRE(a_device_residues != nullptr);
  REQUIRE(b_device_residues != nullptr);
  REQUIRE(c_device_residues != nullptr);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs(static_cast<int64_t>(bound), -static_cast<int64_t>(bound));

  void* a_upload = a_matrix->hip_upload_buffer;
  void* b_upload = b_matrix->hip_upload_buffer;
  void* c_upload = c_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const std::size_t a_upload_bytes = a_matrix->hip_upload_bytes;
  const std::size_t b_upload_bytes = b_matrix->hip_upload_bytes;
  const std::size_t c_upload_bytes = c_matrix->hip_upload_bytes;
  const std::size_t c_export_bytes = c_matrix->hip_export_bytes;
  const std::size_t c_status_bytes = c_matrix->hip_status_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(a_upload != nullptr);
  REQUIRE(b_upload != nullptr);
  CHECK(c_upload == nullptr);
  CHECK(c_upload_bytes == 0);
  REQUIRE(c_export != nullptr);
  REQUIRE(c_status != nullptr);
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs(127 * 127, -(127 * 127));

  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(a_matrix->hip_upload_buffer == a_upload);
  CHECK(b_matrix->hip_upload_buffer == b_upload);
  CHECK(c_matrix->hip_upload_buffer == c_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(a_matrix->hip_upload_bytes == a_upload_bytes);
  CHECK(b_matrix->hip_upload_bytes == b_upload_bytes);
  CHECK(c_matrix->hip_upload_bytes == c_upload_bytes);
  CHECK(c_matrix->hip_export_bytes == c_export_bytes);
  CHECK(c_matrix->hip_status_bytes == c_status_bytes);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent bounded u64 prefix-9 covers exact K-block boundary with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP bounded u64 K-boundary smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK);
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 1;
  constexpr uint64_t sentinel = 0xd1ffd1ffd1ffd1ffull;
  const uint64_t bound = static_cast<uint64_t>(k) * 12u;

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(kk)] = variant == 0 ? 1 : 2;
      A[static_cast<std::size_t>(lda + kk)] = variant == 0 ? 2 : 3;
      B[static_cast<std::size_t>(kk * ldb)] = variant == 0 ? 3 : 1;
      B[static_cast<std::size_t>(kk * ldb + 1)] = variant == 0 ? 5 : 4;
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(cpu_c[static_cast<std::size_t>(row * ldc + n)] == sentinel);
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == sentinel);
    }
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);
  cpu_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  hip_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.adaptive_prefix_active == 0);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;
  REQUIRE(a_device_residues != nullptr);
  REQUIRE(b_device_residues != nullptr);
  REQUIRE(c_device_residues != nullptr);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);

  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  REQUIRE(warmed_snapshot.a_upload != nullptr);
  REQUIRE(warmed_snapshot.b_upload != nullptr);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  REQUIRE(warmed_snapshot.c_export != nullptr);
  // Prefix 9 direct export elides status (structurally covers full u64 range)
  REQUIRE(warmed_snapshot.c_status == nullptr);
  REQUIRE(warmed_snapshot.allocations.allocate_calls > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  CHECK(c_matrix->source_version != 0);
  CHECK(c_matrix->source_version != first_output_version);

  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

