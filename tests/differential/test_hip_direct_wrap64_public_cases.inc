TEST_CASE("direct HIP public wrap64 byte-limb path matches CPU reference") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 4;
  constexpr int64_t ldc = 5;
  const std::vector<uint64_t> A = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      0xaaaaull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x7f7f7f7f7f7f7f7full,
      17,
      0xbbbbull};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      0xccccull,
      29,
      0x8080808080808080ull,
      31,
      0xddddull,
      0x0101010101010101ull,
      0xfefdfcfbfaf9f8f7ull,
      37,
      0xeeeeull,
      41,
      43,
      47,
      0xffffull,
      53,
      59,
      61,
      0x1234ull};
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);

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
  CHECK(cpu_plan->prefix == 0);
  CHECK(hip_plan->prefix == 0);
  rns8_plan_backend_info hip_info{};
  hip_info.struct_size = sizeof(hip_info);
  hip_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(hip_plan, &hip_info) == RNS8_SUCCESS);
  CHECK(std::string(hip_info.selected_kernel) == "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4");
  CHECK(std::string(hip_info.isa_evidence) ==
        "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine");
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
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->desc.logical_ld == lda);
  CHECK(hip_b->desc.logical_ld == ldb);
  CHECK(hip_out->desc.logical_ld == ldc);
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK_FALSE(hip_a->device_byte_limbs_current);
  CHECK_FALSE(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_out->device_byte_limbs_current);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 7) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 11) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 7) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 11) == RNS8_SUCCESS);
  auto hip_pack_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_pack_events, "pack_h2d"));
  CHECK(has_timing_label(hip_pack_events, "pack_kernel"));
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_a->device_byte_limbs_current);
  CHECK(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);

  std::vector<uint64_t> changed_A = A;
  changed_A[0] = 123456789ull;
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_pack_u64(hip, hip_a, changed_A.data(), lda, 7) == RNS8_SUCCESS);
  auto same_source_pack_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK_FALSE(has_timing_label(same_source_pack_events, "pack_h2d"));
  CHECK_FALSE(has_timing_label(same_source_pack_events, "pack_kernel"));
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  std::fill(hip_c.begin(), hip_c.end(), 0x1111111111111111ull);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == 0x1111111111111111ull);
    }
  }
  std::fill(hip_c.begin(), hip_c.end(), 0xdeadbeefdeadbeefull);

  CHECK(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  auto hip_gemm_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_gemm_events, "wrap64_byte_gemm36_tiled_2d_kernel"));
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
  CHECK(hip_out->device_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);

  CHECK(rns8_export_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  auto hip_export_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_export_events, "wrap64_export_kernel"));
  CHECK(has_timing_label(hip_export_events, "wrap64_export_d2h"));
  CHECK(hip_out->hip_export_buffer != nullptr);
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK(hip_c == cpu_c);

  std::fill(hip_c.begin(), hip_c.end(), 0xfeedfeedfeedfeedull);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == 0xfeedfeedfeedfeedull);
    }
  }

  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), 0xababababababababull);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), 0xababababababababull);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_oneshot == cpu_oneshot);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      const uint64_t decomposition =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(decomposition == expected);
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == expected);
    }
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xfeedfeedfeedfeedull);
    CHECK(hip_oneshot[static_cast<std::size_t>(row * ldc + n)] == 0xababababababababull);
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

TEST_CASE("direct HIP public wrap64 tiled byte-limb path matches CPU for random padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP tiled smoke");
  }

  constexpr int64_t m = 19;
  constexpr int64_t n = 18;
  constexpr int64_t k = 33;
  constexpr int64_t lda = 37;
  constexpr int64_t ldb = 23;
  constexpr int64_t ldc = 21;
  constexpr uint64_t c_sentinel = 0x5a5a5a5a5a5a5a5aull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::mt19937_64 rng(0x726e73385f777261ull);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = rng();
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = rng();
    }
  }

  A[0] = 0;
  A[1] = std::numeric_limits<uint64_t>::max();
  A[2] = 0x8080808080808080ull;
  A[static_cast<std::size_t>((m / 2) * lda + (k / 2))] = 0x0102030405060708ull;
  A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = std::numeric_limits<uint64_t>::max() - 1;
  B[0] = std::numeric_limits<uint64_t>::max();
  B[1] = 1;
  B[2] = 0xfefdfcfbfaf9f8f7ull;
  B[static_cast<std::size_t>((k / 2) * ldb + (n / 2))] = 0x7f807f807f807f80ull;
  B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = std::numeric_limits<uint64_t>::max();

  rns8_context* cpu = create_context(RNS8_BACKEND_WRAP64_BYTE_LIMB);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8::detail::hip_direct_allocation_counters_reset();
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
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->desc.logical_ld == lda);
  CHECK(hip_b->desc.logical_ld == ldb);
  CHECK(hip_out->desc.logical_ld == ldc);
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK_FALSE(hip_a->device_byte_limbs_current);
  CHECK_FALSE(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_out->device_byte_limbs_current);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 102) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 102) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;

  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_oneshot == cpu_oneshot);
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  const auto warmed_snapshot = capture_wrap64_resident_snapshot(hip_a, hip_b, hip_out, hip_workspace);

  std::fill(cpu_c.begin(), cpu_c.end(), c_sentinel);
  std::fill(hip_c.begin(), hip_c.end(), c_sentinel);
  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 201) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 202) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 201) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 202) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  check_wrap64_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_out, hip_workspace);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(cpu_c[out_index] == expected);
      CHECK(hip_c[out_index] == expected);
      CHECK(hip_oneshot[out_index] == expected);
    }
    for (int64_t col = n; col < ldc; ++col) {
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(hip_c[out_index] == c_sentinel);
      CHECK(hip_oneshot[out_index] == c_sentinel);
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

TEST_CASE("direct HIP public wrap64 resident path is device-current for large padded tiles") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP resident stress smoke");
  }

  constexpr int64_t m = 33;
  constexpr int64_t n = 35;
  constexpr int64_t k = 65;
  constexpr int64_t lda = 71;
  constexpr int64_t ldb = 43;
  constexpr int64_t ldc = 39;
  constexpr uint64_t a_pad = 0xaaaaaaaaaaaaaaaaull;
  constexpr uint64_t b_pad = 0xbbbbbbbbbbbbbbbbull;
  constexpr uint64_t c_sentinel = 0x9191919191919191ull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), a_pad);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), b_pad);
  std::vector<uint64_t> latest_A;
  std::vector<uint64_t> latest_B;
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);

  auto fill_inputs = [&](uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::fill(A.begin(), A.end(), a_pad);
    std::fill(B.begin(), B.end(), b_pad);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] = rng();
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] = rng();
      }
    }
    A[0] = 0;
    A[1] = std::numeric_limits<uint64_t>::max();
    A[2] = 0x8080808080808080ull;
    A[3] = 0x7f807f807f807f80ull;
    A[static_cast<std::size_t>((m / 2) * lda + (k / 2))] = 0xfefdfcfbfaf9f8f7ull;
    A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = std::numeric_limits<uint64_t>::max() - 1;
    B[0] = std::numeric_limits<uint64_t>::max();
    B[1] = 1;
    B[2] = 0x0102030405060708ull;
    B[3] = 0x8080808080808080ull;
    B[static_cast<std::size_t>((k / 2) * ldb + (n / 2))] = 0x7f807f807f807f80ull;
    B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = std::numeric_limits<uint64_t>::max();
  };

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
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  rns8::detail::hip_direct_allocation_counters_reset();
  auto run_persistent = [&](uint64_t seed, uint64_t a_version, uint64_t b_version) {
    fill_inputs(seed);
    latest_A = A;
    latest_B = B;
    std::fill(cpu_c.begin(), cpu_c.end(), c_sentinel);
    std::fill(hip_c.begin(), hip_c.end(), c_sentinel);

    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, b_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, b_version) == RNS8_SUCCESS);
    CHECK(hip_a->device_byte_limbs_current);
    CHECK(hip_b->device_byte_limbs_current);
    CHECK_FALSE(hip_a->host_byte_limbs_current);
    CHECK_FALSE(hip_b->host_byte_limbs_current);

    std::fill(A.begin(), A.end(), 0x1111111111111111ull);
    std::fill(B.begin(), B.end(), 0x2222222222222222ull);
    REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
    CHECK(hip_c == cpu_c);
    CHECK(hip_a->source_version == a_version);
    CHECK(hip_b->source_version == b_version);
    CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
    CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
    CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
    CHECK(hip_out->device_byte_limbs_current);
    CHECK_FALSE(hip_out->host_byte_limbs_current);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const uint64_t expected =
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(
                latest_A.data(), lda, latest_B.data(), ldb, row, col, k);
        const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
        CHECK(hip_c[out_index] == expected);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  };

  run_persistent(0x7772617036345f31ull, 301, 302);
  void* hip_a_upload = hip_a->hip_upload_buffer;
  void* hip_b_upload = hip_b->hip_upload_buffer;
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_a_upload_bytes = hip_a->hip_upload_bytes;
  const std::size_t hip_b_upload_bytes = hip_b->hip_upload_bytes;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(hip_a_upload != nullptr);
  REQUIRE(hip_b_upload != nullptr);
  REQUIRE(hip_export != nullptr);
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  const auto warmed_snapshot = capture_wrap64_resident_snapshot(hip_a, hip_b, hip_out, hip_workspace);

  run_persistent(0x7772617036345f32ull, 401, 402);
  CHECK(hip_a->hip_upload_buffer == hip_a_upload);
  CHECK(hip_b->hip_upload_buffer == hip_b_upload);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_a->hip_upload_bytes == hip_a_upload_bytes);
  CHECK(hip_b->hip_upload_bytes == hip_b_upload_bytes);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  check_wrap64_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_out, hip_workspace);

  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(
              cpu, &cpu_desc, latest_A.data(), lda, latest_B.data(), ldb, cpu_oneshot.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(
              hip, &hip_desc, latest_A.data(), lda, latest_B.data(), ldb, hip_oneshot.data(), ldc) == RNS8_SUCCESS);
  CHECK(cpu_oneshot == cpu_c);
  CHECK(hip_oneshot == hip_c);

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

