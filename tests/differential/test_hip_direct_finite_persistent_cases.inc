TEST_CASE("direct HIP finite u8 persistent matrices match CPU and reuse resident storage") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 persistent smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 13;
  constexpr int64_t n = 11;
  constexpr int64_t k = 17;
  constexpr int64_t lda = 23;
  constexpr int64_t ldb = 19;
  constexpr int64_t ldc = 17;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          static_cast<uint8_t>((row * 31 + col * 47 + (row ^ (col * 3)) * 5) & 0xff);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          static_cast<uint8_t>((row * 29 + col * 43 + (row + col * 7) * 9) & 0xff);
    }
  }

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
  };
  for (const Case item : {
           Case{RNS8_FINITE_RING_U8, 255},
           Case{RNS8_FINITE_RING_U8, 256},
           Case{RNS8_FINITE_FIELD_U8, 251},
       }) {
    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_workspace* cpu_workspace = nullptr;
    rns8_workspace* hip_workspace = nullptr;
    rns8_matrix* cpu_a = nullptr;
    rns8_matrix* cpu_b = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

    auto a_desc = matrix_desc(m, k, item.semantics, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, item.semantics, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, item.semantics, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, item.modulus, A.data(), lda, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_b, item.modulus, B.data(), ldb, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(cpu, cpu_plan, item.modulus, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, item.modulus, cpu_c, cpu_out.data(), ldc) == RNS8_SUCCESS);

    rns8::detail::hip_direct_allocation_counters_reset();
    REQUIRE(rns8_pack_finite_u8(hip, hip_a, item.modulus, A.data(), lda, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b, item.modulus, B.data(), ldb, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, item.modulus, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, item.modulus, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    const auto warmed_snapshot = capture_finite_resident_snapshot(hip_a, hip_b, hip_c, hip_workspace);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xdd);
    REQUIRE(rns8_pack_finite_u8(hip, hip_a, item.modulus, A.data(), lda, 3) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b, item.modulus, B.data(), ldb, 4) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, item.modulus, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, item.modulus, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    check_finite_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_c, hip_workspace);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xdd);
      }
    }
    CHECK(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_finite_u8(hip, hip_plan, item.modulus == 251 ? 255 : 251, hip_c, hip_out.data(), ldc) ==
          RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_matrix(cpu_b);
    rns8_destroy_matrix(cpu_a);
    rns8_destroy_workspace(hip_workspace);
    rns8_destroy_workspace(cpu_workspace);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 public grouped GEMM matches CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite grouped public API smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 3;
  constexpr int64_t n = 2;
  constexpr int64_t k = 4;
  constexpr uint16_t modulus = 251;
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CPU_REFERENCE, modulus);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIP_DIRECT, modulus);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);

  rns8_workspace* cpu_workspace[2] = {};
  rns8_workspace* hip_workspace[2] = {};
  rns8_matrix* cpu_a[2] = {};
  rns8_matrix* cpu_b[2] = {};
  rns8_matrix* cpu_c[2] = {};
  rns8_matrix* hip_a[2] = {};
  rns8_matrix* hip_b[2] = {};
  rns8_matrix* hip_c[2] = {};
  auto a_desc = matrix_desc(m, k, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE);
  for (uint32_t task = 0; task < 2; ++task) {
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b[task]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c[task]) == RNS8_SUCCESS);
  }

  std::vector<std::vector<uint8_t>> a_values(2, std::vector<uint8_t>(static_cast<std::size_t>(m * k)));
  std::vector<std::vector<uint8_t>> b_values(2, std::vector<uint8_t>(static_cast<std::size_t>(k * n)));
  for (uint32_t task = 0; task < 2; ++task) {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        a_values[task][static_cast<std::size_t>(row * k + col)] =
            static_cast<uint8_t>((task * 17 + row * 11 + col * 13 + 3) % modulus);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        b_values[task][static_cast<std::size_t>(row * n + col)] =
            static_cast<uint8_t>((task * 19 + row * 7 + col * 23 + 5) % modulus);
      }
    }
  }

  rns8_grouped_gemm_task tasks[2] = {};
  std::vector<std::vector<uint8_t>> cpu_out(2, std::vector<uint8_t>(static_cast<std::size_t>(m * n)));
  std::vector<std::vector<uint8_t>> hip_out(2, std::vector<uint8_t>(static_cast<std::size_t>(m * n)));
  for (uint32_t task = 0; task < 2; ++task) {
    const uint64_t a_version = 10 + task * 2;
    const uint64_t b_version = 11 + task * 2;
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_a[task], modulus, a_values[task].data(), k, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_b[task], modulus, b_values[task].data(), n, b_version) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(
                cpu, cpu_plan, modulus, cpu_a[task], cpu_b[task], cpu_c[task], cpu_workspace[task]) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, modulus, cpu_c[task], cpu_out[task].data(), n) ==
            RNS8_SUCCESS);

    REQUIRE(rns8_pack_finite_u8(hip, hip_a[task], modulus, a_values[task].data(), k, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b[task], modulus, b_values[task].data(), n, b_version) == RNS8_SUCCESS);
    tasks[task] = {sizeof(rns8_grouped_gemm_task), RNS8_ABI_VERSION, hip_a[task], hip_b[task], hip_c[task],
                   hip_workspace[task]};
  }

  REQUIRE(rns8_gemm_finite_u8_grouped(hip, hip_plan, modulus, tasks, 2) == RNS8_SUCCESS);
  CHECK(rns8_gemm_finite_u8_grouped(hip, hip_plan, 253, tasks, 2) == RNS8_INVALID_ARGUMENT);
  for (uint32_t task = 0; task < 2; ++task) {
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, modulus, hip_c[task], hip_out[task].data(), n) == RNS8_SUCCESS);
    CHECK(hip_out[task] == cpu_out[task]);
  }
  rns8::test::set_device_residues_current(*hip_a[1], false);
  CHECK(rns8_gemm_finite_u8_grouped(hip, hip_plan, modulus, tasks, 2) == RNS8_INVALID_ARGUMENT);
  rns8::test::set_device_residues_current(*hip_a[1], true);

  for (int task = 1; task >= 0; --task) {
    rns8_destroy_matrix(hip_c[task]);
    rns8_destroy_matrix(hip_b[task]);
    rns8_destroy_matrix(hip_a[task]);
    rns8_destroy_matrix(cpu_c[task]);
    rns8_destroy_matrix(cpu_b[task]);
    rns8_destroy_matrix(cpu_a[task]);
    rns8_destroy_workspace(hip_workspace[task]);
    rns8_destroy_workspace(cpu_workspace[task]);
  }
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 one-shot preserves K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 K-split smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<uint8_t> A(static_cast<std::size_t>(k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k), 255);
  uint8_t cpu_out = 0;
  uint8_t hip_out = 0;
  auto cpu_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), 1, &cpu_out, 1) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), 1, &hip_out, 1) ==
          RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP fixed-prefix plans advertise grouped GEMM kernels") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP fixed-prefix metadata smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr const char* grouped_kernel = "direct_hip_prefix9_grouped_rns_gemm_v1";
  for (const rns8_semantics semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64}) {
    CAPTURE(semantics);
    auto desc = semantics == RNS8_BOUNDED_I64
                    ? signed_desc(32, 32, 32, 32u * 127u * 127u, RNS8_BACKEND_HIP_DIRECT)
                    : unsigned_desc(32, 32, 32, 32u * 127u * 127u, RNS8_BACKEND_HIP_DIRECT);
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(plan->tile_schedule.empty());

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == grouped_kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + grouped_kernel + ";") != std::string::npos);

    rns8_destroy_plan(plan);
  }

  constexpr const char* exact_grouped_kernel = "direct_hip_prefix20_grouped_rns_gemm_v1";
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    CAPTURE(semantics);
    auto desc = semantics == RNS8_EXACT_WIDE_SIGNED ? exact_signed_desc(16, 16, 16, RNS8_BACKEND_HIP_DIRECT)
                                                    : exact_unsigned_desc(16, 16, 16, RNS8_BACKEND_HIP_DIRECT);
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(plan->prefix == RNS8_MAX_SUPPORTED_PREFIX);
    CHECK(plan->tile_schedule.empty());

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == exact_grouped_kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + exact_grouped_kernel + ";") !=
          std::string::npos);

    rns8_destroy_plan(plan);
  }

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto adaptive_desc = per_tile_signed_desc(65, 65, 64, bounds, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* adaptive_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &adaptive_desc, &adaptive_plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(adaptive_plan->tile_schedule.empty());
  rns8_plan_backend_info adaptive_info{};
  adaptive_info.struct_size = sizeof(adaptive_info);
  adaptive_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(adaptive_plan, &adaptive_info) == RNS8_SUCCESS);
  CHECK(std::string(adaptive_info.selected_kernel) == "direct_hip_tiled_active_prefix_rns_gemm_v2");
  CHECK(std::string(adaptive_info.autotune_key).find("kernel=direct_hip_tiled_active_prefix_rns_gemm_v2;") !=
        std::string::npos);
  CHECK(std::string(adaptive_info.workspace_mode) ==
        "resident_device_buffers_with_active_prefix_tiled_schedule");
  std::vector<rns8_plan_tile_schedule_entry> adaptive_entries(adaptive_plan->tile_schedule.size());
  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(
              adaptive_plan, adaptive_entries.data(), adaptive_entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == adaptive_entries.size());
  uint64_t active_entry_count = 0;
  for (const auto& entry : adaptive_entries) {
    active_entry_count += entry.selected_prefix;
  }
  const uint64_t expected_workspace_bytes = active_entry_count * sizeof(rns8_plan_tile_schedule_entry);
  CHECK(adaptive_info.workspace_required_bytes == expected_workspace_bytes);
  rns8_workspace* adaptive_workspace = nullptr;
  REQUIRE(rns8_create_workspace(hip, adaptive_plan, &adaptive_workspace) == RNS8_SUCCESS);
  REQUIRE(adaptive_workspace != nullptr);
  CHECK(adaptive_workspace->hip_tile_schedule == nullptr);
  CHECK(adaptive_workspace->hip_tile_schedule_bytes == 0);
  CHECK(adaptive_workspace->hip_tile_schedule_count == 0);
  CHECK(adaptive_workspace->hip_tile_schedule_active_prefix_count == adaptive_plan->schedule_max_selected_prefix);
  CHECK(adaptive_workspace->hip_tile_schedule_active_entries_count == active_entry_count);
  CHECK(adaptive_workspace->hip_tile_schedule_active_entries_bytes ==
        active_entry_count * sizeof(rns8_plan_tile_schedule_entry));
  REQUIRE(rns8_destroy_workspace(adaptive_workspace) == RNS8_SUCCESS);
  rns8_destroy_plan(adaptive_plan);
  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP finite u8 fixed-modulus kernels preserve K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite reducer smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 3;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 17;
  constexpr int64_t lda = k + 3;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 4;

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
    const char* kernel;
  };
  for (const Case item : {
           Case{RNS8_FINITE_FIELD_U8, 251, "direct_hip_tiled_finite_u8_gemm_mod251_v1"},
           Case{RNS8_FINITE_RING_U8, 255, "direct_hip_tiled_finite_u8_gemm_mod255_v1"},
           Case{RNS8_FINITE_RING_U8, 256, "direct_hip_tiled_finite_u8_gemm_mod256_v1"},
       }) {
    CAPTURE(item.modulus);
    std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
    std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] =
            static_cast<uint8_t>((row * 97 + col * 53 + ((row + 3) ^ col) * 11) % item.modulus);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<uint8_t>((row * 47 + col * 89 + (row ^ (col + 7)) * 13) % item.modulus);
      }
    }

    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    rns8_plan* hip_plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(hip_plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == item.kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + item.kernel + ";") != std::string::npos);
    CHECK(std::string(info.isa_evidence) == "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide");

    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
    if (item.semantics == RNS8_FINITE_FIELD_U8) {
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    }
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "residue_h2d_sync"));
    CHECK(has_timing_label(hip_events, "finite_native_gemm_kernel"));
    CHECK(has_timing_label(hip_events, "finite_export_kernel"));
    CHECK_FALSE(has_timing_label(hip_events, "finite_pack_kernel"));
    CHECK_FALSE(has_timing_label(hip_events, "finite_resident_gemm_kernel"));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
      }
    }

    rns8_destroy_plan(hip_plan);
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 persistent path preserves K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 persistent K-split smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<uint8_t> A(static_cast<std::size_t>(k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k), 255);
  uint8_t cpu_out = 0;
  uint8_t hip_out = 0;
  auto cpu_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(1, k, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, 256, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(cpu, cpu_b, 256, B.data(), 1, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(cpu, cpu_plan, 256, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, 256, cpu_c, &cpu_out, 1) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_finite_u8(hip, hip_a, 256, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(hip, hip_b, 256, B.data(), 1, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, 256, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(hip, hip_plan, 256, hip_c, &hip_out, 1) == RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("private HIP ring GEMM rejects mismatched modulus metadata before launch") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP metadata-contract smoke");
  }

  constexpr int64_t bytes = 1;
  const int8_t a = 1;
  const int8_t b = 1;
  const int8_t sentinel = -77;
  int8_t out = 0;
  void* d_a = nullptr;
  void* d_b = nullptr;
  void* d_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, &a, bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b, &b, bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_c, &sentinel, bytes) == RNS8_SUCCESS);

  CHECK(rns8_hip_direct_ring_gemm_i8_device(
            static_cast<const int8_t*>(d_a),
            static_cast<const int8_t*>(d_b),
            static_cast<int8_t*>(d_c),
            1,
            1,
            1,
            1,
            1,
            1,
            251,
            reciprocal_for_modulus(251),
            1,
            RNS8_DEFAULT_BOUNDED_PREFIX,
            RNS8_SAFE_INT32_K_BLOCK) != 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, &out, d_c, bytes) == RNS8_SUCCESS);
  CHECK(out == sentinel);

  const uint16_t valid_modulus = rns8::detail::kDefaultModuli[0];
  CHECK(rns8_hip_direct_ring_gemm_i8_device(
            static_cast<const int8_t*>(d_a),
            static_cast<const int8_t*>(d_b),
            static_cast<int8_t*>(d_c),
            1,
            1,
            1,
            1,
            1,
            1,
            valid_modulus,
            reciprocal_for_modulus(valid_modulus) + 1u,
            0,
            1,
            RNS8_SAFE_INT32_K_BLOCK) != 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, &out, d_c, bytes) == RNS8_SUCCESS);
  CHECK(out == sentinel);

  CHECK(rns8::detail::hip_direct_free(0, d_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, d_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
#else
  SKIP("direct HIP private metadata-contract smoke requires a HIP build");
#endif
}

