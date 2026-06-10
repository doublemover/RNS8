TEST_CASE("direct HIP bounded GEMM rejects host-current stale device inputs") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded stale-input GEMM rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 2;
  constexpr int8_t c_sentinel = -33;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = signed_desc(m, n, k, 100, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(2)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(3), boost::multiprecision::cpp_int(4)});
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
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);
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
    auto desc = unsigned_desc(m, n, k, 100, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(5), boost::multiprecision::cpp_int(6)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(7), boost::multiprecision::cpp_int(8)});
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
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);
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

TEST_CASE("direct HIP bounded exports reject host-current stale device residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded stale-device export rejection smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const std::vector<int64_t> expected = {-31, 0, 17, 999};
    auto desc = signed_desc(m, n, 1, 1000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(expected[0]),
         boost::multiprecision::cpp_int(expected[1]),
         boost::multiprecision::cpp_int(expected[2]),
         boost::multiprecision::cpp_int(expected[3])});

    std::vector<int64_t> out(static_cast<std::size_t>(m * n), INT64_C(0x7f7f7f7f));
    CHECK(rns8_export_i64(hip, plan, c_matrix, out.data(), n) == RNS8_INVALID_ARGUMENT);
    CHECK(out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x7f7f7f7f)));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    const std::vector<uint64_t> expected = {0, 17, 999, 1000};
    auto desc = unsigned_desc(m, n, 1, 1000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(expected[0]),
         boost::multiprecision::cpp_int(expected[1]),
         boost::multiprecision::cpp_int(expected[2]),
         boost::multiprecision::cpp_int(expected[3])});

    std::vector<uint64_t> out(static_cast<std::size_t>(m * n), UINT64_C(0xffffffffffffffff));
    CHECK(rns8_export_u64(hip, plan, c_matrix, out.data(), n) == RNS8_INVALID_ARGUMENT);
    CHECK(out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xffffffffffffffff)));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP bounded export range errors preserve host output and reuse resident buffers") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded range-error export smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto cpu_desc = signed_desc(m, n, 1, 10, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, 1, 10, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(cpu_plan->prefix == 1u);
    REQUIRE(hip_plan->prefix == 1u);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(-11), boost::multiprecision::cpp_int(7)});
    fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(-11), boost::multiprecision::cpp_int(7)});
    upload_exact_residues_to_hip(hip_c);

    std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    std::vector<int64_t> hip_out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    CHECK(rns8_export_i64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(cpu_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK(hip_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    REQUIRE(hip_c->hip_export_buffer != nullptr);
    REQUIRE(hip_c->hip_status_buffer != nullptr);
    void* export_buffer = hip_c->hip_export_buffer;
    void* status_buffer = hip_c->hip_status_buffer;
    const std::size_t export_bytes = hip_c->hip_export_bytes;
    const std::size_t status_bytes = hip_c->hip_status_bytes;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();

    CHECK(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    CHECK(hip_c->hip_export_buffer == export_buffer);
    CHECK(hip_c->hip_status_buffer == status_buffer);
    CHECK(hip_c->hip_upload_bytes == 0);
    CHECK(hip_c->hip_export_bytes == export_bytes);
    CHECK(hip_c->hip_status_bytes == status_bytes);
    CHECK(hip_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  {
    auto cpu_desc = unsigned_desc(m, n, 1, 10, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, 1, 10, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(cpu_plan->prefix == 1u);
    REQUIRE(hip_plan->prefix == 1u);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(11), boost::multiprecision::cpp_int(3)});
    fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(11), boost::multiprecision::cpp_int(3)});
    upload_exact_residues_to_hip(hip_c);

    std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    CHECK(rns8_export_u64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(cpu_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK(hip_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    REQUIRE(hip_c->hip_export_buffer != nullptr);
    REQUIRE(hip_c->hip_status_buffer != nullptr);
    void* export_buffer = hip_c->hip_export_buffer;
    void* status_buffer = hip_c->hip_status_buffer;
    const std::size_t export_bytes = hip_c->hip_export_bytes;
    const std::size_t status_bytes = hip_c->hip_status_bytes;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();

    CHECK(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    CHECK(hip_c->hip_export_buffer == export_buffer);
    CHECK(hip_c->hip_status_buffer == status_buffer);
    CHECK(hip_c->hip_upload_bytes == 0);
    CHECK(hip_c->hip_export_bytes == export_bytes);
    CHECK(hip_c->hip_status_bytes == status_bytes);
    CHECK(hip_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent per-tile bounded u64 reuses resident storage across same-shape calls") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent per-tile direct HIP bounded smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 2;
  constexpr int64_t lda = k + 1;
  constexpr int64_t ldb = n + 2;
  constexpr int64_t ldc = n + 3;
  constexpr uint64_t sentinel = 0xbad0bad0bad0bad0ull;
  const std::vector<uint64_t> bounds = {64, 4096, 8000000, 2000000000};

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row * lda)] = row < 64 ? (variant == 0 ? 1 : 3) : (variant == 0 ? 1000000 : 500000);
      A[static_cast<std::size_t>(row * lda + 1)] = row < 64 ? (variant == 0 ? 2 : 4) : (variant == 0 ? 3 : 5);
    }
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? (variant == 0 ? 7 : 5) : (variant == 0 ? 1000 : 999);
      B[static_cast<std::size_t>(ldb + col)] = col < 64 ? (variant == 0 ? 11 : 6) : (variant == 0 ? 13 : 7);
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t pad = n; pad < ldc; ++pad) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
        CHECK(cpu_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
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
  CHECK(info.max_selected_prefix < plan->prefix);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
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
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();

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

