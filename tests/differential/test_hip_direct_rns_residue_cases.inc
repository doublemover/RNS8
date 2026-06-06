TEST_CASE("direct HIP residue packing matches CPU reference for i64 and u64") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP pack smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t rows = 3;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<int64_t> src = {
        0,
        1,
        -1,
        127,
        999,
        128,
        -128,
        -129,
        std::numeric_limits<int64_t>::max(),
        999,
        -std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min(),
        251,
        -251,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(cpu, cpu_matrix, src.data(), ld, 11) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(hip, hip_matrix, src.data(), ld, 11) == RNS8_SUCCESS);
    CHECK(hip_matrix->hip_residues != nullptr);
    CHECK(hip_matrix->device_residues_current);
    CHECK_FALSE(hip_matrix->host_residues_current);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 11);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 5;
    const int64_t ld = 6;
    const std::vector<uint64_t> src = {
        0,
        1,
        127,
        128,
        255,
        999,
        256,
        257,
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max() - 1,
        251,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(cpu, cpu_matrix, src.data(), ld, 19) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(hip, hip_matrix, src.data(), ld, 19) == RNS8_SUCCESS);
    CHECK(hip_matrix->hip_residues != nullptr);
    CHECK(hip_matrix->device_residues_current);
    CHECK_FALSE(hip_matrix->host_residues_current);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 19);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 3;
    const int64_t ld = 4;
    const std::vector<int64_t> src = {
        std::numeric_limits<int64_t>::min(),
        -9223372036854775807LL,
        -257,
        999,
        -1,
        0,
        std::numeric_limits<int64_t>::max(),
        999};
    auto desc = matrix_desc(rows, cols, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(cpu, cpu_matrix, src.data(), ld, 23) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(hip, hip_matrix, src.data(), ld, 23) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 23);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 3;
    const int64_t ld = 4;
    const std::vector<uint64_t> src = {
        0,
        1,
        257,
        999,
        std::numeric_limits<uint64_t>::max() - 1,
        std::numeric_limits<uint64_t>::max(),
        1234567890123456789ull,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(cpu, cpu_matrix, src.data(), ld, 29) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(hip, hip_matrix, src.data(), ld, 29) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 29);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent RNS matrices keep device storage through GEMM") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const int64_t A[] = {5, -7, 11, 13, -17, 19, 23, -29};
  const int64_t B[] = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
  int64_t cpu_c[6] = {};
  int64_t hip_c[6] = {};

  auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  REQUIRE(a_matrix->hip_residues != nullptr);
  REQUIRE(b_matrix->hip_residues != nullptr);
  REQUIRE(c_matrix->hip_residues != nullptr);
  CHECK_FALSE(a_matrix->host_residues_current);
  CHECK_FALSE(a_matrix->device_residues_current);
  CHECK_FALSE(b_matrix->host_residues_current);
  CHECK_FALSE(b_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);
  CHECK_FALSE(c_matrix->device_residues_current);
  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;

  REQUIRE(rns8_pack_i64(hip, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B, n, 2) == RNS8_SUCCESS);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(a_matrix->device_residues_current);
  CHECK(b_matrix->device_residues_current);

  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(c_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);

  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c, n) == RNS8_SUCCESS);
  CHECK_FALSE(c_matrix->host_residues_current);
  CHECK(c_matrix->hip_export_buffer != nullptr);
  CHECK(c_matrix->hip_status_buffer != nullptr);
  CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
        std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));

  void* a_upload = a_matrix->hip_upload_buffer;
  void* b_upload = b_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  std::fill(std::begin(hip_c), std::end(hip_c), int64_t{0});

  REQUIRE(rns8_pack_i64(hip, a_matrix, A, k, 3) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B, n, 4) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c, n) == RNS8_SUCCESS);
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(a_matrix->hip_upload_buffer == a_upload);
  CHECK(b_matrix->hip_upload_buffer == b_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
        std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("AUTO direct-HIP RNS GEMM converts current native bounded inputs to RNS residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for AUTO native-to-RNS conversion smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(0, &auto_options, &auto_ctx) == RNS8_SUCCESS);
  REQUIRE(auto_ctx->auto_backend_selection);
  REQUIRE(auto_ctx->backend == RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t m = 2;
    const int64_t n = 3;
    const int64_t k = 4;
    const std::vector<int64_t> A = {5, -7, 11, 13, -17, 19, 23, -29};
    const std::vector<int64_t> B = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> auto_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto auto_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(auto_ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(rns8_pack_i64(auto_ctx, a_matrix, A.data(), k, 101) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(auto_ctx, b_matrix, B.data(), n, 202) == RNS8_SUCCESS);
    REQUIRE(a_matrix->device_native_current);
    REQUIRE(b_matrix->device_native_current);
    REQUIRE(a_matrix->device_residues_current);
    REQUIRE(b_matrix->device_residues_current);

    rns8::test::set_device_residues_current(*a_matrix, false);
    rns8::test::set_device_residues_current(*b_matrix, false);
    REQUIRE(rns8_gemm_rns(auto_ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->device_residues_current);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK(c_matrix->device_residues_current);
    REQUIRE(rns8_export_i64(auto_ctx, plan, c_matrix, auto_c.data(), n) == RNS8_SUCCESS);
    CHECK(auto_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 3;
    const std::vector<uint64_t> A = {5, 7, 11, 13, 17, 19};
    const std::vector<uint64_t> B = {23, 29, 31, 37, 41, 43};
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> auto_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_CPU_REFERENCE);
    auto auto_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(auto_ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(rns8_pack_u64(auto_ctx, a_matrix, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(auto_ctx, b_matrix, B.data(), n, 302) == RNS8_SUCCESS);
    REQUIRE(a_matrix->device_native_current);
    REQUIRE(b_matrix->device_native_current);
    REQUIRE(a_matrix->device_residues_current);
    REQUIRE(b_matrix->device_residues_current);

    rns8::test::set_device_residues_current(*a_matrix, false);
    rns8::test::set_device_residues_current(*b_matrix, false);
    REQUIRE(rns8_gemm_rns(auto_ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->device_residues_current);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK(c_matrix->device_residues_current);
    REQUIRE(rns8_export_u64(auto_ctx, plan, c_matrix, auto_c.data(), n) == RNS8_SUCCESS);
    CHECK(auto_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(auto_ctx);
  rns8_destroy_context(cpu);
}

