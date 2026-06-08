TEST_CASE("vector ALU backend keeps native bounded storage through persistent GEMM") {
  if (!hip_available()) {
    SKIP("no HIP device available for vector ALU persistent bounded smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* vector = create_context(RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  {
    const int64_t m = 2;
    const int64_t n = 3;
    const int64_t k = 4;
    const std::vector<int64_t> A = {5, -7, 11, 13, -17, 19, 23, -29};
    const std::vector<int64_t> B = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8::detail::hip_direct_allocation_counters_reset();
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    rns8_plan_packing_info packing_info{};
    packing_info.struct_size = sizeof(packing_info);
    packing_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &packing_info) == RNS8_SUCCESS);
    CHECK(packing_info.input_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64);
    CHECK(packing_info.output_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64);
    CHECK(packing_info.output_host_current == 0);
    CHECK(packing_info.output_device_current == 1);
    CHECK((packing_info.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) != 0);
    CHECK((packing_info.next_op_flags & RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE) != 0);
    CHECK(std::string(packing_info.output_domain_name) == "native_i64_u64_current");
    CHECK(std::string(packing_info.next_op_hint).find("native-to-RNS conversion") != std::string::npos);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(a_matrix->hip_native_i64 != nullptr);
    REQUIRE(b_matrix->hip_native_i64 != nullptr);
    REQUIRE(c_matrix->hip_native_i64 != nullptr);
    CHECK(a_matrix->hip_residues == nullptr);
    CHECK(b_matrix->hip_residues == nullptr);
    CHECK(c_matrix->hip_residues == nullptr);
    CHECK(a_matrix->hip_native_i64_bytes == static_cast<std::size_t>(m * k) * sizeof(int64_t));
    CHECK(b_matrix->hip_native_i64_bytes == static_cast<std::size_t>(k * n) * sizeof(int64_t));
    CHECK(c_matrix->hip_native_i64_bytes == static_cast<std::size_t>(m * n) * sizeof(int64_t));

    rns8_matrix_storage_info storage_info{};
    storage_info.struct_size = sizeof(storage_info);
    storage_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(a_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.uses_native_storage == 1);
    CHECK(storage_info.uses_residue_storage == 0);
    CHECK(storage_info.device_native_current == 0);
    CHECK(storage_info.device_native_bytes == static_cast<uint64_t>(m * k * sizeof(int64_t)));
    CHECK(std::string(storage_info.layout_version) == "native_i64_rowmajor_v1");
    CHECK(std::string(storage_info.storage_scope) == "native_device_storage");

    void* a_native = a_matrix->hip_native_i64;
    void* b_native = b_matrix->hip_native_i64;
    void* c_native = c_matrix->hip_native_i64;
    REQUIRE(rns8_pack_i64(vector, a_matrix, A.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B.data(), n, 2) == RNS8_SUCCESS);
    CHECK(a_matrix->hip_native_i64 == a_native);
    CHECK(b_matrix->hip_native_i64 == b_native);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK_FALSE(a_matrix->device_residues_current);
    CHECK_FALSE(b_matrix->device_residues_current);

    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(c_matrix->hip_native_i64 == c_native);
    CHECK(c_matrix->device_native_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_status_buffer != nullptr);

    REQUIRE(rns8_export_i64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    REQUIRE(rns8_get_matrix_storage_info(c_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.device_native_current == 1);
    CHECK(storage_info.device_residues_current == 0);

    void* status_buffer = c_matrix->hip_status_buffer;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(warmed_allocations.allocate_calls > 0);
    std::fill(vector_c.begin(), vector_c.end(), int64_t{0});

    REQUIRE(rns8_pack_i64(vector, a_matrix, A.data(), k, 3) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B.data(), n, 4) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(a_matrix->hip_native_i64 == a_native);
    CHECK(b_matrix->hip_native_i64 == b_native);
    CHECK(c_matrix->hip_native_i64 == c_native);
    CHECK(c_matrix->hip_status_buffer == status_buffer);
    CHECK(vector_c == cpu_c);

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
    std::vector<uint64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(a_matrix->hip_native_u64 != nullptr);
    REQUIRE(b_matrix->hip_native_u64 != nullptr);
    REQUIRE(c_matrix->hip_native_u64 != nullptr);
    CHECK(a_matrix->hip_residues == nullptr);
    CHECK(b_matrix->hip_residues == nullptr);
    CHECK(c_matrix->hip_residues == nullptr);

    rns8_matrix_storage_info storage_info{};
    storage_info.struct_size = sizeof(storage_info);
    storage_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(a_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.uses_native_storage == 1);
    CHECK(storage_info.uses_residue_storage == 0);
    CHECK(std::string(storage_info.layout_version) == "native_u64_rowmajor_v1");

    REQUIRE(rns8_pack_u64(vector, a_matrix, A.data(), k, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, b_matrix, B.data(), n, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(c_matrix->device_native_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    REQUIRE(rns8_export_u64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    const int64_t m = 1;
    const int64_t n = 1;
    const int64_t k = 4096;
    std::vector<int64_t> A(static_cast<std::size_t>(m * k));
    std::vector<int64_t> B(static_cast<std::size_t>(k * n));
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(kk)] = static_cast<int64_t>((kk % 9) - 4);
      B[static_cast<std::size_t>(kk)] = static_cast<int64_t>(3 - (kk % 7));
    }
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == "hip_vector_alu_i64_gemv_n1_exact_192b_v1");
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, a_matrix, A.data(), k, 21) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B.data(), n, 22) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    const int64_t m = 3;
    const int64_t n = 1;
    const int64_t k = 4096;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * k));
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t kk = 0; kk < k; ++kk) {
        A[static_cast<std::size_t>(row * k + kk)] = static_cast<uint64_t>(((row + kk) % 11) + 1);
      }
    }
    for (int64_t kk = 0; kk < k; ++kk) {
      B[static_cast<std::size_t>(kk)] = static_cast<uint64_t>((kk % 13) + 1);
    }
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == "hip_vector_alu_u64_gemv_n1_exact_192b_v1");
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, a_matrix, A.data(), k, 31) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, b_matrix, B.data(), n, 32) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    rns8_plan* rejected = nullptr;
    auto exact = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    CHECK(rns8_create_plan(vector, &exact, &rejected) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(rejected == nullptr);

    auto wrap = wrap_desc(1, 1, 1, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    CHECK(rns8_create_plan(vector, &wrap, &rejected) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(rejected == nullptr);
  }

  rns8_destroy_context(vector);
  rns8_destroy_context(cpu);
}

TEST_CASE("vector ALU native output materializes as Direct-HIP RNS input") {
  if (!hip_available()) {
    SKIP("no HIP device available for vector-to-RNS bridge smoke");
  }

  rns8_context* vector = create_context(RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 2;
    constexpr int64_t k0 = 3;
    constexpr int64_t bridge_k = 2;
    constexpr int64_t n = 2;
    const std::vector<int64_t> A = {2, -3, 5, -7, 11, 13};
    const std::vector<int64_t> B = {17, -19, 23, 29, -31, 37};
    const std::vector<int64_t> D = {3, -5, 7, 11};
    std::vector<int64_t> expected_first(static_cast<std::size_t>(m * bridge_k), 0);
    std::vector<int64_t> expected(static_cast<std::size_t>(m * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < bridge_k; ++col) {
        for (int64_t kk = 0; kk < k0; ++kk) {
          expected_first[static_cast<std::size_t>(row * bridge_k + col)] +=
              A[static_cast<std::size_t>(row * k0 + kk)] * B[static_cast<std::size_t>(kk * bridge_k + col)];
        }
      }
    }
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        for (int64_t kk = 0; kk < bridge_k; ++kk) {
          expected[static_cast<std::size_t>(row * n + col)] +=
              expected_first[static_cast<std::size_t>(row * bridge_k + kk)] *
              D[static_cast<std::size_t>(kk * n + col)];
        }
      }
    }

    auto vector_desc = signed_desc(m, bridge_k, k0, 100000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    auto hip_desc = signed_desc(m, n, bridge_k, 10000000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* vector_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_workspace* vector_workspace = nullptr;
    rns8_workspace* hip_workspace = nullptr;
    rns8_matrix* vector_a = nullptr;
    rns8_matrix* vector_b = nullptr;
    rns8_matrix* vector_c = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &vector_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, vector_plan, &vector_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
    auto vector_a_desc = matrix_desc(m, k0, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto vector_b_desc = matrix_desc(k0, bridge_k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto vector_c_desc = matrix_desc(m, bridge_k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto hip_a_desc = matrix_desc(m, bridge_k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto hip_b_desc = matrix_desc(bridge_k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto hip_c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &vector_a_desc, &vector_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &vector_b_desc, &vector_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &vector_c_desc, &vector_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_c_desc, &hip_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, vector_a, A.data(), k0, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, vector_b, B.data(), bridge_k, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, vector_plan, vector_a, vector_b, vector_c, vector_workspace) == RNS8_SUCCESS);
    REQUIRE(vector_c->device_native_current);
    CHECK_FALSE(vector_c->device_residues_current);
    const uint64_t produced_source_version = vector_c->source_version;
    REQUIRE(produced_source_version != 0);
    vector_c->source_version = 0;
    CHECK(rns8::detail::materialize_native_matrix_as_direct_rns(hip, hip_plan, vector_c, hip_a) ==
          RNS8_INVALID_ARGUMENT);
    CHECK_FALSE(hip_a->device_residues_current);
    CHECK(hip_a->source_version == 0);
    vector_c->source_version = produced_source_version;
    REQUIRE(rns8::detail::materialize_native_matrix_as_direct_rns(hip, hip_plan, vector_c, hip_a) == RNS8_SUCCESS);
    CHECK(hip_a->device_residues_current);
    CHECK_FALSE(hip_a->device_native_current);
    CHECK(hip_a->source_version == vector_c->source_version);
    REQUIRE(rns8_pack_i64(hip, hip_b, D.data(), n, 3) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    std::vector<int64_t> out(static_cast<std::size_t>(m * n), 0);
    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, out.data(), n) == RNS8_SUCCESS);
    CHECK(out == expected);

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_matrix(vector_c);
    rns8_destroy_matrix(vector_b);
    rns8_destroy_matrix(vector_a);
    rns8_destroy_workspace(hip_workspace);
    rns8_destroy_workspace(vector_workspace);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(vector_plan);
  }

  {
    constexpr int64_t m = 2;
    constexpr int64_t k0 = 2;
    constexpr int64_t bridge_k = 2;
    constexpr int64_t n = 3;
    const std::vector<uint64_t> A = {2, 3, 5, 7};
    const std::vector<uint64_t> B = {11, 13, 17, 19};
    const std::vector<uint64_t> D = {23, 29, 31, 37, 41, 43};
    std::vector<uint64_t> expected_first(static_cast<std::size_t>(m * bridge_k), 0);
    std::vector<uint64_t> expected(static_cast<std::size_t>(m * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < bridge_k; ++col) {
        for (int64_t kk = 0; kk < k0; ++kk) {
          expected_first[static_cast<std::size_t>(row * bridge_k + col)] +=
              A[static_cast<std::size_t>(row * k0 + kk)] * B[static_cast<std::size_t>(kk * bridge_k + col)];
        }
      }
    }
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        for (int64_t kk = 0; kk < bridge_k; ++kk) {
          expected[static_cast<std::size_t>(row * n + col)] +=
              expected_first[static_cast<std::size_t>(row * bridge_k + kk)] *
              D[static_cast<std::size_t>(kk * n + col)];
        }
      }
    }

    auto vector_desc = unsigned_desc(m, bridge_k, k0, 100000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    auto hip_desc = unsigned_desc(m, n, bridge_k, 10000000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* vector_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_workspace* vector_workspace = nullptr;
    rns8_workspace* hip_workspace = nullptr;
    rns8_matrix* vector_a = nullptr;
    rns8_matrix* vector_b = nullptr;
    rns8_matrix* vector_c = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &vector_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, vector_plan, &vector_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
    auto vector_a_desc = matrix_desc(m, k0, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto vector_b_desc = matrix_desc(k0, bridge_k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto vector_c_desc = matrix_desc(m, bridge_k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto hip_a_desc = matrix_desc(m, bridge_k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto hip_b_desc = matrix_desc(bridge_k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto hip_c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &vector_a_desc, &vector_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &vector_b_desc, &vector_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &vector_c_desc, &vector_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &hip_c_desc, &hip_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, vector_a, A.data(), k0, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, vector_b, B.data(), bridge_k, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, vector_plan, vector_a, vector_b, vector_c, vector_workspace) == RNS8_SUCCESS);
    REQUIRE(vector_c->device_native_current);
    CHECK_FALSE(vector_c->device_residues_current);
    REQUIRE(rns8::detail::materialize_native_matrix_as_direct_rns(hip, hip_plan, vector_c, hip_a) == RNS8_SUCCESS);
    CHECK(hip_a->device_residues_current);
    CHECK_FALSE(hip_a->device_native_current);
    CHECK(hip_a->source_version == vector_c->source_version);
    REQUIRE(rns8_pack_u64(hip, hip_b, D.data(), n, 13) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> out(static_cast<std::size_t>(m * n), 0);
    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, out.data(), n) == RNS8_SUCCESS);
    CHECK(out == expected);

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_matrix(vector_c);
    rns8_destroy_matrix(vector_b);
    rns8_destroy_matrix(vector_a);
    rns8_destroy_workspace(hip_workspace);
    rns8_destroy_workspace(vector_workspace);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(vector_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(vector);
}

TEST_CASE("vector ALU native export range errors leave destination unchanged") {
  if (!hip_available()) {
    SKIP("no HIP device available for vector ALU native export range smoke");
  }

  rns8_context* vector = create_context(RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  {
    constexpr int64_t m = 1;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    const int64_t A[] = {1};
    const int64_t B[] = {7, 11};
    auto desc = signed_desc(m, n, k, 10, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, a_matrix, A, k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B, n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<int64_t> out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    CHECK(rns8_export_i64(vector, plan, c_matrix, out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK(c_matrix->device_native_current);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    constexpr int64_t m = 1;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    const uint64_t A[] = {1};
    const uint64_t B[] = {7, 11};
    auto desc = unsigned_desc(m, n, k, 10, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, a_matrix, A, k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, b_matrix, B, n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<uint64_t> out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    CHECK(rns8_export_u64(vector, plan, c_matrix, out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK(c_matrix->device_native_current);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(vector);
}

