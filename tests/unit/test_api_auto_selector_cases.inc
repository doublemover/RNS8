#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
TEST_CASE("AUTO plan consumes reviewed vector ALU cache entry for final native bounded output") {
  constexpr int64_t m = 32;
  constexpr int64_t n = 32;
  constexpr int64_t k = 32;

  rns8_context_options vector_options{};
  vector_options.struct_size = sizeof(vector_options);
  vector_options.abi_version = RNS8_ABI_VERSION;
  vector_options.requested_backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  rns8_context* vector_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &vector_options, &vector_ctx) == RNS8_SUCCESS);

  rns8_device_info vector_device{};
  vector_device.struct_size = sizeof(vector_device);
  vector_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(vector_ctx, &vector_device) == RNS8_SUCCESS);

  rns8_gemm_desc vector_desc{};
  vector_desc.struct_size = sizeof(vector_desc);
  vector_desc.abi_version = RNS8_ABI_VERSION;
  vector_desc.semantics = RNS8_BOUNDED_U64;
  vector_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  vector_desc.requested_backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  vector_desc.m = m;
  vector_desc.n = n;
  vector_desc.k = k;
  vector_desc.bound = static_cast<uint64_t>(k);
  vector_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* vector_plan = nullptr;
  REQUIRE(rns8_create_plan(vector_ctx, &vector_desc, &vector_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info vector_info{};
  vector_info.struct_size = sizeof(vector_info);
  vector_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(vector_plan, &vector_info) == RNS8_SUCCESS);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-vector-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << vector_info.autotune_key << "\","
          << "\"selected_backend\":\"hip-vector-alu-int64\","
          << "\"selected_kernel\":\"" << vector_info.selected_kernel << "\","
          << "\"target_id\":\"" << vector_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << vector_info.accelerator_version << "\","
          << "\"semantic_contract\":\"bounded_u64\","
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=1;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << vector_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << vector_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << vector_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"" << reviewed_release_status_for_target(vector_device.gcn_arch) << "\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-03T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(vector_plan);
  rns8_destroy_context(vector_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = vector_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == vector_info.autotune_key);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(auto_plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) != 0);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE) != 0);

  auto a_desc = bounded_matrix_desc(m, k, RNS8_BOUNDED_U64);
  auto b_desc = bounded_matrix_desc(k, n, RNS8_BOUNDED_U64);
  auto c_desc = bounded_matrix_desc(m, n, RNS8_BOUNDED_U64);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<uint64_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<uint64_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<uint64_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_u64(auto_ctx, A, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(auto_ctx, B, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(auto_ctx, auto_plan, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(auto_ctx, auto_plan, C, c.data(), n) == RNS8_SUCCESS);
  for (uint64_t value : c) {
    CHECK(value == static_cast<uint64_t>(k));
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}
#endif

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
TEST_CASE("AUTO plan consumes reviewed CK cache entry with HIP resident matrices") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options ck_options{};
  ck_options.struct_size = sizeof(ck_options);
  ck_options.abi_version = RNS8_ABI_VERSION;
  ck_options.requested_backend = RNS8_BACKEND_CK;
  rns8_context* ck_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &ck_options, &ck_ctx) == RNS8_SUCCESS);

  rns8_device_info ck_device{};
  ck_device.struct_size = sizeof(ck_device);
  ck_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ck_ctx, &ck_device) == RNS8_SUCCESS);

  rns8_gemm_desc ck_desc{};
  ck_desc.struct_size = sizeof(ck_desc);
  ck_desc.abi_version = RNS8_ABI_VERSION;
  ck_desc.semantics = RNS8_BOUNDED_I64;
  ck_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  ck_desc.requested_backend = RNS8_BACKEND_CK;
  ck_desc.m = m;
  ck_desc.n = n;
  ck_desc.k = k;
  ck_desc.bound = static_cast<uint64_t>(k);
  ck_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* ck_plan = nullptr;
  REQUIRE(rns8_create_plan(ck_ctx, &ck_desc, &ck_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info ck_info{};
  ck_info.struct_size = sizeof(ck_info);
  ck_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(ck_plan, &ck_info) == RNS8_SUCCESS);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-ck-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << ck_info.autotune_key << "\","
          << "\"selected_backend\":\"ck\","
          << "\"selected_kernel\":\"" << ck_info.selected_kernel << "\","
          << "\"target_id\":\"" << ck_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << ck_info.accelerator_version << "\","
          << "\"semantic_contract\":\"bounded_i64\","
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=1;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << ck_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << ck_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << ck_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"" << reviewed_release_status_for_target(ck_device.gcn_arch) << "\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-02T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(ck_plan);
  rns8_destroy_context(ck_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = ck_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_CK);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == ck_info.autotune_key);

  auto a_desc = bounded_matrix_desc(m, k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(k, n, RNS8_BOUNDED_I64);
  auto c_desc = bounded_matrix_desc(m, n, RNS8_BOUNDED_I64);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<int64_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<int64_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<int64_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_i64(auto_ctx, A, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(auto_ctx, B, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(auto_ctx, auto_plan, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(auto_ctx, auto_plan, C, c.data(), n) == RNS8_SUCCESS);
  for (int64_t value : c) {
    CHECK(value == k);
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}
#endif

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
TEST_CASE("AUTO plan consumes reviewed rocWMMA bounded cache entry with HIP resident matrices") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options wmma_options{};
  wmma_options.struct_size = sizeof(wmma_options);
  wmma_options.abi_version = RNS8_ABI_VERSION;
  wmma_options.requested_backend = RNS8_BACKEND_ROCWMMA;
  rns8_context* wmma_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &wmma_options, &wmma_ctx) == RNS8_SUCCESS);

  rns8_device_info wmma_device{};
  wmma_device.struct_size = sizeof(wmma_device);
  wmma_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(wmma_ctx, &wmma_device) == RNS8_SUCCESS);

  rns8_gemm_desc wmma_desc{};
  wmma_desc.struct_size = sizeof(wmma_desc);
  wmma_desc.abi_version = RNS8_ABI_VERSION;
  wmma_desc.semantics = RNS8_BOUNDED_I64;
  wmma_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  wmma_desc.requested_backend = RNS8_BACKEND_ROCWMMA;
  wmma_desc.m = m;
  wmma_desc.n = n;
  wmma_desc.k = k;
  wmma_desc.bound = static_cast<uint64_t>(k);
  wmma_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* wmma_plan = nullptr;
  REQUIRE(rns8_create_plan(wmma_ctx, &wmma_desc, &wmma_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info wmma_info{};
  wmma_info.struct_size = sizeof(wmma_info);
  wmma_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(wmma_plan, &wmma_info) == RNS8_SUCCESS);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-rocwmma-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << wmma_info.autotune_key << "\","
          << "\"selected_backend\":\"rocwmma\","
          << "\"selected_kernel\":\"" << wmma_info.selected_kernel << "\","
          << "\"target_id\":\"" << wmma_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << wmma_info.accelerator_version << "\","
          << "\"semantic_contract\":\"bounded_i64\","
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=1;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << wmma_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << wmma_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << wmma_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"" << reviewed_release_status_for_target(wmma_device.gcn_arch) << "\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-03T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(wmma_plan);
  rns8_destroy_context(wmma_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = wmma_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_ROCWMMA);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == wmma_info.autotune_key);

  auto a_desc = bounded_matrix_desc(m, k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(k, n, RNS8_BOUNDED_I64);
  auto c_desc = bounded_matrix_desc(m, n, RNS8_BOUNDED_I64);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<int64_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<int64_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<int64_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_i64(auto_ctx, A, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(auto_ctx, B, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(auto_ctx, auto_plan, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(auto_ctx, auto_plan, C, c.data(), n) == RNS8_SUCCESS);
  for (int64_t value : c) {
    CHECK(value == k);
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}

TEST_CASE("AUTO plan consumes reviewed rocWMMA finite-u8 cache entry when modulus is plan keyed") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 64;
  constexpr int64_t k = 64;

  rns8_context_options wmma_options{};
  wmma_options.struct_size = sizeof(wmma_options);
  wmma_options.abi_version = RNS8_ABI_VERSION;
  wmma_options.requested_backend = RNS8_BACKEND_ROCWMMA;
  rns8_context* wmma_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &wmma_options, &wmma_ctx) == RNS8_SUCCESS);

  rns8_device_info wmma_device{};
  wmma_device.struct_size = sizeof(wmma_device);
  wmma_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(wmma_ctx, &wmma_device) == RNS8_SUCCESS);

  rns8_gemm_desc wmma_desc{};
  wmma_desc.struct_size = sizeof(wmma_desc);
  wmma_desc.abi_version = RNS8_ABI_VERSION;
  wmma_desc.semantics = RNS8_FINITE_RING_U8;
  wmma_desc.bound_kind = RNS8_BOUND_NONE;
  wmma_desc.requested_backend = RNS8_BACKEND_ROCWMMA;
  wmma_desc.m = m;
  wmma_desc.n = n;
  wmma_desc.k = k;
  wmma_desc.finite_modulus = 251;

  rns8_plan* wmma_plan = nullptr;
  REQUIRE(rns8_create_plan(wmma_ctx, &wmma_desc, &wmma_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info wmma_info{};
  wmma_info.struct_size = sizeof(wmma_info);
  wmma_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(wmma_plan, &wmma_info) == RNS8_SUCCESS);
  CHECK(std::string(wmma_info.autotune_key).find(";finite_modulus=251;") != std::string::npos);
  CHECK(std::string(wmma_info.autotune_key).find(";epilogue=") != std::string::npos);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-rocwmma-finite-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << wmma_info.autotune_key << "\","
          << "\"selected_backend\":\"rocwmma\","
          << "\"selected_kernel\":\"" << wmma_info.selected_kernel << "\","
          << "\"target_id\":\"" << wmma_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << wmma_info.accelerator_version << "\","
          << "\"semantic_contract\":\"finite_ring_u8\","
          << "\"finite_modulus\":251,"
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=0;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << wmma_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << wmma_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << wmma_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"" << reviewed_release_status_for_target(wmma_device.gcn_arch) << "\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-03T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(wmma_plan);
  rns8_destroy_context(wmma_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = wmma_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_ROCWMMA);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == wmma_info.autotune_key);

  auto a_desc = finite_matrix_desc(m, k, RNS8_FINITE_RING_U8);
  auto b_desc = finite_matrix_desc(k, n, RNS8_FINITE_RING_U8);
  auto c_desc = finite_matrix_desc(m, n, RNS8_FINITE_RING_U8);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<uint8_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<uint8_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<uint8_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_finite_u8(auto_ctx, A, 251, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(auto_ctx, B, 251, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(auto_ctx, auto_plan, 251, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(auto_ctx, auto_plan, 251, C, c.data(), n) == RNS8_SUCCESS);
  for (uint8_t value : c) {
    CHECK(value == static_cast<uint8_t>(k % 251));
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}
#endif

