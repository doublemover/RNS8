TEST_CASE("rocWMMA fused adaptive bounded schedule matches CPU and direct HIP") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 64;
  std::vector<uint64_t> bounds = {64, 1024, uint64_t{1} << 32u, uint64_t{1} << 40u};
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = static_cast<uint64_t>((row + col) & 1);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>(((row ^ col) & 1) == 0 ? 1 : 0);
    }
  }
  std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> rocwmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  auto rocwmma_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_ROCWMMA);

  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(rocwmma, &rocwmma_desc, A.data(), k, B.data(), n, rocwmma_out.data(), n) == RNS8_SUCCESS);
  require_same_u64(cpu_out, hip_out);
  require_same_u64(cpu_out, rocwmma_out);

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA exact-wide RNS output matches CPU and direct HIP limbs") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);
  constexpr int64_t m = 48;
  constexpr int64_t n = 80;
  constexpr int64_t k = 32;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = (row + col) % 3 == 0 ? -7 : 5;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = (row + 2 * col) % 5 - 2;
    }
  }
  constexpr uint32_t limb_count = 2;
  constexpr int64_t limb_ld = n;

  auto run_backend = [&](rns8_context* ctx, rns8_backend_kind backend) {
    auto desc = exact_signed_desc(m, n, k, backend);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a = nullptr;
    rns8_matrix* b = nullptr;
    rns8_matrix* c = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    if (backend == RNS8_BACKEND_ROCWMMA) {
      CHECK(
          std::string(info.selected_kernel) ==
          "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2");
      CHECK(std::string(info.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue_rns_output");
      CHECK(info.workspace_required_bytes > 0);
      CHECK(info.accumulator_uses_int32_inner_product == 1);
      CHECK(info.accumulator_k_block_size == static_cast<uint64_t>(k));
      CHECK(info.accumulator_k_block_cap == 65536);
      CHECK(std::string(info.accumulator_type) == "int32");
      CHECK(std::string(info.accumulator_safety_status) == "safe_int32_k_block_split");
    }
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, A.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, B.data(), n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, a, b, c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs.data(), limb_ld, limb_count) == RNS8_SUCCESS);
    rns8_destroy_matrix(c);
    rns8_destroy_matrix(b);
    rns8_destroy_matrix(a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    return limbs;
  };

  const auto cpu_limbs = run_backend(cpu, RNS8_BACKEND_CPU_REFERENCE);
  const auto hip_limbs = run_backend(hip, RNS8_BACKEND_HIP_DIRECT);
  const auto rocwmma_limbs = run_backend(rocwmma, RNS8_BACKEND_ROCWMMA);
  require_same_u64(cpu_limbs, hip_limbs);
  require_same_u64(cpu_limbs, rocwmma_limbs);

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA finite u8 backend matches CPU and direct HIP across padded strides") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);
  constexpr int64_t m = 48;
  constexpr int64_t n = 80;
  constexpr int64_t k = 32;
  constexpr int64_t lda = k + 3;
  constexpr int64_t ldb = n + 5;
  constexpr int64_t ldc = n + 7;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = static_cast<uint8_t>((row * 17 + col * 23) % 251);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = static_cast<uint8_t>((row * 29 + col * 31 + 7) % 251);
    }
  }
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
  std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
  std::vector<uint8_t> rocwmma_out(static_cast<std::size_t>(m * ldc), 0xcc);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIP_DIRECT);
  auto rocwmma_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_ROCWMMA);

  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              cpu, &cpu_desc, 251, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              hip, &hip_desc, 251, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              rocwmma, &rocwmma_desc, 251, A.data(), lda, B.data(), ldb, rocwmma_out.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(rocwmma_out == cpu_out);

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA finite u8 common moduli report specialized reducer kernels") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);
  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
    const char* selected_kernel;
  };
  const Case cases[] = {
      {RNS8_FINITE_FIELD_U8, 251, "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2"},
      {RNS8_FINITE_RING_U8, 255, "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2"},
      {RNS8_FINITE_RING_U8, 256, "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2"},
  };
  for (const auto& item : cases) {
    auto desc = finite_desc(64, 64, 64, item.semantics, RNS8_BACKEND_ROCWMMA, item.modulus);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(rocwmma, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == item.selected_kernel);
    CHECK(std::string(info.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export");
    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(rocwmma);
}

