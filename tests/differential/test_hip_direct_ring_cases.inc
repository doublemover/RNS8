TEST_CASE("direct HIP ring GEMM matches CPU reference for one modulus") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP smoke");
  }

  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const uint16_t modulus = 255;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20};
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * n), 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 1, 2);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP bounded export uses pinned staging for large padded host outputs") {
  if (!hip_available()) {
    SKIP("no HIP device available for large padded direct HIP export staging smoke");
  }

  constexpr int64_t m = 96;
  constexpr int64_t n = 96;
  constexpr int64_t k = 3;
  constexpr int64_t lda = k + 2;
  constexpr int64_t ldb = n + 3;
  constexpr int64_t ldc = n + 5;
  constexpr int64_t sentinel = std::numeric_limits<int64_t>::min() / 3;
  const uint64_t bound = 8192;

  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), 0);
  std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = (row % 11) - (col % 5);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = (col % 13) - (row % 7);
    }
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);
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
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), lda, 17) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_b, B.data(), ldb, 23) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_a, A.data(), lda, 17) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_b, B.data(), ldb, 23) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);

  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_export_i64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  auto export_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);

  CHECK(has_timing_label(export_events, "crt_export_kernel"));
  CHECK(has_timing_label(export_events, "crt_export_d2h"));
  CHECK(has_timing_label(export_events, "export_host_staging_copy"));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
            cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == sentinel);
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

TEST_CASE("direct HIP ring GEMM reciprocal reduction matches CPU across supported default moduli") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP reciprocal ladder smoke");
  }

  const int64_t m = 5;
  const int64_t n = 7;
  const int64_t k = 67;
  const int64_t lda = k + 3;
  const int64_t ldb = n + 2;
  const int64_t ldc = n + 4;
  const int8_t a_sentinel = static_cast<int8_t>(-88);
  const int8_t b_sentinel = static_cast<int8_t>(99);
  const int8_t c_sentinel = static_cast<int8_t>(-33);
  std::vector<int8_t> A(static_cast<std::size_t>(m * lda), a_sentinel);
  std::vector<int8_t> B(static_cast<std::size_t>(k * ldb), b_sentinel);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      const int pattern = static_cast<int>((row * 37 + kk * 19) % 11);
      const int mixed = static_cast<int>((row + 113 - (2 * kk) % 113) % 113);
      A[static_cast<std::size_t>(row * lda + kk)] =
          pattern == 0 ? static_cast<int8_t>(127)
                       : pattern == 1 ? static_cast<int8_t>(-127)
                                      : static_cast<int8_t>(mixed - 56);
    }
  }
  for (int64_t kk = 0; kk < k; ++kk) {
    for (int64_t col = 0; col < n; ++col) {
      const int pattern = static_cast<int>((kk * 23 + col * 29) % 13);
      B[static_cast<std::size_t>(kk * ldb + col)] =
          pattern == 0 ? static_cast<int8_t>(126)
                       : pattern == 1 ? static_cast<int8_t>(-128)
                                      : static_cast<int8_t>(((3 * kk + col) % 127) - 63);
    }
  }

  for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    std::vector<int8_t> cpu(static_cast<std::size_t>(m * ldc), c_sentinel);
    std::vector<int8_t> gpu(static_cast<std::size_t>(m * ldc), c_sentinel);

    rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, lda, ldb, ldc, modulus);
    run_resident_ring_gemm(A, B, gpu, m, n, k, lda, ldb, ldc, modulus, p, p + 1);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CAPTURE(p, modulus, row, col);
        CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] ==
              cpu[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CAPTURE(p, modulus, row, col);
        CHECK(cpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
        CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  }
}

TEST_CASE("direct HIP ring GEMM covers centered correction boundaries") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP centered-boundary smoke");
  }

  const int64_t m = 1;
  const int64_t n = 4;
  const int64_t k = 2;
  const uint16_t modulus = 255;
  const std::vector<int8_t> A = {1, 1};
  const std::vector<int8_t> B = {
      64,
      127,
      127,
      -64,
      64,
      0,
      127,
      -64};
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * n), 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  REQUIRE(cpu[0] == -127);
  REQUIRE(cpu[1] == 127);
  REQUIRE(cpu[2] == -1);
  REQUIRE(cpu[3] == 127);
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 1, 2);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP ring GEMM splits K above the int32 safe block") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP split smoke");
  }

  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const uint16_t modulus = 251;
  std::vector<int8_t> A(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> B(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> cpu(1, 0);
  std::vector<int8_t> gpu(1, 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 3, 4);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP tiled ring GEMM covers shared-memory tile tails with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP tiled-tail smoke");
  }

  const int64_t m = 17;
  const int64_t n = 19;
  const int64_t k = 130;
  const int64_t lda = 137;
  const int64_t ldb = 23;
  const int64_t ldc = 29;
  const uint16_t modulus = 253;
  const int8_t a_sentinel = static_cast<int8_t>(-99);
  const int8_t b_sentinel = static_cast<int8_t>(77);
  const int8_t c_sentinel = static_cast<int8_t>(44);
  std::vector<int8_t> A(static_cast<std::size_t>(m * lda), a_sentinel);
  std::vector<int8_t> B(static_cast<std::size_t>(k * ldb), b_sentinel);
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * ldc), c_sentinel);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      switch ((row * 17 + kk * 5) % 7) {
        case 0:
          A[static_cast<std::size_t>(row * lda + kk)] = 127;
          break;
        case 1:
          A[static_cast<std::size_t>(row * lda + kk)] = -126;
          break;
        case 2:
          A[static_cast<std::size_t>(row * lda + kk)] = 64;
          break;
        case 3:
          A[static_cast<std::size_t>(row * lda + kk)] = -63;
          break;
        default:
          A[static_cast<std::size_t>(row * lda + kk)] = static_cast<int8_t>((row - kk) % 11);
          break;
      }
    }
  }
  for (int64_t kk = 0; kk < k; ++kk) {
    for (int64_t col = 0; col < n; ++col) {
      switch ((kk * 11 + col * 3) % 7) {
        case 0:
          B[static_cast<std::size_t>(kk * ldb + col)] = 126;
          break;
        case 1:
          B[static_cast<std::size_t>(kk * ldb + col)] = -127;
          break;
        case 2:
          B[static_cast<std::size_t>(kk * ldb + col)] = 63;
          break;
        case 3:
          B[static_cast<std::size_t>(kk * ldb + col)] = -64;
          break;
        default:
          B[static_cast<std::size_t>(kk * ldb + col)] = static_cast<int8_t>((kk + col) % 13 - 6);
          break;
      }
    }
  }

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, lda, ldb, ldc, modulus);
  run_resident_ring_gemm(A, B, gpu, m, n, k, lda, ldb, ldc, modulus, 2, 3);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] ==
            cpu[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(cpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
  }
}

