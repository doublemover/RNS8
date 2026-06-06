TEST_CASE("direct HIP bounded descriptors hard-reject invalid bound contracts") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded descriptor rejection smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* plan = nullptr;
  rns8_matrix* matrix = nullptr;

  auto signed_without_bound = signed_desc(1, 1, 1, 0, RNS8_BACKEND_HIP_DIRECT);
  signed_without_bound.bound_kind = RNS8_BOUND_NONE;
  CHECK(rns8_create_plan(hip, &signed_without_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto signed_with_unsigned_bound = signed_desc(1, 1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  signed_with_unsigned_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  CHECK(rns8_create_plan(hip, &signed_with_unsigned_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto unsigned_with_signed_bound = unsigned_desc(1, 1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  unsigned_with_signed_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  CHECK(rns8_create_plan(hip, &unsigned_with_signed_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto per_tile_with_global_bound = per_tile_unsigned_desc(65, 65, 1, bounds, RNS8_BACKEND_HIP_DIRECT);
  per_tile_with_global_bound.bound = 1;
  CHECK(rns8_create_plan(hip, &per_tile_with_global_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> too_few_bounds = {7, 1000, 7000000};
  auto per_tile_too_few = per_tile_unsigned_desc(65, 65, 1, too_few_bounds, RNS8_BACKEND_HIP_DIRECT);
  CHECK(rns8_create_plan(hip, &per_tile_too_few, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto insufficient_prefix = signed_desc(
      1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), RNS8_BACKEND_HIP_DIRECT);
  insufficient_prefix.max_prefix = 8;
  CHECK(rns8_create_plan(hip, &insufficient_prefix, &plan) == RNS8_RANGE_ERROR);
  CHECK(plan == nullptr);

  auto bad_matrix_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_NONE);
  CHECK(rns8_create_matrix(hip, &bad_matrix_desc, &matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(matrix == nullptr);

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP bounded oneshot matches CPU for signed and unsigned APIs") {
  if (!hip_available()) {
    SKIP("no HIP device available for public bounded GEMM smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 3;
    const int64_t A[] = {7, -3, 5, -11, 13, 17};
    const int64_t B[] = {2, -5, 19, 23, -29, 31};
    int64_t cpu_c[4] = {};
    int64_t hip_c[4] = {};
    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    hip_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, k, B, n, hip_c, n) == RNS8_SUCCESS);
    CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));
  }

  {
    const int64_t m = 1;
    const int64_t n = 1;
    const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
    const uint64_t expected_bound = static_cast<uint64_t>(k) * 127u * 127u;
    std::vector<int64_t> A(static_cast<std::size_t>(k), 127);
    std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
    int64_t cpu_c[1] = {};
    int64_t hip_c[1] = {};
    auto cpu_desc = signed_desc(m, n, k, expected_bound, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, expected_bound, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c, n) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == static_cast<int64_t>(expected_bound));
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::min()};
    const int64_t B[] = {1};
    int64_t cpu_c[1] = {};
    int64_t hip_c[1] = {};
    auto cpu_desc = signed_desc(1, 1, 1, 1ull << 63u, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(1, 1, 1, 1ull << 63u, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, 1, B, 1, cpu_c, 1) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, 1, B, 1, hip_c, 1) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == std::numeric_limits<int64_t>::min());
  }

  {
    const uint64_t A[] = {17, 3, 255, 9, 41, 5};
    const uint64_t B[] = {11, 7, 13, 19, 23, 29};
    uint64_t cpu_c[4] = {};
    uint64_t hip_c[4] = {};
    auto cpu_desc = unsigned_desc(2, 2, 3, 20000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(2, 2, 3, 20000, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 3, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 3, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(std::vector<uint64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<uint64_t>(std::begin(cpu_c), std::end(cpu_c)));
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t cpu_c[1] = {};
    uint64_t hip_c[1] = {};
    auto cpu_desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 1, B, 1, cpu_c, 1) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 1, B, 1, hip_c, 1) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == std::numeric_limits<uint64_t>::max());
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP bounded prefix-9 oneshot skips global RNS pack kernels") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP native-input bounded oneshot timing smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 4;
    constexpr int64_t lda = k + 1;
    constexpr int64_t ldb = n + 1;
    constexpr int64_t ldc = n + 1;
    std::vector<int64_t> A(static_cast<std::size_t>(m * lda), -999);
    std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), -777);
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), INT64_C(-0x12345));
    std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), INT64_C(-0x12345));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] = (row + 1) * (col % 2 == 0 ? 5 : -7);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] = (col + 2) * (row % 2 == 0 ? -3 : 11);
      }
    }
    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    hip_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) ==
            RNS8_SUCCESS);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_c.data(), ldc) ==
            RNS8_SUCCESS);
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(hip_c == cpu_c);
    CHECK(has_timing_label(hip_events, "residue_h2d_sync"));
    CHECK(has_timing_label(hip_events, "rns_gemm_kernel_group"));
    CHECK(has_timing_label(hip_events, "crt_export_status_memset"));
    CHECK(has_timing_label(hip_events, "crt_export_kernel"));
    CHECK(has_timing_label(hip_events, "crt_export_status_d2h"));
    CHECK(has_timing_label(hip_events, "crt_export_d2h"));
    CHECK_FALSE(has_timing_label(hip_events, "pack_kernel"));
  }

  {
    constexpr int64_t m = 3;
    constexpr int64_t n = 2;
    constexpr int64_t k = 5;
    constexpr int64_t lda = k + 2;
    constexpr int64_t ldb = n + 1;
    constexpr int64_t ldc = n + 1;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xabcdefu);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0x123456u);
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
    std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] = static_cast<uint64_t>((row + 3) * (col + 5));
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] = static_cast<uint64_t>((row + 7) * (col + 11));
      }
    }
    auto cpu_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    hip_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) ==
            RNS8_SUCCESS);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_c.data(), ldc) ==
            RNS8_SUCCESS);
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(hip_c == cpu_c);
    CHECK(has_timing_label(hip_events, "residue_h2d_sync"));
    CHECK(has_timing_label(hip_events, "rns_gemm_kernel_group"));
    CHECK(has_timing_label(hip_events, "crt_export_status_memset"));
    CHECK(has_timing_label(hip_events, "crt_export_kernel"));
    CHECK(has_timing_label(hip_events, "crt_export_status_d2h"));
    CHECK(has_timing_label(hip_events, "crt_export_d2h"));
    CHECK_FALSE(has_timing_label(hip_events, "pack_kernel"));
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP bounded oneshot matches CPU for cancellation and full-width stressors") {
  if (!hip_available()) {
    SKIP("no HIP device available for public bounded GEMM stress smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 4;
    const int64_t A[] = {
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min() + 1,
        37,
        -37,
        1000000000,
        -1000000000,
        -12345,
        12345};
    const int64_t B[] = {1, 1, 1, 1, 1000000, 1000000, 1000000, -1000000};
    int64_t cpu_c[4] = {};
    int64_t hip_c[4] = {};
    auto cpu_desc = signed_desc(m, n, k, 30000000000ull, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 30000000000ull, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, k, B, n, hip_c, n) == RNS8_SUCCESS);
    CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));
    CHECK(hip_c[0] == 0);
    CHECK(hip_c[1] == 74000000);
    CHECK(hip_c[2] == 0);
    CHECK(hip_c[3] == -24690000000LL);
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1, 0, 0, 1};
    int64_t cpu_c[2] = {};
    int64_t hip_c[2] = {};
    auto cpu_desc = signed_desc(1, 2, 2, 1ull << 63u, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(1, 2, 2, 1ull << 63u, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, 2, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, 2, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == std::numeric_limits<int64_t>::min());
    CHECK(hip_c[1] == std::numeric_limits<int64_t>::max());
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 42};
    const uint64_t B[] = {1, 0, 0, 1};
    uint64_t cpu_c[2] = {};
    uint64_t hip_c[2] = {};
    auto cpu_desc = unsigned_desc(1, 2, 2, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(1, 2, 2, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 2, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 2, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == std::numeric_limits<uint64_t>::max());
    CHECK(hip_c[1] == std::numeric_limits<uint64_t>::max() - 42);
  }

  {
    const int64_t m = 1;
    const int64_t n = 2;
    const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
    std::vector<uint64_t> A(static_cast<std::size_t>(k), 255);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
    for (int64_t kk = 0; kk < k; ++kk) {
      B[static_cast<std::size_t>(kk * n)] = 1;
      B[static_cast<std::size_t>(kk * n + 1)] = 2;
    }
    uint64_t cpu_c[2] = {};
    uint64_t hip_c[2] = {};
    const uint64_t bound = static_cast<uint64_t>(k) * 255u * 2u;
    auto cpu_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c, n) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == static_cast<uint64_t>(k) * 255u);
    CHECK(hip_c[1] == bound);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded oneshot matches CPU for signed and unsigned APIs") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile bounded GEMM smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 65;
    constexpr int64_t n = 65;
    constexpr int64_t k = 1;
    constexpr int64_t ldc = n + 1;
    std::vector<uint64_t> A(m * k);
    std::vector<uint64_t> B(k * n);
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
    std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
    }
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? 7 : 1000;
    }
    const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
    auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xdeadbeefdeadbeefull);
    }

    const std::vector<uint64_t> too_small_bounds = {6, 1000, 7000000, 1000000000};
    auto bad_cpu_desc = per_tile_unsigned_desc(m, n, k, too_small_bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto bad_hip_desc = per_tile_unsigned_desc(m, n, k, too_small_bounds, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &bad_cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) ==
          RNS8_RANGE_ERROR);
    CHECK(rns8_gemm_u64_oneshot(hip, &bad_hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) ==
          RNS8_RANGE_ERROR);

    const std::vector<uint64_t> zero_bad_bounds = {0, 1000, 7000000, 1000000000};
    auto zero_bad_cpu_desc = per_tile_unsigned_desc(m, n, k, zero_bad_bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto zero_bad_hip_desc = per_tile_unsigned_desc(m, n, k, zero_bad_bounds, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &zero_bad_cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) ==
          RNS8_RANGE_ERROR);
    CHECK(rns8_gemm_u64_oneshot(hip, &zero_bad_hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) ==
          RNS8_RANGE_ERROR);

    std::vector<uint64_t> zero_b = B;
    for (int64_t col = 0; col < n; ++col) {
      zero_b[static_cast<std::size_t>(col)] = col < 64 ? 0 : 1000;
    }
    const std::vector<uint64_t> zero_bounds = {0, 1000, 0, 1000000000};
    auto zero_cpu_desc = per_tile_unsigned_desc(m, n, k, zero_bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto zero_hip_desc = per_tile_unsigned_desc(m, n, k, zero_bounds, RNS8_BACKEND_HIP_DIRECT);
    zero_cpu_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    zero_hip_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    std::fill(cpu_c.begin(), cpu_c.end(), 0xdeadbeefdeadbeefull);
    std::fill(hip_c.begin(), hip_c.end(), 0xdeadbeefdeadbeefull);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &zero_cpu_desc, A.data(), k, zero_b.data(), n, cpu_c.data(), ldc) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_gemm_u64_oneshot(hip, &zero_hip_desc, A.data(), k, zero_b.data(), n, hip_c.data(), ldc) ==
            RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xdeadbeefdeadbeefull);
    }
  }

  {
    constexpr int64_t m = 65;
    constexpr int64_t n = 65;
    constexpr int64_t k = 1;
    constexpr int64_t ldc = n + 1;
    std::vector<int64_t> A(m * k);
    std::vector<int64_t> B(k * n);
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), INT64_C(0x123456789abcdef));
    std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), INT64_C(0x123456789abcdef));
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row)] = row < 64 ? -2 : -1000000;
    }
    A[0] = 0;
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? 3 : -1000;
    }
    const std::vector<uint64_t> bounds = {6, 2000, 3000000, 1000000000};
    auto cpu_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == INT64_C(0x123456789abcdef));
    }
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded K-split oneshot matches CPU selected prefix") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile bounded K-split selected-prefix smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  constexpr int64_t ldc = n + 1;
  const uint64_t ku = static_cast<uint64_t>(k);
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k));
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
  for (int64_t kk = 0; kk < k; ++kk) {
    A[static_cast<std::size_t>(kk)] = 1;
    A[static_cast<std::size_t>(k + kk)] = 2;
    B[static_cast<std::size_t>(kk * n)] = 3;
    B[static_cast<std::size_t>(kk * n + 1)] = 5;
  }

  const std::vector<uint64_t> bounds = {10u * ku};
  auto cpu_desc = unsigned_desc(m, n, k, 0, RNS8_BACKEND_CPU_REFERENCE);
  cpu_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  cpu_desc.tile_m = 64;
  cpu_desc.tile_n = 64;
  cpu_desc.tile_bounds = bounds.data();
  cpu_desc.tile_bounds_count = bounds.size();
  auto hip_desc = cpu_desc;
  hip_desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
            cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xccccccccccccccccull);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

