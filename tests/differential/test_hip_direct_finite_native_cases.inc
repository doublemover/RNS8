TEST_CASE("direct HIP finite u8 one-shot matches CPU for explicit ring and field moduli") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 17;
  constexpr int64_t n = 19;
  constexpr int64_t k = 33;
  constexpr int64_t lda = 40;
  constexpr int64_t ldb = 23;
  constexpr int64_t ldc = 29;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          static_cast<uint8_t>((row * 37 + col * 19 + (row ^ col) * 11) & 0xff);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          static_cast<uint8_t>((row * 23 + col * 41 + (row + col) * 7) & 0xff);
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
    CHECK_FALSE(has_timing_label(hip_events, "finite_ring_gemm_kernel"));

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
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 native A resident B path matches CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite native-A smoke");
  }

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
  };
  constexpr int64_t m = 7;
  constexpr int64_t n = 5;
  constexpr int64_t k = 9;
  constexpr int64_t lda = 13;
  constexpr int64_t ldb = 11;
  constexpr int64_t ldc = 8;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  for (const Case item : {
           Case{RNS8_FINITE_RING_U8, 255},
           Case{RNS8_FINITE_FIELD_U8, 251},
           Case{RNS8_FINITE_RING_U8, 256},
       }) {
    std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xee);
    std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0xdd);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] =
            static_cast<uint8_t>((row * 19 + col * 23 + item.modulus / 3) % item.modulus);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<uint8_t>((row * 29 + col * 31 + item.modulus / 5) % item.modulus);
      }
    }

    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);

    if (item.semantics == RNS8_FINITE_FIELD_U8) {
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
    }

    rns8_plan* hip_plan = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto b_desc = matrix_desc(k, n, item.semantics, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, item.semantics, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b, item.modulus, B.data(), ldb, 2) == RNS8_SUCCESS);

    void* d_a = nullptr;
    const std::size_t a_bytes = static_cast<std::size_t>(m * lda) * sizeof(uint8_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &d_a) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, A.data(), a_bytes) == RNS8_SUCCESS);

    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_finite_u8_native_a_resident_b_device(
                0,
                d_a,
                hip_b->hip_residues,
                hip_c->hip_residues,
                m,
                n,
                k,
                lda,
                hip_b->desc.logical_ld,
                hip_c->desc.logical_ld,
                item.modulus) == RNS8_SUCCESS);
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "finite_native_a_gemm_kernel"));
    CHECK_FALSE(has_timing_label(hip_events, "finite_resident_gemm_kernel"));

    rns8::test::set_host_residues_current(*hip_c, false);
    rns8::test::set_device_residues_current(*hip_c, true);
    rns8::test::set_host_byte_limbs_current(*hip_c, false);
    rns8::test::set_device_byte_limbs_current(*hip_c, false);
    rns8::test::set_host_native_current(*hip_c, false);
    rns8::test::set_device_native_current(*hip_c, false);
    hip_c->finite_modulus = item.modulus;
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, item.modulus, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);

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

    CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_plan(hip_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP bounded i64 native prefix9 colpair path matches CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded i64 native colpair smoke");
  }

  constexpr int64_t m = 17;
  constexpr int64_t n = 19;
  constexpr int64_t k = 23;
  constexpr int64_t lda = 29;
  constexpr int64_t ldb = 31;
  constexpr int64_t ldc = 23;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), -777);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), -888);
  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * ldc), -999);
  std::vector<int64_t> hip_out(static_cast<std::size_t>(m * ldc), -999);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = static_cast<int64_t>((row * 17 + col * 5) % 23) - 11;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = static_cast<int64_t>((row * 7 + col * 13) % 19) - 9;
    }
  }

  auto cpu_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_DIRECT);
  cpu_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  hip_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
          RNS8_SUCCESS);

  rns8_plan* hip_plan = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  void* d_a = nullptr;
  void* d_b = nullptr;
  const std::size_t a_bytes = A.size() * sizeof(int64_t);
  const std::size_t b_bytes = B.size() * sizeof(int64_t);
  REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &d_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, b_bytes, &d_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, A.data(), a_bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b, B.data(), b_bytes) == RNS8_SUCCESS);

  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8::detail::hip_direct_gemm_i64_native_prefix9_colpair_device(
              0, d_a, d_b, hip_c->hip_residues, m, n, k, lda, ldb, hip_c->desc.logical_ld) == RNS8_SUCCESS);
  const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_events, "rns_gemm_kernel_group"));

  rns8::test::set_host_residues_current(*hip_c, false);
  rns8::test::set_device_residues_current(*hip_c, true);
  rns8::test::set_host_byte_limbs_current(*hip_c, false);
  rns8::test::set_device_byte_limbs_current(*hip_c, false);
  rns8::test::set_host_native_current(*hip_c, false);
  rns8::test::set_device_native_current(*hip_c, false);
  REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
            cpu_out[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
    }
  }

  CHECK(rns8::detail::hip_direct_free(0, d_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
  rns8_destroy_matrix(hip_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP bounded native A resident B path matches CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded native A resident B smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  SECTION("bounded i64") {
    constexpr int64_t m = 7;
    constexpr int64_t n = 5;
    constexpr int64_t k = 9;
    constexpr int64_t lda = 13;
    constexpr int64_t ldb = 11;
    constexpr int64_t ldc = 8;
    std::vector<int64_t> A(static_cast<std::size_t>(m * lda), -777);
    std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), -888);
    std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * ldc), -999);
    std::vector<int64_t> hip_out(static_cast<std::size_t>(m * ldc), -999);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] = static_cast<int64_t>((row * 17 + col * 5) % 23) - 11;
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] = static_cast<int64_t>((row * 7 + col * 13) % 19) - 9;
      }
    }

    auto cpu_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
            RNS8_SUCCESS);

    rns8_plan* hip_plan = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(hip, hip_a, A.data(), lda, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(hip, hip_b, B.data(), ldb, 2) == RNS8_SUCCESS);

    void* d_a = nullptr;
    const std::size_t a_bytes = static_cast<std::size_t>(m * lda) * sizeof(int64_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &d_a) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, A.data(), a_bytes) == RNS8_SUCCESS);

    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
                0, d_a, hip_b, hip_c, m, n, k, lda, 5) == RNS8_SUCCESS);
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(hip_events, "rns_gemm_kernel_group"));
    CHECK(hip_c->source_version == 5);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == -999);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_matrix(
                0, d_a, hip_b, hip_c, m, n, k, lda, 7) == RNS8_SUCCESS);
    const auto uniform_small_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(uniform_small_events, "bounded_uniform_small_native_a_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(uniform_small_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 7);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    void* d_b = nullptr;
    const std::size_t b_bytes = static_cast<std::size_t>(k * ldb) * sizeof(int64_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, b_bytes, &d_b) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b, B.data(), b_bytes) == RNS8_SUCCESS);

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix(
                0, hip_a, d_b, hip_c, m, n, k, ldb, 17) == RNS8_SUCCESS);
    const auto hip_native_b_colpair_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_native_b_colpair_events, "bounded_native_b_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(hip_native_b_colpair_events, "bounded_native_a_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 17);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    std::vector<int8_t> A_i8(static_cast<std::size_t>(m * lda), 0);
    std::vector<int8_t> B_i8(static_cast<std::size_t>(k * ldb), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A_i8[static_cast<std::size_t>(row * lda + col)] =
            static_cast<int8_t>(A[static_cast<std::size_t>(row * lda + col)]);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B_i8[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<int8_t>(B[static_cast<std::size_t>(row * ldb + col)]);
      }
    }
    void* d_a_i8 = nullptr;
    void* d_b_i8 = nullptr;
    const std::size_t a_i8_bytes = A_i8.size() * sizeof(int8_t);
    const std::size_t b_i8_bytes = B_i8.size() * sizeof(int8_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, a_i8_bytes, &d_a_i8) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_allocate(0, b_i8_bytes, &d_b_i8) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a_i8, A_i8.data(), a_i8_bytes) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b_i8, B_i8.data(), b_i8_bytes) == RNS8_SUCCESS);

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 9) == RNS8_SUCCESS);
    const auto uniform_small_i8_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(uniform_small_i8_events, "bounded_uniform_small_i8_ab_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(uniform_small_i8_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 9);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 11) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_events,
        "bounded_uniform_small_i8_ab_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 11);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 13) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_reuse_a_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_reuse_a_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_reuse_a_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 13);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), -999);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 15) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_transient_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 15);

    REQUIRE(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == -999);
      }
    }

    CHECK(rns8::detail::hip_direct_free(0, d_b_i8) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_a_i8) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_b) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_plan(hip_plan);
  }

  SECTION("bounded u64") {
    constexpr int64_t m = 6;
    constexpr int64_t n = 7;
    constexpr int64_t k = 8;
    constexpr int64_t lda = 10;
    constexpr int64_t ldb = 12;
    constexpr int64_t ldc = 9;
    std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xddddddddddddddddull);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xeeeeeeeeeeeeeeeeull);
    std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xffffffffffffffffull);
    std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * ldc), 0xffffffffffffffffull);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] = static_cast<uint64_t>((row * 11 + col * 3) % 17);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] = static_cast<uint64_t>((row * 5 + col * 7) % 13);
      }
    }

    auto cpu_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
            RNS8_SUCCESS);

    rns8_plan* hip_plan = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 3) == RNS8_SUCCESS);

    void* d_a = nullptr;
    const std::size_t a_bytes = static_cast<std::size_t>(m * lda) * sizeof(uint64_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &d_a) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, A.data(), a_bytes) == RNS8_SUCCESS);

    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
                0, d_a, hip_b, hip_c, m, n, k, lda, 6) == RNS8_SUCCESS);
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(hip_events, "rns_gemm_kernel_group"));
    CHECK(hip_c->source_version == 6);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
                0, d_a, hip_b, hip_c, m, n, k, lda, 16) == RNS8_SUCCESS);
    const auto hip_colpair_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_colpair_events, "bounded_native_a_colpair_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(hip_colpair_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 16);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_matrix(
                0, d_a, hip_b, hip_c, m, n, k, lda, 8) == RNS8_SUCCESS);
    const auto uniform_small_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(uniform_small_events, "bounded_uniform_small_native_a_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(uniform_small_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 8);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    void* d_b = nullptr;
    const std::size_t b_bytes = static_cast<std::size_t>(k * ldb) * sizeof(uint64_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, b_bytes, &d_b) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b, B.data(), b_bytes) == RNS8_SUCCESS);

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
                0, hip_a, d_b, hip_c, m, n, k, ldb, 18) == RNS8_SUCCESS);
    const auto hip_native_b_colpair_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_native_b_colpair_events, "bounded_native_b_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(hip_native_b_colpair_events, "bounded_native_a_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 18);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::vector<int8_t> A_i8(static_cast<std::size_t>(m * lda), 0);
    std::vector<int8_t> B_i8(static_cast<std::size_t>(k * ldb), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A_i8[static_cast<std::size_t>(row * lda + col)] =
            static_cast<int8_t>(A[static_cast<std::size_t>(row * lda + col)]);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B_i8[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<int8_t>(B[static_cast<std::size_t>(row * ldb + col)]);
      }
    }
    void* d_a_i8 = nullptr;
    void* d_b_i8 = nullptr;
    const std::size_t a_i8_bytes = A_i8.size() * sizeof(int8_t);
    const std::size_t b_i8_bytes = B_i8.size() * sizeof(int8_t);
    REQUIRE(rns8::detail::hip_direct_allocate(0, a_i8_bytes, &d_a_i8) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_allocate(0, b_i8_bytes, &d_b_i8) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a_i8, A_i8.data(), a_i8_bytes) == RNS8_SUCCESS);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b_i8, B_i8.data(), b_i8_bytes) == RNS8_SUCCESS);

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 10) == RNS8_SUCCESS);
    const auto uniform_small_i8_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(uniform_small_i8_events, "bounded_uniform_small_i8_ab_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(uniform_small_i8_events, "bounded_native_a_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 10);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 12) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_events,
        "bounded_uniform_small_i8_ab_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 12);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 14) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_reuse_a_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_reuse_a_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_reuse_a_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 14);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xffffffffffffffffull);
    rns8::detail::hip_direct_timing_set_enabled(true);
    rns8::detail::hip_direct_timing_reset();
    REQUIRE(rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
                0, d_a_i8, d_b_i8, hip_c, m, n, k, lda, ldb, 20) == RNS8_SUCCESS);
    const auto uniform_small_i8_colpair_transient_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"));
    CHECK_FALSE(has_timing_label(
        uniform_small_i8_colpair_transient_events,
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"));
    CHECK(hip_c->source_version == 20);

    REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xffffffffffffffffull);
      }
    }

    CHECK(rns8::detail::hip_direct_free(0, d_b_i8) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_a_i8) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_b) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_plan(hip_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

