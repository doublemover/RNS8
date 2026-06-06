TEST_CASE("rocWMMA finite u8 K-split preserves signed centered accumulation") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* rocwmma = create_backend_context(RNS8_BACKEND_ROCWMMA);
  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  const int64_t k = 65537;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * n), 255);
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> rocwmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  auto rocwmma_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_ROCWMMA, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), n, cpu_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), n, hip_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(rocwmma, &rocwmma_desc, 256, A.data(), k, B.data(), n, rocwmma_out.data(), n) ==
          RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(rocwmma_out == cpu_out);

  rns8_destroy_context(rocwmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA wrap64 byte-GEMM36 candidate matches direct HIP and CPU oracle across tails") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  const std::vector<Wrap64CandidateShape> cases = {
      {"single-cell K tail", 1, 1, 1, 1, 1, 1, 0x6436776d6d617572ull},
      {"exact WMMA tile", 16, 16, 16, 2, 3, 4, 0x77726f6336343136ull},
      {"padded carry-heavy tile tails", 17, 19, 33, 3, 5, 7, 0x6436776d6d617572ull},
      {"ragged two-tile output", 31, 33, 47, 1, 2, 3, 0x7461696c73363477ull},
      {"release-shape 64x64x64 full output", 64, 64, 64, 0, 0, 0, 0x3634783634783634ull},
      {"release-shape 128x128x128 full output", 128, 128, 128, 0, 0, 0, 0x3132387831323878ull},
  };
  for (const auto& candidate_case : cases) {
    require_wrap64_rocwmma_candidate_matches_direct_hip_and_oracle(candidate_case);
  }
}

TEST_CASE("rocWMMA wrap64 byte-GEMM36 candidate enforces K boundary") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  constexpr int device_id = 0;
  require_wrap64_rocwmma_candidate_matches_direct_hip_and_oracle(
      {"maximum accepted K", 1, 1, 32768, 0, 0, 0, 0x3332373638776d6dull});

  HipBuffer a_limbs(8);
  HipBuffer b_limbs(8);
  HipBuffer c_limbs(8);
  CHECK(rns8::detail::rocwmma_wrap64_gemm_byte_limbs_candidate_device(
            device_id, a_limbs.get(), b_limbs.get(), c_limbs.get(), 1, 1, 32769) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("rocWMMA wrap64 byte-GEMM36 candidate matches direct HIP on large release shapes") {
  if (!rocwmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  const std::vector<Wrap64CandidateShape> cases = {
      {"release-shape 512x512x512 full direct-HIP output", 512, 512, 512, 0, 0, 0, 0x3531327835313278ull},
      {"release-shape 1024x1024x1024 full direct-HIP output", 1024, 1024, 1024, 0, 0, 0, 0x3130323478313032ull},
  };
  for (const auto& candidate_case : cases) {
    require_wrap64_rocwmma_candidate_matches_direct_hip_and_oracle(candidate_case, false);
  }
}
