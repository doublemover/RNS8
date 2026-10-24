TEST_CASE("direct HIP finite INT4 WMMA matches exact reference across tails and reduction slabs") {
  if (!hip_available()) SKIP("no HIP device for finite INT4 differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (const int k : {1, 15, 16, 17, 65535, 65536, 65537}) {
    const int m = k < 32 ? 19 : 2, n = k < 32 ? 21 : 3;
    const int lda = k + 1, ldb = n + 2, ldc = n + 3;
    std::vector<int8_t> a(m * lda, 99), b(k * ldb, 99), c(m * ldc, 99);
    for (int row = 0; row < m; ++row)
      for (int q = 0; q < k; ++q) a[row * lda + q] = static_cast<int8_t>((row * 71 + q * 19) % 256 - 128);
    for (int q = 0; q < k; ++q)
      for (int col = 0; col < n; ++col) b[q * ldb + col] = static_cast<int8_t>((q * 37 + col * 11) % 256 - 128);
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, c.size(), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) == RNS8_SUCCESS);
    bool unsupported = false;
    for (uint32_t modulus = 2; modulus <= 16; ++modulus) {
      std::fill(c.begin(), c.end(), 99);
      REQUIRE(hip_direct_copy_host_to_device(0, dc, c.data(), c.size()) == RNS8_SUCCESS);
      const auto status = hip_direct_gemm_finite_int4_device(0, static_cast<const int8_t*>(da),
          static_cast<const int8_t*>(db), static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, modulus);
      REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc, c.size()) == RNS8_SUCCESS);
      if (status == RNS8_UNSUPPORTED_BACKEND) {
        REQUIRE(std::all_of(c.begin(), c.end(), [](int8_t x) { return x == 99; }));
        unsupported = true;
        break;
      }
      REQUIRE(status == RNS8_SUCCESS);
      for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
          cpp_int expected = 0;
          for (int q = 0; q < k; ++q) expected += cpp_int(a[row * lda + q]) * b[q * ldb + col];
          int residue = (expected % modulus).convert_to<int>();
          if (residue < 0) residue += modulus;
          if (residue >= static_cast<int>((modulus + 1) / 2)) residue -= modulus;
          REQUIRE(c[row * ldc + col] == residue);
        }
        for (int col = n; col < ldc; ++col) REQUIRE(c[row * ldc + col] == 99);
      }
    }
    REQUIRE(hip_direct_gemm_finite_int4_device(0, static_cast<const int8_t*>(da),
        static_cast<const int8_t*>(db), static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, 17) == RNS8_INVALID_ARGUMENT);
    REQUIRE(hip_direct_gemm_finite_int4_device(0, static_cast<const int8_t*>(da),
        static_cast<const int8_t*>(db), static_cast<int8_t*>(da), m, n, k, lda, ldb, ldc, 16) == RNS8_INVALID_ARGUMENT);
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
    if (unsupported) SKIP("compiled device target lacks RDNA3 INT4 WMMA");
  }
}
