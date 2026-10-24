
TEST_CASE("rocWMMA fused epilogue matches integer reference across tails and safe K slabs", "[hip][rocwmma][epilogue]") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (int64_t k : {1, 17, 65535, 65536, 65537}) for (uint32_t prefix : {1u, 3u, 20u}) {
    const int64_t m = k == 17 ? 17 : 2, n = k == 17 ? 19 : 3;
    const int64_t lda = k + 1, ldb = n + 2, ldd = n + 1, ldc = n + 3;
    std::vector<int8_t> a(prefix * m * lda), b(prefix * k * ldb), d(prefix * m * ldd);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>(static_cast<int>((i * 37) % 256) - 128);
    for (size_t i = 0; i < b.size(); ++i) b[i] = i % 2 ? int8_t{-128} : int8_t{127};
    for (size_t i = 0; i < d.size(); ++i) d[i] = static_cast<int8_t>(-1 - static_cast<int>(i % 128));
    std::vector<int64_t> dots(prefix * m * n);
    for (uint32_t p = 0; p < prefix; ++p) for (int64_t row = 0; row < m; ++row)
      for (int64_t col = 0; col < n; ++col) for (int64_t q = 0; q < k; ++q)
        dots[(p * m + row) * n + col] += int64_t{a[(p * m + row) * lda + q]} * b[(p * k + q) * ldb + col];
    std::vector<int8_t> output(prefix * m * ldc, 99);
    void *da = nullptr, *db = nullptr, *dd = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, d.size(), &dd) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, output.size(), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dd, d.data(), d.size()) == RNS8_SUCCESS);
    for (auto coefficients : {std::pair<int64_t, int64_t>{0, 0}, {0, 1}, {1, 0}, {-1, 1},
                             {INT64_MIN, INT64_MAX}, {INT64_MAX, INT64_MIN}, {INT64_MIN, INT64_MIN}}) {
      const auto [alpha, beta] = coefficients;
      std::fill(output.begin(), output.end(), int8_t{99});
      REQUIRE(hip_direct_copy_host_to_device(0, dc, output.data(), output.size()) == RNS8_SUCCESS);
      REQUIRE(rocwmma_exact_axpby_device(0, da, db, dd, dd, m, n, k, lda, ldb, ldd, ldc, prefix, alpha, beta)
              == RNS8_INVALID_ARGUMENT);
      REQUIRE(rocwmma_exact_axpby_device(0, da, db, dd, dc, m, n, k, lda, ldb, ldd, ldc, prefix, alpha, beta)
              == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0, output.data(), dc, output.size()) == RNS8_SUCCESS);
      for (uint32_t p = 0; p < prefix; ++p) for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < ldc; ++col) {
          int expected = 99;
          if (col < n) {
            const cpp_int value = cpp_int(alpha) * dots[(p * m + row) * n + col] +
                                  cpp_int(beta) * d[(p * m + row) * ldd + col];
            const int modulus = kDefaultModuli[p];
            expected = static_cast<int>(value % modulus);
            if (expected < 0) expected += modulus;
            if (expected >= (modulus + 1) / 2) expected -= modulus;
          }
          REQUIRE(output[(p * m + row) * ldc + col] == expected);
        }
    }
    for (void* buffer : {da, db, dd, dc}) REQUIRE(hip_direct_free(0, buffer) == RNS8_SUCCESS);
  }
}
