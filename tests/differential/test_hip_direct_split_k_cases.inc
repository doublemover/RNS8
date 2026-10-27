TEST_CASE("direct HIP explicit split K reduces byte partials through bounded merge levels") {
  if (!hip_available()) SKIP("no HIP device for split-K differential");
  using namespace rns8::detail;
  constexpr int64_t m = 2, n = 3, ldc = 5;
  for (const int64_t k : {1, 7, 65535, 65536, 65537, 131073}) {
    const int64_t lda = k + 3, ldb = n + 2;
    for (const uint32_t prefix : {1u, 9u, 20u}) {
      std::vector<int8_t> a(prefix * m * lda, 77), b(prefix * k * ldb, 88);
      std::vector<int8_t> expected(prefix * m * ldc, 99), actual(expected.size());
      for (uint32_t p = 0; p < prefix; ++p) {
        for (int64_t row = 0; row < m; ++row)
          for (int64_t q = 0; q < k; ++q)
            a[p * m * lda + row * lda + q] = row == 0 ? -128 : (q % 2 ? 127 : -128);
        for (int64_t q = 0; q < k; ++q)
          for (int64_t col = 0; col < n; ++col)
            b[p * k * ldb + q * ldb + col] = col == 0 ? -128 : (col == 1 ? 127 : (q % 2 ? -128 : 127));
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < n; ++col) {
            // At these dimensions the exact sum is <2^32 in magnitude.
            int64_t sum = 0;
            for (int64_t q = 0; q < k; ++q)
              sum += static_cast<int64_t>(a[p * m * lda + row * lda + q]) * b[p * k * ldb + q * ldb + col];
            expected[p * m * ldc + row * ldc + col] = centered_residue(
                boost::multiprecision::cpp_int(sum), kDefaultModuli[p]);
          }
      }
      void *da = nullptr, *db = nullptr, *dc = nullptr;
      REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, expected.size(), &dc) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) == RNS8_SUCCESS);
      const uint32_t minimum = static_cast<uint32_t>((k + 65535) / 65536);
      for (const uint32_t splits : {minimum, minimum + 1, 129u}) {
        if (splits > static_cast<uint64_t>(k)) continue;
        std::fill(actual.begin(), actual.end(), int8_t{99});
        REQUIRE(hip_direct_copy_host_to_device(0, dc, actual.data(), actual.size()) == RNS8_SUCCESS);
        const uint64_t scratch = m * n * prefix * (splits + (splits + 127) / 128);
        CHECK(hip_direct_gemm_rns_split_k_device(0, da, db, dc, m, n, k, lda, ldb, ldc,
                                               prefix, splits, scratch - 1) == RNS8_WORKSPACE_TOO_SMALL);
        REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
        REQUIRE(std::all_of(actual.begin(), actual.end(), [](int8_t x) { return x == 99; }));
        REQUIRE(hip_direct_gemm_rns_split_k_device(0, da, db, dc, m, n, k, lda, ldb, ldc,
                                                 prefix, splits, scratch) == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
        REQUIRE(actual == expected);
      }
      if (minimum > 1)
        CHECK(hip_direct_gemm_rns_split_k_device(0, da, db, dc, m, n, k, lda, ldb, ldc,
                                               prefix, minimum - 1, UINT64_MAX) == RNS8_RANGE_ERROR);
      CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
    }
  }
}
