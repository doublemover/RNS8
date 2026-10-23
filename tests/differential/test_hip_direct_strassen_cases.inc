TEST_CASE(
    "one-level HIP Strassen matches exact classical residues with odd tails") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  if (!hip_available())
    SKIP("no HIP device for Strassen");
  for (uint32_t prefix : {1u, 3u, 9u, 20u})
    for (int k : {1, 3, 17, 131073}) {
      const int m = k > 100 ? 1 : 5, n = k > 100 ? 1 : 7, lda = k + 1,
                ldb = n + 2, ldc = n + 3;
      std::vector<int8_t> a(prefix * m * lda, 99), b(prefix * k * ldb, 99),
          expected(prefix * m * ldc, 77), observed(expected.size(), 77);
      for (uint32_t p = 0; p < prefix; ++p) {
        const int modulus = kDefaultModuli[p];
        for (int row = 0; row < m; ++row)
          for (int q = 0; q < k; ++q)
            a[(p * m + row) * lda + q] =
                centered_residue(cpp_int((row + q) % 2 ? -128 : 127), modulus);
        for (int q = 0; q < k; ++q)
          for (int col = 0; col < n; ++col)
            b[(p * k + q) * ldb + col] =
                centered_residue(cpp_int((q + col) % 3 ? -127 : 128), modulus);
        for (int row = 0; row < m; ++row)
          for (int col = 0; col < n; ++col) {
            cpp_int sum = 0;
            for (int q = 0; q < k; ++q)
              sum += cpp_int(a[(p * m + row) * lda + q]) *
                     b[(p * k + q) * ldb + col];
            expected[(p * m + row) * ldc + col] =
                centered_residue(sum, modulus);
          }
      }
      void *da = nullptr, *db = nullptr, *dc = nullptr;
      REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, observed.size(), &dc) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) ==
              RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) ==
              RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, dc, observed.data(),
                                             observed.size()) == RNS8_SUCCESS);
      bool committed = true;
      REQUIRE(hip_direct_gemm_strassen_device(0, da, db, dc, m, n, k, lda, ldb,
                                              ldc, prefix, 0, committed) ==
              RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE_FALSE(committed);
      REQUIRE(hip_direct_copy_device_to_host(0, observed.data(), dc,
                                             observed.size()) == RNS8_SUCCESS);
      REQUIRE(std::all_of(observed.begin(), observed.end(),
                          [](int8_t v) { return v == 77; }));
      REQUIRE(hip_direct_gemm_strassen_device(0, da, db, dc, m, n, k, lda, ldb,
                                              ldc, prefix, UINT64_MAX,
                                              committed) == RNS8_SUCCESS);
      REQUIRE(committed);
      REQUIRE(hip_direct_copy_device_to_host(0, observed.data(), dc,
                                             observed.size()) == RNS8_SUCCESS);
      REQUIRE(observed == expected);
      CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
    }
}
