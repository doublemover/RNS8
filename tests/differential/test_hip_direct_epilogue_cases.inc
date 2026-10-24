TEST_CASE("direct HIP fused modular expression matches exact integer reference") {
  if (!hip_available()) SKIP("no HIP device for fused epilogue differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr int64_t m = 2, n = 3, ldc = 5, ldd = 4;
  constexpr uint32_t prefix = 20;
  for (const int64_t k : {1, 65537}) {
    const int64_t lda = k + 1, ldb = n + 1;
    std::vector<int8_t> a(prefix * m * lda, -128), b(prefix * k * ldb, 127), d(prefix * m * ldd, -127);
    std::vector<int8_t> output(prefix * m * ldc, 99);
    void *da = nullptr, *db = nullptr, *dd = nullptr, *dc = nullptr, *dx = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, d.size(), &dd) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, output.size(), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, output.size(), &dx) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dd, d.data(), d.size()) == RNS8_SUCCESS);
    for (bool fused : {false, true}) for (const int64_t alpha : {INT64_MIN, int64_t{-1}, int64_t{0}, INT64_MAX}) {
      const int64_t beta = alpha == INT64_MIN ? INT64_MAX : INT64_MIN;
      std::fill(output.begin(), output.end(), int8_t{99});
      REQUIRE(hip_direct_copy_host_to_device(0, dc, output.data(), output.size()) == RNS8_SUCCESS);
      CHECK(hip_direct_gemm_rns_axpby_device(0, da, db, dd, dd, m, n, k, lda, ldb, ldd, ldc,
                                            prefix, alpha, beta) == RNS8_INVALID_ARGUMENT);
      if (fused) {
        REQUIRE(hip_direct_gemm_rns_axpby_device(0, da, db, dd, dc, m, n, k, lda, ldb, ldd, ldc,
                                                prefix, alpha, beta) == RNS8_SUCCESS);
      } else {
        REQUIRE(hip_direct_gemm_rns_device(0, da, db, dx, m, n, k, lda, ldb, ldc, prefix) == RNS8_SUCCESS);
        REQUIRE(hip_direct_axpby_residues_device(0, dx, dd, dd, m, n, ldc, ldd, ldc, prefix, alpha, beta)
                == RNS8_INVALID_ARGUMENT);
        REQUIRE(hip_direct_axpby_residues_device(0, dx, dd, dc, m, n, ldc, ldd, ldc, prefix, alpha, beta)
                == RNS8_SUCCESS);
      }
      REQUIRE(hip_direct_copy_device_to_host(0, output.data(), dc, output.size()) == RNS8_SUCCESS);
      const cpp_int expected = cpp_int(alpha) * (-128 * 127) * k + cpp_int(beta) * -127;
      for (uint32_t p = 0; p < prefix; ++p)
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < ldc; ++col)
            REQUIRE(output[p * m * ldc + row * ldc + col] ==
                    (col < n ? centered_residue(expected, kDefaultModuli[p]) : int8_t{99}));
    }
    CHECK(hip_direct_free(0, dx) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, dd) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  }
}
