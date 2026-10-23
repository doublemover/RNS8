#include "test_hip_direct_native_byte_cases.inc"
TEST_CASE("direct HIP native narrow admission proves partial sums from actual device inputs") {
  if (!hip_available()) SKIP("no HIP device for native narrow differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  auto run = [](const std::vector<int64_t>& av, const std::vector<int64_t>& bv,
                int64_t m, int64_t n, int64_t k, rns8_status expected_status) {
    const int64_t lda = k + 1, ldb = n + 1, ldc = n + 2;
    std::vector<int64_t> a(m * lda, INT64_MIN), b(k * ldb, INT64_MIN), c(m * ldc, 99);
    for (int64_t row = 0; row < m; ++row)
      std::copy_n(av.data() + row * k, k, a.data() + row * lda);
    for (int64_t row = 0; row < k; ++row)
      std::copy_n(bv.data() + row * n, n, b.data() + row * ldb);
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size() * sizeof(int64_t), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size() * sizeof(int64_t), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, c.size() * sizeof(int64_t), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, c.data(), c.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    uint64_t bound = UINT64_MAX;
    REQUIRE(hip_direct_gemm_i64_narrow_checked_device(0, static_cast<const int64_t*>(da), static_cast<const int64_t*>(db),
                                                      static_cast<int64_t*>(dc), m, n, k, lda, ldb, ldc, &bound) == expected_status);
    REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc, c.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    if (expected_status == RNS8_SUCCESS) {
      REQUIRE(bound <= static_cast<uint64_t>(INT64_MAX));
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
          cpp_int expected = 0;
          for (int64_t q = 0; q < k; ++q) expected += cpp_int(av[row * k + q]) * bv[q * n + col];
          REQUIRE(cpp_int(c[row * ldc + col]) == expected);
          REQUIRE(boost::multiprecision::abs(expected) <= bound);
        }
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = n; col < ldc; ++col) REQUIRE(c[row * ldc + col] == 99);
    } else {
      REQUIRE(bound == UINT64_MAX);
      REQUIRE(std::all_of(c.begin(), c.end(), [](int64_t value) { return value == 99; }));
    }
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  };
  run({INT64_MAX}, {1}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {0}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {-1}, 1, 1, 1, RNS8_RANGE_ERROR);
  run({INT64_MAX, INT64_MAX}, {1, -1}, 1, 1, 2, RNS8_RANGE_ERROR); // final zero is insufficient
  for (const int64_t k : {1, 7, 65537}) {
    constexpr int64_t m = 3, n = 5;
    std::vector<int64_t> a(m * k), b(k * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int64_t>(i % 23) - 11;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int64_t>(i % 19) - 9;
    run(a, b, m, n, k, RNS8_SUCCESS);
  }
}
