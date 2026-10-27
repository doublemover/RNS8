#include "core/hip_resources.hpp"
TEST_CASE(
    "raw byte WMMA requires actual operand domains and safe partial sums") {
  if (!hip_available())
    SKIP("no HIP device for raw byte WMMA");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (bool unsigned_values : {false, true}) {
    const auto run = [&](int64_t m, int64_t n, int64_t k, int mode,
                         rns8_status expected_status) {
      const int64_t lda = k + 1, ldb = n + 2, ldc = n + 3;
      std::vector<uint64_t> a(m * lda, UINT64_MAX), b(k * ldb, UINT64_MAX),
          c(m * ldc, 91);
      for (int64_t row = 0; row < m; ++row)
        for (int64_t q = 0; q < k; ++q)
          a[row * lda + q] =
              unsigned_values
                  ? (mode ? 255 : (row + q) % 256)
                  : static_cast<uint64_t>(mode ? -128 : (row + q) % 256 - 128);
      for (int64_t q = 0; q < k; ++q)
        for (int64_t col = 0; col < n; ++col)
          b[q * ldb + col] = unsigned_values
                                 ? (mode ? 255 : (q + col * 3) % 256)
                                 : static_cast<uint64_t>(
                                       mode ? -128 : (q + col * 3) % 256 - 128);
      if (mode == 2)
        a[0] = unsigned_values ? 256 : 128;
      if (mode == 3)
        a[0] = unsigned_values ? UINT64_MAX : static_cast<uint64_t>(INT64_MIN);
      // Tiny/zero final values cannot admit invalid byte operands.
      if (mode == 4) {
        a[0] = unsigned_values ? 256 : 128;
        std::fill(b.begin(), b.end(), 0);
      }
      hip_direct_device_buffer da, db, dc;
      REQUIRE(da.allocate(0, a.size() * 8) == RNS8_SUCCESS);
      REQUIRE(db.allocate(0, b.size() * 8) == RNS8_SUCCESS);
      REQUIRE(dc.allocate(0, c.size() * 8) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, da.get(), a.data(),
                                             a.size() * 8) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db.get(), b.data(),
                                             b.size() * 8) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, dc.get(), c.data(),
                                             c.size() * 8) == RNS8_SUCCESS);
      uint64_t bound = UINT64_MAX;
      REQUIRE(hip_direct_gemm_native_byte_wmma_checked_device(
                  0, da.get(), db.get(), da.get(), m, n, k, lda, ldb, ldc,
                  unsigned_values, &bound) == RNS8_INVALID_ARGUMENT);
      const auto status = hip_direct_gemm_native_byte_wmma_checked_device(
          0, da.get(), db.get(), dc.get(), m, n, k, lda, ldb, ldc,
          unsigned_values, &bound);
      if (status == RNS8_UNSUPPORTED_BACKEND)
        SKIP("raw-byte WMMA requires qualified RDNA3 instruction support");
      REQUIRE(status == expected_status);
      REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc.get(),
                                             c.size() * 8) == RNS8_SUCCESS);
      if (status == RNS8_SUCCESS) {
        const auto value = [&](uint64_t bits) -> cpp_int {
          return unsigned_values ? cpp_int(bits)
                                 : cpp_int(static_cast<int64_t>(bits));
        };
        REQUIRE(bound <= INT32_MAX);
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < n; ++col) {
            cpp_int expected = 0;
            for (int64_t q = 0; q < k; ++q)
              expected += value(a[row * lda + q]) * value(b[q * ldb + col]);
            REQUIRE(value(c[row * ldc + col]) == expected);
            REQUIRE(boost::multiprecision::abs(expected) <= bound);
          }
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = n; col < ldc; ++col)
            REQUIRE(c[row * ldc + col] == 91);
        // Reuse the same addresses with invalid data: each call must rescan.
        a[0] = unsigned_values ? 256 : 128;
        REQUIRE(hip_direct_copy_host_to_device(0, da.get(), a.data(),
                                               a.size() * 8) == RNS8_SUCCESS);
        const auto before = c;
        uint64_t rejected_bound = UINT64_MAX;
        REQUIRE(hip_direct_gemm_native_byte_wmma_checked_device(
                    0, da.get(), db.get(), dc.get(), m, n, k, lda, ldb, ldc,
                    unsigned_values, &rejected_bound) == RNS8_RANGE_ERROR);
        REQUIRE(rejected_bound == UINT64_MAX);
        REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc.get(),
                                               c.size() * 8) == RNS8_SUCCESS);
        REQUIRE(c == before);
      } else {
        REQUIRE(bound == UINT64_MAX);
        REQUIRE(std::all_of(c.begin(), c.end(),
                            [](uint64_t x) { return x == 91; }));
      }
    };
    for (int64_t k : {1, 17, 65537})
      run(17, 19, k, 0,
          unsigned_values && k == 65537 ? RNS8_RANGE_ERROR : RNS8_SUCCESS);
    const int64_t cap =
        unsigned_values ? INT32_MAX / (255 * 255) : INT32_MAX / (128 * 128);
    run(1, 1, cap, 1, RNS8_SUCCESS);
    run(1, 1, cap + 1, 1, RNS8_RANGE_ERROR);
    for (int mode : {2, 3, 4})
      run(1, 1, 1, mode, RNS8_RANGE_ERROR);
  }
}
