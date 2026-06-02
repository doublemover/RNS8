#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"

namespace {

uint64_t low64(boost::multiprecision::cpp_int value) {
  const boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << 64;
  value %= modulus;
  if (value < 0) {
    value += modulus;
  }
  return static_cast<uint64_t>(value);
}

}  // namespace

TEST_CASE("wrap64 byte-limb product matches low 64-bit multiprecision product") {
  const std::vector<uint64_t> values = {
      0,
      1,
      255,
      256,
      0x0102030405060708ull,
      0x8080808080808080ull,
      std::numeric_limits<uint64_t>::max() - 1,
      std::numeric_limits<uint64_t>::max()};

  for (const uint64_t a : values) {
    for (const uint64_t b : values) {
      const uint64_t expected = low64(boost::multiprecision::cpp_int(a) * boost::multiprecision::cpp_int(b));
      CHECK(rns8::detail::wrap64_byte_limb_product(a, b) == expected);
    }
  }
}

TEST_CASE("wrap64 byte-limb GEMM cell matches multiprecision modulo 2^64") {
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 3;
  const uint64_t A[] = {
      std::numeric_limits<uint64_t>::max(),
      0x0102030405060708ull,
      17,
      0x8080808080808080ull,
      std::numeric_limits<uint64_t>::max() - 1,
      255};
  const uint64_t B[] = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31};

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int exact = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        exact += boost::multiprecision::cpp_int(A[row * k + kk]) * boost::multiprecision::cpp_int(B[kk * n + col]);
      }
      CHECK(rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, row, col, k) == low64(exact));
    }
  }
}
