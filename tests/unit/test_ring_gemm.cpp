#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <vector>

#include "core/internal.hpp"

namespace {

int8_t exact_cell_mod(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    int64_t n,
    int64_t k,
    int64_t row,
    int64_t col,
    uint16_t modulus) {
  boost::multiprecision::cpp_int acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += boost::multiprecision::cpp_int(A[row * k + kk]) * boost::multiprecision::cpp_int(B[kk * n + col]);
  }
  return rns8::detail::centered_residue(acc, modulus);
}

}  // namespace

TEST_CASE("scalar ring GEMM handles prime and composite moduli") {
  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20};
  for (uint16_t modulus : {uint16_t(255), uint16_t(251)}) {
    std::vector<int8_t> C(static_cast<std::size_t>(m * n), 0);
    rns8::detail::ring_gemm_modulus(A.data(), B.data(), C.data(), m, n, k, k, n, n, modulus);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[row * n + col] == exact_cell_mod(A, B, n, k, row, col, modulus));
      }
    }
  }
}

TEST_CASE("scalar ring GEMM splits K above the int32 safe block") {
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<int8_t> A(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> B(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> C(1, 0);
  rns8::detail::ring_gemm_modulus(A.data(), B.data(), C.data(), 1, 1, k, k, 1, 1, 256);

  boost::multiprecision::cpp_int expected = boost::multiprecision::cpp_int(127) * 127 * k;
  CHECK(C[0] == rns8::detail::centered_residue(expected, 256));
}

