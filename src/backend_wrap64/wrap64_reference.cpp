#include "core/internal.hpp"

namespace rns8::detail {

uint64_t wrap64_byte_limb_product(uint64_t a, uint64_t b) {
  uint64_t out = 0;
  uint64_t carry = 0;
  for (uint32_t diagonal = 0; diagonal < 8; ++diagonal) {
    uint64_t column = carry;
    for (uint32_t i = 0; i <= diagonal; ++i) {
      const uint32_t j = diagonal - i;
      const uint64_t a_byte = (a >> (8u * i)) & 0xffu;
      const uint64_t b_byte = (b >> (8u * j)) & 0xffu;
      column += a_byte * b_byte;
    }
    out |= (column & 0xffu) << (8u * diagonal);
    carry = column >> 8u;
  }
  return out;
}

uint64_t wrap64_byte_limb_gemm_cell(
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  uint64_t acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += wrap64_byte_limb_product(A[row * lda + kk], B[kk * ldb + col]);
  }
  return acc;
}

}  // namespace rns8::detail
