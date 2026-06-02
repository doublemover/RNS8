#include "core/internal.hpp"

namespace rns8::detail {

namespace {

std::size_t wrap_byte_limb_index(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t limb) {
  const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix.desc.cols) +
                           static_cast<std::size_t>(col);
  return cell * 8u + limb;
}

}  // namespace

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

void pack_wrap_u64_matrix(rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      set_wrap_u64_matrix_cell(matrix, row, col, src[row * ld + col]);
    }
  }
}

uint64_t wrap_u64_matrix_cell(const rns8_matrix& matrix, int64_t row, int64_t col) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < 8; ++limb) {
    value |= static_cast<uint64_t>(matrix.byte_limbs[wrap_byte_limb_index(matrix, row, col, limb)]) << (8u * limb);
  }
  return value;
}

void set_wrap_u64_matrix_cell(rns8_matrix& matrix, int64_t row, int64_t col, uint64_t value) {
  for (uint32_t limb = 0; limb < 8; ++limb) {
    matrix.byte_limbs[wrap_byte_limb_index(matrix, row, col, limb)] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
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

rns8_status cpu_gemm_wrap_u64(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B, rns8_matrix& C) {
  if (plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 || A.desc.semantics != RNS8_WRAP_U64_MOD_2_64 ||
      B.desc.semantics != RNS8_WRAP_U64_MOD_2_64 || C.desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.byte_limbs.empty() || B.byte_limbs.empty() || C.byte_limbs.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      uint64_t acc = 0;
      for (int64_t kk = 0; kk < plan.desc.k; ++kk) {
        acc += wrap64_byte_limb_product(wrap_u64_matrix_cell(A, row, kk), wrap_u64_matrix_cell(B, kk, col));
      }
      set_wrap_u64_matrix_cell(C, row, col, acc);
    }
  }
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail
