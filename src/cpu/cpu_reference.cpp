#include "core/internal.hpp"

#include <algorithm>
#include <limits>

namespace rns8::detail {

uint32_t canonical_residue(const cpp_int& value, uint16_t modulus) {
  cpp_int residue = value % modulus;
  if (residue < 0) {
    residue += modulus;
  }
  return static_cast<uint32_t>(residue);
}

uint32_t canonical_from_centered(int8_t residue, uint16_t modulus) {
  const int value = static_cast<int>(residue);
  if (value < 0) {
    return static_cast<uint32_t>(value + static_cast<int>(modulus));
  }
  return static_cast<uint32_t>(value);
}

int8_t centered_residue(const cpp_int& value, uint16_t modulus) {
  const uint32_t canonical = canonical_residue(value, modulus);
  const uint32_t threshold = (static_cast<uint32_t>(modulus) + 1u) / 2u;
  const int centered = canonical >= threshold ? static_cast<int>(canonical) - static_cast<int>(modulus)
                                              : static_cast<int>(canonical);
  return static_cast<int8_t>(centered);
}

int8_t reduce_to_centered(int64_t value, uint16_t modulus) {
  int64_t residue = value % static_cast<int64_t>(modulus);
  if (residue < 0) {
    residue += modulus;
  }
  const int64_t threshold = (static_cast<int64_t>(modulus) + 1) / 2;
  if (residue >= threshold) {
    residue -= modulus;
  }
  return static_cast<int8_t>(residue);
}

std::size_t residue_index(const rns8_matrix& matrix, uint32_t modulus_index, int64_t row, int64_t col) {
  return (static_cast<std::size_t>(modulus_index) * static_cast<std::size_t>(matrix.desc.rows) +
          static_cast<std::size_t>(row)) *
             static_cast<std::size_t>(matrix.desc.cols) +
         static_cast<std::size_t>(col);
}

void pack_i64_matrix(rns8_matrix& matrix, const int64_t* src, int64_t ld) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = kDefaultModuli[p];
    for (int64_t row = 0; row < matrix.desc.rows; ++row) {
      for (int64_t col = 0; col < matrix.desc.cols; ++col) {
        matrix.residues[residue_index(matrix, p, row, col)] =
            centered_residue(cpp_int(src[row * ld + col]), modulus);
      }
    }
  }
}

void pack_u64_matrix(rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = kDefaultModuli[p];
    for (int64_t row = 0; row < matrix.desc.rows; ++row) {
      for (int64_t col = 0; col < matrix.desc.cols; ++col) {
        matrix.residues[residue_index(matrix, p, row, col)] =
            centered_residue(cpp_int(src[row * ld + col]), modulus);
      }
    }
  }
}

void ring_gemm_modulus(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      int8_t residue_acc = 0;
      int64_t offset = 0;
      while (offset < k) {
        const int64_t block = std::min<int64_t>(RNS8_SAFE_INT32_K_BLOCK, k - offset);
        int32_t block_acc = 0;
        for (int64_t kk = 0; kk < block; ++kk) {
          const int av = static_cast<int>(A[row * lda + offset + kk]);
          const int bv = static_cast<int>(B[(offset + kk) * ldb + col]);
          block_acc += av * bv;
        }
        residue_acc = reduce_to_centered(static_cast<int64_t>(residue_acc) + block_acc, modulus);
        offset += block;
      }
      C[row * ldc + col] = residue_acc;
    }
  }
}

rns8_status cpu_gemm_rns(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B, rns8_matrix& C) {
  if (A.desc.rows != plan.desc.m || A.desc.cols != plan.desc.k || B.desc.rows != plan.desc.k ||
      B.desc.cols != plan.desc.n || C.desc.rows != plan.desc.m || C.desc.cols != plan.desc.n) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.prefix < plan.prefix || B.prefix < plan.prefix || C.prefix < plan.prefix) {
    return RNS8_INVALID_ARGUMENT;
  }

  if (!plan.tile_schedule.empty()) {
    for (const auto& entry : plan.tile_schedule) {
      for (uint32_t p = 0; p < entry.selected_prefix; ++p) {
        const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(A.desc.rows) *
                                     static_cast<std::size_t>(A.desc.cols);
        const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(B.desc.rows) *
                                     static_cast<std::size_t>(B.desc.cols);
        const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(C.desc.rows) *
                                     static_cast<std::size_t>(C.desc.cols);
        ring_gemm_modulus(
            A.residues.data() + a_offset + static_cast<std::size_t>(entry.row_offset) *
                                      static_cast<std::size_t>(A.desc.cols),
            B.residues.data() + b_offset + static_cast<std::size_t>(entry.col_offset),
            C.residues.data() + c_offset + static_cast<std::size_t>(entry.row_offset) *
                                      static_cast<std::size_t>(C.desc.cols) +
                                      static_cast<std::size_t>(entry.col_offset),
            entry.row_extent,
            entry.col_extent,
            plan.desc.k,
            A.desc.cols,
            B.desc.cols,
            C.desc.cols,
            kDefaultModuli[p]);
      }
    }
    return RNS8_SUCCESS;
  }

  for (uint32_t p = 0; p < plan.prefix; ++p) {
    const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(A.desc.rows) *
                                 static_cast<std::size_t>(A.desc.cols);
    const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(B.desc.rows) *
                                 static_cast<std::size_t>(B.desc.cols);
    const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(C.desc.rows) *
                                 static_cast<std::size_t>(C.desc.cols);
    ring_gemm_modulus(
        A.residues.data() + a_offset,
        B.residues.data() + b_offset,
        C.residues.data() + c_offset,
        plan.desc.m,
        plan.desc.n,
        plan.desc.k,
        A.desc.cols,
        B.desc.cols,
        C.desc.cols,
        kDefaultModuli[p]);
  }
  return RNS8_SUCCESS;
}

cpp_int exact_i64_gemm_cell(
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  cpp_int acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += cpp_int(A[row * lda + kk]) * cpp_int(B[kk * ldb + col]);
  }
  return acc;
}

cpp_int exact_u64_gemm_cell(
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  cpp_int acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += cpp_int(A[row * lda + kk]) * cpp_int(B[kk * ldb + col]);
  }
  return acc;
}

}  // namespace rns8::detail
