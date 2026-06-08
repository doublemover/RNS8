#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <vector>

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
  const int64_t plane_rows = static_cast<int64_t>(matrix.prefix) * matrix.desc.rows;
  const auto pack_plane_row = [&](int64_t plane_row) {
    const auto p = static_cast<uint32_t>(plane_row / matrix.desc.rows);
    const int64_t row = plane_row % matrix.desc.rows;
    const uint16_t modulus = kDefaultModuli[p];
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      matrix.residues[residue_index(matrix, p, row, col)] =
          centered_residue(cpp_int(src[row * ld + col]), modulus);
    }
  };
  const uint64_t work =
      cpu_parallel_saturating_mul3(static_cast<uint64_t>(matrix.prefix), static_cast<uint64_t>(matrix.desc.rows),
                                   static_cast<uint64_t>(matrix.desc.cols));
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (cpu_parallel_should_use(work)) {
#  pragma omp parallel for schedule(static)
    for (int64_t plane_row = 0; plane_row < plane_rows; ++plane_row) {
      pack_plane_row(plane_row);
    }
    return;
  }
#endif
  for (int64_t plane_row = 0; plane_row < plane_rows; ++plane_row) {
    pack_plane_row(plane_row);
  }
}

void pack_u64_matrix(rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  const int64_t plane_rows = static_cast<int64_t>(matrix.prefix) * matrix.desc.rows;
  const auto pack_plane_row = [&](int64_t plane_row) {
    const auto p = static_cast<uint32_t>(plane_row / matrix.desc.rows);
    const int64_t row = plane_row % matrix.desc.rows;
    const uint16_t modulus = kDefaultModuli[p];
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      matrix.residues[residue_index(matrix, p, row, col)] =
          centered_residue(cpp_int(src[row * ld + col]), modulus);
    }
  };
  const uint64_t work =
      cpu_parallel_saturating_mul3(static_cast<uint64_t>(matrix.prefix), static_cast<uint64_t>(matrix.desc.rows),
                                   static_cast<uint64_t>(matrix.desc.cols));
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (cpu_parallel_should_use(work)) {
#  pragma omp parallel for schedule(static)
    for (int64_t plane_row = 0; plane_row < plane_rows; ++plane_row) {
      pack_plane_row(plane_row);
    }
    return;
  }
#endif
  for (int64_t plane_row = 0; plane_row < plane_rows; ++plane_row) {
    pack_plane_row(plane_row);
  }
}

void pack_finite_u8_matrix(rns8_matrix& matrix, const uint8_t* src, int64_t ld, uint16_t modulus) {
  const auto pack_row = [&](int64_t row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      matrix.residues[static_cast<std::size_t>(row * matrix.desc.cols + col)] =
          centered_residue(cpp_int(src[row * ld + col]), modulus);
    }
  };
  const uint64_t work = cpu_parallel_saturating_mul(
      static_cast<uint64_t>(matrix.desc.rows), static_cast<uint64_t>(matrix.desc.cols));
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (cpu_parallel_should_use(work)) {
#  pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < matrix.desc.rows; ++row) {
      pack_row(row);
    }
    return;
  }
#endif
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    pack_row(row);
  }
}

void export_finite_u8_matrix(const rns8_matrix& matrix, uint8_t* dst, int64_t ld, uint16_t modulus) {
  const auto export_row = [&](int64_t row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      dst[row * ld + col] = static_cast<uint8_t>(
          canonical_from_centered(matrix.residues[static_cast<std::size_t>(row * matrix.desc.cols + col)], modulus));
    }
  };
  const uint64_t work = cpu_parallel_saturating_mul(
      static_cast<uint64_t>(matrix.desc.rows), static_cast<uint64_t>(matrix.desc.cols));
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (cpu_parallel_should_use(work)) {
#  pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < matrix.desc.rows; ++row) {
      export_row(row);
    }
    return;
  }
#endif
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    export_row(row);
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
  const auto compute_row = [&](int64_t row, std::vector<int32_t>& block_acc) {
    int8_t* c_row = C + row * ldc;
    std::fill(c_row, c_row + n, int8_t{0});
    int64_t offset = 0;
    while (offset < k) {
      const int64_t block = std::min<int64_t>(RNS8_SAFE_INT32_K_BLOCK, k - offset);
      std::fill(block_acc.begin(), block_acc.end(), 0);
      for (int64_t kk = 0; kk < block; ++kk) {
        const int av = static_cast<int>(A[row * lda + offset + kk]);
        const int8_t* b_row = B + (offset + kk) * ldb;
        for (int64_t col = 0; col < n; ++col) {
          block_acc[static_cast<std::size_t>(col)] += av * static_cast<int>(b_row[col]);
        }
      }
      for (int64_t col = 0; col < n; ++col) {
        c_row[col] =
            reduce_to_centered(static_cast<int64_t>(c_row[col]) + block_acc[static_cast<std::size_t>(col)], modulus);
      }
      offset += block;
    }
  };
  const uint64_t work =
      cpu_parallel_saturating_mul3(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k));
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (cpu_parallel_should_use(work)) {
#  pragma omp parallel
    {
      std::vector<int32_t> block_acc(static_cast<std::size_t>(n), 0);
#  pragma omp for schedule(static)
      for (int64_t row = 0; row < m; ++row) {
        compute_row(row, block_acc);
      }
    }
    return;
  }
#endif
  std::vector<int32_t> block_acc(static_cast<std::size_t>(n), 0);
  for (int64_t row = 0; row < m; ++row) {
    compute_row(row, block_acc);
  }
}

void fill_tile_modulus(int8_t* C, int64_t row_extent, int64_t col_extent, int64_t ldc, int8_t value) {
  for (int64_t row = 0; row < row_extent; ++row) {
    std::fill(C + row * ldc, C + row * ldc + col_extent, value);
  }
}

uint32_t cpu_rns_storage_prefix_for_plan(const rns8_plan& plan) {
  if (!plan.tile_schedule.empty() && plan.schedule_max_selected_prefix > 0) {
    return plan.schedule_max_selected_prefix;
  }
  return plan.prefix;
}

rns8_status cpu_gemm_rns(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B, rns8_matrix& C) {
  if (A.desc.rows != plan.desc.m || A.desc.cols != plan.desc.k || B.desc.rows != plan.desc.k ||
      B.desc.cols != plan.desc.n || C.desc.rows != plan.desc.m || C.desc.cols != plan.desc.n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = cpu_rns_storage_prefix_for_plan(plan);
  if (A.prefix < storage_prefix || B.prefix < storage_prefix || C.prefix < storage_prefix) {
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
        auto* c_tile = C.residues.data() + c_offset + static_cast<std::size_t>(entry.row_offset) *
                                                  static_cast<std::size_t>(C.desc.cols) +
                                              static_cast<std::size_t>(entry.col_offset);
        if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
          fill_tile_modulus(c_tile, entry.row_extent, entry.col_extent, C.desc.cols, 0);
          continue;
        }
        ring_gemm_modulus(
            A.residues.data() + a_offset + static_cast<std::size_t>(entry.row_offset) *
                                      static_cast<std::size_t>(A.desc.cols),
            B.residues.data() + b_offset + static_cast<std::size_t>(entry.col_offset),
            c_tile,
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

rns8_status cpu_gemm_finite_u8(
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    rns8_matrix& C) {
  if (A.desc.rows != plan.desc.m || A.desc.cols != plan.desc.k || B.desc.rows != plan.desc.k ||
      B.desc.cols != plan.desc.n || C.desc.rows != plan.desc.m || C.desc.cols != plan.desc.n ||
      A.residues.size() != static_cast<std::size_t>(A.desc.rows * A.desc.cols) ||
      B.residues.size() != static_cast<std::size_t>(B.desc.rows * B.desc.cols) ||
      C.residues.size() != static_cast<std::size_t>(C.desc.rows * C.desc.cols)) {
    return RNS8_INVALID_ARGUMENT;
  }
  ring_gemm_modulus(
      A.residues.data(),
      B.residues.data(),
      C.residues.data(),
      plan.desc.m,
      plan.desc.n,
      plan.desc.k,
      A.desc.cols,
      B.desc.cols,
      C.desc.cols,
      modulus);
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
