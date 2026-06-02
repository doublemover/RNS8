#include "core/internal.hpp"

#include <array>
#include <limits>

namespace rns8::detail {

uint32_t wrap64_unsigned_byte_product_from_signed_i8(uint8_t a, uint8_t b);

namespace {

constexpr uint32_t kWrap64ByteLimbs = 8;
constexpr uint32_t kWrap64LowProductDiagonals = kWrap64ByteLimbs;
constexpr uint32_t kWrap64LowProductPairCount =
    (kWrap64LowProductDiagonals * (kWrap64LowProductDiagonals + 1u)) / 2u;
static_assert(kWrap64LowProductPairCount == 36u, "low 64-bit wrap64 uses exactly 36 byte-product pairs");

std::size_t wrap_compact_limb_index(int64_t row, int64_t col, int64_t cols, uint32_t limb) {
  const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                           static_cast<std::size_t>(col);
  return cell * kWrap64ByteLimbs + limb;
}

std::size_t wrap_byte_limb_index(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t limb) {
  return wrap_compact_limb_index(row, col, matrix.desc.cols, limb);
}

bool expected_wrap_byte_limb_count(int64_t rows, int64_t cols, std::size_t& count) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  constexpr uint64_t limbs_per_cell = 8;
  const uint64_t max_count = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_count / u_cols / limbs_per_cell) {
    return false;
  }
  count = static_cast<std::size_t>(u_rows * u_cols * limbs_per_cell);
  return true;
}

bool wrap64_cpu_storage_matches(const rns8_matrix& matrix, int64_t rows, int64_t cols) {
  std::size_t expected_limbs = 0;
  return expected_wrap_byte_limb_count(rows, cols, expected_limbs) &&
         matrix.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB &&
         matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64 &&
         matrix.desc.bound_kind == RNS8_BOUND_NONE &&
         matrix.desc.rows == rows &&
         matrix.desc.cols == cols &&
         matrix.desc.logical_layout == RNS8_LAYOUT_ROW_MAJOR &&
         matrix.desc.logical_ld >= matrix.desc.cols &&
         matrix.prefix == 0 &&
         matrix.desc.max_prefix == 0 &&
         matrix.residues.empty() &&
         matrix.byte_limbs.size() == expected_limbs &&
         !matrix.host_residues_current &&
         !matrix.device_residues_current &&
         !matrix.device_byte_limbs_current &&
         matrix.hip_residues == nullptr &&
         matrix.hip_residue_bytes == 0 &&
         matrix.hip_byte_limbs == nullptr &&
         matrix.hip_byte_limb_bytes == 0 &&
         matrix.hip_upload_buffer == nullptr &&
         matrix.hip_upload_bytes == 0 &&
         matrix.hip_export_buffer == nullptr &&
         matrix.hip_export_bytes == 0 &&
         matrix.hip_status_buffer == nullptr &&
         matrix.hip_status_bytes == 0;
}

bool wrap64_cpu_plan_metadata_is_clean(const rns8_plan& plan) {
  return plan.desc.tile_bounds == nullptr &&
         plan.desc.tile_bounds_count == 0 &&
         plan.tile_bounds.empty() &&
         plan.tile_schedule.empty() &&
         plan.schedule_min_required_prefix == 0 &&
         plan.schedule_max_required_prefix == 0 &&
         plan.schedule_min_selected_prefix == 0 &&
         plan.schedule_max_selected_prefix == 0 &&
         plan.schedule_prefix_group_count == 0 &&
         plan.schedule_range_bit_length == 0 &&
         plan.schedule_adaptive_prefix_active == 0 &&
         plan.schedule_adaptive_skip_active == 0 &&
         plan.schedule_flags == 0;
}

void accumulate_wrap64_low_diagonals_from_limbs(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    std::array<cpp_int, kWrap64LowProductDiagonals>& diagonals) {
  for (uint32_t a_limb = 0; a_limb < kWrap64ByteLimbs; ++a_limb) {
    for (uint32_t b_limb = 0; b_limb + a_limb < kWrap64LowProductDiagonals; ++b_limb) {
      diagonals[a_limb + b_limb] += wrap64_unsigned_byte_product_from_signed_i8(a_limbs[a_limb], b_limbs[b_limb]);
    }
  }
}

uint64_t wrap64_low64_from_diagonals(const std::array<cpp_int, kWrap64LowProductDiagonals>& diagonals) {
  uint64_t out = 0;
  cpp_int carry = 0;
  for (uint32_t diagonal = 0; diagonal < kWrap64LowProductDiagonals; ++diagonal) {
    const cpp_int column = diagonals[diagonal] + carry;
    const uint64_t byte = static_cast<uint64_t>(column & 0xffu);
    out |= byte << (8u * diagonal);
    carry = column >> 8u;
  }
  return out;
}

uint64_t wrap64_compact_byte_limb_gemm_cell(
    const rns8_matrix& A,
    const rns8_matrix& B,
    int64_t row,
    int64_t col,
    int64_t k) {
  std::array<cpp_int, kWrap64LowProductDiagonals> diagonals{};
  for (int64_t kk = 0; kk < k; ++kk) {
    const uint8_t* a_limbs = A.byte_limbs.data() + wrap_byte_limb_index(A, row, kk, 0);
    const uint8_t* b_limbs = B.byte_limbs.data() + wrap_byte_limb_index(B, kk, col, 0);
    accumulate_wrap64_low_diagonals_from_limbs(a_limbs, b_limbs, diagonals);
  }
  return wrap64_low64_from_diagonals(diagonals);
}

}  // namespace

int32_t wrap64_signed_i8_product_correction(uint8_t a, uint8_t b) {
  const int32_t unsigned_product = static_cast<int32_t>(a) * static_cast<int32_t>(b);
  const int32_t signed_a = static_cast<int32_t>(static_cast<int8_t>(a));
  const int32_t signed_b = static_cast<int32_t>(static_cast<int8_t>(b));
  return unsigned_product - signed_a * signed_b;
}

uint32_t wrap64_unsigned_byte_product_from_signed_i8(uint8_t a, uint8_t b) {
  const int32_t signed_a = static_cast<int32_t>(static_cast<int8_t>(a));
  const int32_t signed_b = static_cast<int32_t>(static_cast<int8_t>(b));
  const int32_t corrected = signed_a * signed_b + wrap64_signed_i8_product_correction(a, b);
  return static_cast<uint32_t>(corrected);
}

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

uint64_t wrap64_low_diagonal_byte_pair_gemm_cell(
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  std::array<cpp_int, kWrap64LowProductDiagonals> diagonals{};
  for (int64_t kk = 0; kk < k; ++kk) {
    const uint64_t a_value = A[row * lda + kk];
    const uint64_t b_value = B[kk * ldb + col];
    for (uint32_t a_limb = 0; a_limb < kWrap64ByteLimbs; ++a_limb) {
      const auto a_byte = static_cast<uint8_t>((a_value >> (8u * a_limb)) & 0xffu);
      for (uint32_t b_limb = 0; b_limb + a_limb < kWrap64LowProductDiagonals; ++b_limb) {
        const auto b_byte = static_cast<uint8_t>((b_value >> (8u * b_limb)) & 0xffu);
        diagonals[a_limb + b_limb] += wrap64_unsigned_byte_product_from_signed_i8(a_byte, b_byte);
      }
    }
  }

  return wrap64_low64_from_diagonals(diagonals);
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
  if (plan.backend != RNS8_BACKEND_WRAP64_BYTE_LIMB ||
      plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 ||
      plan.desc.bound_kind != RNS8_BOUND_NONE ||
      plan.desc.bound != 0 ||
      plan.prefix != 0 ||
      plan.desc.max_prefix != 0 ||
      !wrap64_cpu_plan_metadata_is_clean(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap64_cpu_storage_matches(A, plan.desc.m, plan.desc.k) ||
      !wrap64_cpu_storage_matches(B, plan.desc.k, plan.desc.n) ||
      !wrap64_cpu_storage_matches(C, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!A.host_byte_limbs_current || !B.host_byte_limbs_current) {
    return RNS8_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      set_wrap_u64_matrix_cell(C, row, col, wrap64_compact_byte_limb_gemm_cell(A, B, row, col, plan.desc.k));
    }
  }
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail
