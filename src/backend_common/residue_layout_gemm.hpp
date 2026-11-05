#pragma once
#include "residue_layout.hpp"
#include "residue_dot.hpp"
#include "checked_arithmetic.hpp"
#include "rns8/status.h"
#include <algorithm>
#include <array>

namespace rns8::detail {
// This consumer operates on explicitly described centered byte residues. Its
// output is a residue representation; integer uniqueness belongs to its owner.
inline bool compatible_layout_gemm(const ResidueLayout& a, const ResidueLayout& b,
                                   const ResidueLayout& c, uint32_t prefix) {
  return valid_residue_layout(a) && valid_residue_layout(b) && valid_residue_layout(c) &&
         a.basis == b.basis && a.basis == c.basis && a.cols == b.rows && c.rows == a.rows &&
         c.cols == b.cols && prefix && prefix <= a.prefix && prefix <= b.prefix &&
         prefix <= c.prefix;
}
// These layout kernels use 64-bit byte offsets. Bound actual launch geometry,
// not the full allocation to INT32 merely because accumulation is INT32.
inline rns8_status query_hip_layout_conversion_support(const ResidueLayout& source,
                                                       const ResidueLayout& destination,
                                                       uint32_t first, uint32_t end) {
  if (!compatible_residue_planes(source, destination, first, end))
    return RNS8_INVALID_ARGUMENT;
  uint64_t cells = 0, work = 0;
  if (!checked::multiply(source.rows, source.cols, cells) ||
      !checked::multiply(cells, uint64_t(end - first), work))
    return RNS8_RANGE_ERROR;
  return work / 256 + uint64_t(work % 256 != 0) > uint64_t(INT32_MAX) ? RNS8_UNSUPPORTED_BACKEND
                                                                      : RNS8_SUCCESS;
}
inline rns8_status query_hip_layout_gemm_support(const ResidueLayout& a, const ResidueLayout& b,
                                                 const ResidueLayout& c, uint32_t prefix) {
  if (!compatible_layout_gemm(a, b, c, prefix))
    return RNS8_INVALID_ARGUMENT;
  return a.rows / 16 + uint64_t(a.rows % 16 != 0) > 65535 ||
                 b.cols / 16 + uint64_t(b.cols % 16 != 0) > uint64_t(INT32_MAX)
             ? RNS8_UNSUPPORTED_BACKEND
             : RNS8_SUCCESS;
}
inline bool residue_layout_gemm_host(const int8_t* a, const ResidueLayout& al, const int8_t* b,
                                     const ResidueLayout& bl, int8_t* c, const ResidueLayout& cl,
                                     uint32_t prefix) {
  if (!compatible_layout_gemm(al, bl, cl, prefix) ||
      !nonoverlapping_byte_ranges(a, al.bytes, c, cl.bytes) ||
      !nonoverlapping_byte_ranges(b, bl.bytes, c, cl.bytes))
    return false;
  static constexpr finite_u8::ByteModulusReciprocals reciprocals;
  for (uint32_t p = 0; p < prefix; ++p) {
    const uint32_t modulus = al.basis.moduli[p], reciprocal = reciprocals.values[modulus];
    for (uint64_t row = 0; row < al.rows; ++row)
      for (uint64_t col = 0; col < bl.cols; col += 16) {
        const auto width = static_cast<unsigned>(std::min(uint64_t(16), bl.cols - col));
        std::array<int8_t, 16> carry{};
        const bool contiguous_a = al.kind == ResidueLayoutKind::modulus_major;
        const bool contiguous_b = bl.kind == ResidueLayoutKind::modulus_major ||
                                  col / bl.tile_cols == (col + width - 1) / bl.tile_cols;
        const auto* arow = contiguous_a ? a + al.offset(p, row, 0) : nullptr;
        for (uint64_t begin = 0; begin < al.cols;) {
          const uint64_t extent = std::min(uint64_t(kResidueDotKBlock), al.cols - begin);
          std::array<int32_t, 16> sums{};
          for (uint64_t inner = begin; inner < begin + extent; ++inner) {
            const int32_t av = contiguous_a ? arow[inner] : a[al.offset(p, row, inner)];
            if (contiguous_b) {
              const auto* brow = b + bl.offset(p, inner, col);
              for (unsigned lane = 0; lane < width; ++lane)
                sums[lane] += av * int32_t(brow[lane]);
            } else {
              for (unsigned lane = 0; lane < width; ++lane)
                sums[lane] += av * int32_t(b[bl.offset(p, inner, col + lane)]);
            }
          }
          for (unsigned lane = 0; lane < width; ++lane)
            carry[lane] = reduce_residue_slab(sums[lane], carry[lane], modulus, reciprocal);
          begin += extent;
        }
        for (unsigned lane = 0; lane < width; ++lane)
          c[cl.offset(p, row, col + lane)] = carry[lane];
      }
  }
  return true;
}
// Flat kernel argument: no standard-library host member calls in device code.
inline constexpr const char* kResidueLayoutGemmPolicy = "centered-byte-blocked-k65536-v2";
struct ResidueLayoutModuli {
  uint16_t values[20]{};
  uint32_t reciprocals[20]{};
};
inline ResidueLayoutModuli residue_layout_moduli(const OrderedBasisIdentity& basis) {
  static constexpr finite_u8::ByteModulusReciprocals table;
  ResidueLayoutModuli result;
  for (uint32_t p = 0; p < basis.count; ++p) {
    result.values[p] = basis.moduli[p];
    result.reciprocals[p] = table.values[basis.moduli[p]];
  }
  return result;
}
} // namespace rns8::detail
