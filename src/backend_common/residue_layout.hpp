#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include "address_domain.hpp"

#if defined(__HIPCC__)
#define RNS8_LAYOUT_HD __host__ __device__
#else
#define RNS8_LAYOUT_HD
#endif

namespace rns8::detail {
enum class ResidueLayoutKind : uint32_t { modulus_major = 1, tile_major = 2 };

// Uniform-prefix byte layouts only. Version 1 always uses the ordered default
// ladder; a converter cannot invent missing channels or change the basis.
struct ResidueLayout {
  uint32_t version = 1;
  uint32_t basis_version = 1;
  ResidueLayoutKind kind = ResidueLayoutKind::modulus_major;
  uint32_t prefix = 0;
  uint32_t tile_rows = 0, tile_cols = 0;
  uint64_t rows = 0, cols = 0, ld = 0;
  uint64_t grid_rows = 0, grid_cols = 0, bytes = 0;

  RNS8_LAYOUT_HD uint64_t offset(uint32_t plane, uint64_t row, uint64_t col) const {
    if (kind == ResidueLayoutKind::modulus_major) return (static_cast<uint64_t>(plane) * rows + row) * ld + col;
    const uint64_t tile = (row / tile_rows) * grid_cols + col / tile_cols;
    return (((tile * prefix + plane) * tile_rows + row % tile_rows) * tile_cols + col % tile_cols);
  }
};

inline bool make_residue_layout(ResidueLayoutKind kind, uint64_t rows, uint64_t cols, uint64_t ld,
                                uint32_t prefix, uint32_t tile_rows, uint32_t tile_cols, ResidueLayout& out) {
  if (!rows || !cols || ld < cols || !prefix || prefix > 20) return false;
  ResidueLayout candidate;
  candidate.kind = kind;
  candidate.rows = rows; candidate.cols = cols; candidate.ld = ld; candidate.prefix = prefix;
  uint64_t bytes = prefix;
  auto multiply = [&](uint64_t factor) {
    if (!factor || bytes > std::numeric_limits<std::size_t>::max() / factor) return false;
    bytes *= factor;
    return true;
  };
  if (kind == ResidueLayoutKind::modulus_major) {
    if (tile_rows || tile_cols || !multiply(rows) || !multiply(ld)) return false;
  } else if (kind == ResidueLayoutKind::tile_major) {
    if (!tile_rows || !tile_cols || tile_rows > 128 || tile_cols > 128 || ld != cols) return false;
    candidate.tile_rows = tile_rows; candidate.tile_cols = tile_cols;
    candidate.grid_rows = 1 + (rows - 1) / tile_rows;
    candidate.grid_cols = 1 + (cols - 1) / tile_cols;
    if (!multiply(candidate.grid_rows) || !multiply(candidate.grid_cols) || !multiply(tile_rows) || !multiply(tile_cols))
      return false;
  } else return false;
  candidate.bytes = bytes;
  out = candidate;
  return true;
}

inline bool valid_residue_layout(const ResidueLayout& value) {
  ResidueLayout expected;
  return value.version == 1 && value.basis_version == 1 &&
         make_residue_layout(value.kind, value.rows, value.cols, value.ld, value.prefix,
                             value.tile_rows, value.tile_cols, expected) &&
         value.grid_rows == expected.grid_rows && value.grid_cols == expected.grid_cols && value.bytes == expected.bytes;
}

inline bool compatible_residue_layouts(const ResidueLayout& a, const ResidueLayout& b) {
  return valid_residue_layout(a) && valid_residue_layout(b) && a.rows == b.rows && a.cols == b.cols && a.prefix == b.prefix;
}

inline bool convert_residue_layout_host(const int8_t* src, const ResidueLayout& source,
                                        int8_t* dst, const ResidueLayout& destination) {
  if (!compatible_residue_layouts(source, destination) ||
      !nonoverlapping_byte_ranges(src, source.bytes, dst, destination.bytes)) return false;
  for (uint32_t p = 0; p < source.prefix; ++p)
    for (uint64_t row = 0; row < source.rows; ++row)
      for (uint64_t col = 0; col < source.cols; ++col)
        dst[destination.offset(p, row, col)] = src[source.offset(p, row, col)];
  return true;
}
}
#undef RNS8_LAYOUT_HD
