#ifndef RNS8_ADDRESS_DOMAIN_HPP
#define RNS8_ADDRESS_DOMAIN_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rns8::detail {

struct AddressDomain {
  uint64_t last_element = 0;
  uint64_t byte_extent = 0;
};

// Host planes are rows*ld apart. Prove only accessed bytes, within the
// representable C++ object/difference domain; an unused last pitch is harmless.
inline bool checked_host_plane_address(uint64_t rows, uint64_t cols, uint64_t ld,
                                       uint64_t planes, uint64_t element_bytes,
                                       AddressDomain& out) {
  constexpr uint64_t object_limit = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  constexpr uint64_t size_limit = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  constexpr uint64_t limit = object_limit < size_limit ? object_limit : size_limit;
  if (!rows || !cols || ld < cols || !planes || !element_bytes ||
      element_bytes > limit || rows > UINT64_MAX / planes) return false;
  const uint64_t cells_limit = limit / element_bytes;
  const uint64_t logical_rows = rows * planes;
  if (cols > cells_limit || logical_rows - 1 > (cells_limit - cols) / ld) return false;
  const uint64_t cells = (logical_rows - 1) * ld + cols;
  out = {cells - 1, cells * element_bytes};
  return true;
}

inline bool nonoverlapping_byte_ranges(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
  constexpr auto limit = std::numeric_limits<uintptr_t>::max();
  const auto first = reinterpret_cast<uintptr_t>(a), second = reinterpret_cast<uintptr_t>(b);
  if (!a || !b || !a_bytes || !b_bytes || a_bytes > limit || b_bytes > limit ||
      first > limit - a_bytes || second > limit - b_bytes) return false;
  return first + a_bytes <= second || second + b_bytes <= first;
}

// Prove the last accessed element and byte extent without evaluating an
// overflowing product. Plane stride is in elements; zero means dense planes.
inline bool checked_address_domain(uint64_t rows, uint64_t cols, uint64_t ld,
                                   uint64_t element_bytes, uint64_t planes,
                                   uint64_t plane_stride, AddressDomain& out) {
  constexpr uint64_t limit = std::numeric_limits<std::size_t>::max();
  if (!rows || !cols || ld < cols || !element_bytes || !planes ||
      rows > limit || cols > limit || ld > limit || element_bytes > limit) return false;
  if (rows - 1 > (limit - (cols - 1)) / ld) return false;
  uint64_t last = (rows - 1) * ld + cols - 1;
  if (!plane_stride) {
    if (rows > limit / ld) return false;
    plane_stride = rows * ld;
  }
  if (plane_stride <= last || planes - 1 > (limit - last) / plane_stride) return false;
  last += (planes - 1) * plane_stride;
  if (last >= limit / element_bytes) return false;
  out = {last, (last + 1) * element_bytes};
  return true;
}

// Conservative qualification for kernels whose global intermediate products
// still use signed int. Reserve a K slab for rounded grids/tile-tail additions.
// This is a backend restriction, never a restriction on CPU descriptors.
inline bool qualified_i32_matrix_address(int64_t rows, int64_t cols, int64_t ld,
                                        uint64_t planes = 1, uint64_t element_bytes = 1) {
  constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
  constexpr int64_t dimension_limit = std::numeric_limits<int32_t>::max() - 65536;
  if (rows <= 0 || cols <= 0 || ld < cols || rows > dimension_limit ||
      cols > dimension_limit || ld > dimension_limit) return false;
  AddressDomain domain;
  if (!checked_address_domain(rows, cols, ld, element_bytes, planes, 0, domain)) return false;
  // Full padded plane products can be formed before a kernel's bounds guard.
  return static_cast<uint64_t>(rows) <= limit / static_cast<uint64_t>(ld) / planes;
}

inline bool qualified_i32_gemm_address(int64_t m, int64_t n, int64_t k,
                                      int64_t lda, int64_t ldb, int64_t ldc,
                                      uint64_t planes = 1) {
  return qualified_i32_matrix_address(m, k, lda, planes, 8) &&
         qualified_i32_matrix_address(k, n, ldb, planes, 8) &&
         qualified_i32_matrix_address(m, n, ldc, planes, 8);
}

} // namespace rns8::detail
#endif
