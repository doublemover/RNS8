#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace rns8::bench {
inline constexpr std::array<std::string_view, 8> block_sparse_patterns = {
    "dense", "mixed", "disjoint", "half", "diagonal", "banded", "skewed", "clustered"};

// Logical native-integer zero masks, not per-residue or hardware 2:4 masks.
inline bool block_sparse_pattern_active(std::string_view pattern, bool lhs, uint64_t tile,
                                        uint64_t kb, uint64_t k_tiles) {
  if (!k_tiles || kb >= k_tiles)
    return false;
  if (pattern == "dense")
    return true;
  if (pattern == "disjoint")
    return kb % 2 == (lhs ? 0u : 1u);
  if (pattern == "half")
    return kb % 2 == 0;
  const auto center = tile % k_tiles;
  if (pattern == "diagonal")
    return kb == center;
  if (pattern == "banded")
    return (center >= kb ? center - kb : kb - center) <= 1;
  if (pattern == "skewed")
    return tile % 4 == 0 || kb == center;
  if (pattern == "clustered")
    return (kb / 4) % 2 == (tile / 2) % 2;
  if (pattern == "mixed")
    return tile % 3 == 0 || (tile % 3 == 2 ? kb == 0 : kb % 2 == (lhs ? 0u : 1u));
  return false;
}
} // namespace rns8::bench
