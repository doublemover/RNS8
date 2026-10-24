#pragma once
#include <algorithm>
#include <vector>
#include "residue_layout.hpp"
#include "native_tile_bound.hpp"
#include "garner_constants.hpp"
#if !defined(__HIP_DEVICE_COMPILE__)
#include "rns8/status.h"
#endif

namespace rns8::detail {
struct RaggedResidueTile {
  uint64_t byte_offset = 0;
  uint64_t magnitude_bound[3]{};
  uint32_t prefix = 0;
  uint32_t reserved = 0;
};
// A present tile has an exact unique value at every logical cell. Missing
// suffix planes are absent information, never stored zeros. Identity names the
// logical value whose range proof authorized the directory.
struct RaggedResidueLayout {
  uint32_t version = 1, basis_version = 1;
  uint64_t rows = 0, cols = 0, grid_rows = 0, grid_cols = 0, bytes = 0;
  uint32_t tile_rows = 0, tile_cols = 0;
  uint64_t value_instance = 0, value_generation = 0;
  bool signed_value = true;
  std::vector<RaggedResidueTile> tiles;
};
inline bool valid_ragged_residue_layout(
    const RaggedResidueLayout& layout, uint64_t expected_instance, uint64_t expected_generation,
    uint32_t target_prefix) {
  if (layout.version != 1 || layout.basis_version != 1 || !layout.rows || !layout.cols ||
      !layout.tile_rows || !layout.tile_cols || layout.tile_rows > 128 || layout.tile_cols > 128 ||
      !layout.value_instance || layout.value_instance != expected_instance ||
      layout.value_generation != expected_generation || layout.value_generation == UINT64_MAX ||
      !target_prefix || target_prefix > 20) return false;
  const uint64_t rows = 1 + (layout.rows - 1) / layout.tile_rows;
  const uint64_t cols = 1 + (layout.cols - 1) / layout.tile_cols;
  if (rows != layout.grid_rows || cols != layout.grid_cols || rows > SIZE_MAX / cols ||
      rows * cols != layout.tiles.size()) return false;
  constexpr auto basis = garner::make_constants();
  const uint64_t area = uint64_t{layout.tile_rows} * layout.tile_cols;
  uint64_t offset = 0;
  for (const auto& tile : layout.tiles) {
    if (!tile.prefix || tile.prefix > 20 || tile.reserved || tile.byte_offset != offset ||
        (tile.magnitude_bound[2] >> 32)) return false;
    NativeTileBound bound{{tile.magnitude_bound[0],tile.magnitude_bound[1],tile.magnitude_bound[2]},0,0};
    select_native_bound_prefix(bound, layout.signed_value,
                              std::min(tile.prefix, target_prefix), basis.products);
    if (bound.required_prefix == 21) return false;
    const uint64_t bytes = area * tile.prefix;
    if (offset > SIZE_MAX - bytes) return false;
    offset += bytes;
  }
  return offset == layout.bytes;
}
inline bool compatible_ragged_destination(
    const RaggedResidueLayout& source, const ResidueLayout& destination,
    uint64_t instance, uint64_t generation) {
  return valid_residue_layout(destination) && destination.rows == source.rows && destination.cols == source.cols &&
         valid_ragged_residue_layout(source, instance, generation, destination.prefix);
}
#if !defined(__HIP_DEVICE_COMPILE__)
rns8_status materialize_ragged_residues_host(
    const int8_t* source, const RaggedResidueLayout& layout, int8_t* destination,
    const ResidueLayout& target, uint64_t expected_instance, uint64_t expected_generation,
    uint64_t max_staged_bytes);
#endif
} // namespace rns8::detail
