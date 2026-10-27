#pragma once
#include <algorithm>
#include <vector>
#include "residue_layout.hpp"
#include "native_tile_bound.hpp"
#include "garner_constants.hpp"
#include "basis_word_arithmetic.hpp"
#if !defined(__HIP_DEVICE_COMPILE__)
#include "rns8/status.h"
#endif

namespace rns8::detail {
inline BasisPacket ragged_basis_packet(const OrderedBasisIdentity& identity=OrderedBasisIdentity{}) {
  BasisPacket packet;packet.count=identity.count;packet.identity=identity.fingerprint;packet.products[0][0]=1;
  for(uint32_t p=0;p<packet.count;++p) {
    const auto q=identity.moduli[p];packet.moduli[p]=q;
    packet.inverse_prefix[p]=uint16_t(garner::inverse(basis_words_modulo(packet.products[p],q),q));
    basis_add_scaled_words(packet.products[p+1],packet.products[p],q);
  }
  for(uint32_t p=0;p<=packet.count;++p) {
    // ceil(product/2), with a carry from the next higher word.
    for(unsigned w=0;w<3;++w)packet.negative_thresholds[p][w]=(packet.products[p][w]>>1)|(w<2?packet.products[p][w+1]<<63:0);
    uint64_t carry=packet.products[p][0]&1;
    for(unsigned w=0;w<3 && carry;++w){++packet.negative_thresholds[p][w];carry=!packet.negative_thresholds[p][w];}
  }
  return packet;
}
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
  BasisPacket basis=ragged_basis_packet();
  std::vector<RaggedResidueTile> tiles;
};
inline bool valid_ragged_residue_layout(
    const RaggedResidueLayout& layout, uint64_t expected_instance, uint64_t expected_generation,
    uint32_t target_prefix) {
  const auto identity=ordered_basis_identity(layout.basis);
  if (layout.version != 1 || layout.basis_version != 1 || !valid_ordered_basis(identity) || !layout.rows || !layout.cols ||
      !layout.tile_rows || !layout.tile_cols || layout.tile_rows > 128 || layout.tile_cols > 128 ||
      !layout.value_instance || layout.value_instance != expected_instance ||
      layout.value_generation != expected_generation || layout.value_generation == UINT64_MAX ||
      !target_prefix || target_prefix > identity.count) return false;
  const auto checked=ragged_basis_packet(identity);
  for(uint32_t p=0;p<=identity.count;++p) {
    if(p<identity.count && layout.basis.inverse_prefix[p]!=checked.inverse_prefix[p])return false;
    for(unsigned w=0;w<3;++w)if(layout.basis.products[p][w]!=checked.products[p][w] ||
        layout.basis.negative_thresholds[p][w]!=checked.negative_thresholds[p][w])return false;
  }
  const uint64_t rows = 1 + (layout.rows - 1) / layout.tile_rows;
  const uint64_t cols = 1 + (layout.cols - 1) / layout.tile_cols;
  if (rows != layout.grid_rows || cols != layout.grid_cols || rows > SIZE_MAX / cols ||
      rows * cols != layout.tiles.size()) return false;
  const auto& basis = layout.basis;
  const uint64_t area = uint64_t{layout.tile_rows} * layout.tile_cols;
  uint64_t offset = 0;
  for (const auto& tile : layout.tiles) {
    if (!tile.prefix || tile.prefix > basis.count || tile.reserved || tile.byte_offset != offset ||
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
  return valid_residue_layout(destination) && destination.basis==ordered_basis_identity(source.basis) &&
      destination.rows == source.rows && destination.cols == source.cols &&
         valid_ragged_residue_layout(source, instance, generation, destination.prefix);
}
#if !defined(__HIP_DEVICE_COMPILE__)
rns8_status materialize_ragged_residues_host(
    const int8_t* source, const RaggedResidueLayout& layout, int8_t* destination,
    const ResidueLayout& target, uint64_t expected_instance, uint64_t expected_generation,
    uint64_t max_staged_bytes);
#endif
} // namespace rns8::detail
