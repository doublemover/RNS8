#include "backend_common/ragged_residues.hpp"
#include "core/internal.hpp"
#include <algorithm>
#include <new>

namespace rns8::detail {
rns8_status materialize_ragged_residues_host(
    const int8_t* source, const RaggedResidueLayout& layout, int8_t* destination,
    const ResidueLayout& target, uint64_t expected_instance, uint64_t expected_generation,
    uint64_t max_staged_bytes) {
  if (!compatible_ragged_destination(layout, target, expected_instance, expected_generation) ||
      !nonoverlapping_byte_ranges(source, layout.bytes, destination, target.bytes)) return RNS8_INVALID_ARGUMENT;
  if (target.bytes > max_staged_bytes) return RNS8_WORKSPACE_TOO_SMALL;
  try {
    std::vector<int8_t> staged(destination, destination + target.bytes), residues(20);
    const auto& basis = layout.basis;
    const uint64_t area = uint64_t{layout.tile_rows} * layout.tile_cols;
    using boost::multiprecision::cpp_int;
    for (uint64_t row = 0; row < layout.rows; ++row)
      for (uint64_t col = 0; col < layout.cols; ++col) {
        const auto& tile = layout.tiles[(row / layout.tile_rows) * layout.grid_cols + col / layout.tile_cols];
        const uint64_t local = (row % layout.tile_rows) * layout.tile_cols + col % layout.tile_cols;
        for (uint32_t p = 0; p < tile.prefix; ++p) {
          const int value = source[tile.byte_offset + p * area + local];
          if (value < -static_cast<int>(basis.moduli[p] / 2) ||
              value > static_cast<int>((basis.moduli[p] - 1) / 2)) return RNS8_INVALID_ARGUMENT;
          residues[p] = static_cast<int8_t>(value);
        }
        const auto& product_bits = basis.products[tile.prefix];
        const cpp_int product = cpp_int(product_bits[0]) + (cpp_int(product_bits[1]) << 64) +
                                (cpp_int(product_bits[2]) << 128);
        cpp_int value=0;
        if(ordered_basis_identity(basis)==OrderedBasisIdentity{})value=reconstruct_canonical(residues,tile.prefix);
        else for(uint32_t p=0;p<tile.prefix;++p) {
          const auto q=basis.moduli[p];int canonical=residues[p];if(canonical<0)canonical+=q;
          const auto digit=uint32_t((canonical+q-(value%q).convert_to<uint32_t>())*uint32_t(basis.inverse_prefix[p])%q);
          const cpp_int weight=cpp_int(basis.products[p][0])+(cpp_int(basis.products[p][1])<<64)+(cpp_int(basis.products[p][2])<<128);
          value+=weight*digit;
        }
        if (layout.signed_value && value >= (product + 1) / 2) value -= product;
        const cpp_int bound = cpp_int(tile.magnitude_bound[0]) + (cpp_int(tile.magnitude_bound[1]) << 64) +
                              (cpp_int(tile.magnitude_bound[2]) << 128);
        if (boost::multiprecision::abs(value) > bound) return RNS8_RANGE_ERROR;
        for (uint32_t p = 0; p < target.prefix; ++p) {
          int residue = (value % basis.moduli[p]).convert_to<int>();
          if (residue < 0) residue += basis.moduli[p];
          if (residue >= static_cast<int>((basis.moduli[p] + 1) / 2)) residue -= basis.moduli[p];
          staged[target.offset(p,row,col)] = static_cast<int8_t>(residue);
        }
      }
    std::copy(staged.begin(), staged.end(), destination);
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) { return RNS8_INTERNAL_ERROR; }
}
} // namespace rns8::detail
