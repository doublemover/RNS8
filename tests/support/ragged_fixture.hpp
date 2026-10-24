#pragma once
#include "backend_common/ragged_residues.hpp"
#include "core/internal.hpp"

struct RaggedFixture {
  rns8::detail::RaggedResidueLayout layout;
  std::vector<boost::multiprecision::cpp_int> values;
  std::vector<int8_t> bytes;
};
inline RaggedFixture make_ragged_fixture(bool signed_values) {
  using boost::multiprecision::cpp_int;
  RaggedFixture f;
  auto& l = f.layout;
  l.rows=3; l.cols=5; l.tile_rows=2; l.tile_cols=2; l.grid_rows=2; l.grid_cols=3;
  l.value_instance=71; l.value_generation=9; l.signed_value=signed_values;
  const uint32_t prefixes[] = {1,2,9,3,12,20};
  const uint32_t bits[] = {4,8,60,16,80,140};
  l.tiles.resize(6);
  for (int t=0; t<6; ++t) {
    l.tiles[t].byte_offset=l.bytes; l.tiles[t].prefix=prefixes[t];
    l.bytes += prefixes[t]*4;
  }
  f.bytes.assign(l.bytes,99); f.values.resize(15);
  std::vector<cpp_int> bounds(6);
  for (uint64_t row=0; row<l.rows; ++row) for (uint64_t col=0; col<l.cols; ++col) {
    const uint64_t t=(row/2)*3+col/2, local=(row%2)*2+col%2;
    cpp_int value=(cpp_int(1)<<bits[t])+row*3+col;
    if (signed_values && (row+col)%2) value=-value;
    f.values[row*5+col]=value;
    const cpp_int magnitude=boost::multiprecision::abs(value);
    if (magnitude>bounds[t]) bounds[t]=magnitude;
    for (uint32_t p=0; p<prefixes[t]; ++p)
      f.bytes[l.tiles[t].byte_offset+p*4+local]=rns8::detail::centered_residue(value,rns8::detail::kDefaultModuli[p]);
  }
  const cpp_int mask=(cpp_int(1)<<64)-1;
  for (int t=0; t<6; ++t)
    for (int limb=0; limb<3; ++limb)
      l.tiles[t].magnitude_bound[limb]=((bounds[t]>>(64*limb))&mask).convert_to<uint64_t>();
  return f;
}
