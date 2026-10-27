#include <catch2/catch_test_macros.hpp>
#include "../support/ragged_fixture.hpp"

TEST_CASE("ragged exact prefixes materialize values across a different tile layout") {
  using namespace rns8::detail;
  for (bool signed_values : {false,true}) {
    auto f=make_ragged_fixture(signed_values);
    REQUIRE(valid_ragged_residue_layout(f.layout,71,9,20));
    REQUIRE_FALSE(valid_ragged_residue_layout(f.layout,71,10,20));
    REQUIRE_FALSE(valid_ragged_residue_layout(f.layout,71,9,9));
    for (auto kind : {ResidueLayoutKind::modulus_major,ResidueLayoutKind::tile_major}) {
      ResidueLayout target;
      const bool tiled=kind==ResidueLayoutKind::tile_major;
      REQUIRE(make_residue_layout(kind,3,5,tiled?5:7,20,tiled?2:0,tiled?3:0,target));
      std::vector<int8_t> output(target.bytes,99), expected=output;
      for (uint32_t p=0;p<20;++p) for (uint64_t row=0;row<3;++row) for(uint64_t col=0;col<5;++col)
        expected[target.offset(p,row,col)]=centered_residue(f.values[row*5+col],kDefaultModuli[p]);
      REQUIRE(materialize_ragged_residues_host(f.bytes.data(),f.layout,output.data(),target,71,9,target.bytes)==RNS8_SUCCESS);
      REQUIRE(output==expected);
      const auto saved=output;
      REQUIRE(materialize_ragged_residues_host(f.bytes.data(),f.layout,output.data(),target,71,10,target.bytes)==RNS8_INVALID_ARGUMENT);
      REQUIRE(output==saved);
      REQUIRE(materialize_ragged_residues_host(f.bytes.data(),f.layout,output.data(),target,71,9,0)==RNS8_WORKSPACE_TOO_SMALL);
      auto invalid=f.layout;
      invalid.tiles[0].magnitude_bound[0]=0;
      REQUIRE(materialize_ragged_residues_host(f.bytes.data(),invalid,output.data(),target,71,9,target.bytes)==RNS8_RANGE_ERROR);
      REQUIRE(output==saved);
      invalid=f.layout; invalid.tiles[1].byte_offset++;
      REQUIRE_FALSE(valid_ragged_residue_layout(invalid,71,9,20));
    }
  }
}
