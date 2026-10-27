#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "backend_common/residue_layout.hpp"
#include "backend_common/fragment_layout.hpp"
#include "backend_rocwmma/rocwmma_packing_policy.hpp"
#include "backend_common/native_tiled_resources.hpp"
#include <array>
#include "../support/layout_gemm_fixture.hpp"

TEST_CASE("layout-aware residue products preserve exact planes tails and full native boundaries") {
  using namespace rns8::detail;
  for(const auto [k,prefix]:rns8::test::layout_gemm_cases()) {
    rns8::test::LayoutGemmFixture f(k,prefix,k%2!=0,k<32);
    std::vector<int8_t> packed(f.packed.bytes,99),actual(f.c.bytes,77);
    REQUIRE(convert_residue_planes_host(f.bv.data(),f.b,packed.data(),f.packed,0,prefix));
    REQUIRE(packed==f.packed_expected);
    REQUIRE(residue_layout_gemm_host(f.av.data(),f.a,packed.data(),f.packed,actual.data(),f.c,prefix));
    REQUIRE(actual==f.expected);
    const auto before=actual;
    auto wrong=f.packed;std::swap(wrong.basis.moduli[0],wrong.basis.moduli[1]);
    wrong.basis.fingerprint=ordered_basis_fingerprint(wrong.basis.moduli.data(),wrong.basis.count);
    CHECK_FALSE(residue_layout_gemm_host(f.av.data(),f.a,packed.data(),wrong,actual.data(),f.c,prefix));
    CHECK_FALSE(convert_residue_planes_host(f.bv.data(),f.b,packed.data(),f.packed,prefix,prefix));
    CHECK_FALSE(convert_residue_planes_host(f.bv.data(),f.b,packed.data(),f.packed,0,21));
    CHECK_FALSE(residue_layout_gemm_host(f.av.data(),f.a,packed.data(),f.packed,const_cast<int8_t*>(f.av.data()),f.c,prefix));
    CHECK(actual==before);
  }
}

TEST_CASE("binary fragment algebra preserves image replicas inverse and composition") {
  namespace binary=rns8::detail::binary_layout;namespace fragment=rns8::detail::fragment_layout;
  for(const auto f:{fragment::a,fragment::b,fragment::c}) {
    const auto map=fragment::linear_map(f);REQUIRE(binary::valid(map));
    const auto elimination=binary::eliminate(map);CHECK(elimination.rank==8);
    CHECK(elimination.nullity==map.input_bits-8);
    const auto inverse=binary::right_inverse(map);REQUIRE(inverse.valid);
    const auto identity=binary::compose(map,inverse.map);REQUIRE(identity.valid);
    for(unsigned logical=0;logical<256;++logical) {
      CHECK(identity.map.apply(logical)==logical);
      const auto solved=binary::solve(map,logical);REQUIRE(solved.valid);CHECK(map.apply(solved.value)==logical);
      for(unsigned i=0;i<elimination.nullity;++i)CHECK(map.apply(solved.value^elimination.kernel[i])==logical);
    }
    for(unsigned slot=0;slot<(1u<<map.input_bits);++slot)CHECK(map.apply(slot)==f.logical(slot));
  }
  binary::Physical a{fragment::linear_map(fragment::a),5,4,0,8},b{fragment::linear_map(fragment::b),5,4,0,8};
  CHECK(binary::conversion(a,a)==binary::Conversion::register_only);
  CHECK(binary::conversion(a,b)==binary::Conversion::within_wave);
  binary::Physical source{{6,6,{1,2,4,8,16,32}},5,0,1,8},destination=source;
  std::swap(destination.map.columns[0],destination.map.columns[5]);
  CHECK(binary::conversion(source,destination)==binary::Conversion::cross_wave_lds);
  source.map.columns[5]=0;
  CHECK(binary::conversion(source,destination)==binary::Conversion::unsupported);
  CHECK_FALSE(binary::right_inverse(source.map).valid);
  auto malformed=source.map;malformed.columns[15]=1;CHECK_FALSE(binary::valid(malformed));
  CHECK_FALSE(binary::compose(source.map,fragment::linear_map(fragment::a)).valid);
  CHECK_FALSE(binary::solve(source.map,64).valid);
  CHECK_FALSE(binary::solve(source.map,0,1,2).valid);
}

TEST_CASE("GF2 paired columns cover exactly two adjacent instruction outputs") {
  namespace binary=rns8::detail::binary_layout;
  std::array<unsigned,32> visits{};const auto inverse=binary::right_inverse(binary::paired_columns);
  REQUIRE(inverse.valid);
  for(unsigned axis=0;axis<16;++axis)for(unsigned pair=0;pair<2;++pair) {
    const auto col=binary::paired_column(axis,pair);CHECK(col==axis*2+pair);++visits[col];
    CHECK(inverse.map.apply(col)==(axis|(pair<<4)));
  }
  for(auto count:visits)CHECK(count==1);
  const auto identity=binary::fingerprint(binary::paired_columns,0x1100,32,1,8);
  CHECK(identity!=binary::fingerprint(binary::paired_columns,0x1100,64,1,8));
  CHECK(identity!=binary::fingerprint(binary::paired_columns,0x1101,32,1,8));
  CHECK(identity!=binary::fingerprint(binary::paired_columns,0x1100,32,2,8));
  for(unsigned group:{1u,2u}) {
    const auto r=rns8::detail::query_native_tiled_resources(16,0,0,rns8::detail::NativeConversion::paired_columns,group,1);
    REQUIRE(r.status==RNS8_SUCCESS);CHECK(r.output_rows==16);CHECK(r.output_cols==32);
    CHECK(r.accumulator_i32_per_lane==16*group);CHECK(r.fragment_words_per_lane==12*group);CHECK(r.total_lds_bytes==0);
    CHECK(r.conversions_per_panel==1536*group);CHECK(r.raw_operand_scalars_per_panel==768);
  }
  CHECK(rns8::detail::query_native_tiled_resources(16,0,0,rns8::detail::NativeConversion::paired_columns,4,1).status==RNS8_INVALID_ARGUMENT);
}

TEST_CASE("LDS bank model requires explicit scalar instruction phases and handles broadcasts") {
  namespace binary=rns8::detail::binary_layout;
  std::array<uint64_t,64> addresses{};binary::LdsContract c;c.opcode=binary::LdsOpcode::read_b32;
  for(unsigned i=0;i<64;++i)addresses[i]=4*i;
  auto model=binary::analyze_banks(addresses.data(),32,c);REQUIRE(model.supported);CHECK(model.max_bank_words==1);CHECK_FALSE(model.measured);
  for(unsigned i=0;i<32;++i)addresses[i]=128*i;
  CHECK(binary::analyze_banks(addresses.data(),32,c).max_bank_words==32);
  addresses.fill(0);CHECK(binary::analyze_banks(addresses.data(),32,c).max_bank_words==1);
  c.opcode=binary::LdsOpcode::write_b32;CHECK_FALSE(binary::analyze_banks(addresses.data(),32,c).supported);
  c.opcode=binary::LdsOpcode::read_b32;c.wave_lanes=64;
  CHECK(binary::analyze_banks(addresses.data(),64,c).phases==2);
  c.opcode=binary::LdsOpcode::unknown;CHECK_FALSE(binary::analyze_banks(addresses.data(),64,c).supported);
  c.opcode=binary::LdsOpcode::read_b32;addresses[7]=1;CHECK_FALSE(binary::analyze_banks(addresses.data(),64,c).supported);
}

TEST_CASE("rocWMMA direct-A admission proves complete aligned fragment loads") {
  using namespace rns8::detail;
  CHECK(rocwmma_direct_a_eligible(256,16,65536,65552));
  CHECK_FALSE(rocwmma_direct_a_eligible(257,16,16,16));
  CHECK_FALSE(rocwmma_direct_a_eligible(256,15,16,16));
  CHECK_FALSE(rocwmma_direct_a_eligible(256,16,17,32));
  CHECK_FALSE(rocwmma_direct_a_eligible(256,16,16,17));
  CHECK_FALSE(rocwmma_direct_a_eligible(0,16,16,16));
  RocwmmaPanelKey key{256,16,17,32,34,0};
  for(unsigned field=0;field<6;++field) {
    auto changed=key;
    switch(field){case 0:++changed.source;break;case 1:++changed.extent;break;case 2:++changed.valid_k;break;
      case 3:++changed.padded_k;break;case 4:++changed.ld;break;case 5:++changed.k_offset;break;}
    CHECK_FALSE(key==changed);
  }
}

TEST_CASE("RDNA3 executable fragment maps agree with independent coordinate enumeration") {
  namespace f = rns8::detail::fragment_layout;
  for (const auto map : {f::a,f::b,f::c}) {
    REQUIRE(f::admitted(map));
    std::array<unsigned,256> visits{};
    for(unsigned lane=0;lane<32;++lane) {
      if(map.role==f::Role::accumulator) {
        for(unsigned reg=0;reg<8;++reg) {
          // Independent legacy instruction convention: lane half chooses row
          // parity; low lane chooses column; each register adds two rows.
          const unsigned expected_row=2*reg+lane/16, expected_col=lane%16;
          const auto coordinate=f::c_coordinate(lane,reg);
          REQUIRE(f::row(coordinate)==expected_row);
          REQUIRE(f::col(coordinate)==expected_col);
          ++visits[coordinate];
        }
      } else {
        for(unsigned word=0;word<4;++word) for(unsigned byte=0;byte<4;++byte) {
          const auto coordinate=map.logical(f::input_slot(lane,word,byte));
          const unsigned contraction=4*word+byte, axis=lane%16;
          REQUIRE(f::row(coordinate)==(map.role==f::Role::a?axis:contraction));
          REQUIRE(f::col(coordinate)==(map.role==f::Role::a?contraction:axis));
          ++visits[coordinate];
        }
      }
    }
    for(unsigned logical=0;logical<256;++logical) {
      REQUIRE(visits[logical]==map.replicas);
      const auto canonical=f::canonical_slot(map,logical);
      REQUIRE(canonical<(1u<<map.physical_bits));
      REQUIRE(map.logical(canonical)==logical);
      if(map.replicas==2) REQUIRE((canonical&16)==0);
    }
    for(unsigned rows : {0,1,7,15,16}) for(unsigned cols : {0,1,9,15,16}) {
      unsigned active=0;
      for(unsigned slot=0;slot<(1u<<map.physical_bits);++slot)
        active+=f::active(map,slot,rows,cols);
      REQUIRE(active==rows*cols*map.replicas);
    }
    REQUIRE_FALSE(f::active(map,1u<<map.physical_bits,16,16));
    REQUIRE_FALSE(f::active(map,0,17,16));
  }
}

TEST_CASE("fragment conversion preserves replicas and rejects altered instruction contracts") {
  namespace f = rns8::detail::fragment_layout;
  REQUIRE(f::conversion(f::a,f::a)==f::Conversion::register_only);
  REQUIRE(f::conversion(f::a,f::b)==f::Conversion::within_wave);
  REQUIRE(f::conversion(f::a,f::c)==f::Conversion::unsupported);
  for(unsigned slot=0;slot<512;++slot) {
    REQUIRE((f::source_slot(f::a,f::a,slot)&31)==(slot&31));
    const auto selected=f::source_slot(f::a,f::b,slot);
    REQUIRE(selected<512);
    REQUIRE(f::a.logical(selected)==f::b.logical(slot));
    const auto reverse=f::source_slot(f::b,f::a,selected);
    REQUIRE(f::b.logical(reverse)==f::a.logical(selected));
  }
  for(unsigned mutation=0;mutation<7;++mutation) {
    auto bad=f::a;
    switch(mutation) {
      case 0: bad.basis[0]=2; break;
      case 1: bad.basis[7]=32; break;
      case 2: bad.instruction_version=2; break;
      case 3: bad.wave_size=64; break;
      case 4: bad.masked_edges=false; break;
      case 5: bad.replicas=1; break;
      case 6: bad.physical_bits=32; break;
    }
    REQUIRE_FALSE(f::admitted(bad));
    REQUIRE(f::canonical_slot(bad,0)==UINT32_MAX);
    REQUIRE(f::source_slot(bad,f::b,0)==UINT32_MAX);
    REQUIRE(f::conversion(bad,f::b)==f::Conversion::unsupported);
  }
}

TEST_CASE("uniform tile-major residue mappings are bijective and preserve padding") {
  using namespace rns8::detail;
  for (const uint64_t rows : {1, 3, 7, 17})
    for (const uint64_t cols : {1, 5, 19})
      for (uint32_t prefix = 1; prefix <= 20; ++prefix)
        for (const uint32_t tile : {1, 4, 16}) {
          ResidueLayout plain, tiled;
          REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major, rows, cols, cols + 3, prefix, 0, 0, plain));
          REQUIRE(make_residue_layout(ResidueLayoutKind::tile_major, rows, cols, cols, prefix, tile, tile * 2, tiled));
          std::vector<int8_t> source(plain.bytes, 77), packed(tiled.bytes, 99), restored(plain.bytes, 77);
          std::vector<uint8_t> visited(tiled.bytes, 0);
          for (uint32_t p = 0; p < prefix; ++p)
            for (uint64_t row = 0; row < rows; ++row)
              for (uint64_t col = 0; col < cols; ++col) {
                source[plain.offset(p, row, col)] = static_cast<int8_t>(static_cast<int>((p * 13 + row * 5 + col) % 255) - 127);
                const auto offset = tiled.offset(p, row, col);
                REQUIRE(offset < tiled.bytes);
                REQUIRE(visited[offset] == 0);
                visited[offset] = 1;
              }
          REQUIRE(convert_residue_layout_host(source.data(), plain, packed.data(), tiled));
          REQUIRE(convert_residue_layout_host(packed.data(), tiled, restored.data(), plain));
          REQUIRE(restored == source);
          for (std::size_t offset = 0; offset < packed.size(); ++offset)
            if (!visited[offset]) REQUIRE(packed[offset] == 99);
          auto bad = tiled;
          --bad.bytes;
          CHECK_FALSE(valid_residue_layout(bad));
          bad = tiled; ++bad.basis_version;
          CHECK_FALSE(compatible_residue_layouts(plain, bad));
          CHECK_FALSE(convert_residue_layout_host(source.data(), plain, source.data(), plain));
        }
}

TEST_CASE("residue layout admission rejects overflowing or missing-plane descriptors") {
  using namespace rns8::detail;
  ResidueLayout layout, other;
  CHECK_FALSE(make_residue_layout(ResidueLayoutKind::modulus_major, UINT64_MAX, 2, 2, 20, 0, 0, layout));
  CHECK_FALSE(make_residue_layout(ResidueLayoutKind::tile_major, UINT64_MAX, 1, 1, 20, 16, 16, layout));
  CHECK_FALSE(make_residue_layout(ResidueLayoutKind::tile_major, 1, 1, 1, 1, 0, 16, layout));
  REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major, 3, 5, 5, 1, 0, 0, layout));
  REQUIRE(make_residue_layout(ResidueLayoutKind::tile_major, 3, 5, 5, 2, 4, 8, other));
  CHECK_FALSE(compatible_residue_layouts(layout, other));
}

TEST_CASE("layout conversion binds complete ordered experimental bases") {
  using namespace rns8::detail;
  auto basis=OrderedBasisIdentity{};std::swap(basis.moduli[0],basis.moduli[1]);
  basis.fingerprint=ordered_basis_fingerprint(basis.moduli.data(),basis.count);
  ResidueLayout source,tiled;
  REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major,3,5,7,20,0,0,source,basis));
  REQUIRE(make_residue_layout(ResidueLayoutKind::tile_major,3,5,5,20,4,8,tiled,basis));
  std::vector<int8_t> bytes(source.bytes,71),packed(tiled.bytes,99),restored(source.bytes,71);
  for(unsigned p=0;p<20;++p)for(unsigned r=0;r<3;++r)for(unsigned c=0;c<5;++c)
    bytes[source.offset(p,r,c)]=int8_t(int((p*17+r*5+c)%127)-63);
  REQUIRE(convert_residue_layout_host(bytes.data(),source,packed.data(),tiled));
  REQUIRE(convert_residue_layout_host(packed.data(),tiled,restored.data(),source));
  CHECK(restored==bytes);
  auto altered=tiled;altered.basis=OrderedBasisIdentity{};
  CHECK_FALSE(compatible_residue_layouts(source,altered));
  const auto unchanged=packed;
  CHECK_FALSE(convert_residue_layout_host(bytes.data(),source,packed.data(),altered));CHECK(packed==unchanged);
  altered=tiled;altered.basis.fingerprint^=1;CHECK_FALSE(valid_residue_layout(altered));
  altered=tiled;altered.version=1;CHECK_FALSE(valid_residue_layout(altered));
  auto short_basis=basis;short_basis.count=2;
  std::fill(short_basis.moduli.begin()+2,short_basis.moduli.end(),0);
  short_basis.fingerprint=ordered_basis_fingerprint(short_basis.moduli.data(),short_basis.count);
  CHECK_FALSE(make_residue_layout(ResidueLayoutKind::modulus_major,3,5,5,3,0,0,altered,short_basis));
  REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major,3,5,5,2,0,0,altered,short_basis));
  altered.basis.moduli[19]=7;CHECK_FALSE(valid_residue_layout(altered));
  short_basis.moduli[1]=short_basis.moduli[0];
  short_basis.fingerprint=ordered_basis_fingerprint(short_basis.moduli.data(),short_basis.count);
  CHECK_FALSE(valid_ordered_basis(short_basis));
}
