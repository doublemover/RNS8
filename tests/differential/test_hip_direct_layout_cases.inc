#include "../support/layout_gemm_fixture.hpp"
TEST_CASE("HIP layout-aware products consume shared-prefix packed B without reading poison planes") {
  if(!hip_available())SKIP("no HIP device for layout-aware residue product");
  using namespace rns8::detail;
  for(const auto [k,prefix]:rns8::test::layout_gemm_cases()) {
    rns8::test::LayoutGemmFixture f(k,prefix,k%2!=0,k<32);
    hip_direct_device_buffer da,db,dp,dc;
    REQUIRE(da.allocate(0,f.a.bytes)==RNS8_SUCCESS);
    REQUIRE(db.allocate(0,f.b.bytes)==RNS8_SUCCESS);
    REQUIRE(dp.allocate(0,f.packed.bytes)==RNS8_SUCCESS);
    REQUIRE(dc.allocate(0,f.c.bytes)==RNS8_SUCCESS);
    std::vector<int8_t> packed(f.packed.bytes,99),actual(f.c.bytes,77);
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),f.av.data(),f.av.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db.get(),f.bv.data(),f.bv.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,dp.get(),packed.data(),packed.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
    const uint32_t middle=prefix/2;
    if(middle)REQUIRE(hip_direct_convert_residue_planes(0,db.get(),f.b,dp.get(),f.packed,0,middle)==RNS8_SUCCESS);
    REQUIRE(hip_direct_convert_residue_planes(0,db.get(),f.b,dp.get(),f.packed,middle,prefix)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,packed.data(),dp.get(),packed.size())==RNS8_SUCCESS);
    REQUIRE(packed==f.packed_expected);
    REQUIRE(hip_direct_gemm_residue_layout(0,da.get(),f.a,dp.get(),f.packed,dc.get(),f.c,prefix)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
    REQUIRE(actual==f.expected);
    CHECK(hip_direct_gemm_residue_layout(0,da.get(),f.a,dp.get(),f.packed,da.get(),f.c,prefix)==RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("direct HIP tile-major residue conversion matches host mapping and round trips") {
  if (!hip_available()) SKIP("no HIP device for residue layout differential");
  using namespace rns8::detail;
  for (const uint64_t rows : {1, 3, 17})
    for (const uint32_t prefix : {1, 9, 20}) {
      constexpr uint64_t cols = 19;
      ResidueLayout plain, tiled;
      REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major, rows, cols, cols + 3, prefix, 0, 0, plain));
      REQUIRE(make_residue_layout(ResidueLayoutKind::tile_major, rows, cols, cols, prefix, 4, 8, tiled));
      std::vector<int8_t> source(plain.bytes, 77), expected(tiled.bytes, 99), actual(tiled.bytes, 99), restored(plain.bytes, 77);
      for (uint32_t p = 0; p < prefix; ++p)
        for (uint64_t row = 0; row < rows; ++row)
          for (uint64_t col = 0; col < cols; ++col)
            source[plain.offset(p, row, col)] = static_cast<int8_t>(static_cast<int>((p * 13 + row * 5 + col) % 255) - 127);
      REQUIRE(convert_residue_layout_host(source.data(), plain, expected.data(), tiled));
      void *ds = nullptr, *dt = nullptr, *dr = nullptr;
      REQUIRE(hip_direct_allocate(0, plain.bytes, &ds) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, tiled.bytes, &dt) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, plain.bytes, &dr) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, ds, source.data(), source.size()) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, dt, actual.data(), actual.size()) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, dr, restored.data(), restored.size()) == RNS8_SUCCESS);
      auto bad = tiled; ++bad.grid_cols;
      CHECK(hip_direct_convert_residue_layout(0, ds, plain, dt, bad) == RNS8_INVALID_ARGUMENT);
      CHECK(hip_direct_convert_residue_layout(0, ds, plain, ds, plain) == RNS8_INVALID_ARGUMENT);
      REQUIRE(hip_direct_convert_residue_layout(0, ds, plain, dt, tiled) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dt, actual.size()) == RNS8_SUCCESS);
      REQUIRE(actual == expected);
      REQUIRE(hip_direct_convert_residue_layout(0, dt, tiled, dr, plain) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0, restored.data(), dr, restored.size()) == RNS8_SUCCESS);
      REQUIRE(restored == source);
      CHECK(hip_direct_free(0, dr) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, dt) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, ds) == RNS8_SUCCESS);
    }
}
