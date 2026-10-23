#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "backend_common/residue_layout.hpp"

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
