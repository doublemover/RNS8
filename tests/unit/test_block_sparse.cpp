#include "backend_common/block_sparse.hpp"
#include "core/internal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("native block sparsity is exact and dense materialization preserves "
          "every plane") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (auto role : {BlockSparseRole::lhs, BlockSparseRole::rhs})
    for (uint32_t prefix : {1u, 9u, 20u}) {
      constexpr uint64_t rows = 65, cols = 65, ld = 67;
      std::vector<int64_t> source(rows * ld, 99);
      for (uint64_t row = 0; row < rows; ++row)
        for (uint64_t col = 0; col < cols; ++col)
          source[row * ld + col] =
              (row + col) % 3 == 0 ? 256
                                   : ((row + col) % 2 ? INT64_MIN : INT64_MAX);
      BlockSparseOperand layout;
      std::vector<int8_t> packed;
      REQUIRE(pack_block_sparse_i64(source.data(), rows, cols, ld, prefix, role,
                                    7, 3, UINT64_MAX, layout,
                                    packed) == RNS8_SUCCESS);
      REQUIRE(valid_block_sparse_operand(layout, 7, 3));
      REQUIRE_FALSE(valid_block_sparse_operand(layout, 7, 4));
      std::vector<int8_t> dense(prefix * rows * ld, 77), expected = dense;
      for (uint32_t p = 0; p < prefix; ++p)
        for (uint64_t row = 0; row < rows; ++row)
          for (uint64_t col = 0; col < cols; ++col)
            expected[(p * rows + row) * ld + col] = centered_residue(
                cpp_int(source[row * ld + col]), kDefaultModuli[p]);
      REQUIRE(materialize_block_sparse_host(layout, packed.data(), dense.data(),
                                            ld, 7, 3) == RNS8_SUCCESS);
      REQUIRE(dense == expected);
      const auto saved = packed;
      REQUIRE(pack_block_sparse_i64(source.data(), rows, cols, ld, prefix, role,
                                    7, 3, 0, layout,
                                    packed) == RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE(packed == saved);
      auto malformed = layout;
      REQUIRE(malformed.offsets[1] >= 2);
      malformed.k_blocks[1] = malformed.k_blocks[0];
      REQUIRE_FALSE(valid_block_sparse_operand(malformed, 7, 3));
    }
  // A mathematically nonzero block divisible by the only stored modulus must
  // remain structurally present, even when its entire residue payload is zero.
  const uint64_t value = 256;
  BlockSparseOperand layout;
  std::vector<int8_t> packed;
  REQUIRE(pack_block_sparse_u64(&value, 1, 1, 1, 1, BlockSparseRole::lhs, 8, 1,
                                4096, layout, packed) == RNS8_SUCCESS);
  REQUIRE(layout.k_blocks.size() == 1);
  REQUIRE(std::all_of(packed.begin(), packed.end(),
                      [](int8_t x) { return x == 0; }));
  const uint64_t zero = 0;
  REQUIRE(pack_block_sparse_u64(&zero, 1, 1, 1, 1, BlockSparseRole::lhs, 8, 2,
                                4096, layout, packed) == RNS8_SUCCESS);
  REQUIRE(layout.k_blocks.empty());
  REQUIRE(packed.empty());
}
