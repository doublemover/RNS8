#include "backend_common/block_sparse.hpp"
#include "../../benchmarks/block_sparse_patterns.hpp"
#include <set>
#include "backend_common/block_sparse_directory.hpp"
#include "core/internal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("prepared sparse directory admission binds device extents roles and generations") {
  using namespace rns8::detail;
  const int64_t native = 7;
  BlockSparseOperand layout;
  std::vector<int8_t> payload;
  REQUIRE(pack_block_sparse_i64(&native, 1, 1, 1, 9, BlockSparseRole::lhs, 21, 4, UINT64_MAX,
                                layout, payload) == RNS8_SUCCESS);
  uint64_t bytes = 0;
  REQUIRE(block_sparse_directory_bytes(layout, bytes));
  std::vector<uint32_t> storage(static_cast<size_t>(bytes / sizeof(uint32_t)));
  BlockSparseDeviceDirectory directory;
  REQUIRE(make_block_sparse_device_directory(layout, 3, storage.data(), bytes, directory));
  REQUIRE(block_sparse_directory_matches(directory, layout, 3, 21, 4));
  for (unsigned mutation = 0; mutation < 12; ++mutation) {
    CAPTURE(mutation);
    auto changed = directory;
    switch (mutation) {
    case 0:
      ++changed.version;
      break;
    case 1:
      ++changed.basis_version;
      break;
    case 2:
      ++changed.prefix;
      break;
    case 3:
      changed.role = BlockSparseRole::rhs;
      break;
    case 4:
      ++changed.device_id;
      break;
    case 5:
      ++changed.instance;
      break;
    case 6:
      ++changed.generation;
      break;
    case 7:
      ++changed.allocation_bytes;
      break;
    case 8:
      ++changed.offset_count;
      break;
    case 9:
      ++changed.block_count;
      break;
    case 10:
      changed.offsets = nullptr;
      break;
    default:
      changed.k_blocks = nullptr;
      break;
    }
    CHECK_FALSE(block_sparse_directory_matches(changed, layout, 3, 21, 4));
  }
  const auto original = directory;
  CHECK_FALSE(make_block_sparse_device_directory(layout, 3, storage.data(), bytes - 1, directory));
  CHECK(directory.allocation == original.allocation);
  CHECK(directory.allocation_bytes == original.allocation_bytes);
  auto malformed = layout;
  malformed.k_blocks[0] = UINT32_MAX;
  CHECK_FALSE(make_block_sparse_device_directory(malformed, 3, storage.data(), bytes, directory));
  CHECK_FALSE(block_sparse_directory_matches(directory, layout, 3, 21, 5));
  CHECK_FALSE(block_sparse_directory_matches(directory, layout, 4, 21, 4));
}

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
              (row + col) % 3 == 0 ? 256 : ((row + col) % 2 ? INT64_MIN : INT64_MAX);
      BlockSparseOperand layout;
      std::vector<int8_t> packed;
      REQUIRE(pack_block_sparse_i64(source.data(), rows, cols, ld, prefix, role, 7, 3, UINT64_MAX,
                                    layout, packed) == RNS8_SUCCESS);
      REQUIRE(valid_block_sparse_operand(layout, 7, 3));
      REQUIRE_FALSE(valid_block_sparse_operand(layout, 7, 4));
      std::vector<int8_t> dense(prefix * rows * ld, 77), expected = dense;
      for (uint32_t p = 0; p < prefix; ++p)
        for (uint64_t row = 0; row < rows; ++row)
          for (uint64_t col = 0; col < cols; ++col)
            expected[(p * rows + row) * ld + col] =
                centered_residue(cpp_int(source[row * ld + col]), kDefaultModuli[p]);
      REQUIRE(materialize_block_sparse_host(layout, packed.data(), dense.data(), ld, 7, 3) ==
              RNS8_SUCCESS);
      REQUIRE(dense == expected);
      const auto saved = packed;
      REQUIRE(pack_block_sparse_i64(source.data(), rows, cols, ld, prefix, role, 7, 3, 0, layout,
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
  REQUIRE(pack_block_sparse_u64(&value, 1, 1, 1, 1, BlockSparseRole::lhs, 8, 1, 4096, layout,
                                packed) == RNS8_SUCCESS);
  REQUIRE(layout.k_blocks.size() == 1);
  REQUIRE(std::all_of(packed.begin(), packed.end(), [](int8_t x) { return x == 0; }));
  const uint64_t zero = 0;
  REQUIRE(pack_block_sparse_u64(&zero, 1, 1, 1, 1, BlockSparseRole::lhs, 8, 2, 4096, layout,
                                packed) == RNS8_SUCCESS);
  REQUIRE(layout.k_blocks.empty());
  REQUIRE(packed.empty());
}

TEST_CASE("natural block-sparse benchmark recipes preserve explicit integer-zero masks") {
  using namespace rns8::bench;
  using Masks = std::array<std::set<uint64_t>, 4>;
  const std::array<std::pair<std::string_view, Masks>, 5> cases{{
      {"half", {{{0, 2, 4, 6}, {0, 2, 4, 6}, {0, 2, 4, 6}, {0, 2, 4, 6}}}},
      {"diagonal", {{{0}, {1}, {2}, {3}}}},
      {"banded", {{{0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}}}},
      {"skewed", {{{0, 1, 2, 3, 4, 5, 6, 7}, {1}, {2}, {3}}}},
      {"clustered", {{{0, 1, 2, 3}, {0, 1, 2, 3}, {4, 5, 6, 7}, {4, 5, 6, 7}}}},
  }};
  for (const auto& example : cases)
    for (bool lhs : {false, true})
      for (uint64_t tile = 0; tile < 4; ++tile) {
        std::set<uint64_t> observed;
        for (uint64_t k = 0; k < 8; ++k)
          if (block_sparse_pattern_active(example.first, lhs, tile, k, 8))
            observed.insert(k);
        CHECK(observed == example.second[tile]);
      }
  for (auto pattern : block_sparse_patterns) {
    CHECK_FALSE(block_sparse_pattern_active(pattern, true, 0, 0, 0));
    CHECK_FALSE(block_sparse_pattern_active(pattern, true, 0, 8, 8));
  }
  CHECK(block_sparse_pattern_active("banded", true, UINT64_MAX - 1, UINT64_MAX - 2, UINT64_MAX));
  CHECK_FALSE(block_sparse_pattern_active("banded", true, UINT64_MAX - 1, 0, UINT64_MAX));
}
