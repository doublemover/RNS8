#include "backend_common/block_sparse.hpp"
#include "backend_common/garner_constants.hpp"
#include <algorithm>
#include <new>
#include <type_traits>

namespace rns8::detail {
namespace {
template <typename T>
rns8_status pack_native(const T *source, uint64_t rows, uint64_t cols,
                        uint64_t ld, uint32_t prefix, BlockSparseRole role,
                        uint64_t instance, uint64_t generation, uint64_t budget,
                        BlockSparseOperand &out, std::vector<int8_t> &payload) {
  AddressDomain address;
  if (!source || !prefix || prefix > 20 || !instance ||
      generation == UINT64_MAX ||
      (role != BlockSparseRole::lhs && role != BlockSparseRole::rhs) ||
      !checked_address_domain(rows, cols, ld, sizeof(T), 1, 0, address))
    return RNS8_INVALID_ARGUMENT;
  const uint64_t outer = role == BlockSparseRole::lhs ? rows : cols;
  const uint64_t inner = role == BlockSparseRole::lhs ? cols : rows;
  const uint64_t groups = 1 + (outer - 1) / 16, kb = 1 + (inner - 1) / 64;
  if (groups >= UINT32_MAX || kb > UINT32_MAX)
    return RNS8_RANGE_ERROR;
  const uint64_t offset_bytes = (groups + 1) * sizeof(uint32_t);
  if (offset_bytes > budget || offset_bytes > SIZE_MAX)
    return RNS8_WORKSPACE_TOO_SMALL;
  try {
    BlockSparseOperand layout;
    layout.role = role;
    layout.rows = rows;
    layout.cols = cols;
    layout.prefix = prefix;
    layout.value_instance = instance;
    layout.value_generation = generation;
    layout.offsets.resize(groups + 1);
    for (uint64_t group = 0; group < groups; ++group) {
      layout.offsets[group] = static_cast<uint32_t>(layout.k_blocks.size());
      for (uint64_t block = 0; block < kb; ++block) {
        bool nonzero = false;
        for (uint64_t i = 0; i < 16 && group * 16 + i < outer && !nonzero; ++i)
          for (uint64_t q = 0; q < 64 && block * 64 + q < inner; ++q) {
            const uint64_t row =
                role == BlockSparseRole::lhs ? group * 16 + i : block * 64 + q;
            const uint64_t col =
                role == BlockSparseRole::lhs ? block * 64 + q : group * 16 + i;
            if (source[row * ld + col] != 0) {
              nonzero = true;
              break;
            }
          }
        if (!nonzero)
          continue;
        const uint64_t entries = uint64_t{layout.k_blocks.size()} + 1;
        const uint64_t entry_bytes = uint64_t{1024} * prefix + sizeof(uint32_t);
        if (entries > UINT32_MAX || entries > SIZE_MAX / entry_bytes ||
            entries > (budget - offset_bytes) / entry_bytes)
          return RNS8_WORKSPACE_TOO_SMALL;
        layout.k_blocks.push_back(static_cast<uint32_t>(block));
      }
    }
    layout.offsets.back() = static_cast<uint32_t>(layout.k_blocks.size());
    layout.payload_bytes = layout.k_blocks.size() * uint64_t{1024} * prefix;
    std::vector<int8_t> packed(layout.payload_bytes, 0);
    constexpr auto basis = garner::make_constants();
    for (uint64_t group = 0; group < groups; ++group)
      for (uint32_t entry = layout.offsets[group];
           entry < layout.offsets[group + 1]; ++entry)
        for (uint64_t i = 0; i < 16 && group * 16 + i < outer; ++i)
          for (uint64_t q = 0;
               q < 64 && uint64_t{layout.k_blocks[entry]} * 64 + q < inner;
               ++q) {
            const uint64_t row =
                role == BlockSparseRole::lhs
                    ? group * 16 + i
                    : uint64_t{layout.k_blocks[entry]} * 64 + q;
            const uint64_t col = role == BlockSparseRole::lhs
                                     ? uint64_t{layout.k_blocks[entry]} * 64 + q
                                     : group * 16 + i;
            const T value = source[row * ld + col];
            for (uint32_t p = 0; p < prefix; ++p) {
              int residue;
              if constexpr (std::is_signed_v<T>) {
                residue = static_cast<int>(
                    value % static_cast<int64_t>(basis.moduli[p]));
                if (residue < 0)
                  residue += static_cast<int>(basis.moduli[p]);
              } else
                residue = static_cast<int>(value % basis.moduli[p]);
              if (residue >= static_cast<int>((basis.moduli[p] + 1) / 2))
                residue -= static_cast<int>(basis.moduli[p]);
              const uint64_t local =
                  role == BlockSparseRole::lhs ? i * 64 + q : q * 16 + i;
              packed[(uint64_t{entry} * prefix + p) * 1024 + local] =
                  static_cast<int8_t>(residue);
            }
          }
    out = std::move(layout);
    payload = std::move(packed);
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc &) {
    return RNS8_INTERNAL_ERROR;
  }
}
} // namespace
rns8_status pack_block_sparse_i64(const int64_t *source, uint64_t rows,
                                  uint64_t cols, uint64_t ld, uint32_t prefix,
                                  BlockSparseRole role, uint64_t instance,
                                  uint64_t generation, uint64_t budget,
                                  BlockSparseOperand &out,
                                  std::vector<int8_t> &payload) {
  return pack_native(source, rows, cols, ld, prefix, role, instance, generation,
                     budget, out, payload);
}
rns8_status pack_block_sparse_u64(const uint64_t *source, uint64_t rows,
                                  uint64_t cols, uint64_t ld, uint32_t prefix,
                                  BlockSparseRole role, uint64_t instance,
                                  uint64_t generation, uint64_t budget,
                                  BlockSparseOperand &out,
                                  std::vector<int8_t> &payload) {
  return pack_native(source, rows, cols, ld, prefix, role, instance, generation,
                     budget, out, payload);
}
rns8_status materialize_block_sparse_host(const BlockSparseOperand &layout,
                                          const int8_t *payload, int8_t *dense,
                                          uint64_t ld, uint64_t instance,
                                          uint64_t generation) {
  AddressDomain domain;
  if (!valid_block_sparse_operand(layout, instance, generation) || !dense ||
      !checked_address_domain(layout.rows, layout.cols, ld, 1, layout.prefix, 0,
                              domain) ||
      (layout.payload_bytes &&
       !nonoverlapping_byte_ranges(payload, layout.payload_bytes, dense,
                                   domain.byte_extent)))
    return RNS8_INVALID_ARGUMENT;
  for (uint32_t p = 0; p < layout.prefix; ++p)
    for (uint64_t r = 0; r < layout.rows; ++r)
      std::fill_n(dense + (uint64_t{p} * layout.rows + r) * ld, layout.cols,
                  int8_t{0});
  for (uint64_t group = 0; group + 1 < layout.offsets.size(); ++group)
    for (uint32_t entry = layout.offsets[group];
         entry < layout.offsets[group + 1]; ++entry)
      for (uint64_t i = 0; i < 16; ++i)
        for (uint64_t q = 0; q < 64; ++q) {
          const uint64_t row = layout.role == BlockSparseRole::lhs
                                   ? group * 16 + i
                                   : uint64_t{layout.k_blocks[entry]} * 64 + q;
          const uint64_t col = layout.role == BlockSparseRole::lhs
                                   ? uint64_t{layout.k_blocks[entry]} * 64 + q
                                   : group * 16 + i;
          if (row >= layout.rows || col >= layout.cols)
            continue;
          const uint64_t local =
              layout.role == BlockSparseRole::lhs ? i * 64 + q : q * 16 + i;
          for (uint32_t p = 0; p < layout.prefix; ++p)
            dense[(uint64_t{p} * layout.rows + row) * ld + col] =
                payload[(uint64_t{entry} * layout.prefix + p) * 1024 + local];
        }
  return RNS8_SUCCESS;
}
} // namespace rns8::detail
