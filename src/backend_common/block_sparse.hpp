#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
#include <cstdint>
#include <vector>

namespace rns8::detail {
enum class BlockSparseRole : uint32_t { lhs = 1, rhs = 2 };
// LHS: CSR of 16-row tiles with 16x64 blocks. RHS: CSC of 16-column
// tiles with 64x16 blocks. Each entry stores prefix contiguous 1024-byte
// planes. Absence defines an exact zero block; native packing checks actual
// zeros.
struct BlockSparseOperand {
  uint32_t version = 1, basis_version = 1, prefix = 0;
  BlockSparseRole role = BlockSparseRole::lhs;
  uint64_t rows = 0, cols = 0, value_instance = 0, value_generation = 0,
           payload_bytes = 0;
  std::vector<uint32_t> offsets, k_blocks;
};
inline bool valid_block_sparse_operand(const BlockSparseOperand &value,
                                       uint64_t instance, uint64_t generation) {
  if (value.version != 1 || value.basis_version != 1 || !value.prefix ||
      value.prefix > 20 || !value.rows || !value.cols ||
      !value.value_instance || value.value_instance != instance ||
      value.value_generation != generation || generation == UINT64_MAX ||
      (value.role != BlockSparseRole::lhs &&
       value.role != BlockSparseRole::rhs))
    return false;
  const uint64_t outer =
      value.role == BlockSparseRole::lhs ? value.rows : value.cols;
  const uint64_t inner =
      value.role == BlockSparseRole::lhs ? value.cols : value.rows;
  const uint64_t groups = 1 + (outer - 1) / 16, kb = 1 + (inner - 1) / 64;
  if (groups >= UINT32_MAX || kb > UINT32_MAX ||
      value.k_blocks.size() > UINT32_MAX ||
      value.offsets.size() != groups + 1 || value.offsets.front() != 0 ||
      value.offsets.back() != value.k_blocks.size())
    return false;
  for (uint64_t g = 0; g < groups; ++g) {
    const uint32_t begin = value.offsets[g], end = value.offsets[g + 1];
    if (begin > end || end > value.k_blocks.size())
      return false;
    for (uint32_t i = begin; i < end; ++i)
      if (value.k_blocks[i] >= kb ||
          (i > begin && value.k_blocks[i - 1] >= value.k_blocks[i]))
        return false;
  }
  const uint64_t block_bytes = uint64_t{1024} * value.prefix;
  return value.k_blocks.size() <= SIZE_MAX / block_bytes &&
         value.payload_bytes == value.k_blocks.size() * block_bytes;
}
rns8_status pack_block_sparse_i64(const int64_t *source, uint64_t rows,
                                  uint64_t cols, uint64_t ld, uint32_t prefix,
                                  BlockSparseRole role, uint64_t instance,
                                  uint64_t generation, uint64_t budget,
                                  BlockSparseOperand &out,
                                  std::vector<int8_t> &payload);
rns8_status pack_block_sparse_u64(const uint64_t *source, uint64_t rows,
                                  uint64_t cols, uint64_t ld, uint32_t prefix,
                                  BlockSparseRole role, uint64_t instance,
                                  uint64_t generation, uint64_t budget,
                                  BlockSparseOperand &out,
                                  std::vector<int8_t> &payload);
rns8_status materialize_block_sparse_host(const BlockSparseOperand &layout,
                                          const int8_t *payload, int8_t *dense,
                                          uint64_t ld, uint64_t instance,
                                          uint64_t generation);
} // namespace rns8::detail
