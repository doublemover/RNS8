#pragma once

#include "block_sparse.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace rns8::detail {
struct BlockSparseDeviceCapability;
using BlockSparseCapabilityProvider = rns8_status (*)(int, BlockSparseDeviceCapability&);
struct BlockSparseDeviceCapability {
  int device_id = -1;
  bool supported = false;
  std::array<char, 64> target{};
  // Process-local provider identity: never serialize or reuse across libraries.
  BlockSparseCapabilityProvider provider = nullptr;
};

struct BlockSparseDeviceDirectory {
  uint32_t version = 1, basis_version = 0, prefix = 0;
  BlockSparseRole role = BlockSparseRole::lhs;
  int device_id = -1;
  uint64_t rows = 0, cols = 0, instance = 0, generation = 0;
  const void* allocation = nullptr;
  uint64_t allocation_bytes = 0, offset_count = 0, block_count = 0;
  const uint32_t* offsets = nullptr;
  const uint32_t* k_blocks = nullptr;
};

inline bool block_sparse_directory_bytes(const BlockSparseOperand& layout, uint64_t& bytes) {
  const auto maximum = uint64_t(PTRDIFF_MAX) / sizeof(uint32_t);
  if (layout.offsets.size() > maximum || layout.k_blocks.size() > maximum - layout.offsets.size())
    return false;
  bytes = (uint64_t(layout.offsets.size()) + layout.k_blocks.size()) * sizeof(uint32_t);
  return bytes != 0;
}

inline bool block_sparse_directory_matches(const BlockSparseDeviceDirectory& directory,
                                           const BlockSparseOperand& layout, int device_id,
                                           uint64_t instance, uint64_t generation) {
  uint64_t bytes = 0;
  if (device_id < 0 || directory.version != 1 || directory.device_id != device_id || !instance ||
      generation == UINT64_MAX || directory.instance != instance ||
      directory.generation != generation || directory.basis_version != layout.basis_version ||
      layout.version != 1 || layout.basis_version != 1 || !layout.prefix || layout.prefix > 20 ||
      (layout.role != BlockSparseRole::lhs && layout.role != BlockSparseRole::rhs) ||
      !layout.rows || !layout.cols || layout.value_instance != instance ||
      layout.value_generation != generation || directory.role != layout.role ||
      directory.rows != layout.rows || directory.cols != layout.cols ||
      directory.prefix != layout.prefix || directory.offset_count != layout.offsets.size() ||
      directory.block_count != layout.k_blocks.size() || !directory.allocation ||
      reinterpret_cast<uintptr_t>(directory.allocation) % alignof(uint32_t) ||
      !block_sparse_directory_bytes(layout, bytes) || directory.allocation_bytes != bytes)
    return false;
  const auto base = reinterpret_cast<uintptr_t>(directory.allocation);
  if (base > UINTPTR_MAX - bytes || directory.offsets != directory.allocation ||
      reinterpret_cast<uintptr_t>(directory.k_blocks) !=
          base + layout.offsets.size() * sizeof(uint32_t))
    return false;
  const auto outer = layout.role == BlockSparseRole::lhs ? layout.rows : layout.cols;
  const auto blocks = layout.k_blocks.size();
  return layout.offsets.size() == 2 + (outer - 1) / 16 && blocks <= UINT32_MAX &&
         blocks <= UINT64_MAX / (uint64_t(1024) * layout.prefix) &&
         layout.payload_bytes == uint64_t(blocks) * 1024 * layout.prefix;
}

// Full structural validation belongs to preparation. The opaque public owner
// keeps the admitted host directory immutable and checks generations on reuse.
inline bool make_block_sparse_device_directory(const BlockSparseOperand& layout, int device_id,
                                               const void* allocation, uint64_t bytes,
                                               BlockSparseDeviceDirectory& out) {
  uint64_t expected_bytes = 0;
  if (!valid_block_sparse_operand(layout, layout.value_instance, layout.value_generation) ||
      !block_sparse_directory_bytes(layout, expected_bytes) || expected_bytes != bytes ||
      !allocation || reinterpret_cast<uintptr_t>(allocation) % alignof(uint32_t))
    return false;
  BlockSparseDeviceDirectory candidate;
  candidate.device_id = device_id;
  candidate.basis_version = layout.basis_version;
  candidate.prefix = layout.prefix;
  candidate.role = layout.role;
  candidate.rows = layout.rows;
  candidate.cols = layout.cols;
  candidate.instance = layout.value_instance;
  candidate.generation = layout.value_generation;
  candidate.allocation = allocation;
  candidate.allocation_bytes = bytes;
  candidate.offset_count = layout.offsets.size();
  candidate.block_count = layout.k_blocks.size();
  candidate.offsets = static_cast<const uint32_t*>(allocation);
  const auto base = reinterpret_cast<uintptr_t>(allocation);
  const auto offset_bytes = layout.offsets.size() * sizeof(uint32_t);
  if (base > UINTPTR_MAX - offset_bytes)
    return false;
  candidate.k_blocks = reinterpret_cast<const uint32_t*>(base + offset_bytes);
  if (!block_sparse_directory_matches(candidate, layout, device_id, layout.value_instance,
                                      layout.value_generation))
    return false;
  out = candidate;
  return true;
}
} // namespace rns8::detail
