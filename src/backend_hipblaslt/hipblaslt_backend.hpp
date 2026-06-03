#ifndef RNS8_BACKEND_HIPBLASLT_HPP
#define RNS8_BACKEND_HIPBLASLT_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/backend_common.hpp"
#include "rns8/rns8.h"

namespace rns8::detail {

constexpr std::size_t kHipblasLtAlignment = 16u;
constexpr std::size_t kHipblasLtBaselineWorkspaceBytes = 32u * 1024u * 1024u;

inline bool hipblaslt_round_up_aligned(uint64_t value, uint64_t& rounded) {
  constexpr uint64_t alignment = static_cast<uint64_t>(kHipblasLtAlignment);
  if (!round_up_aligned_u64(value, alignment, rounded)) {
    return false;
  }
  return rounded <= static_cast<uint64_t>(std::numeric_limits<int>::max());
}

inline bool hipblaslt_baseline_workspace_requirements(
    int64_t m,
    int64_t n,
    int64_t k,
    std::size_t& int32_scratch_bytes,
    std::size_t& workspace_bytes) {
  int32_scratch_bytes = 0;
  workspace_bytes = 0;
  if (m <= 0 || n <= 0 || k <= 0) {
    return false;
  }
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  uint64_t padded_k = 0;
  const uint64_t max_k_block =
      static_cast<uint64_t>(k) < static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK)
          ? static_cast<uint64_t>(k)
          : static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK);
  if (!hipblaslt_round_up_aligned(static_cast<uint64_t>(m), padded_m) ||
      !hipblaslt_round_up_aligned(static_cast<uint64_t>(n), padded_n) ||
      !hipblaslt_round_up_aligned(max_k_block, padded_k)) {
    return false;
  }
  uint64_t scratch_elements = 0;
  if (!checked_mul_u64(padded_m, padded_n, scratch_elements) ||
      scratch_elements > std::numeric_limits<std::size_t>::max() / sizeof(int32_t)) {
    return false;
  }
  int32_scratch_bytes = static_cast<std::size_t>(scratch_elements) * sizeof(int32_t);

  uint64_t pack_a_bytes = 0;
  uint64_t pack_b_bytes = 0;
  uint64_t pack_bytes = 0;
  if (!checked_mul_u64(padded_m, padded_k, pack_a_bytes) ||
      !checked_mul_u64(padded_n, padded_k, pack_b_bytes) ||
      pack_a_bytes > std::numeric_limits<std::size_t>::max() ||
      pack_b_bytes > std::numeric_limits<std::size_t>::max() ||
      pack_a_bytes > std::numeric_limits<uint64_t>::max() - pack_b_bytes) {
    return false;
  }
  pack_bytes = pack_a_bytes + pack_b_bytes;
  if (pack_bytes > std::numeric_limits<uint64_t>::max() - kHipblasLtBaselineWorkspaceBytes ||
      pack_bytes + kHipblasLtBaselineWorkspaceBytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  workspace_bytes = static_cast<std::size_t>(pack_bytes + kHipblasLtBaselineWorkspaceBytes);
  return true;
}

bool hipblaslt_compiled();
rns8_status hipblaslt_create_context(int device_id, rns8_device_info& out, void** handle, std::string& version);
rns8_status hipblaslt_destroy_context(int device_id, void* handle);
rns8_status hipblaslt_gemm_rns_device(
    int device_id,
    void* handle,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* int32_scratch,
    std::size_t int32_scratch_bytes,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix);
rns8_status hipblaslt_gemm_finite_u8_device(
    int device_id,
    void* handle,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* int32_scratch,
    std::size_t int32_scratch_bytes,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);

}  // namespace rns8::detail

#endif
