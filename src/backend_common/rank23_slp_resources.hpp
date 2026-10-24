#pragma once
#include "bilinear_resources.hpp"
namespace rns8::detail {
// Separate experimental SLP schedule: all 23 transformed operands and products
// are materialized. Local-register liveness does NOT eliminate these buffers.
inline BilinearResources query_rank23_slp_resources(
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint64_t budget) {
  auto result = query_bilinear_resources(m,n,k,lda,ldb,ldc,prefix,3,UINT64_MAX);
  if (result.status != RNS8_SUCCESS) return result;
  uint64_t total = 0;
  const uint64_t bytes[]{result.a_bytes, result.b_bytes, result.product_bytes, result.output_bytes};
  uint64_t regions[4]{};
  for (int i = 0; i < 4; ++i) {
    const uint64_t copies = i == 3 ? 1 : 23;
    if (bytes[i] > SIZE_MAX / copies || bytes[i] * copies > SIZE_MAX - total ||
        bytes[i] * copies > uint64_t(PTRDIFF_MAX) - total) {
      result.status = RNS8_RANGE_ERROR;
      return result;
    }
    regions[i] = bytes[i] * copies;
    total += regions[i];
  }
  result.b_offset = regions[0];
  result.product_offset = result.b_offset + regions[1];
  result.output_offset = result.product_offset + regions[2];
  result.total_bytes = total;
  result.status = total > budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  return result;
}
} // namespace rns8::detail
