#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
namespace rns8::detail {
// A status-bearing query: failure is never a valid zero-byte workspace.
// Regions are byte-aligned INT8 storage, disjoint for their complete lifetimes.
struct BilinearResources {
  rns8_status status = RNS8_INVALID_ARGUMENT;
  uint64_t block_m = 0, block_n = 0, block_k = 0;
  uint64_t a_bytes = 0, b_bytes = 0, product_bytes = 0, output_bytes = 0;
  uint64_t b_offset = 0, product_offset = 0, output_offset = 0, total_bytes = 0;
  AddressDomain source_a{}, source_b{}, destination{};
};

inline BilinearResources
query_bilinear_resources(int64_t m, int64_t n, int64_t k, int64_t lda,
                         int64_t ldb, int64_t ldc, uint32_t prefix,
                         uint32_t blocks, uint64_t budget) {
  BilinearResources result;
  if (m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !prefix ||
      prefix > 20 || (blocks != 2 && blocks != 3 && blocks != 4))
    return result;
  // Original transforms/commit compute (plane*rows+row)*ld in int64. Their
  // scalar dimensions and one-plane grid domains are still qualified as i32.
  if (!qualified_i32_matrix_address(m, k, lda) ||
      !qualified_i32_matrix_address(k, n, ldb) ||
      !qualified_i32_matrix_address(m, n, ldc)) {
    result.status = RNS8_UNSUPPORTED_BACKEND;
    return result;
  }
  if (!checked_host_plane_address(m, k, lda, prefix, 1, result.source_a) ||
      !checked_host_plane_address(k, n, ldb, prefix, 1, result.source_b) ||
      !checked_host_plane_address(m, n, ldc, prefix, 1, result.destination)) {
    result.status = RNS8_RANGE_ERROR;
    return result;
  }
  result.block_m = (uint64_t(m) + blocks - 1) / blocks;
  result.block_n = (uint64_t(n) + blocks - 1) / blocks;
  result.block_k = (uint64_t(k) + blocks - 1) / blocks;
  // The actual sub-GEMM retains the stricter full-stack i32 contract.
  if (!qualified_i32_gemm_address(result.block_m, result.block_n,
                                  result.block_k, result.block_k,
                                  result.block_n, result.block_n, prefix)) {
    result.status = RNS8_UNSUPPORTED_BACKEND;
    return result;
  }
  // Above address proofs bound each product by INT32_MAX before multiplying
  // by at most 20 planes. Accumulate scratch with an explicit size_t check.
  result.a_bytes = result.block_m * result.block_k * prefix;
  result.b_bytes = result.block_k * result.block_n * prefix;
  result.product_bytes = result.block_m * result.block_n * prefix;
  result.output_bytes = uint64_t(m) * n * prefix;
  uint64_t total = 0;
  const uint64_t sizes[]{result.a_bytes, result.b_bytes, result.product_bytes,
                         result.output_bytes};
  for (uint64_t bytes : sizes) {
    if (bytes > SIZE_MAX - total || bytes > uint64_t(PTRDIFF_MAX) - total) {
      result.status = RNS8_RANGE_ERROR;
      return result;
    }
    total += bytes;
  }
  result.b_offset = result.a_bytes;
  result.product_offset = result.b_offset + result.b_bytes;
  result.output_offset = result.product_offset + result.product_bytes;
  result.total_bytes = total;
  result.status = total > budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  return result;
}
// Borrowed arena admission covers the full arena, not just each disjoint
// subrange. Lifetime/draining is owned by the public workspace lease.
inline rns8_status validate_bilinear_arena(const BilinearResources& r,const void* scratch,
    uint64_t bytes,const void* a,const void* b,const void* c) {
  if(!scratch)return bytes?RNS8_INVALID_ARGUMENT:RNS8_SUCCESS;
  if(bytes<r.total_bytes)return RNS8_WORKSPACE_TOO_SMALL;
  if(!nonoverlapping_byte_ranges(scratch,bytes,a,r.source_a.byte_extent) ||
     !nonoverlapping_byte_ranges(scratch,bytes,b,r.source_b.byte_extent) ||
     !nonoverlapping_byte_ranges(scratch,bytes,c,r.destination.byte_extent))return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
} // namespace rns8::detail
