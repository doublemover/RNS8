#pragma once
#include "address_domain.hpp"
#include "native_tile_bound.hpp"
#include "rns8/rns8.h"

namespace rns8::detail {
// Static operation admission, not a device qualification or measured-cost record.
enum class AdaptiveNativeConstraint : uint32_t {
  none, shape, prefix, group, method, panel_depth, workers, address_span,
  panel_count, host_size, scratch_budget
};
struct AdaptiveNativeSupport {
  uint32_t version = 1;
  rns8_status status = RNS8_INVALID_ARGUMENT;
  AdaptiveNativeConstraint constraint = AdaptiveNativeConstraint::shape;
  bool resource_requirement_known = false;
  bool requires_device_capability_check = true;
  bool requires_data_bound_check = true;
  uint64_t tiles = 0, stage_bytes = 0, bound_bytes = 0, metadata_bytes = 0;
  uint64_t reserved_bytes = 0, discovery_bytes = 0, peak_scratch_bytes = 0;
};
inline AdaptiveNativeSupport query_adaptive_native_support(
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t capacity, uint32_t group, NativeBoundMethod method, uint32_t panel_k,
    uint32_t compact_workers, uint64_t budget) {
  AdaptiveNativeSupport out;
  const auto reject = [&](rns8_status status, AdaptiveNativeConstraint constraint) {
    out.status = status; out.constraint = constraint; return out;
  };
  if (m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n)
    return out;
  if (!capacity || capacity > 20) return reject(RNS8_INVALID_ARGUMENT, AdaptiveNativeConstraint::prefix);
  if (group != 1 && group != 2 && group != 4) return reject(RNS8_INVALID_ARGUMENT, AdaptiveNativeConstraint::group);
  if (method != NativeBoundMethod::tile_max && method != NativeBoundMethod::axis_l1 && method != NativeBoundMethod::panel_l1)
    return reject(RNS8_INVALID_ARGUMENT, AdaptiveNativeConstraint::method);
  if (method == NativeBoundMethod::panel_l1 && (!panel_k || panel_k > 65536))
    return reject(RNS8_INVALID_ARGUMENT, AdaptiveNativeConstraint::panel_depth);
  if (compact_workers > 65535) return reject(RNS8_INVALID_ARGUMENT, AdaptiveNativeConstraint::workers);
  if (!qualified_i32_matrix_address(m,k,lda,1,8) || !qualified_i32_matrix_address(k,n,ldb,1,8) ||
      !qualified_i32_matrix_address(m,n,ldc,capacity))
    return reject(RNS8_UNSUPPORTED_BACKEND, AdaptiveNativeConstraint::address_span);
  // Qualified input/output full-plane spans bound these products below UINT64_MAX.
  const uint64_t rows=(uint64_t(m)+15)/16, cols=(uint64_t(n)+15)/16;
  const uint64_t extent=method == NativeBoundMethod::panel_l1 ? panel_k : uint64_t(k);
  const uint64_t panels=(uint64_t(k)+extent-1)/extent;
  if (panels > 65535) return reject(RNS8_UNSUPPORTED_BACKEND, AdaptiveNativeConstraint::panel_count);
  out.tiles=rows*cols;
  out.stage_bytes=uint64_t(m)*uint64_t(n)*capacity;
  out.bound_bytes=out.tiles*sizeof(NativeTileBound);
  out.metadata_bytes=(out.tiles+23)*sizeof(uint32_t);
  out.reserved_bytes=out.stage_bytes+out.bound_bytes+out.metadata_bytes;
  out.discovery_bytes=(rows+cols)*panels*
      (method == NativeBoundMethod::tile_max ? sizeof(uint64_t) : sizeof(NativeAxisSummary))+43*sizeof(uint32_t);
  if (out.reserved_bytes > SIZE_MAX || out.discovery_bytes > SIZE_MAX-out.reserved_bytes)
    return reject(RNS8_RANGE_ERROR, AdaptiveNativeConstraint::host_size);
  out.peak_scratch_bytes=out.reserved_bytes+out.discovery_bytes;
  out.resource_requirement_known=true;
  if (out.peak_scratch_bytes > budget) return reject(RNS8_WORKSPACE_TOO_SMALL, AdaptiveNativeConstraint::scratch_budget);
  return reject(RNS8_SUCCESS, AdaptiveNativeConstraint::none);
}
} // namespace rns8::detail
