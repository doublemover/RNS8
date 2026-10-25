#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"

namespace rns8::detail {
// Static provider records describe implemented operations, not measured devices.
// Dynamic pointer/alias, generation, range and stream ownership remain separate.
enum class OperationConstraint : uint32_t { none, shape, prefix, address_span, scratch_budget, modulus, execution_mode, operation };
struct OperationSupport {
  uint32_t version = 1;
  rns8_status status = RNS8_INVALID_ARGUMENT;
  OperationConstraint constraint = OperationConstraint::shape;
  bool resources_known = false;
  bool requires_value_proof = true;
  bool device_qualified = false;
  uint64_t payload_bytes = 0;
  uint64_t scratch_bytes = 0;
};
inline OperationSupport query_residue_gemm_support(int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix) {
  OperationSupport out;
  if (m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldc<n) return out;
  if (!prefix || prefix>20) { out.constraint=OperationConstraint::prefix; return out; }
  if (!qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,prefix)) {
    out.status=RNS8_UNSUPPORTED_BACKEND; out.constraint=OperationConstraint::address_span; return out;
  }
  out.status=RNS8_SUCCESS; out.constraint=OperationConstraint::none; out.resources_known=true;
  return out;
}
inline OperationSupport query_canonical_crt_support(int64_t cells, uint32_t prefix) {
  OperationSupport out;
  if (cells<=0) return out;
  if (!prefix || prefix>20) { out.constraint=OperationConstraint::prefix; return out; }
  if (!qualified_i32_matrix_address(1,cells,cells,prefix,24)) {
    out.status=RNS8_UNSUPPORTED_BACKEND; out.constraint=OperationConstraint::address_span; return out;
  }
  out.status=RNS8_SUCCESS; out.constraint=OperationConstraint::none; out.resources_known=true;
  out.requires_value_proof=false; // canonical ring reconstruction, not a signed integer lift
  out.payload_bytes=uint64_t(cells)*24;
  return out;
}
inline OperationSupport query_finite_verify_support(int64_t m, int64_t n, int64_t k,
    uint32_t modulus, bool gpu, uint64_t budget) {
  OperationSupport out;
  out.requires_value_proof=false;
  if(m<=0 || n<=0 || k<=0) return out;
  if(modulus<2 || modulus>256) { out.constraint=OperationConstraint::modulus; return out; }
  const uint64_t limit=uint64_t(PTRDIFF_MAX), um=uint64_t(m),un=uint64_t(n),uk=uint64_t(k);
  const uint64_t copies=gpu?2:1, overhead=gpu?8:0;
  if(um>limit/uk || uk>limit/un || um>limit/un || uk>(limit-overhead)/4 ||
      un>(limit-overhead-uk*4)/copies) {
    out.status=RNS8_RANGE_ERROR; out.constraint=OperationConstraint::address_span; return out;
  }
  out.scratch_bytes=un*copies+uk*4+overhead;
  out.resources_known=true;
  out.status=out.scratch_bytes>budget?RNS8_RANGE_ERROR:RNS8_SUCCESS;
  out.constraint=out.status==RNS8_SUCCESS?OperationConstraint::none:OperationConstraint::scratch_budget;
  return out;
}
inline OperationSupport query_axpby_support(int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldd, int64_t ldc, uint32_t prefix) {
  OperationSupport out;
  if (m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldd<n || ldc<n) return out;
  if (!prefix || prefix>20) { out.constraint=OperationConstraint::prefix; return out; }
  if (!qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,prefix) ||
      !qualified_i32_matrix_address(m,n,ldd,prefix)) {
    out.status=RNS8_UNSUPPORTED_BACKEND; out.constraint=OperationConstraint::address_span; return out;
  }
  out.status=RNS8_SUCCESS; out.constraint=OperationConstraint::none; out.resources_known=true;
  return out; // fused epilogue has no private global scratch
}
inline OperationSupport query_exact_lift_support(int64_t cells, uint32_t source,
                                                uint32_t target, uint64_t payload_budget) {
  OperationSupport out;
  if (cells<=0) return out;
  if (!source || source>=target || target>20) { out.constraint=OperationConstraint::prefix; return out; }
  if (cells>INT32_MAX-256 || uint64_t(cells)>(SIZE_MAX-sizeof(int))/(target-source)) {
    out.status=RNS8_RANGE_ERROR; out.constraint=OperationConstraint::address_span; return out;
  }
  out.payload_bytes=uint64_t(cells)*(target-source);
  out.scratch_bytes=out.payload_bytes+sizeof(int); // suffix plus device proof status
  out.resources_known=true;
  out.status=out.payload_bytes>payload_budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  out.constraint=out.status==RNS8_SUCCESS ? OperationConstraint::none : OperationConstraint::scratch_budget;
  return out;
}
// Compact persistent output; native staging and the selected input scan overlap.
inline OperationSupport query_narrow_rns_support(int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, uint32_t prefix, bool byte_wmma, uint64_t budget,bool fused=false) {
  OperationSupport out;
  if (m<=0 || n<=0 || k<=0 || lda<k || ldb<n) return out;
  if (!prefix || prefix>20) { out.constraint=OperationConstraint::prefix; return out; }
  if (!qualified_i32_gemm_address(m,n,k,lda,ldb,n) ||
      !qualified_i32_matrix_address(m,n,n,prefix)) {
    out.status=RNS8_UNSUPPORTED_BACKEND; out.constraint=OperationConstraint::address_span; return out;
  }
  if(byte_wmma && (!qualified_i32_matrix_address(m,k,lda,1,8) || !qualified_i32_matrix_address(k,n,ldb,1,8))) {
    out.status=RNS8_UNSUPPORTED_BACKEND;out.constraint=OperationConstraint::address_span;return out;
  }
  out.payload_bytes=fused?0:uint64_t(m)*uint64_t(n)*sizeof(uint64_t);
  const uint64_t scan_bytes=(byte_wmma ? 4u : 2u)*sizeof(uint64_t);
  if (out.payload_bytes>SIZE_MAX-scan_bytes) {
    out.status=RNS8_RANGE_ERROR; out.constraint=OperationConstraint::address_span; return out;
  }
  out.scratch_bytes=out.payload_bytes+scan_bytes;
  out.resources_known=true;
  out.status=out.scratch_bytes>budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  out.constraint=out.status==RNS8_SUCCESS ? OperationConstraint::none : OperationConstraint::scratch_budget;
  return out;
}
} // namespace rns8::detail
