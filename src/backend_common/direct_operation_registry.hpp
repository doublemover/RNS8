#pragma once
#include "operation_support.hpp"
#include <array>
namespace rns8::detail {
// Internal static inventory, deliberately separate from runtime device probes,
// bound operands, arithmetic evidence and reviewed AUTO promotion.
enum class DirectOperation : uint32_t {
  residue_gemm, exact_axpby, exact_lift, narrow_scalar, narrow_byte_wmma,
  canonical_direct_crt, canonical_balanced_crt, finite_verify
};
enum class DirectExecution : uint32_t { synchronous, borrowed_stream, owned_capture, distributed };
struct DirectOperationRecord {
  DirectOperation operation;
  const char* implementation;
  const char* input_domain;
  const char* output_domain;
  uint32_t index_bits;
  bool native_value_scan;
  bool matrix_instruction_probe;
};
inline constexpr std::array<DirectOperationRecord,8> direct_operation_records{{
  {DirectOperation::residue_gemm,"hip_direct_gemm_rns_device","default_rns_centered","default_rns_centered",32,false,false},
  {DirectOperation::exact_axpby,"hip_direct_gemm_rns_axpby_device","default_rns_centered","default_rns_centered",32,false,false},
  {DirectOperation::exact_lift,"hip_direct_lift_exact_suffix","default_rns_unique_integer","default_rns_suffix",32,false,false},
  {DirectOperation::narrow_scalar,"hip_direct_gemm_narrow_rns_device.scalar","native_i64","default_rns_centered",32,true,false},
  {DirectOperation::narrow_byte_wmma,"hip_direct_gemm_narrow_rns_device.byte","native_i64_or_u64_byte_domain","default_rns_centered",32,true,true},
  {DirectOperation::canonical_direct_crt,"hip_direct_export_canonical_direct_crt_device","default_rns_centered","canonical_u192",32,false,false},
  {DirectOperation::canonical_balanced_crt,"hip_direct_export_canonical_balanced_crt_device","default_rns_centered","canonical_u192",32,false,false},
  {DirectOperation::finite_verify,"hip_finite_verify_trial","finite_centered_ring","verification_status",64,false,false}
}};
struct DirectOperationProblem {
  uint32_t version=1;
  DirectOperation operation=DirectOperation::residue_gemm;
  DirectExecution execution=DirectExecution::synchronous;
  int64_t m=0,n=0,k=0,lda=0,ldb=0,ldc=0,ldd=0;
  uint32_t prefix=0,source_prefix=0,modulus=0;
  uint64_t budget=UINT64_MAX;
};
struct DirectOperationQuery {
  uint32_t version=1;
  const DirectOperationRecord* implementation=nullptr;
  OperationSupport static_support;
  bool compiled=false;
  // No inference from compilation or math compatibility to a device/evidence.
  bool runtime_observed=false, exact_qualified=false, performance_promoted=false;
  bool requires_operand_binding=true;
};
inline DirectOperationQuery query_direct_operation(const DirectOperationProblem& p) {
  DirectOperationQuery out;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  out.compiled=true;
#endif
  if(p.version!=1) { out.static_support.constraint=OperationConstraint::operation; return out; }
  for(const auto& record:direct_operation_records)
    if(record.operation==p.operation) { out.implementation=&record; break; }
  if(!out.implementation) { out.static_support.constraint=OperationConstraint::operation; return out; }
  if(p.execution!=DirectExecution::synchronous) {
    out.static_support.status=RNS8_UNSUPPORTED_BACKEND;
    out.static_support.constraint=OperationConstraint::execution_mode; return out;
  }
  switch(p.operation) {
    case DirectOperation::residue_gemm:
      out.static_support=query_residue_gemm_support(p.m,p.n,p.k,p.lda,p.ldb,p.ldc,p.prefix); break;
    case DirectOperation::exact_axpby:
      out.static_support=query_axpby_support(p.m,p.n,p.k,p.lda,p.ldb,p.ldd,p.ldc,p.prefix); break;
    case DirectOperation::exact_lift:
      // m is the number of cells. Existing lift budget is suffix payload, while
      // its resource report also discloses the separately allocated status word.
      out.static_support=query_exact_lift_support(p.m,p.source_prefix,p.prefix,p.budget); break;
    case DirectOperation::narrow_scalar: case DirectOperation::narrow_byte_wmma:
      out.static_support=query_narrow_rns_support(p.m,p.n,p.k,p.lda,p.ldb,p.prefix,
          p.operation==DirectOperation::narrow_byte_wmma,p.budget); break;
    case DirectOperation::canonical_direct_crt: case DirectOperation::canonical_balanced_crt:
      out.static_support=query_canonical_crt_support(p.m,p.prefix); break;
    case DirectOperation::finite_verify:
      out.static_support=query_finite_verify_support(p.m,p.n,p.k,p.modulus,true,p.budget); break;
  }
  return out;
}
}
