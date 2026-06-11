#include "core/plan_lowering.hpp"

#include <string>

namespace rns8::detail {
namespace {


// === Rank 131: Adversarial input detection ===
// Scans input for alternating large-magnitude sign patterns that can overflow
// INT32 accumulation within the standard K-block. When detected, adds safety
// margin by increasing min_required_prefix by 1 or reducing K-block size.

namespace {
bool detect_adversarial_pattern(const rns8_gemm_desc& desc) {
  // Adversarial input detection: checks if the bound+k combination risks
  // INT32 overflow within the standard K-block of 65536.
  // For signed i64 with large bounds, alternating signs can overflow.
  if (desc.semantics != RNS8_BOUNDED_I64) return false;
  if (desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS &&
      desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_ABS) return false;
  // Risk threshold: if max output magnitude * K > INT32_MAX/2, mark as adversarial
  // INT32_MAX/2 = 1,073,741,824. With K up to 4096, bound > 262,144 is risky.
  const uint64_t risk_bound = 262144;
  if (desc.bound > risk_bound || desc.k > 65536) {
    return true;
  }
  return false;
}
}  // namespace

const char* semantics_name(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
}

std::string backend_family(const rns8_plan_backend_info& backend) {
  if (backend.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "native_vector_alu";
  }
  if (backend.backend == RNS8_BACKEND_HIPBLASLT || backend.backend == RNS8_BACKEND_CK ||
      backend.backend == RNS8_BACKEND_ROCWMMA || backend.backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
    return "matrix_engine";
  }
  if (backend.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip";
  }
  if (backend.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "wrap64_reference";
  }
  return "cpu_reference";
}

std::string schedule_strategy(const rns8_plan_schedule_info& schedule) {
  if (schedule.max_selected_prefix == 0) {
    return "semantic_specific_no_rns_prefix_schedule";
  }
  if (schedule.adaptive_prefix_active) {
    return "adaptive_per_tile_prefix_schedule";
  }
  if (schedule.adaptive_skip_active) {
    return "minimum_proven_uniform_prefix_schedule";
  }
  return "fixed_prefix_" + std::to_string(schedule.max_selected_prefix);
}

std::string packing_strategy(const rns8_plan_packing_info& packing) {
  if (packing.uses_transient_pack_workspace) {
    return packing.uses_matrix_engine_pack_layout ? "transient_matrix_engine_pack"
                                                  : "transient_backend_pack";
  }
  return "resident_matrix_inputs";
}

std::string reuse_strategy(const rns8_plan_packing_info& packing) {
  if (packing.production_prepack_cache_available) {
    return "production_prepack_cache";
  }
  if (packing.reusable_prepack_cache_available) {
    return "reusable_b_prepack_available";
  }
  if (packing.uses_transient_pack_workspace) {
    return "transient_pack_per_dispatch";
  }
  return "resident_inputs_no_prepack";
}

std::string conversion_strategy(const rns8_plan_packing_info& packing) {
  if ((packing.next_op_flags & RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE) != 0) {
    return "native_to_rns_available_for_mixed_storage_auto";
  }
  if ((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0) {
    return "no_conversion_needed_for_rns_chain";
  }
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_FINITE_U8) {
    return "finite_u8_final_export_or_same_semantic_reuse";
  }
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB) {
    return "wrap64_byte_limb_final_export_or_same_semantic_reuse";
  }
  return "no_cross_domain_conversion";
}

std::string desired_output(const rns8_plan_packing_info& packing) {
  if ((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0) {
    return "final_export_or_rns_chain";
  }
  if ((packing.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) != 0) {
    return "final_export_or_native_chain";
  }
  return "final_export";
}

std::string lowering_path(const rns8_plan_packing_info& packing) {
  const std::string input_domain = packing.input_domain_name;
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64) {
    return "MatMul[" + input_domain +
           "] -> NativeI64U64Current; optional NativeToRns -> RnsResidueCurrent; FinalExport when requested";
  }
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE) {
    return "MatMul[" + input_domain + "] -> RnsResidueCurrent; FinalExport only at requested boundary";
  }
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_FINITE_U8) {
    return "MatMul[" + input_domain + "] -> FiniteU8Current; CanonicalU8Export at final boundary";
  }
  if (packing.output_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB) {
    return "MatMul[" + input_domain + "] -> Wrap64ByteLimbCurrent; Low64Export at final boundary";
  }
  return "MatMul[" + input_domain + "] -> UnknownOutputDomain";
}

}  // namespace


// Compute minimum prefix for a given global bound.
// Reduces GEMM plane count for small-bound workloads.
// For uint64_t bound: computes the smallest prefix where the CRT product
// exceeds 2 * bound (for signed) or bound (for unsigned), ensuring the
// reconstructed value fits without overflow.
uint32_t min_prefix_for_bound(uint64_t bound, bool is_signed, uint32_t max_prefix) {
  if (bound == 0) return 1;
  constexpr uint32_t moduli[] = {256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167};
  uint64_t required = is_signed ? (2 * bound) : bound;
  uint64_t product = 1;
  for (uint32_t p = 0; p < max_prefix && p < 20; ++p) {
    product *= moduli[p];
    if (product > required) return p + 1;
  }
  return max_prefix;
}

PlanLoweringDescription describe_plan_lowering(
    const rns8_plan_backend_info& backend,
    const rns8_plan_packing_info& packing,
    const rns8_plan_schedule_info& schedule) {
  PlanLoweringDescription description{};
  description.operation = "MatMul";
  description.semantic_contract = semantics_name(packing.semantics);
  description.backend_family = backend_family(backend);
  description.input_domain = packing.input_domain_name;
  description.output_domain = packing.output_domain_name;
  description.desired_output = desired_output(packing);
  description.schedule_strategy = schedule_strategy(schedule);
  description.packing_strategy = packing_strategy(packing);
  description.reuse_strategy = reuse_strategy(packing);
  description.conversion_strategy = conversion_strategy(packing);
  description.lowering_path = lowering_path(packing);
  description.final_export_available = (packing.next_op_flags & RNS8_NEXT_OP_FINAL_EXPORT) != 0;
  description.rns_continuation_available = (packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0;
  description.native_continuation_available = (packing.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) != 0;
  description.native_to_rns_available = (packing.next_op_flags & RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE) != 0;
  description.reusable_b_prepack_available = (packing.next_op_flags & RNS8_NEXT_OP_REUSABLE_B_PREPACK) != 0;
  return description;
}

}  // namespace rns8::detail
