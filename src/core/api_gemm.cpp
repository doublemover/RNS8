#include "core/api_internal.hpp"

namespace rns8::detail::api {

rns8_status validate_rns_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.k, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, plan.desc.semantics, plan.desc.bound_kind, plan.desc.k, plan.desc.n, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.n, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, plan.prefix) ||
      !rns_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n, plan.prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, plan.prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_residue_state_current_for_backend(A, plan.backend) ||
      !rns_residue_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_finite_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (!uses_finite_storage(plan.desc.semantics) ||
      !rns8::detail::valid_finite_modulus_for_semantics(plan.desc.semantics, modulus) ||
      plan.desc.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.k, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!finite_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k) ||
      !finite_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !finite_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_residue_state_current_for_backend(A, plan.backend) ||
      !rns_residue_state_current_for_backend(B, plan.backend) || A.finite_modulus != modulus ||
      B.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_wrap_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 || plan.prefix != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.k, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.hip_residues || B.hip_residues || C.hip_residues) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k) ||
      !wrap_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_byte_limb_state_current_for_backend(A, plan.backend) ||
      !wrap_byte_limb_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_rns(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = true;
        C->device_residues_current = false;
        C->host_byte_limbs_current = false;
        C->device_byte_limbs_current = false;
        if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
          C->source_version = gemm_output_source_version(*A, *B);
        }
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::hip_direct_gemm_rns_tiled_device_schedule(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            workspace->hip_tile_schedule,
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::hip_direct_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (!ctx->hipblaslt_handle || plan->schedule_adaptive_prefix_active || plan->schedule_adaptive_skip_active ||
          !plan->tile_schedule.empty()) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status status = rns8::detail::hipblaslt_gemm_rns_device(
          ctx->device_id,
          ctx->hipblaslt_handle,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->hipblaslt_int32_scratch,
          workspace->hipblaslt_int32_scratch_bytes,
          workspace->hipblaslt_workspace,
          workspace->hipblaslt_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          plan->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::ck_gemm_rns_tiled_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::ck_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_WMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::wmma_gemm_rns_tiled_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::wmma_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_finite_gemm_operands(*ctx, *plan, modulus, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_finite_u8(*plan, modulus, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = true;
        C->device_residues_current = false;
        C->host_byte_limbs_current = false;
        C->device_byte_limbs_current = false;
        C->finite_modulus = modulus;
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_gemm_finite_u8_resident_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (!ctx->hipblaslt_handle) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status status = rns8::detail::hipblaslt_gemm_finite_u8_device(
          ctx->device_id,
          ctx->hipblaslt_handle,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->hipblaslt_int32_scratch,
          workspace->hipblaslt_int32_scratch_bytes,
          workspace->hipblaslt_workspace,
          workspace->hipblaslt_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      const rns8_status status = rns8::detail::ck_gemm_finite_u8_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->accelerator_workspace,
          workspace->accelerator_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_WMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      const rns8_status status = rns8::detail::wmma_gemm_finite_u8_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->accelerator_workspace,
          workspace->accelerator_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_wrap_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      const rns8_status status = rns8::detail::cpu_gemm_wrap_u64(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = false;
        C->device_residues_current = false;
        C->host_byte_limbs_current = true;
        C->device_byte_limbs_current = false;
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
          ctx->device_id,
          A->hip_byte_limbs,
          B->hip_byte_limbs,
          C->hip_byte_limbs,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = false;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = true;
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}
