#include "core/api_internal.hpp"

namespace rns8::detail::api {

rns8_status validate_rns_gemm_prepacked_b_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_prepack_cache& B,
    const rns8_matrix& C) {
  if (!context_accepts_backend(ctx, plan.backend) || !rocwmma_b_prepack_cache_supported(plan) ||
      !prepack_cache_matches_plan(B, plan) || B.hip_device_id != ctx.device_id ||
      B.target_id != prepack_target_id_for_context(ctx) ||
      !matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) && (A.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  const rns8_bound_kind storage_bound_kind = storage_bound_kind_for_plan(plan);
  if (!matrix_descriptor_matches(
          A,
          plan.desc.semantics,
          storage_bound_kind,
          plan.desc.m,
          plan.desc.k,
          storage_prefix,
          plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C,
          plan.desc.semantics,
          storage_bound_kind,
          plan.desc.m,
          plan.desc.n,
          storage_prefix,
          plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, storage_prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, storage_prefix) ||
      !rns_residue_state_current_for_backend(A, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}
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
  if (hip_device_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  const rns8_bound_kind storage_bound_kind = storage_bound_kind_for_plan(plan);
  if (!matrix_descriptor_matches(
          A, plan.desc.semantics, storage_bound_kind, plan.desc.m, plan.desc.k, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, plan.desc.semantics, storage_bound_kind, plan.desc.k, plan.desc.n, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, storage_bound_kind, plan.desc.m, plan.desc.n, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (native_vector_backend(plan.backend)) {
    if (plan.desc.semantics != RNS8_BOUNDED_I64 && plan.desc.semantics != RNS8_BOUNDED_U64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!bounded_native_storage_matches(A, plan.desc.semantics, plan.desc.m, plan.desc.k) ||
        !bounded_native_storage_matches(B, plan.desc.semantics, plan.desc.k, plan.desc.n) ||
        !bounded_native_storage_matches(C, plan.desc.semantics, plan.desc.m, plan.desc.n) ||
        !bounded_native_state_current(A) || !bounded_native_state_current(B)) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, storage_prefix) ||
      !rns_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n, storage_prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, storage_prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && plan_all_zero_output_tiles(plan)) {
    return RNS8_SUCCESS;
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

rns8_status ensure_logically_const_rns_input_current(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& matrix) {
  // AUTO can populate internal residue cache state without changing the logical matrix value.
  return ensure_bounded_native_residues_current_for_rns_plan(
      ctx, plan, *const_cast<rns8_matrix*>(&matrix));
}

rns8_status build_public_grouped_gemm_descriptor(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count,
    std::vector<rns8::detail::hip_direct_grouped_gemm_task>& internal_tasks,
    rns8::detail::hip_direct_grouped_gemm_bucket_plan& bucket_plan,
    const rns8::detail::hip_direct_grouped_gemm_descriptor** out_descriptor) {
  if (!tasks || !out_descriptor || task_count <= 1) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out_descriptor = nullptr;
  internal_tasks.clear();
  internal_tasks.reserve(task_count);
  for (uint32_t index = 0; index < task_count; ++index) {
    const rns8_grouped_gemm_task& task = tasks[index];
    if (!rns8::detail::valid_abi(task.struct_size, task.abi_version, sizeof(task)) ||
        !task.a || !task.b || !task.c || !task.workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(ctx, plan, *task.workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    internal_tasks.push_back(
        {const_cast<rns8_matrix*>(task.a), const_cast<rns8_matrix*>(task.b), task.c, task.workspace});
  }
  if (plan.backend != RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled()) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (!context_accepts_backend(ctx, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = rns8::detail::hip_direct_build_same_shape_grouped_bucket_plan(
      &plan,
      internal_tasks.data(),
      task_count,
      plan.desc.semantics,
      plan.desc.m,
      plan.desc.n,
      plan.desc.k,
      plan.prefix,
      &bucket_plan);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const auto* descriptor = rns8::detail::hip_direct_single_bucket_descriptor(bucket_plan);
  if (!descriptor) {
    return RNS8_INVALID_ARGUMENT;
  }
  int descriptor_device_id = -1;
  status = rns8::detail::hip_direct_validate_grouped_gemm_descriptor_setup(*descriptor, &descriptor_device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (descriptor_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out_descriptor = descriptor;
  return RNS8_SUCCESS;
}

rns8_status execute_public_grouped_gemm(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count,
    bool finite_u8,
    uint16_t modulus) {
  if ((finite_u8 && (!uses_finite_storage(plan.desc.semantics) || plan.desc.finite_modulus != modulus ||
                     !rns8::detail::valid_finite_modulus_for_semantics(plan.desc.semantics, modulus))) ||
      (!finite_u8 && !uses_rns_storage(plan.desc.semantics))) {
    return RNS8_INVALID_ARGUMENT;
  }

  std::vector<rns8::detail::hip_direct_grouped_gemm_task> internal_tasks;
  rns8::detail::hip_direct_grouped_gemm_bucket_plan bucket_plan;
  const rns8::detail::hip_direct_grouped_gemm_descriptor* descriptor = nullptr;
  rns8_status status =
      build_public_grouped_gemm_descriptor(ctx, plan, tasks, task_count, internal_tasks, bucket_plan, &descriptor);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  rns8::detail::hip_direct_grouped_device_resources resources;
  status = rns8::detail::hip_direct_allocate_grouped_task_device_resources(*descriptor, 0, 0, 0, 0, &resources);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_prepare_grouped_task_residue_pointers(*descriptor, resources);
  if (status == RNS8_SUCCESS) {
    status = finite_u8 ? rns8::detail::hip_direct_gemm_grouped_finite_u8_task_outputs(
                             *descriptor,
                             resources,
                             modulus)
                       : rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(*descriptor, resources);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(*descriptor);
  }
  const rns8_status reset_status = resources.reset();
  return status != RNS8_SUCCESS ? status : reset_status;
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
    const bool trusted_all_zero_direct_hip =
        plan->backend == RNS8_BACKEND_HIP_DIRECT && plan_all_zero_output_tiles(*plan);
    if (!trusted_all_zero_direct_hip) {
      rns8_status conversion_status =
          ensure_logically_const_rns_input_current(*ctx, *plan, *A);
      if (conversion_status != RNS8_SUCCESS) {
        return conversion_status;
      }
      conversion_status = ensure_logically_const_rns_input_current(*ctx, *plan, *B);
      if (conversion_status != RNS8_SUCCESS) {
        return conversion_status;
      }
    }
    const rns8_status operand_status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      auto* mutable_c = C;
      rns8_status status = rns8::detail::hip_direct_ensure_upload_buffer(
          ctx->device_id, sizeof(uint32_t), &mutable_c->hip_status_buffer, &mutable_c->hip_status_bytes);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      status = run_timed_api_status("vector_alu_status_memset", [&]() {
        return rns8::detail::hip_direct_zero(ctx->device_id, mutable_c->hip_status_buffer, sizeof(uint32_t));
      });
      if (status != RNS8_SUCCESS) {
        return status;
      }
      if (plan->desc.semantics == RNS8_BOUNDED_I64) {
        status = rns8::detail::vector_alu_gemm_i64_device(
            ctx->device_id,
            A->hip_native_i64,
            B->hip_native_i64,
            mutable_c->hip_native_i64,
            mutable_c->hip_status_buffer,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k);
      } else {
        status = rns8::detail::vector_alu_gemm_u64_device(
            ctx->device_id,
            A->hip_native_u64,
            B->hip_native_u64,
            mutable_c->hip_native_u64,
            mutable_c->hip_status_buffer,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      uint32_t host_status = 0;
      status = run_timed_api_status("vector_alu_status_d2h", [&]() {
        return rns8::detail::hip_direct_copy_device_to_host(
            ctx->device_id, &host_status, mutable_c->hip_status_buffer, sizeof(host_status));
      });
      if (status != RNS8_SUCCESS) {
        return status;
      }
      if (host_status != 0) {
        return RNS8_RANGE_ERROR;
      }
      mark_output_device_native_current(*mutable_c);
      mutable_c->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_rns(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        mark_output_host_residues_current(*C);
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
            workspace->hip_tile_schedule_active_entries,
            workspace->hip_zero_a_rows,
            workspace->hip_zero_b_cols,
            workspace->hip_tile_schedule_active_offsets,
            workspace->hip_tile_schedule_active_counts,
            workspace->hip_tile_schedule_active_prefix_count,
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
      mark_output_device_residues_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (!ctx->hipblaslt_handle || plan->schedule_adaptive_prefix_active || !plan->tile_schedule.empty()) {
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
          plan->prefix,
          workspace,
          A->source_version,
          B->source_version);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      mark_output_device_residues_current(*C);
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
      mark_output_device_residues_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_ROCWMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::rocwmma_gemm_rns_tiled_device(
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
        status = rns8::detail::rocwmma_gemm_rns_device(
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
      mark_output_device_residues_current(*C);
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

rns8_status rns8_gemm_rns_grouped(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan) {
      return RNS8_INVALID_ARGUMENT;
    }
    return execute_public_grouped_gemm(*ctx, *plan, tasks, task_count, false, 0);
  });
}

rns8_status rns8_gemm_rns_prepacked_b(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_prepack_cache* B,
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
    const rns8_status operand_status = validate_rns_gemm_prepacked_b_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend != RNS8_BACKEND_ROCWMMA) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    const rns8_status status = rns8::detail::rocwmma_gemm_rns_prepacked_b_device(
        ctx->device_id,
        A->hip_residues,
        B->device_data,
        C->hip_residues,
        workspace->accelerator_workspace,
        workspace->accelerator_workspace_bytes,
        plan->desc.m,
        plan->desc.n,
        plan->desc.k,
        A->desc.cols,
        C->desc.cols,
        plan->prefix);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    mark_output_device_residues_current(*C);
    if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
      C->source_version = gemm_output_source_version_values(A->source_version, B->source_version);
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_finite_u8_grouped(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan) {
      return RNS8_INVALID_ARGUMENT;
    }
    return execute_public_grouped_gemm(*ctx, *plan, tasks, task_count, true, modulus);
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
        mark_output_host_residues_current(*C);
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
      mark_output_device_residues_current(*C);
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
      mark_output_device_residues_current(*C);
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
      mark_output_device_residues_current(*C);
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_ROCWMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      const rns8_status status = rns8::detail::rocwmma_gemm_finite_u8_device(
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
      mark_output_device_residues_current(*C);
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
        mark_output_host_byte_limbs_current(*C);
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
      mark_output_device_byte_limbs_current(*C);
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}
