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

bool result_cache_contract_flags_valid(uint32_t flags) {
  constexpr uint32_t kKnownFlags =
      RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
      RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
      RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
      RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
  return (flags & ~kKnownFlags) == 0;
}

uint32_t result_cache_default_flags(uint32_t flags) {
  if (flags != 0) {
    return flags;
  }
  return RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
         RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
         RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
         RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
}

uint64_t result_cache_key_hash_for_plan(const rns8_context& ctx, const rns8_plan& plan) {
  uint64_t hash = plan_workspace_fingerprint(plan);
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(ctx.device_id));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.backend));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.desc.semantics));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.m));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.n));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.k));
  hash = workspace_fingerprint_mix(hash, plan.prefix);
  hash = workspace_fingerprint_mix(hash, plan.desc.finite_modulus);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_target_id);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_selected_kernel);
  return hash;
}

uint32_t result_cache_flags(const rns8_result_cache& cache) {
  uint32_t flags = 0;
  if (cache.initialized) flags |= RNS8_RESULT_CACHE_FLAG_INITIALIZED;
  if (cache.last_cache_hit) flags |= RNS8_RESULT_CACHE_FLAG_LAST_CALL_HIT;
  if (cache.last_cache_miss) flags |= RNS8_RESULT_CACHE_FLAG_LAST_CALL_MISS;
  if (cache.last_recomputed_region_count != 0) flags |= RNS8_RESULT_CACHE_FLAG_LAST_CALL_PARTIAL_RECOMPUTE;
  if (cache.last_full_fallback) flags |= RNS8_RESULT_CACHE_FLAG_LAST_CALL_FULL_FALLBACK;
  if (cache.last_stale_rejection) flags |= RNS8_RESULT_CACHE_FLAG_LAST_CALL_STALE_REJECTED;
  return flags;
}

void reset_result_cache_last_call(rns8_result_cache& cache) {
  cache.last_cache_hit = false;
  cache.last_cache_miss = false;
  cache.last_full_fallback = false;
  cache.last_stale_rejection = false;
  cache.last_dirty_region_count = 0;
  cache.last_recomputed_region_count = 0;
  cache.copied_from_cache_bytes = 0;
  cache.recomputed_cell_count = 0;
  cache.stale_reason.clear();
  cache.fallback_reason.clear();
  cache.detail.clear();
}

rns8_status free_result_cache_snapshot(rns8_result_cache& cache) {
  rns8_status status = RNS8_SUCCESS;
  if (cache.hip_snapshot_residues) {
    status = rns8::detail::hip_direct_free(cache.hip_device_id, cache.hip_snapshot_residues);
  }
  cache.hip_snapshot_residues = nullptr;
  cache.hip_snapshot_residue_bytes = 0;
  cache.cache_allocation_bytes = 0;
  return status;
}

rns8_status ensure_result_cache_snapshot(rns8_context& ctx, rns8_result_cache& cache, std::size_t bytes) {
  if (bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (cache.hip_snapshot_residues && cache.hip_snapshot_residue_bytes == bytes) {
    return RNS8_SUCCESS;
  }
  const rns8_status free_status = free_result_cache_snapshot(cache);
  if (free_status != RNS8_SUCCESS) {
    return free_status;
  }
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, bytes, &cache.hip_snapshot_residues);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  cache.hip_snapshot_residue_bytes = bytes;
  cache.cache_allocation_bytes = static_cast<uint64_t>(bytes);
  return RNS8_SUCCESS;
}

rns8_status snapshot_result_cache_from_output(rns8_context& ctx, rns8_result_cache& cache, const rns8_matrix& C) {
  const rns8_status status = ensure_result_cache_snapshot(ctx, cache, C.hip_residue_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return rns8::detail::hip_direct_copy_device_to_device(
      ctx.device_id, cache.hip_snapshot_residues, C.hip_residues, C.hip_residue_bytes);
}

rns8_status restore_result_cache_to_output(rns8_context& ctx, rns8_result_cache& cache, rns8_matrix& C) {
  if (!cache.initialized || !cache.hip_snapshot_residues || cache.hip_snapshot_residue_bytes != C.hip_residue_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = rns8::detail::hip_direct_copy_device_to_device(
      ctx.device_id, C.hip_residues, cache.hip_snapshot_residues, C.hip_residue_bytes);
  if (status == RNS8_SUCCESS) {
    cache.copied_from_cache_bytes += static_cast<uint64_t>(C.hip_residue_bytes);
  }
  return status;
}

void bind_result_cache_after_compute(
    rns8_result_cache& cache,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C,
    const rns8_workspace& workspace) {
  cache.initialized = true;
  cache.workspace_fingerprint = workspace.schedule_fingerprint;
  cache.a_matrix_instance_id = A.matrix_instance_id;
  cache.b_matrix_instance_id = B.matrix_instance_id;
  cache.c_matrix_instance_id = C.matrix_instance_id;
  cache.a_source_version = A.source_version;
  cache.b_source_version = B.source_version;
  cache.c_source_version = C.source_version;
}

bool dirty_region_covers_full_output(const rns8_dirty_region& region, const rns8_plan& plan) {
  return region.row_offset == 0 && region.col_offset == 0 && region.row_extent == plan.desc.m &&
         region.col_extent == plan.desc.n;
}

rns8_status validate_dirty_regions(
    const rns8_plan& plan,
    const rns8_dirty_region* dirty_regions,
    uint32_t dirty_region_count,
    uint32_t max_dirty_regions,
    bool& out_full_output) {
  out_full_output = false;
  if (dirty_region_count == 0) {
    return dirty_regions ? RNS8_INVALID_ARGUMENT : RNS8_SUCCESS;
  }
  if (!dirty_regions || dirty_region_count > max_dirty_regions) {
    return RNS8_INVALID_ARGUMENT;
  }
  for (uint32_t index = 0; index < dirty_region_count; ++index) {
    const rns8_dirty_region& region = dirty_regions[index];
    if (!rns8::detail::valid_abi(region.struct_size, region.abi_version, sizeof(region)) || region.flags != 0 ||
        region.row_offset < 0 || region.col_offset < 0 || region.row_extent <= 0 || region.col_extent <= 0 ||
        region.row_offset > plan.desc.m || region.col_offset > plan.desc.n ||
        region.row_extent > plan.desc.m - region.row_offset ||
        region.col_extent > plan.desc.n - region.col_offset) {
      return RNS8_INVALID_ARGUMENT;
    }
    out_full_output = out_full_output || dirty_region_covers_full_output(region, plan);
  }
  return RNS8_SUCCESS;
}

rns8_status validate_result_cache_for_call(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace,
    rns8_result_cache& cache,
    uint16_t modulus,
    bool finite) {
  if (plan.backend != RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled() || !context_accepts_backend(ctx, plan.backend)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (cache.backend != plan.backend || cache.semantics != plan.desc.semantics || cache.bound_kind != plan.desc.bound_kind ||
      cache.m != plan.desc.m || cache.n != plan.desc.n || cache.k != plan.desc.k || cache.prefix != plan.prefix ||
      cache.finite_modulus != (finite ? modulus : 0u) || cache.hip_device_id != ctx.device_id ||
      cache.plan_fingerprint != plan_workspace_fingerprint(plan) ||
      cache.result_cache_key_hash != result_cache_key_hash_for_plan(ctx, plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (cache.workspace_fingerprint != 0 && cache.workspace_fingerprint != workspace.schedule_fingerprint) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status reject_stale_result_cache(rns8_result_cache& cache, const char* reason) {
  cache.last_stale_rejection = true;
  cache.stale_reason = reason;
  return RNS8_INVALID_ARGUMENT;
}

rns8_status execute_incremental_full_recompute(
    rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    rns8_matrix& C,
    rns8_workspace& workspace,
    rns8_result_cache& cache,
    bool finite,
    const char* fallback_reason) {
  cache.last_cache_miss = true;
  cache.last_full_fallback = fallback_reason != nullptr;
  if (fallback_reason) {
    cache.fallback_reason = fallback_reason;
  }
  const rns8_status status = finite
      ? rns8_gemm_finite_u8(&ctx, &plan, modulus, &A, &B, &C, &workspace)
      : rns8_gemm_rns(&ctx, &plan, &A, &B, &C, &workspace);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  rns8_status snapshot_status = snapshot_result_cache_from_output(ctx, cache, C);
  if (snapshot_status != RNS8_SUCCESS) {
    return snapshot_status;
  }
  bind_result_cache_after_compute(cache, A, B, C, workspace);
  return RNS8_SUCCESS;
}

rns8_status execute_incremental_region_recompute(
    rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    rns8_matrix& C,
    rns8_workspace& workspace,
    rns8_result_cache& cache,
    const rns8_dirty_region* dirty_regions,
    uint32_t dirty_region_count,
    bool finite) {
  rns8_status status = restore_result_cache_to_output(ctx, cache, C);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  for (uint32_t index = 0; index < dirty_region_count; ++index) {
    const rns8_dirty_region& region = dirty_regions[index];
    status = finite ? rns8::detail::hip_direct_gemm_finite_u8_region_device(
                          ctx.device_id,
                          A.hip_residues,
                          B.hip_residues,
                          C.hip_residues,
                          plan.desc.m,
                          plan.desc.n,
                          plan.desc.k,
                          A.desc.cols,
                          B.desc.cols,
                          C.desc.cols,
                          region.row_offset,
                          region.col_offset,
                          region.row_extent,
                          region.col_extent,
                          modulus)
                    : rns8::detail::hip_direct_gemm_rns_region_device(
                          ctx.device_id,
                          A.hip_residues,
                          B.hip_residues,
                          C.hip_residues,
                          plan.desc.m,
                          plan.desc.n,
                          plan.desc.k,
                          A.desc.cols,
                          B.desc.cols,
                          C.desc.cols,
                          region.row_offset,
                          region.col_offset,
                          region.row_extent,
                          region.col_extent,
                          plan.prefix);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    cache.recomputed_cell_count += static_cast<uint64_t>(region.row_extent) * static_cast<uint64_t>(region.col_extent);
  }
  mark_output_device_residues_current(C);
  C.source_version = gemm_output_source_version(A, B);
  if (finite) {
    C.finite_modulus = modulus;
  }
  status = snapshot_result_cache_from_output(ctx, cache, C);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  cache.last_recomputed_region_count = dirty_region_count;
  bind_result_cache_after_compute(cache, A, B, C, workspace);
  return RNS8_SUCCESS;
}

rns8_status execute_incremental_gemm(
    rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    rns8_matrix& C,
    rns8_workspace& workspace,
    rns8_result_cache& cache,
    const rns8_dirty_region* dirty_regions,
    uint32_t dirty_region_count,
    bool finite) {
  reset_result_cache_last_call(cache);
  cache.last_dirty_region_count = dirty_region_count;
  const rns8_status cache_status = validate_result_cache_for_call(ctx, plan, workspace, cache, modulus, finite);
  if (cache_status != RNS8_SUCCESS) {
    return cache_status;
  }
  if (A.source_version == 0 || B.source_version == 0) {
    return reject_stale_result_cache(cache, "source_version_zero");
  }
  bool full_output_dirty = false;
  rns8_status status =
      validate_dirty_regions(plan, dirty_regions, dirty_region_count, cache.max_dirty_regions, full_output_dirty);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (!cache.initialized) {
    return execute_incremental_full_recompute(
        ctx, plan, modulus, A, B, C, workspace, cache, finite, dirty_region_count == 0 ? nullptr : "initial_cache_fill");
  }
  if (cache.a_matrix_instance_id != A.matrix_instance_id || cache.b_matrix_instance_id != B.matrix_instance_id) {
    return reject_stale_result_cache(cache, "matrix_identity_mismatch");
  }
  if ((cache.a_source_version != A.source_version || cache.b_source_version != B.source_version) &&
      dirty_region_count == 0) {
    return reject_stale_result_cache(cache, "source_version_mismatch_without_dirty_regions");
  }
  if (dirty_region_count == 0) {
    status = restore_result_cache_to_output(ctx, cache, C);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    mark_output_device_residues_current(C);
    C.source_version = cache.c_source_version;
    if (finite) {
      C.finite_modulus = modulus;
    }
    cache.last_cache_hit = true;
    cache.detail = "restored_cached_output";
    return RNS8_SUCCESS;
  }
  if (full_output_dirty) {
    return execute_incremental_full_recompute(
        ctx, plan, modulus, A, B, C, workspace, cache, finite, "dirty_regions_cover_full_output");
  }
  return execute_incremental_region_recompute(
      ctx, plan, modulus, A, B, C, workspace, cache, dirty_regions, dirty_region_count, finite);
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

rns8_status rns8_create_result_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_result_cache_desc* desc,
    rns8_result_cache** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc)) ||
        !result_cache_contract_flags_valid(desc->flags) || desc->reserved0 != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend != RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled() || !context_accepts_backend(*ctx, plan->backend)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!uses_rns_storage(plan->desc.semantics) && !uses_finite_storage(plan->desc.semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    const uint32_t flags = result_cache_default_flags(desc->flags);
    constexpr uint32_t kRequiredFlags =
        RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
        RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
        RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
        RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
    if ((flags & kRequiredFlags) != kRequiredFlags) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* cache = new (std::nothrow) rns8_result_cache();
    if (!cache) {
      return RNS8_INTERNAL_ERROR;
    }
    cache->backend = plan->backend;
    cache->semantics = plan->desc.semantics;
    cache->bound_kind = plan->desc.bound_kind;
    cache->m = plan->desc.m;
    cache->n = plan->desc.n;
    cache->k = plan->desc.k;
    cache->prefix = plan->prefix;
    cache->finite_modulus = uses_finite_storage(plan->desc.semantics) ? plan->desc.finite_modulus : 0;
    cache->max_dirty_regions = desc->max_dirty_regions == 0 ? 64u : desc->max_dirty_regions;
    cache->plan_fingerprint = plan_workspace_fingerprint(*plan);
    cache->result_cache_key_hash = result_cache_key_hash_for_plan(*ctx, *plan);
    cache->target_id = plan->backend_target_id;
    cache->selected_kernel = plan->backend_selected_kernel;
    cache->hip_device_id = ctx->device_id;
    cache->detail = "explicit_result_cache_contract_v1";
    *out = cache;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_result_cache(rns8_result_cache* cache) {
  if (cache) {
    const rns8_status status = free_result_cache_snapshot(*cache);
    delete cache;
    return status;
  }
  delete cache;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_result_cache_info(const rns8_result_cache* cache, rns8_result_cache_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!cache || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = cache->backend;
    out->semantics = cache->semantics;
    out->bound_kind = cache->bound_kind;
    out->m = cache->m;
    out->n = cache->n;
    out->k = cache->k;
    out->max_prefix = cache->prefix;
    out->finite_modulus = cache->finite_modulus;
    out->flags = result_cache_flags(*cache);
    out->hip_device_id = cache->hip_device_id;
    out->initialized = cache->initialized ? 1u : 0u;
    out->max_dirty_regions = cache->max_dirty_regions;
    out->last_dirty_region_count = cache->last_dirty_region_count;
    out->last_recomputed_region_count = cache->last_recomputed_region_count;
    out->last_full_fallback = cache->last_full_fallback ? 1u : 0u;
    out->last_cache_hit = cache->last_cache_hit ? 1u : 0u;
    out->last_cache_miss = cache->last_cache_miss ? 1u : 0u;
    out->last_stale_rejection = cache->last_stale_rejection ? 1u : 0u;
    out->plan_fingerprint = cache->plan_fingerprint;
    out->workspace_fingerprint = cache->workspace_fingerprint;
    out->result_cache_key_hash = cache->result_cache_key_hash;
    out->a_matrix_instance_id = cache->a_matrix_instance_id;
    out->b_matrix_instance_id = cache->b_matrix_instance_id;
    out->c_matrix_instance_id = cache->c_matrix_instance_id;
    out->a_source_version = cache->a_source_version;
    out->b_source_version = cache->b_source_version;
    out->c_source_version = cache->c_source_version;
    out->snapshot_device_bytes = static_cast<uint64_t>(cache->hip_snapshot_residue_bytes);
    out->copied_from_cache_bytes = cache->copied_from_cache_bytes;
    out->recomputed_cell_count = cache->recomputed_cell_count;
    out->cache_allocation_bytes = cache->cache_allocation_bytes;
    set_text(out->target_id, sizeof(out->target_id), cache->target_id);
    set_text(out->selected_backend, sizeof(out->selected_backend), backend_name(cache->backend));
    set_text(out->selected_kernel, sizeof(out->selected_kernel), cache->selected_kernel);
    set_text(out->dirty_region_contract, sizeof(out->dirty_region_contract), "output_rectangles_full_k_recompute");
    set_text(out->source_identity_policy, sizeof(out->source_identity_policy), "matrix_instance_id_exact_match");
    set_text(out->source_version_policy, sizeof(out->source_version_policy), "nonzero_source_versions_required");
    set_text(out->result_lifetime_policy, sizeof(out->result_lifetime_policy), "explicit_cache_handle_lifetime");
    set_text(out->stale_reason, sizeof(out->stale_reason), cache->stale_reason);
    set_text(out->fallback_reason, sizeof(out->fallback_reason), cache->fallback_reason);
    set_text(out->detail, sizeof(out->detail), cache->detail);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_invalidate_result_cache(rns8_result_cache* cache) {
  return guard_api([&]() -> rns8_status {
    if (!cache) {
      return RNS8_INVALID_ARGUMENT;
    }
    reset_result_cache_last_call(*cache);
    cache->initialized = false;
    cache->workspace_fingerprint = 0;
    cache->a_matrix_instance_id = 0;
    cache->b_matrix_instance_id = 0;
    cache->c_matrix_instance_id = 0;
    cache->a_source_version = 0;
    cache->b_source_version = 0;
    cache->c_source_version = 0;
    cache->detail = "invalidated";
    return RNS8_SUCCESS;
  });
}

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

rns8_status rns8_gemm_rns_incremental(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace,
    rns8_result_cache* cache,
    const rns8_dirty_region* dirty_regions,
    uint32_t dirty_region_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace || !cache) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    rns8_status conversion_status = ensure_logically_const_rns_input_current(*ctx, *plan, *A);
    if (conversion_status != RNS8_SUCCESS) {
      return conversion_status;
    }
    conversion_status = ensure_logically_const_rns_input_current(*ctx, *plan, *B);
    if (conversion_status != RNS8_SUCCESS) {
      return conversion_status;
    }
    const rns8_status operand_status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    return execute_incremental_gemm(
        *ctx,
        *plan,
        0,
        *A,
        *B,
        *C,
        *workspace,
        *cache,
        dirty_regions,
        dirty_region_count,
        false);
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

rns8_status rns8_gemm_finite_u8_incremental(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace,
    rns8_result_cache* cache,
    const rns8_dirty_region* dirty_regions,
    uint32_t dirty_region_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace || !cache) {
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
    return execute_incremental_gemm(
        *ctx,
        *plan,
        modulus,
        *A,
        *B,
        *C,
        *workspace,
        *cache,
        dirty_regions,
        dirty_region_count,
        true);
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
