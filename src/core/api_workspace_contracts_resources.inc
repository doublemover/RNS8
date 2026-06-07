void initialize_workspace_identity(rns8_workspace& workspace, const rns8_context& ctx, const rns8_plan& plan) {
  workspace.semantics = plan.desc.semantics;
  workspace.bound_kind = plan.desc.bound_kind;
  workspace.m = plan.desc.m;
  workspace.n = plan.desc.n;
  workspace.k = plan.desc.k;
  workspace.bound = plan.desc.bound;
  workspace.finite_modulus = plan.desc.finite_modulus;
  workspace.tile_m = plan.desc.tile_m;
  workspace.tile_n = plan.desc.tile_n;
  workspace.prefix = plan.prefix;
  workspace.backend = plan.backend;
  workspace.hip_device_id = hip_device_backend(plan.backend) ? ctx.device_id : -1;
}

void initialize_workspace_schedule_contract(rns8_workspace& workspace, const rns8_plan& plan) {
  workspace.schedule_tile_rows = plan.schedule_tile_rows;
  workspace.schedule_tile_cols = plan.schedule_tile_cols;
  workspace.schedule_tile_count = plan.schedule_tile_count;
  workspace.schedule_min_required_prefix = plan.schedule_min_required_prefix;
  workspace.schedule_max_required_prefix = plan.schedule_max_required_prefix;
  workspace.schedule_min_selected_prefix = plan.schedule_min_selected_prefix;
  workspace.schedule_max_selected_prefix = plan.schedule_max_selected_prefix;
  workspace.schedule_prefix_group_count = plan.schedule_prefix_group_count;
  workspace.schedule_range_bit_length = plan.schedule_range_bit_length;
  workspace.schedule_adaptive_prefix_active = plan.schedule_adaptive_prefix_active;
  workspace.schedule_adaptive_skip_active = plan.schedule_adaptive_skip_active;
  workspace.schedule_flags = plan.schedule_flags;
  workspace.zero_a_row_count = plan.zero_a_row_count;
  workspace.zero_b_col_count = plan.zero_b_col_count;
  workspace.zero_row_col_product_count = plan.zero_row_col_product_count;
  workspace.schedule_fingerprint = plan_workspace_fingerprint(plan);
}

void initialize_workspace_backend_metadata(rns8_workspace& workspace, const rns8_plan& plan) {
  workspace.backend_workspace_required_bytes = plan.backend_workspace_required_bytes;
  workspace.backend_selected_kernel = plan.backend_selected_kernel;
  workspace.backend_library = plan.backend_library;
  workspace.backend_library_version = plan.backend_library_version;
  workspace.backend_capability_status = plan.backend_capability_status;
  workspace.backend_epilogue_mode = plan.backend_epilogue_mode;
  workspace.backend_workspace_mode = plan.backend_workspace_mode;
  workspace.backend_isa_evidence = plan.backend_isa_evidence;
  workspace.backend_target_id = plan.backend_target_id;
  workspace.backend_autotune_key = plan.backend_autotune_key;
  workspace.backend_performance_validated = plan.backend_performance_validated;
}

bool workspace_identity_matches_plan(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace) {
  if (!context_accepts_backend(ctx, plan.backend) || workspace.backend != plan.backend) {
    return false;
  }
  if (workspace.semantics != plan.desc.semantics || workspace.bound_kind != plan.desc.bound_kind) {
    return false;
  }
  if (workspace.m != plan.desc.m || workspace.n != plan.desc.n || workspace.k != plan.desc.k ||
      workspace.prefix != plan.prefix) {
    return false;
  }
  if (hip_device_backend(plan.backend) && workspace.hip_device_id != ctx.device_id) {
    return false;
  }
  return true;
}

bool workspace_schedule_contract_matches_plan(const rns8_plan& plan, const rns8_workspace& workspace) {
  return workspace.bound == plan.desc.bound &&
         workspace.finite_modulus == plan.desc.finite_modulus &&
         workspace.tile_m == plan.desc.tile_m &&
         workspace.tile_n == plan.desc.tile_n &&
         workspace.schedule_tile_rows == plan.schedule_tile_rows &&
         workspace.schedule_tile_cols == plan.schedule_tile_cols &&
         workspace.schedule_tile_count == plan.schedule_tile_count &&
         workspace.schedule_min_required_prefix == plan.schedule_min_required_prefix &&
         workspace.schedule_max_required_prefix == plan.schedule_max_required_prefix &&
         workspace.schedule_min_selected_prefix == plan.schedule_min_selected_prefix &&
         workspace.schedule_max_selected_prefix == plan.schedule_max_selected_prefix &&
         workspace.schedule_prefix_group_count == plan.schedule_prefix_group_count &&
         workspace.schedule_range_bit_length == plan.schedule_range_bit_length &&
         workspace.schedule_adaptive_prefix_active == plan.schedule_adaptive_prefix_active &&
         workspace.schedule_adaptive_skip_active == plan.schedule_adaptive_skip_active &&
         workspace.schedule_flags == plan.schedule_flags &&
         workspace.zero_a_row_count == plan.zero_a_row_count &&
         workspace.zero_b_col_count == plan.zero_b_col_count &&
         workspace.zero_row_col_product_count == plan.zero_row_col_product_count &&
         workspace.schedule_fingerprint == plan_workspace_fingerprint(plan);
}

bool workspace_backend_metadata_matches_plan(const rns8_plan& plan, const rns8_workspace& workspace) {
  return workspace.backend_workspace_required_bytes == plan.backend_workspace_required_bytes &&
         workspace.backend_selected_kernel == plan.backend_selected_kernel &&
         workspace.backend_library == plan.backend_library &&
         workspace.backend_library_version == plan.backend_library_version &&
         workspace.backend_capability_status == plan.backend_capability_status &&
         workspace.backend_epilogue_mode == plan.backend_epilogue_mode &&
         workspace.backend_workspace_mode == plan.backend_workspace_mode &&
         workspace.backend_isa_evidence == plan.backend_isa_evidence &&
         workspace.backend_target_id == plan.backend_target_id &&
         workspace.backend_autotune_key == plan.backend_autotune_key &&
         workspace.backend_performance_validated == plan.backend_performance_validated;
}

template <typename T>
rns8_status release_workspace_device_buffer(int device_id, T*& ptr, std::size_t& bytes) {
  if (!ptr) {
    bytes = 0;
    return RNS8_SUCCESS;
  }
  const rns8_status status = rns8::detail::hip_direct_free(device_id, ptr);
  ptr = nullptr;
  bytes = 0;
  return status;
}

bool workspace_uses_hip_arena(const rns8_workspace& workspace) {
  return workspace.hip_workspace_arena != nullptr;
}

void clear_workspace_arena_subranges(rns8_workspace& workspace) {
  workspace.hip_tile_schedule = nullptr;
  workspace.hip_tile_schedule_bytes = 0;
  workspace.hip_tile_schedule_count = 0;
  workspace.hip_tile_schedule_active_entries = nullptr;
  workspace.hip_tile_schedule_active_entries_bytes = 0;
  workspace.hip_tile_schedule_active_entries_count = 0;
  workspace.hip_zero_a_rows = nullptr;
  workspace.hip_zero_a_rows_bytes = 0;
  workspace.hip_zero_b_cols = nullptr;
  workspace.hip_zero_b_cols_bytes = 0;
  workspace.hip_workspace_scratch = nullptr;
  workspace.hip_workspace_scratch_bytes = 0;
}

void clear_workspace_arena_contract(rns8_workspace& workspace) {
  workspace.hip_workspace_arena_bytes = 0;
  workspace.hip_workspace_arena_high_water_mark_bytes = 0;
  workspace.hip_workspace_arena_suballocation_count = 0;
  workspace.hip_workspace_arena_plan_fingerprint = 0;
  workspace.hip_workspace_arena_target_id.clear();
  workspace.hip_workspace_arena_policy.clear();
}

void reset_workspace_active_schedule_contract(rns8_workspace& workspace) {
  workspace.hip_tile_schedule_active_prefix_count = 0;
  for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    workspace.hip_tile_schedule_active_offsets[p] = 0;
    workspace.hip_tile_schedule_active_counts[p] = 0;
  }
}

void reset_workspace_hipblaslt_a_prepack_contract(rns8_workspace& workspace) {
  workspace.hipblaslt_a_prepack_current = false;
  workspace.hipblaslt_a_prepack_source_version = 0;
  workspace.hipblaslt_a_prepack_m = 0;
  workspace.hipblaslt_a_prepack_k = 0;
  workspace.hipblaslt_a_prepack_lda = 0;
  workspace.hipblaslt_a_prepack_prefix = 0;
  workspace.hipblaslt_a_prepack_device_id = -1;
}

void reset_workspace_hipblaslt_b_prepack_contract(rns8_workspace& workspace) {
  workspace.hipblaslt_b_prepack_current = false;
  workspace.hipblaslt_b_prepack_source_version = 0;
  workspace.hipblaslt_b_prepack_k = 0;
  workspace.hipblaslt_b_prepack_n = 0;
  workspace.hipblaslt_b_prepack_ldb = 0;
  workspace.hipblaslt_b_prepack_prefix = 0;
  workspace.hipblaslt_b_prepack_device_id = -1;
}

rns8_status release_workspace_resources(rns8_workspace& workspace) {
  rns8_status status = RNS8_SUCCESS;
  const auto keep_first_error = [&](rns8_status free_status) {
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
  };
  if (workspace_uses_hip_arena(workspace)) {
    clear_workspace_arena_subranges(workspace);
  } else {
    keep_first_error(release_workspace_device_buffer(
        workspace.hip_device_id, workspace.hip_tile_schedule, workspace.hip_tile_schedule_bytes));
    workspace.hip_tile_schedule_count = 0;
    keep_first_error(release_workspace_device_buffer(
        workspace.hip_device_id,
        workspace.hip_tile_schedule_active_entries,
        workspace.hip_tile_schedule_active_entries_bytes));
    workspace.hip_tile_schedule_active_entries_count = 0;
    keep_first_error(release_workspace_device_buffer(
        workspace.hip_device_id, workspace.hip_zero_a_rows, workspace.hip_zero_a_rows_bytes));
    keep_first_error(release_workspace_device_buffer(
        workspace.hip_device_id, workspace.hip_zero_b_cols, workspace.hip_zero_b_cols_bytes));
  }
  reset_workspace_active_schedule_contract(workspace);
  keep_first_error(release_workspace_device_buffer(
      workspace.hip_device_id, workspace.hip_workspace_arena, workspace.hip_workspace_arena_bytes));
  clear_workspace_arena_contract(workspace);
  keep_first_error(release_workspace_device_buffer(
      workspace.hip_device_id, workspace.hipblaslt_int32_scratch, workspace.hipblaslt_int32_scratch_bytes));
  keep_first_error(release_workspace_device_buffer(
      workspace.hip_device_id, workspace.hipblaslt_workspace, workspace.hipblaslt_workspace_bytes));
  keep_first_error(release_workspace_device_buffer(
      workspace.hipblaslt_a_prepack_device_id,
      workspace.hipblaslt_a_prepack_cache,
      workspace.hipblaslt_a_prepack_cache_bytes));
  reset_workspace_hipblaslt_a_prepack_contract(workspace);
  keep_first_error(release_workspace_device_buffer(
      workspace.hipblaslt_b_prepack_device_id,
      workspace.hipblaslt_b_prepack_cache,
      workspace.hipblaslt_b_prepack_cache_bytes));
  reset_workspace_hipblaslt_b_prepack_contract(workspace);
  keep_first_error(release_workspace_device_buffer(
      workspace.hip_device_id, workspace.accelerator_auxiliary, workspace.accelerator_auxiliary_bytes));
  keep_first_error(release_workspace_device_buffer(
      workspace.hip_device_id, workspace.accelerator_workspace, workspace.accelerator_workspace_bytes));
  return status;
}

rns8_status validate_plan_context_workspace(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace) {
  if (!plan_schedule_contract_matches(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) || workspace.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.semantics != plan.desc.semantics || workspace.bound_kind != plan.desc.bound_kind) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.m != plan.desc.m || workspace.n != plan.desc.n || workspace.k != plan.desc.k ||
      workspace.prefix != plan.prefix) {
    return RNS8_WORKSPACE_TOO_SMALL;
  }
  if (!workspace_schedule_contract_matches_plan(plan, workspace) ||
      !workspace_backend_metadata_matches_plan(plan, workspace)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) && workspace.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    const bool zero_row_col_masks =
        (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0;
    const std::size_t expected_zero_a_bytes =
        zero_row_col_masks ? static_cast<std::size_t>(plan.desc.m) * sizeof(uint8_t) : 0;
    const std::size_t expected_zero_b_bytes =
        zero_row_col_masks ? static_cast<std::size_t>(plan.desc.n) * sizeof(uint8_t) : 0;
    if (workspace.hip_zero_a_rows_bytes != expected_zero_a_bytes ||
        workspace.hip_zero_b_cols_bytes != expected_zero_b_bytes ||
        (expected_zero_a_bytes == 0 ? workspace.hip_zero_a_rows != nullptr
                                    : workspace.hip_zero_a_rows == nullptr) ||
        (expected_zero_b_bytes == 0 ? workspace.hip_zero_b_cols != nullptr
                                    : workspace.hip_zero_b_cols == nullptr)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const bool scheduled = !plan.tile_schedule.empty();
    if (scheduled && plan.tile_schedule.size() >
                         std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_schedule_bytes =
        scheduled && hip_direct_gemm_requires_device_tile_schedule(plan)
            ? plan.tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry)
            : 0;
    if (scheduled) {
      const uint64_t expected_schedule_count =
          expected_schedule_bytes == 0 ? 0 : static_cast<uint64_t>(plan.tile_schedule.size());
      if (workspace.hip_tile_schedule_bytes != expected_schedule_bytes ||
          workspace.hip_tile_schedule_count != expected_schedule_count ||
          (expected_schedule_bytes == 0 ? workspace.hip_tile_schedule != nullptr
                                        : workspace.hip_tile_schedule == nullptr) ||
          !active_prefix_tile_schedule_matches_workspace(plan, workspace)) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
               workspace.hip_tile_schedule_count != 0 ||
               !active_prefix_tile_schedule_workspace_empty(workspace)) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
             workspace.hip_tile_schedule_count != 0 ||
             !active_prefix_tile_schedule_workspace_empty(workspace) ||
             workspace.hip_zero_a_rows || workspace.hip_zero_a_rows_bytes != 0 ||
             workspace.hip_zero_b_cols || workspace.hip_zero_b_cols_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    std::size_t expected_scratch_bytes = 0;
    std::size_t expected_workspace_bytes = 0;
    if (!hipblaslt_scratch_bytes_for_plan(plan, expected_scratch_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!hipblaslt_workspace_bytes_for_plan(plan, expected_workspace_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!workspace.hipblaslt_int32_scratch ||
        workspace.hipblaslt_int32_scratch_bytes != expected_scratch_bytes ||
        !workspace.hipblaslt_workspace ||
        workspace.hipblaslt_workspace_bytes != expected_workspace_bytes) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hipblaslt_int32_scratch || workspace.hipblaslt_int32_scratch_bytes != 0 ||
             workspace.hipblaslt_workspace || workspace.hipblaslt_workspace_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rns8::detail::accelerator_backend_kind(plan.backend)) {
    if (!rns8::detail::accelerator_backend_compiled(plan.backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_workspace_bytes =
        plan.backend_workspace_required_bytes > std::numeric_limits<std::size_t>::max()
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(plan.backend_workspace_required_bytes);
    if (expected_workspace_bytes != 0) {
      if (!workspace.accelerator_workspace ||
          workspace.accelerator_workspace_bytes != expected_workspace_bytes) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0 ||
             workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}
