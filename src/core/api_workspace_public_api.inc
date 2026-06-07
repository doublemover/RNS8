rns8_status rns8_create_workspace(rns8_context* ctx, const rns8_plan* plan, rns8_workspace** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!context_accepts_backend(*ctx, plan->backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace = new (std::nothrow) rns8_workspace();
    if (!workspace) {
      return RNS8_INTERNAL_ERROR;
    }
    initialize_workspace_identity(*workspace, *ctx, *plan);
    initialize_workspace_schedule_contract(*workspace, *plan);
    initialize_workspace_backend_metadata(*workspace, *plan);
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      std::vector<rns8_plan_tile_schedule_entry> active_entries;
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_schedule.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
          delete workspace;
          return RNS8_RANGE_ERROR;
        }
        if (!build_active_prefix_tile_schedule(
                *plan,
                active_entries,
                workspace->hip_tile_schedule_active_offsets,
                workspace->hip_tile_schedule_active_counts,
                workspace->hip_tile_schedule_active_prefix_count)) {
          delete workspace;
          return RNS8_INVALID_ARGUMENT;
        }
        if (active_entries.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
          delete workspace;
          return RNS8_RANGE_ERROR;
        }
      }
      const bool zero_masks_requested = (plan->desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0;
      const bool upload_device_tile_schedule =
          !plan->tile_schedule.empty() && hip_direct_gemm_requires_device_tile_schedule(*plan);
      const std::size_t zero_a_bytes = zero_masks_requested ? plan->zero_a_rows.size() * sizeof(uint8_t) : 0;
      const std::size_t zero_b_bytes = zero_masks_requested ? plan->zero_b_cols.size() * sizeof(uint8_t) : 0;
      const std::size_t schedule_bytes =
          upload_device_tile_schedule ? plan->tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry) : 0;
      const std::size_t active_schedule_bytes = active_entries.size() * sizeof(rns8_plan_tile_schedule_entry);
      std::size_t explicitly_assigned_workspace_bytes = 0;
      const auto count_explicit_workspace_region = [&](std::size_t bytes) -> bool {
        if (explicitly_assigned_workspace_bytes > std::numeric_limits<std::size_t>::max() - bytes) {
          return false;
        }
        explicitly_assigned_workspace_bytes += bytes;
        return true;
      };
      if (!count_explicit_workspace_region(zero_a_bytes) || !count_explicit_workspace_region(zero_b_bytes) ||
          !count_explicit_workspace_region(schedule_bytes) ||
          !count_explicit_workspace_region(active_schedule_bytes)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      if (plan->backend_workspace_required_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t declared_workspace_bytes = static_cast<std::size_t>(plan->backend_workspace_required_bytes);
      const std::size_t scratch_bytes =
          declared_workspace_bytes > explicitly_assigned_workspace_bytes
              ? declared_workspace_bytes - explicitly_assigned_workspace_bytes
              : 0;
      const auto align_up = [](std::size_t value) -> std::size_t {
        constexpr std::size_t alignment = 256;
        return (value + alignment - 1) / alignment * alignment;
      };
      std::size_t arena_bytes = 0;
      uint32_t suballocation_count = 0;
      const auto reserve_region = [&](std::size_t bytes) -> bool {
        if (bytes == 0) {
          return true;
        }
        const std::size_t aligned = align_up(arena_bytes);
        if (aligned > std::numeric_limits<std::size_t>::max() - bytes) {
          return false;
        }
        arena_bytes = aligned + bytes;
        ++suballocation_count;
        return true;
      };
      if (!reserve_region(zero_a_bytes) || !reserve_region(zero_b_bytes) ||
          !reserve_region(schedule_bytes) || !reserve_region(active_schedule_bytes) ||
          !reserve_region(scratch_bytes)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      if (arena_bytes != 0) {
        rns8_status status =
            rns8::detail::hip_direct_allocate(ctx->device_id, arena_bytes, &workspace->hip_workspace_arena);
        if (status != RNS8_SUCCESS) {
          delete workspace;
          return status;
        }
        workspace->hip_workspace_arena_bytes = arena_bytes;
        workspace->hip_workspace_arena_high_water_mark_bytes = arena_bytes;
        workspace->hip_workspace_arena_suballocation_count = suballocation_count;
        workspace->hip_workspace_arena_plan_fingerprint = plan_workspace_fingerprint(*plan);
        workspace->hip_workspace_arena_target_id = plan->backend_target_id;
        workspace->hip_workspace_arena_policy = "direct_hip_workspace_owned_suballocation_v1";
        auto* arena = static_cast<unsigned char*>(workspace->hip_workspace_arena);
        std::size_t offset = 0;
        const auto assign_region = [&](std::size_t bytes) -> void* {
          if (bytes == 0) {
            return nullptr;
          }
          offset = align_up(offset);
          void* ptr = arena + offset;
          offset += bytes;
          return ptr;
        };
        workspace->hip_zero_a_rows = assign_region(zero_a_bytes);
        workspace->hip_zero_a_rows_bytes = zero_a_bytes;
        workspace->hip_zero_b_cols = assign_region(zero_b_bytes);
        workspace->hip_zero_b_cols_bytes = zero_b_bytes;
        workspace->hip_tile_schedule = assign_region(schedule_bytes);
        workspace->hip_tile_schedule_bytes = schedule_bytes;
        workspace->hip_tile_schedule_count = schedule_bytes != 0 ? static_cast<uint64_t>(plan->tile_schedule.size()) : 0;
        workspace->hip_tile_schedule_active_entries = assign_region(active_schedule_bytes);
        workspace->hip_tile_schedule_active_entries_bytes = active_schedule_bytes;
        workspace->hip_tile_schedule_active_entries_count = static_cast<uint64_t>(active_entries.size());
        workspace->hip_workspace_scratch = assign_region(scratch_bytes);
        workspace->hip_workspace_scratch_bytes = scratch_bytes;
        if (zero_a_bytes != 0) {
          status = rns8::detail::hip_direct_copy_host_to_device(
              ctx->device_id, workspace->hip_zero_a_rows, plan->zero_a_rows.data(), zero_a_bytes);
        }
        if (status == RNS8_SUCCESS && zero_b_bytes != 0) {
          status = rns8::detail::hip_direct_copy_host_to_device(
              ctx->device_id, workspace->hip_zero_b_cols, plan->zero_b_cols.data(), zero_b_bytes);
        }
        if (status == RNS8_SUCCESS && schedule_bytes != 0) {
          status = rns8::detail::hip_direct_copy_host_to_device(
              ctx->device_id, workspace->hip_tile_schedule, plan->tile_schedule.data(), schedule_bytes);
        }
        if (status == RNS8_SUCCESS && active_schedule_bytes != 0) {
          status = rns8::detail::hip_direct_copy_host_to_device(
              ctx->device_id,
              workspace->hip_tile_schedule_active_entries,
              active_entries.data(),
              active_schedule_bytes);
        }
        if (status != RNS8_SUCCESS) {
          (void)release_workspace_resources(*workspace);
          delete workspace;
          return status;
        }
      }
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
      if (!ctx->hipblaslt_handle) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      std::size_t scratch_bytes = 0;
      std::size_t workspace_bytes = 0;
      if (!hipblaslt_scratch_bytes_for_plan(*plan, scratch_bytes) ||
          !hipblaslt_workspace_bytes_for_plan(*plan, workspace_bytes)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      rns8_status status =
          rns8::detail::hip_direct_allocate(ctx->device_id, scratch_bytes, &workspace->hipblaslt_int32_scratch);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hipblaslt_int32_scratch_bytes = scratch_bytes;
      status = rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->hipblaslt_workspace);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hipblaslt_int32_scratch);
        delete workspace;
        return status;
      }
      workspace->hipblaslt_workspace_bytes = workspace_bytes;
    }
    if (rns8::detail::accelerator_backend_kind(plan->backend)) {
      if (!rns8::detail::accelerator_backend_compiled(plan->backend)) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      if (plan->backend_workspace_required_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t workspace_bytes = static_cast<std::size_t>(plan->backend_workspace_required_bytes);
      if (workspace_bytes != 0) {
        rns8_status status =
            rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->accelerator_workspace);
        if (status != RNS8_SUCCESS) {
          delete workspace;
          return status;
        }
        workspace->accelerator_workspace_bytes = workspace_bytes;
      }
    }
    *out = workspace;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_workspace(rns8_workspace* workspace) {
  if (workspace) {
    const rns8_status status = release_workspace_resources(*workspace);
    delete workspace;
    return status;
  }
  delete workspace;
  return RNS8_SUCCESS;
}

