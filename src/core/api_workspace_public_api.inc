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
    const auto free_workspace_zero_masks = [&]() {
      if (workspace->hip_zero_b_cols) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_b_cols);
        workspace->hip_zero_b_cols = nullptr;
        workspace->hip_zero_b_cols_bytes = 0;
      }
      if (workspace->hip_zero_a_rows) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_a_rows);
        workspace->hip_zero_a_rows = nullptr;
        workspace->hip_zero_a_rows_bytes = 0;
      }
    };
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT &&
        (plan->desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0) {
      const std::size_t zero_a_bytes = plan->zero_a_rows.size() * sizeof(uint8_t);
      const std::size_t zero_b_bytes = plan->zero_b_cols.size() * sizeof(uint8_t);
      rns8_status status = rns8::detail::hip_direct_allocate(ctx->device_id, zero_a_bytes, &workspace->hip_zero_a_rows);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hip_zero_a_rows_bytes = zero_a_bytes;
      status = rns8::detail::hip_direct_copy_host_to_device(
          ctx->device_id, workspace->hip_zero_a_rows, plan->zero_a_rows.data(), zero_a_bytes);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_a_rows);
        delete workspace;
        return status;
      }
      status = rns8::detail::hip_direct_allocate(ctx->device_id, zero_b_bytes, &workspace->hip_zero_b_cols);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_a_rows);
        delete workspace;
        return status;
      }
      workspace->hip_zero_b_cols_bytes = zero_b_bytes;
      status = rns8::detail::hip_direct_copy_host_to_device(
          ctx->device_id, workspace->hip_zero_b_cols, plan->zero_b_cols.data(), zero_b_bytes);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_b_cols);
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_zero_a_rows);
        delete workspace;
        return status;
      }
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT && !plan->tile_schedule.empty()) {
      if (plan->tile_schedule.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
        free_workspace_zero_masks();
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      std::vector<rns8_plan_tile_schedule_entry> active_entries;
      if (!build_active_prefix_tile_schedule(
              *plan,
              active_entries,
              workspace->hip_tile_schedule_active_offsets,
              workspace->hip_tile_schedule_active_counts,
              workspace->hip_tile_schedule_active_prefix_count)) {
        free_workspace_zero_masks();
        delete workspace;
        return RNS8_INVALID_ARGUMENT;
      }
      if (active_entries.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
        free_workspace_zero_masks();
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const bool upload_device_tile_schedule = hip_direct_gemm_requires_device_tile_schedule(*plan);
      const std::size_t schedule_bytes =
          upload_device_tile_schedule ? plan->tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry) : 0;
      const std::size_t active_schedule_bytes = active_entries.size() * sizeof(rns8_plan_tile_schedule_entry);
      rns8_status status = RNS8_SUCCESS;
      if (schedule_bytes != 0) {
        status = rns8::detail::hip_direct_allocate(ctx->device_id, schedule_bytes, &workspace->hip_tile_schedule);
        if (status != RNS8_SUCCESS) {
          free_workspace_zero_masks();
          delete workspace;
          return status;
        }
        workspace->hip_tile_schedule_bytes = schedule_bytes;
        workspace->hip_tile_schedule_count = static_cast<uint64_t>(plan->tile_schedule.size());
        status = rns8::detail::hip_direct_copy_host_to_device(
            ctx->device_id, workspace->hip_tile_schedule, plan->tile_schedule.data(), schedule_bytes);
        if (status != RNS8_SUCCESS) {
          (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule);
          free_workspace_zero_masks();
          delete workspace;
          return status;
        }
      }
      workspace->hip_tile_schedule_active_entries_bytes = active_schedule_bytes;
      workspace->hip_tile_schedule_active_entries_count = static_cast<uint64_t>(active_entries.size());
      if (active_schedule_bytes != 0) {
        status = rns8::detail::hip_direct_allocate(
            ctx->device_id, active_schedule_bytes, &workspace->hip_tile_schedule_active_entries);
        if (status != RNS8_SUCCESS) {
          (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule);
          free_workspace_zero_masks();
          delete workspace;
          return status;
        }
        status = rns8::detail::hip_direct_copy_host_to_device(
            ctx->device_id,
            workspace->hip_tile_schedule_active_entries,
            active_entries.data(),
            active_schedule_bytes);
        if (status != RNS8_SUCCESS) {
          (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule_active_entries);
          (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule);
          free_workspace_zero_masks();
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

