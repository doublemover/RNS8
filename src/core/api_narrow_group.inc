// Independent same-plan tasks share launches, never output or proof authority.
rns8_status rns8_gemm_narrow_exact_grouped(rns8_context* ctx, const rns8_plan* plan,
                                           const rns8_grouped_gemm_task* tasks, uint32_t count,
                                           rns8_narrow_product_method method, uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !plan || !tasks ||
        reinterpret_cast<uintptr_t>(tasks) % alignof(rns8_grouped_gemm_task) || count < 2 ||
        count > 4096 || uint32_t(method) >= native_narrow_method_count)
      return RNS8_INVALID_ARGUMENT;
    std::unordered_set<const rns8_matrix*> inputs, outputs;
    std::unordered_set<const rns8_workspace*> workspaces;
    inputs.reserve(size_t(count) * 2);
    outputs.reserve(count);
    workspaces.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& task = tasks[i];
      if (!minimum_abi_header(&task) || !task.a || !task.b || !task.c || !task.workspace ||
          !outputs.insert(task.c).second || !workspaces.insert(task.workspace).second)
        return RNS8_INVALID_ARGUMENT;
      inputs.insert(task.a);
      inputs.insert(task.b);
    }
    for (const auto* output : outputs)
      if (inputs.count(output))
        return RNS8_INVALID_ARGUMENT;
    if (plan->backend != RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(plan->desc.semantics) ||
        !native_narrow_fused(method))
      return RNS8_UNSUPPORTED_BACKEND;
    const bool byte_wmma = native_narrow_byte_wmma(method);
    const bool unsigned_inputs = plan->desc.semantics == RNS8_EXACT_WIDE_UNSIGNED;
    if (unsigned_inputs && !byte_wmma)
      return RNS8_UNSUPPORTED_BACKEND;
    const auto support = query_narrow_group_support(plan->desc.m, plan->desc.n, plan->desc.k,
                                                    plan->prefix, method, count, budget);
    if (support.status != RNS8_SUCCESS)
      return support.status;
    rns8_operation_query query{};
    query.struct_size = sizeof(query);
    query.abi_version = RNS8_ABI_VERSION;
    query.operation = RNS8_OP_NARROW_NATIVE;
    query.execution_modes = RNS8_EXEC_GROUP;
    query.problem = plan->desc;
    query.algorithm = method;
    query.proof_flags = RNS8_QUERY_NATIVE_RETAINED | RNS8_QUERY_NARROW_INPUT_PROOF;
    query.scratch_budget = UINT64_MAX;
    auto status = operation_support(*ctx, query, plan->backend).status;
    if (status != RNS8_SUCCESS)
      return status;
    std::vector<cpp_int> bounds(count);
    std::vector<rns8_exact_matrix_ranges> ranges(count);
    auto native = std::make_shared<std::vector<NativeNarrowTask>>(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& task = tasks[i];
      status = validate_plan_context_workspace(*ctx, *plan, *task.workspace);
      if (status == RNS8_SUCCESS)
        status = validate_rns_gemm_operands(*ctx, *plan, *task.a, *task.b, *task.c, true);
      if (status != RNS8_SUCCESS)
        return status;
      if (task.c->generation_exhausted)
        return RNS8_RANGE_ERROR;
      auto& entry = (*native)[i];
      entry.a = narrow_owned_native(*ctx, *task.a, unsigned_inputs);
      entry.b = narrow_owned_native(*ctx, *task.b, unsigned_inputs);
      entry.c = static_cast<int8_t*>(task.c->hip_residues);
      if (!entry.a || !entry.b)
        return RNS8_INVALID_ARGUMENT;
      status = exact_gemm_range(*plan, *task.a, *task.b, bounds[i], &ranges[i]);
      if (status != RNS8_SUCCESS)
        return status;
    }
    bool commit_started = false;
    std::vector<SynchronousLease> leases;
    leases.reserve(count);
    const auto finish = [&](rns8_status result) {
      for (auto& lease : leases)
        lease.finish(result);
      return result;
    };
    for (uint32_t i = 0; i < count; ++i) {
      size_t token = 0;
      auto& workspace = *tasks[i].workspace;
      status =
          begin_synchronous_lease(*ctx, *tasks[i].a, *tasks[i].b, *tasks[i].c, workspace, token);
      if (status != RNS8_SUCCESS)
        return finish(status);
      leases.emplace_back(*ctx, token);
      leases.back().track_write(commit_started);
      for (auto* buffer : {&workspace.narrow_product_owner, &workspace.narrow_scan_owner,
                           &workspace.narrow_group_owner}) {
        status = leases.back().hold_buffer(*buffer);
        if (status != RNS8_SUCCESS)
          return finish(status);
        if (buffer->pending())
          return finish(RNS8_BACKEND_FAILURE);
        if (*buffer && buffer->device_id() != ctx->device_id)
          return finish(RNS8_INVALID_ARGUMENT);
      }
    }
    status = synchronous_dispatch_checkpoint(*ctx, "narrow_group_prepare");
    if (status != RNS8_SUCCESS)
      return finish(status);
    const auto per_task = support.payload_bytes / count;
    const auto bytes = size_t(count) * sizeof(NativeNarrowTask);
    // Retire inactive storage before scans can allocate. The budget describes
    // simultaneous live owners, not just the eventual replacement allocation.
    for (uint32_t i = 0; i < count; ++i) {
      auto& workspace = *tasks[i].workspace;
      if (workspace.narrow_product_owner.bytes() != per_task) {
        status = workspace.narrow_product_owner.reset();
        if (status != RNS8_SUCCESS)
          return finish(status);
      }
      if (i || workspace.narrow_group_owner.bytes() != bytes) {
        status = workspace.narrow_group_owner.reset();
        if (status != RNS8_SUCCESS)
          return finish(status);
        workspace.narrow_group_tasks.clear();
      }
    }
    if (byte_wmma) {
      status = ensure_native_wmma_capability(*ctx);
      if (status != RNS8_SUCCESS)
        return finish(status);
      if (!ctx->native_wmma_capability.supported)
        return finish(RNS8_UNSUPPORTED_BACKEND);
    }
    // All numerical admissions precede partial/output allocation and dispatch.
    for (uint32_t i = 0; i < count; ++i) {
      auto& workspace = *tasks[i].workspace;
      workspace.narrow_capability_observation = {};
      if (native_wmma_capability_current(*ctx, ctx->native_wmma_capability))
        workspace.narrow_capability_observation = ctx->native_wmma_capability;
      status = prepare_narrow_domains(*ctx, *tasks[i].a, *tasks[i].b, (*native)[i].a,
                                      (*native)[i].b, workspace);
      if (status != RNS8_SUCCESS)
        return finish(status);
      uint64_t bound = 0;
      status = native_narrow_domain_admission(workspace.narrow_domains[0].range,
                                              workspace.narrow_domains[1].range,
                                              uint64_t(plan->desc.k), byte_wmma, bound);
      if (status != RNS8_SUCCESS)
        return finish(status);
    }
    for (uint32_t i = 0; i < count; ++i) {
      auto& workspace = *tasks[i].workspace;
      auto& product = workspace.narrow_product_owner;
      if (product.bytes() != per_task) {
        status = per_task ? product.allocate(ctx->device_id, size_t(per_task)) : product.reset();
        if (status != RNS8_SUCCESS)
          return finish(status);
      }
      (*native)[i].partials = static_cast<int64_t*>(product.get());
    }
    auto& first = *tasks[0].workspace;
    auto& directory = first.narrow_group_owner;
    if (directory.bytes() != bytes || first.narrow_group_tasks != *native) {
      first.narrow_group_tasks.clear();
      if (directory.bytes() != bytes) {
        status = directory.allocate(ctx->device_id, bytes);
        if (status != RNS8_SUCCESS)
          return finish(status);
      }
      status = directory.retain_host_payload(native);
      if (status != RNS8_SUCCESS)
        return finish(status);
      directory.mark_pending();
      status = hip_direct_copy_owned_host_to_device(ctx->device_id, directory.get(),
                                                    {native, native->data()}, bytes);
      if (status != RNS8_SUCCESS)
        return finish(status);
      directory.mark_completed();
      first.narrow_group_tasks = *native;
    }
    directory.mark_pending();
    for (uint32_t i = 0; i < count; ++i)
      if (per_task)
        tasks[i].workspace->narrow_product_owner.mark_pending();
    status = synchronous_dispatch_checkpoint(*ctx, "narrow_group_compute", &commit_started);
    if (status == RNS8_SUCCESS)
      status = hip_direct_launch_admitted_narrow_group(
          ctx->device_id, static_cast<const NativeNarrowTask*>(directory.get()), count,
          plan->desc.m, plan->desc.n, plan->desc.k, plan->prefix, method, unsigned_inputs,
          commit_started);
    if (status == RNS8_SUCCESS)
      for (uint32_t i = 0; i < count; ++i) {
        mark_output_device_residues_current(*tasks[i].c, plan);
        commit_exact_range(*tasks[i].c, *plan, bounds[i], &ranges[i]);
      }
    return finish(status);
  });
}
