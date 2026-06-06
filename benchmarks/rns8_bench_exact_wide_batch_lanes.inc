BenchmarkResult run_exact_wide_signed_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_limb_elements(args, args.exact_wide_limb_count, "host batch C");
  std::vector<std::vector<int64_t>> batch_a(task_count, std::vector<int64_t>(a_elements));
  std::vector<std::vector<int64_t>> batch_b(task_count, std::vector<int64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = dist(rng);
    for (auto& value : batch_b[task_index]) value = dist(rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch exact-wide signed)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  GroupedExactWideRnsBatch<int64_t> grouped_batch;
  const bool grouped_export_slab_enabled = grouped_dispatch_requested(args) &&
                                           selected_backend == RNS8_BACKEND_HIP_DIRECT &&
                                           !exact_wide_export_status_check_required(args) &&
                                           args.output_ld_padding == 0;
  const bool grouped_pack_slab_enabled = grouped_export_slab_enabled;
  const bool grouped_gemm_slab_enabled = grouped_pack_slab_enabled;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_gemm_slab_enabled ? "device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h"
                                    : "host_phase_loop_per_task_export";
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  enforce_grouped_dispatch_task_setup_contract(
      args, plan, tasks, RNS8_EXACT_WIDE_SIGNED, matrix_prefix, "grouped exact-wide signed setup");
  if (grouped_export_slab_enabled) {
    initialize_grouped_exact_wide_rns_batch(
        args,
        plan,
        RNS8_EXACT_WIDE_SIGNED,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        result,
        "grouped exact-wide signed");
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  auto& grouped_resources = grouped_batch.device_resources;
  const auto grouped_bucket_plan =
      make_grouped_dispatch_bucket_plan(args, plan, tasks, RNS8_EXACT_WIDE_SIGNED, matrix_prefix);
  const auto& grouped_descriptor =
      single_grouped_dispatch_descriptor(grouped_bucket_plan, "grouped exact-wide signed execution");

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_pack_grouped_i64_task_inputs(
            grouped_descriptor,
            grouped_resources,
            grouped_batch.a_host_slab.data(),
            grouped_batch.b_host_slab.data(),
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_grouped_i64_task_inputs(host API batch exact-wide signed)", status);
      }
      (void)task;
      return;
    }
    status = rns8_pack_i64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch exact-wide A)", status);
    status = rns8_pack_i64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch exact-wide B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_gemm_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(
            grouped_descriptor,
            grouped_resources);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_grouped_rns_task_outputs(host API batch exact-wide signed)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch exact-wide signed)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_export_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_export_grouped_exact_wide_task_outputs_to_host(
            grouped_descriptor,
            grouped_resources,
            args.exact_wide_limb_count,
            true,
            grouped_batch.export_host_slab.data(),
            grouped_batch.export_host_slab.size());
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_export_grouped_exact_wide_task_outputs_to_host(host API batch exact-wide signed)", status);
        }
      }
      return;
    }
    status = rns8_export_exact_wide_signed_limbs(
        ctx, plan, task.c, batch_c[task_index].data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_EXACT_WIDE_SIGNED,
        matrix_prefix,
        static_cast<uint64_t>(r) + 1,
        nullptr,
        pack_task,
        gemm_task,
        export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_EXACT_WIDE_SIGNED,
        matrix_prefix,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  if (grouped_export_slab_enabled) {
    scatter_grouped_limb_slab_to_outputs(args, grouped_batch.export_host_slab, batch_c);
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_limb_matrix(
        batch_c[task_index], args.m, args.n, ldc, args.exact_wide_limb_count, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_unsigned_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_limb_elements(args, args.exact_wide_limb_count, "host batch C");
  std::vector<std::vector<uint64_t>> batch_a(task_count, std::vector<uint64_t>(a_elements));
  std::vector<std::vector<uint64_t>> batch_b(task_count, std::vector<uint64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = dist(rng);
    for (auto& value : batch_b[task_index]) value = dist(rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch exact-wide unsigned)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  GroupedExactWideRnsBatch<uint64_t> grouped_batch;
  const bool grouped_export_slab_enabled = grouped_dispatch_requested(args) &&
                                           selected_backend == RNS8_BACKEND_HIP_DIRECT &&
                                           !exact_wide_export_status_check_required(args) &&
                                           args.output_ld_padding == 0;
  const bool grouped_pack_slab_enabled = grouped_export_slab_enabled;
  const bool grouped_gemm_slab_enabled = grouped_pack_slab_enabled;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_gemm_slab_enabled ? "device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h"
                                    : "host_phase_loop_per_task_export";
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  enforce_grouped_dispatch_task_setup_contract(
      args, plan, tasks, RNS8_EXACT_WIDE_UNSIGNED, matrix_prefix, "grouped exact-wide unsigned setup");
  if (grouped_export_slab_enabled) {
    initialize_grouped_exact_wide_rns_batch(
        args,
        plan,
        RNS8_EXACT_WIDE_UNSIGNED,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        result,
        "grouped exact-wide unsigned");
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  auto& grouped_resources = grouped_batch.device_resources;
  const auto grouped_bucket_plan =
      make_grouped_dispatch_bucket_plan(args, plan, tasks, RNS8_EXACT_WIDE_UNSIGNED, matrix_prefix);
  const auto& grouped_descriptor =
      single_grouped_dispatch_descriptor(grouped_bucket_plan, "grouped exact-wide unsigned execution");

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_pack_grouped_u64_task_inputs(
            grouped_descriptor,
            grouped_resources,
            grouped_batch.a_host_slab.data(),
            grouped_batch.b_host_slab.data(),
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_grouped_u64_task_inputs(host API batch exact-wide unsigned)", status);
      }
      (void)task;
      return;
    }
    status = rns8_pack_u64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch exact-wide A)", status);
    status = rns8_pack_u64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch exact-wide B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_gemm_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(
            grouped_descriptor,
            grouped_resources);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_grouped_rns_task_outputs(host API batch exact-wide unsigned)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch exact-wide unsigned)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_export_slab_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_export_grouped_exact_wide_task_outputs_to_host(
            grouped_descriptor,
            grouped_resources,
            args.exact_wide_limb_count,
            false,
            grouped_batch.export_host_slab.data(),
            grouped_batch.export_host_slab.size());
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_export_grouped_exact_wide_task_outputs_to_host(host API batch exact-wide unsigned)", status);
        }
      }
      return;
    }
    status = rns8_export_exact_wide_unsigned_limbs(
        ctx, plan, task.c, batch_c[task_index].data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_EXACT_WIDE_UNSIGNED,
        matrix_prefix,
        static_cast<uint64_t>(r) + 1,
        nullptr,
        pack_task,
        gemm_task,
        export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_EXACT_WIDE_UNSIGNED,
        matrix_prefix,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  if (grouped_export_slab_enabled) {
    scatter_grouped_limb_slab_to_outputs(args, grouped_batch.export_host_slab, batch_c);
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_limb_matrix(
        batch_c[task_index], args.m, args.n, ldc, args.exact_wide_limb_count, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

