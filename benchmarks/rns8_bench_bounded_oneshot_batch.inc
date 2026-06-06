BenchmarkResult run_bounded_i64_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  auto desc = gemm_desc(args, bound);
  result = initialize_bounded_oneshot_result(ctx, args, bound, desc, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    const rns8_status status =
        rns8_gemm_i64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_i64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(args, result, selected_backend, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto repeat_end = std::chrono::steady_clock::now();

    if (samples) {
      const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
      samples->pack_us.push_back(0);
      samples->gemm_us.push_back(elapsed);
      samples->export_us.push_back(0);
      samples->end_to_end_us.push_back(elapsed);
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_bounded_u64_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  auto desc = gemm_desc(args, bound);
  result = initialize_bounded_oneshot_result(ctx, args, bound, desc, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    const rns8_status status =
        rns8_gemm_u64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_u64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(args, result, selected_backend, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto repeat_end = std::chrono::steady_clock::now();

    if (samples) {
      const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
      samples->pack_us.push_back(0);
      samples->gemm_us.push_back(elapsed);
      samples->export_us.push_back(0);
      samples->end_to_end_us.push_back(elapsed);
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_bounded_i64_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch C");
  std::vector<std::vector<int64_t>> batch_a(task_count, std::vector<int64_t>(a_elements));
  std::vector<std::vector<int64_t>> batch_b(task_count, std::vector<int64_t>(b_elements));
  std::vector<std::vector<int64_t>> batch_c(
      task_count, std::vector<int64_t>(c_elements, INT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    fill_bounded_i64_inputs(args, batch_a[task_index], batch_b[task_index], rng);
  }

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, batch_a.front(), batch_b.front(), bound, result);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  const bool grouped_pack_gemm_export_enabled =
      grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_pack_gemm_export_enabled ? "device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h"
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
      args, plan, tasks, RNS8_BOUNDED_I64, matrix_prefix, "grouped bounded i64 setup");
  GroupedNativeRnsBatch<int64_t> grouped_batch;
  if (grouped_pack_gemm_export_enabled) {
    initialize_grouped_native_rns_batch(
        args,
        plan,
        RNS8_BOUNDED_I64,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        "grouped bounded i64");
    result.grouped_dispatch_batched_export_enabled = true;
    result.grouped_dispatch_device_output_slab_bytes =
        static_cast<uint64_t>(grouped_batch.device_resources.c_slab.bytes);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  auto& grouped_resources = grouped_batch.device_resources;
  const auto grouped_bucket_plan =
      make_grouped_dispatch_bucket_plan(args, plan, tasks, RNS8_BOUNDED_I64, matrix_prefix);
  const auto& grouped_descriptor =
      single_grouped_dispatch_descriptor(grouped_bucket_plan, "grouped bounded i64 execution");

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_pack_grouped_i64_task_inputs(
            grouped_descriptor,
            grouped_resources,
            grouped_batch.a_host_slab.data(),
            grouped_batch.b_host_slab.data(),
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_grouped_i64_task_inputs(grouped bounded i64)", status);
      }
      (void)task;
      return;
    }
    status = rns8_pack_i64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch A)", status);
    status = rns8_pack_i64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(
            grouped_descriptor,
            grouped_resources);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_grouped_rns_task_outputs(grouped bounded i64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch i64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_export_grouped_i64_task_outputs_to_host(
            grouped_descriptor,
            grouped_resources,
            bound,
            grouped_batch.c_host_slab.data(),
            grouped_batch.c_host_slab.size());
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_export_grouped_i64_task_outputs_to_host(grouped bounded i64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_export_i64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_BOUNDED_I64,
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
        RNS8_BOUNDED_I64,
        matrix_prefix,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  if (grouped_pack_gemm_export_enabled) {
    scatter_grouped_native_slab_to_outputs(args, grouped_batch.c_host_slab, batch_c, "grouped bounded i64 export");
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_u64_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch C");
  std::vector<std::vector<uint64_t>> batch_a(task_count, std::vector<uint64_t>(a_elements));
  std::vector<std::vector<uint64_t>> batch_b(task_count, std::vector<uint64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    fill_bounded_u64_inputs(args, batch_a[task_index], batch_b[task_index], rng);
  }

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, batch_a.front(), batch_b.front(), bound, result);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  const bool grouped_pack_gemm_export_enabled =
      grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_pack_gemm_export_enabled ? "device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h"
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
      args, plan, tasks, RNS8_BOUNDED_U64, matrix_prefix, "grouped bounded u64 setup");
  GroupedNativeRnsBatch<uint64_t> grouped_batch;
  if (grouped_pack_gemm_export_enabled) {
    initialize_grouped_native_rns_batch(
        args,
        plan,
        RNS8_BOUNDED_U64,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        "grouped bounded u64");
    result.grouped_dispatch_batched_export_enabled = true;
    result.grouped_dispatch_device_output_slab_bytes =
        static_cast<uint64_t>(grouped_batch.device_resources.c_slab.bytes);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  auto& grouped_resources = grouped_batch.device_resources;
  const auto grouped_bucket_plan =
      make_grouped_dispatch_bucket_plan(args, plan, tasks, RNS8_BOUNDED_U64, matrix_prefix);
  const auto& grouped_descriptor =
      single_grouped_dispatch_descriptor(grouped_bucket_plan, "grouped bounded u64 execution");

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_pack_grouped_u64_task_inputs(
            grouped_descriptor,
            grouped_resources,
            grouped_batch.a_host_slab.data(),
            grouped_batch.b_host_slab.data(),
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_grouped_u64_task_inputs(grouped bounded u64)", status);
      }
      (void)task;
      return;
    }
    status = rns8_pack_u64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch A)", status);
    status = rns8_pack_u64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(
            grouped_descriptor,
            grouped_resources);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_grouped_rns_task_outputs(grouped bounded u64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch u64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_export_grouped_u64_task_outputs_to_host(
            grouped_descriptor,
            grouped_resources,
            bound,
            grouped_batch.c_host_slab.data(),
            grouped_batch.c_host_slab.size());
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_export_grouped_u64_task_outputs_to_host(grouped bounded u64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_export_u64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        RNS8_BOUNDED_U64,
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
        RNS8_BOUNDED_U64,
        matrix_prefix,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  if (grouped_pack_gemm_export_enabled) {
    scatter_grouped_native_slab_to_outputs(args, grouped_batch.c_host_slab, batch_c, "grouped bounded u64 export");
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

