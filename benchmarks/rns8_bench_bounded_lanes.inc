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
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  const bool grouped_pack_gemm_enabled =
      grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_pack_gemm_enabled ? "device_grouped_pack_gemm_host_exports" : "host_phase_loop_per_task_export";
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  GroupedNativeRnsBatch<int64_t> grouped_batch;
  if (grouped_pack_gemm_enabled) {
    initialize_grouped_native_rns_batch(
        args,
        RNS8_BOUNDED_I64,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        "grouped bounded i64");
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_gemm_enabled) {
      if (task_index == 0) {
        status = hip_direct_pack_grouped_native_matrices_device(
            grouped_batch.a_device_slab.device_id,
            grouped_batch.a_host_slab.data(),
            grouped_batch.a_device_slab.ptr,
            grouped_batch.a_device_slab.bytes,
            grouped_batch.a_matrices.data(),
            task_count,
            RNS8_BOUNDED_I64,
            grouped_batch.a_residue_ptrs.ptr,
            args.m,
            args.k,
            args.k,
            matrix_prefix,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_i64_grouped_matrices_device(grouped bounded A)", status);
        status = hip_direct_pack_grouped_native_matrices_device(
            grouped_batch.b_device_slab.device_id,
            grouped_batch.b_host_slab.data(),
            grouped_batch.b_device_slab.ptr,
            grouped_batch.b_device_slab.bytes,
            grouped_batch.b_matrices.data(),
            task_count,
            RNS8_BOUNDED_I64,
            grouped_batch.b_residue_ptrs.ptr,
            args.k,
            args.n,
            args.n,
            matrix_prefix,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_i64_grouped_matrices_device(grouped bounded B)", status);
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
    if (grouped_pack_gemm_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_rns_grouped_matrices_device(
            grouped_batch.a_device_slab.device_id,
            grouped_batch.a_matrices.data(),
            grouped_batch.b_matrices.data(),
            grouped_batch.c_matrices.data(),
            task_count,
            RNS8_BOUNDED_I64,
            grouped_batch.a_residue_ptrs.ptr,
            grouped_batch.b_residue_ptrs.ptr,
            grouped_batch.c_residue_ptrs.ptr,
            args.m,
            args.n,
            args.k,
            matrix_prefix);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_rns_grouped_matrices_device(grouped bounded i64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch i64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_i64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
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
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  const bool grouped_pack_gemm_enabled =
      grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_pack_gemm_enabled ? "device_grouped_pack_gemm_host_exports" : "host_phase_loop_per_task_export";
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  GroupedNativeRnsBatch<uint64_t> grouped_batch;
  if (grouped_pack_gemm_enabled) {
    initialize_grouped_native_rns_batch(
        args,
        RNS8_BOUNDED_U64,
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        "grouped bounded u64");
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_gemm_enabled) {
      if (task_index == 0) {
        status = hip_direct_pack_grouped_native_matrices_device(
            grouped_batch.a_device_slab.device_id,
            grouped_batch.a_host_slab.data(),
            grouped_batch.a_device_slab.ptr,
            grouped_batch.a_device_slab.bytes,
            grouped_batch.a_matrices.data(),
            task_count,
            RNS8_BOUNDED_U64,
            grouped_batch.a_residue_ptrs.ptr,
            args.m,
            args.k,
            args.k,
            matrix_prefix,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_u64_grouped_matrices_device(grouped bounded A)", status);
        status = hip_direct_pack_grouped_native_matrices_device(
            grouped_batch.b_device_slab.device_id,
            grouped_batch.b_host_slab.data(),
            grouped_batch.b_device_slab.ptr,
            grouped_batch.b_device_slab.bytes,
            grouped_batch.b_matrices.data(),
            task_count,
            RNS8_BOUNDED_U64,
            grouped_batch.b_residue_ptrs.ptr,
            args.k,
            args.n,
            args.n,
            matrix_prefix,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_u64_grouped_matrices_device(grouped bounded B)", status);
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
    if (grouped_pack_gemm_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_rns_grouped_matrices_device(
            grouped_batch.a_device_slab.device_id,
            grouped_batch.a_matrices.data(),
            grouped_batch.b_matrices.data(),
            grouped_batch.c_matrices.data(),
            task_count,
            RNS8_BOUNDED_U64,
            grouped_batch.a_residue_ptrs.ptr,
            grouped_batch.b_residue_ptrs.ptr,
            grouped_batch.c_residue_ptrs.ptr,
            args.m,
            args.n,
            args.k,
            matrix_prefix);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_rns_grouped_matrices_device(grouped bounded u64)", status);
        }
      }
      (void)task;
      return;
    }
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch u64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_u64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
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

BenchmarkResult run_bounded_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_bounded_i64_oneshot(ctx, args, bound);
  }
  if (grouped_task_executor_requested(args)) {
    return run_bounded_i64_host_api_batch(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_i64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_native_b_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_transient_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_a = bounded_uniform_small_i8_ab_reuse_a_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  const bool use_uniform_small_i8_ab_transient = bounded_uniform_small_i8_ab_transient_path(args, result);
  const bool use_uniform_small_i8_ab_reuse = use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_reuse_b;
  const bool use_uniform_small_i8_ab_device_inputs =
      use_uniform_small_i8_ab_reuse || use_uniform_small_i8_ab_transient;
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a = make_uniform_small_i8_inputs(A);
    uniform_small_b = make_uniform_small_i8_inputs(B);
  }
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  DeviceBuffer native_a;
  DeviceBuffer uniform_small_a_device;
  DeviceBuffer uniform_small_b_device;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes =
      use_native_a_reuse_b && !use_uniform_small_i8_ab_reuse_b
          ? checked_bytes(A.size(), sizeof(int64_t), "native A")
          : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A")
          : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B")
          : 0;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded i64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded i64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded i64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (!use_uniform_small_i8_ab_device_inputs) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  const bool skip_all_zero_direct_hip_input_pack =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_transient) {
      (void)source_version;
      status = run_timed_status_operation("bounded_uniform_small_i8_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 uniform-small A)", status);
      return;
    }
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_i64(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
        return rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
      });
    } else {
      status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_device_inputs) {
      (void)source_version;
      status = run_timed_status_operation(bounded_native_a_reuse_b_b_h2d_label(args), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_b_device.ptr, uniform_small_b.data(), uniform_small_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 uniform-small B)", status);
      return;
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
        return rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
      });
    } else {
      status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (!use_native_a_reuse_b && b_matrix && should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    if (residue_chain_independent_final_export_requested(args)) {
      uint64_t pack_total_us = 0;
      uint64_t gemm_total_us = 0;
      uint64_t export_total_us = 0;
      begin_gpu_event_phase(collect_gpu_events);

      auto phase_start = std::chrono::steady_clock::now();
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
      auto phase_end = std::chrono::steady_clock::now();
      pack_total_us += elapsed_us(phase_start, phase_end);

      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        phase_start = std::chrono::steady_clock::now();
        status = run_rns_gemm_with_optional_b_cache(ctx, plan, a_matrix, b_matrix, nullptr, c_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_gemm_rns(independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        gemm_total_us += elapsed_us(phase_start, phase_end);

        phase_start = std::chrono::steady_clock::now();
        status = rns8_export_i64(ctx, plan, c_matrix, C.data(), ldc);
        if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(independent final-output chain)", status);
        phase_end = std::chrono::steady_clock::now();
        export_total_us += elapsed_us(phase_start, phase_end);

        if (chain_index + 1 < args.residue_chain_length) {
          phase_start = std::chrono::steady_clock::now();
          status = rns8_pack_i64(ctx, a_matrix, C.data(), ldc, source_version + chain_index + 2);
          if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(intermediate chain output)", status);
          phase_end = std::chrono::steady_clock::now();
          pack_total_us += elapsed_us(phase_start, phase_end);
        }
      }
      latest_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      const auto repeat_end = std::chrono::steady_clock::now();
      if (samples) {
        samples->pack_us.push_back(pack_total_us);
        samples->gemm_us.push_back(gemm_total_us);
        samples->export_us.push_back(export_total_us);
        samples->end_to_end_us.push_back(elapsed_us(repeat_start, repeat_end));
      }
      return;
    }
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_uniform_small_i8_ab_transient) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_uniform_small_i8_ab_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_b_input(source_version);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_a_reuse_b_pack_h2d_label(args), [&]() {
        return use_uniform_small_i8_ab_reuse_b
            ? rns8::detail::hip_direct_copy_host_to_device(
                  args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes)
            : rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 native A)", status);
      if (collect_gpu_events) {
        collect_bounded_native_a_pack_gpu_events(args, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    if (use_uniform_small_i8_ab_transient || use_uniform_small_i8_ab_reuse_a || use_native_a_reuse_b) {
      if (use_uniform_small_i8_ab_transient) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_a) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_b) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix", status);
        }
      } else {
        status = rns8::detail::hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix", status);
      }
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        if (use_uniform_small_i8_ab_transient) {
          collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(result.gpu_events);
        } else if (use_uniform_small_i8_ab_reuse_a) {
          collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(result.gpu_events);
        } else {
          collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
        }
      }
    } else {
      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        status = run_rns_gemm_with_optional_b_cache(
            ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
        }
        final_output_matrix = out_matrix;
        lhs_matrix = out_matrix;
        out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
      }
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    latest_output_matrix = final_output_matrix;
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_i64(ctx, plan, final_output_matrix, C.data(), ldc);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_i64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(
          (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) ? 0
                                                                                  : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_i64(ctx, plan, latest_output_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(final chain checksum)", status);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  if (b_matrix) {
    rns8_destroy_matrix(b_matrix);
  }
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_bounded_u64_oneshot(ctx, args, bound);
  }
  if (grouped_task_executor_requested(args)) {
    return run_bounded_u64_host_api_batch(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_u64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_native_b_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_transient_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_native_b_reuse_a = bounded_native_b_reuse_a_u64_large_colpair_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_a = bounded_uniform_small_i8_ab_reuse_a_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  const bool use_uniform_small_i8_ab_transient = bounded_uniform_small_i8_ab_transient_path(args, result);
  const bool use_uniform_small_i8_ab_reuse = use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_reuse_b;
  const bool use_uniform_small_i8_ab_device_inputs =
      use_uniform_small_i8_ab_reuse || use_uniform_small_i8_ab_transient;
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a = make_uniform_small_i8_inputs(A);
    uniform_small_b = make_uniform_small_i8_inputs(B);
  }
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  DeviceBuffer native_a;
  DeviceBuffer native_b;
  DeviceBuffer uniform_small_a_device;
  DeviceBuffer uniform_small_b_device;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes =
      use_native_a_reuse_b && !use_uniform_small_i8_ab_reuse_b
          ? checked_bytes(A.size(), sizeof(uint64_t), "native A")
          : 0;
  const std::size_t native_b_bytes =
      use_native_b_reuse_a ? checked_bytes(B.size(), sizeof(uint64_t), "native B") : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A")
          : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B")
          : 0;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded u64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded u64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded u64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (use_native_b_reuse_a) {
    native_b.allocate(args.device_id, native_b_bytes, "hip_direct_allocate(bounded u64 native B)");
  } else if (!use_uniform_small_i8_ab_device_inputs) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  const bool skip_all_zero_direct_hip_input_pack =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_transient) {
      (void)source_version;
      status = run_timed_status_operation("bounded_uniform_small_i8_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 uniform-small A)", status);
      return;
    }
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_u64(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
        return rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
      });
    } else {
      status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_device_inputs) {
      (void)source_version;
      status = run_timed_status_operation(bounded_native_a_reuse_b_b_h2d_label(args), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_b_device.ptr, uniform_small_b.data(), uniform_small_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 uniform-small B)", status);
      return;
    }
    if (use_native_b_reuse_a) {
      fail_status("rns8_pack_u64(B native-B reuse-A path)", RNS8_INVALID_ARGUMENT);
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
        return rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
      });
    } else {
      status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (!use_native_a_reuse_b && b_matrix && should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    if (residue_chain_independent_final_export_requested(args)) {
      uint64_t pack_total_us = 0;
      uint64_t gemm_total_us = 0;
      uint64_t export_total_us = 0;
      begin_gpu_event_phase(collect_gpu_events);

      auto phase_start = std::chrono::steady_clock::now();
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
      auto phase_end = std::chrono::steady_clock::now();
      pack_total_us += elapsed_us(phase_start, phase_end);

      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        phase_start = std::chrono::steady_clock::now();
        status = run_rns_gemm_with_optional_b_cache(ctx, plan, a_matrix, b_matrix, nullptr, c_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_gemm_rns(independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        gemm_total_us += elapsed_us(phase_start, phase_end);

        phase_start = std::chrono::steady_clock::now();
        status = rns8_export_u64(ctx, plan, c_matrix, C.data(), ldc);
        if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(independent final-output chain)", status);
        phase_end = std::chrono::steady_clock::now();
        export_total_us += elapsed_us(phase_start, phase_end);

        if (chain_index + 1 < args.residue_chain_length) {
          phase_start = std::chrono::steady_clock::now();
          status = rns8_pack_u64(ctx, a_matrix, C.data(), ldc, source_version + chain_index + 2);
          if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(intermediate chain output)", status);
          phase_end = std::chrono::steady_clock::now();
          pack_total_us += elapsed_us(phase_start, phase_end);
        }
      }
      latest_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      const auto repeat_end = std::chrono::steady_clock::now();
      if (samples) {
        samples->pack_us.push_back(pack_total_us);
        samples->gemm_us.push_back(gemm_total_us);
        samples->export_us.push_back(export_total_us);
        samples->end_to_end_us.push_back(elapsed_us(repeat_start, repeat_end));
      }
      return;
    }
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_uniform_small_i8_ab_transient) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_uniform_small_i8_ab_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_b_input(source_version);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_b_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label(), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_b.ptr, B.data(), native_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 native B)", status);
      if (collect_gpu_events) {
        collect_bounded_native_b_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_a_reuse_b_pack_h2d_label(args), [&]() {
        return use_uniform_small_i8_ab_reuse_b
            ? rns8::detail::hip_direct_copy_host_to_device(
                  args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes)
            : rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 native A)", status);
      if (collect_gpu_events) {
        collect_bounded_native_a_pack_gpu_events(args, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    if (use_uniform_small_i8_ab_transient || use_uniform_small_i8_ab_reuse_a ||
        use_native_a_reuse_b || use_native_b_reuse_a) {
      if (use_uniform_small_i8_ab_transient) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_a) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_b) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix", status);
        }
      } else if (use_native_a_reuse_b && bounded_native_a_reuse_b_u64_large_colpair(args)) {
        status = rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix", status);
        }
      } else if (use_native_b_reuse_a) {
        status = rns8::detail::hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
            args.device_id,
            a_matrix,
            native_b.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix", status);
        }
      } else {
        status = rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix", status);
      }
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        if (use_uniform_small_i8_ab_transient) {
          collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(result.gpu_events);
        } else if (use_uniform_small_i8_ab_reuse_a) {
          collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(result.gpu_events);
        } else if (use_native_b_reuse_a) {
          collect_bounded_native_b_reuse_a_gemm_gpu_events(result.gpu_events);
        } else {
          collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
        }
      }
    } else {
      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        status = run_rns_gemm_with_optional_b_cache(
            ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
        }
        final_output_matrix = out_matrix;
        lhs_matrix = out_matrix;
        out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
      }
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    latest_output_matrix = final_output_matrix;
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_u64(ctx, plan, final_output_matrix, C.data(), ldc);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_u64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(
          (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) ? 0
                                                                                  : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_u64(ctx, plan, latest_output_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(final chain checksum)", status);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  if (b_matrix) {
    rns8_destroy_matrix(b_matrix);
  }
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}
