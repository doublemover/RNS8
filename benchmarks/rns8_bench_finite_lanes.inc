BenchmarkResult run_finite_u8_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
  for (auto& value : A) value = static_cast<uint8_t>(dist(rng));
  for (auto& value : B) value = static_cast<uint8_t>(dist(rng));

  auto desc = gemm_desc(args, 0);
  BenchmarkResult result = initialize_bounded_oneshot_result(ctx, args, 0, desc);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_status status = RNS8_SUCCESS;
    if (args.semantics == BenchSemantics::FiniteFieldU8) {
      status = rns8_gemm_finite_field_u8_oneshot(
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    } else {
      status = rns8_gemm_finite_ring_u8_oneshot(
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_finite_oneshot_gpu_events(result.gpu_events);
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

BenchmarkResult run_finite_u8_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch finite A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch finite B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch finite C");
  std::vector<std::vector<uint8_t>> batch_a(task_count, std::vector<uint8_t>(a_elements));
  std::vector<std::vector<uint8_t>> batch_b(task_count, std::vector<uint8_t>(b_elements));
  std::vector<std::vector<uint8_t>> batch_c(
      task_count, std::vector<uint8_t>(c_elements, 0x5au));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = static_cast<uint8_t>(dist(rng));
    for (auto& value : batch_b[task_index]) value = static_cast<uint8_t>(dist(rng));
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch finite)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  enforce_grouped_dispatch_task_setup_contract(
      args, plan, tasks, c_semantics(args.semantics), matrix_prefix, "grouped finite-u8 setup");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_finite_u8(
        ctx, task.a, args.finite_modulus, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch A)", status);
    status = rns8_pack_finite_u8(
        ctx, task.b, args.finite_modulus, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8(host API batch)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_finite_u8(
        ctx, plan, args.finite_modulus, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_finite_u8(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        plan,
        tasks,
        c_semantics(args.semantics),
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
        c_semantics(args.semantics),
        matrix_prefix,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch finite C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_finite_u8(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_finite_u8_oneshot(ctx, args, bound);
  }
  if (grouped_task_executor_requested(args)) {
    return run_finite_u8_host_api_batch(ctx, args, bound);
  }
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
  for (auto& value : A) value = static_cast<uint8_t>(dist(rng));
  for (auto& value : B) value = static_cast<uint8_t>(dist(rng));

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  apply_finite_native_a_reuse_b_backend_metadata(args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = finite_native_a_reuse_b_path(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  DeviceBuffer native_a;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes = use_native_a_reuse_b ? checked_bytes(A.size(), sizeof(uint8_t), "native A") : 0;
  if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(finite native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_finite_u8(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    status = rns8_pack_finite_u8(ctx, a_matrix, args.finite_modulus, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_finite_u8(ctx, b_matrix, args.finite_modulus, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation("finite_native_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(finite native A)", status);
      if (collect_gpu_events) {
        collect_finite_native_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    if (use_native_a_reuse_b) {
      status = rns8::detail::hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
          args.device_id,
          native_a.ptr,
          b_matrix,
          c_matrix,
          args.m,
          args.n,
          args.k,
          args.k,
          args.finite_modulus,
          source_version);
      if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_finite_u8_native_a_resident_b_matrix", status);
      if (collect_gpu_events) {
        collect_finite_native_a_gemm_gpu_events(result.gpu_events);
      }
    } else {
      status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, a_matrix, b_matrix, c_matrix, workspace);
      if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8", status);
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
      }
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_finite_u8(ctx, plan, args.finite_modulus, c_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_finite_u8", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}
