uint8_t sample_finite_u8_input(std::mt19937_64& rng, const Args& args) {
  const uint32_t modulus = args.finite_modulus == 256 ? 256u : static_cast<uint32_t>(args.finite_modulus);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  switch (args.input_profile) {
    case InputProfile::FiniteBinary: {
      std::uniform_int_distribution<uint32_t> dist(0, 1);
      return static_cast<uint8_t>(dist(rng) % modulus);
    }
    case InputProfile::FiniteSparse: {
      std::uniform_int_distribution<uint32_t> zero_gate(0, 9);
      if (zero_gate(rng) < 9) {
        return 0;
      }
      std::uniform_int_distribution<uint32_t> nonzero(1, high);
      return static_cast<uint8_t>(nonzero(rng));
    }
    case InputProfile::FiniteLowHamming: {
      static constexpr uint32_t kChoices[] = {0u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
      std::uniform_int_distribution<std::size_t> index(0, (sizeof(kChoices) / sizeof(kChoices[0])) - 1);
      return static_cast<uint8_t>(kChoices[index(rng)] % modulus);
    }
    case InputProfile::FiniteSmallCentered: {
      std::uniform_int_distribution<int32_t> dist(-2, 2);
      const int32_t value = dist(rng);
      const int32_t mod = static_cast<int32_t>(modulus);
      return static_cast<uint8_t>((value % mod + mod) % mod);
    }
    case InputProfile::FiniteFullUniform:
    case InputProfile::UniformSmall: {
      std::uniform_int_distribution<uint32_t> dist(0, high);
      return static_cast<uint8_t>(dist(rng));
    }
    case InputProfile::AdaptiveBands:
      usage_error("finite-u8 lanes do not support adaptive-bands input profile");
  }
  return 0;
}

void fill_finite_u8_input(std::mt19937_64& rng, const Args& args, std::vector<uint8_t>& values) {
  for (auto& value : values) {
    value = sample_finite_u8_input(rng, args);
  }
}

uint8_t sample_nonzero_finite_u8_input(std::mt19937_64& rng, const Args& args) {
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> nonzero(1, high);
  return static_cast<uint8_t>(nonzero(rng));
}

void fill_sparse_a_4_to_2_finite_u8_input(std::mt19937_64& rng, const Args& args, std::vector<uint8_t>& values) {
  if (args.k % 4 != 0 || values.size() != checked_elements(args.m, args.k, "sparse A")) {
    usage_error("internal sparse-A 4:2 fixture shape mismatch");
  }
  std::fill(values.begin(), values.end(), uint8_t{0});
  std::uniform_int_distribution<uint32_t> first_dist(0, 3);
  std::uniform_int_distribution<uint32_t> second_offset_dist(1, 3);
  for (int64_t row = 0; row < args.m; ++row) {
    for (int64_t group = 0; group < args.k / 4; ++group) {
      const uint32_t first = first_dist(rng);
      const uint32_t second = (first + second_offset_dist(rng)) % 4u;
      const uint32_t lo = std::min(first, second);
      const uint32_t hi = std::max(first, second);
      const std::size_t base = static_cast<std::size_t>(row * args.k + group * 4);
      values[base + lo] = sample_nonzero_finite_u8_input(rng, args);
      values[base + hi] = sample_nonzero_finite_u8_input(rng, args);
    }
  }
}

BenchmarkResult run_finite_u8_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
  fill_finite_u8_input(rng, args, A);
  fill_finite_u8_input(rng, args, B);

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
  record_allocation_after_repeats(result);
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_finite_u8_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch finite A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch finite B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch finite C");
  std::vector<std::vector<uint8_t>> batch_a(task_count, std::vector<uint8_t>(a_elements));
  std::vector<std::vector<uint8_t>> batch_b(task_count, std::vector<uint8_t>(b_elements));
  std::vector<std::vector<uint8_t>> batch_c(
      task_count, std::vector<uint8_t>(c_elements, 0x5au));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    fill_finite_u8_input(rng, args, batch_a[task_index]);
    fill_finite_u8_input(rng, args, batch_b[task_index]);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch finite)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_amdgpu_builtin_dense_variant_backend_metadata(args, result);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  const bool grouped_pack_gemm_export_enabled =
      grouped_dispatch_requested(args) && selected_backend == RNS8_BACKEND_HIP_DIRECT;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_pack_gemm_export_enabled ? "device_grouped_pack_gemm_and_finite_export_kernel_batched_d2h"
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
      args, plan, tasks, c_semantics(args.semantics), matrix_prefix, "grouped finite-u8 setup");
  GroupedNativeRnsBatch<uint8_t> grouped_batch;
  if (grouped_pack_gemm_export_enabled) {
    initialize_grouped_native_rns_batch(
        args,
        plan,
        c_semantics(args.semantics),
        matrix_prefix,
        tasks,
        batch_a,
        batch_b,
        a_elements,
        b_elements,
        grouped_batch,
        "grouped finite-u8");
    result.grouped_dispatch_batched_export_enabled = true;
    result.grouped_dispatch_device_output_slab_bytes =
        static_cast<uint64_t>(grouped_batch.device_resources.c_slab.bytes);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  auto& grouped_resources = grouped_batch.device_resources;
  const auto grouped_bucket_plan =
      make_grouped_dispatch_bucket_plan(args, plan, tasks, c_semantics(args.semantics), matrix_prefix);
  const auto& grouped_descriptor =
      single_grouped_dispatch_descriptor(grouped_bucket_plan, "grouped finite execution");

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_pack_grouped_finite_u8_task_inputs(
            grouped_descriptor,
            grouped_resources,
            grouped_batch.a_host_slab.data(),
            grouped_batch.b_host_slab.data(),
            args.k,
            args.n,
            static_cast<uint16_t>(args.finite_modulus),
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_pack_grouped_finite_u8_task_inputs(grouped finite)", status);
      }
      (void)task;
      return OptionalPackOperandTiming{};
    }
    OptionalPackOperandTiming timing{};
    timing.available = true;
    const auto pack_a_start = std::chrono::steady_clock::now();
    status = rns8_pack_finite_u8(
        ctx, task.a, args.finite_modulus, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch A)", status);
    const auto pack_a_end = std::chrono::steady_clock::now();
    timing.timing.a_us = elapsed_us(pack_a_start, pack_a_end);
    const auto pack_b_start = std::chrono::steady_clock::now();
    status = rns8_pack_finite_u8(
        ctx, task.b, args.finite_modulus, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch B)", status);
    const auto pack_b_end = std::chrono::steady_clock::now();
    timing.timing.b_us = elapsed_us(pack_b_start, pack_b_end);
    return timing;
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_gemm_grouped_finite_u8_task_outputs(
            grouped_descriptor,
            grouped_resources,
            static_cast<uint16_t>(args.finite_modulus));
        if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_grouped_finite_u8_task_outputs(grouped finite)", status);
      }
      (void)task;
      return;
    }
    status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8(host API batch)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_pack_gemm_export_enabled) {
      if (task_index == 0) {
        status = rns8::detail::hip_direct_export_grouped_finite_u8_task_outputs_to_host(
            grouped_descriptor,
            grouped_resources,
            static_cast<uint16_t>(args.finite_modulus),
            grouped_batch.c_host_slab.data(),
            grouped_batch.c_host_slab.size());
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_export_grouped_finite_u8_task_outputs_to_host(grouped finite)", status);
        }
      }
      (void)task;
      return;
    }
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
  record_allocation_after_repeats(result);
  if (grouped_pack_gemm_export_enabled) {
    scatter_grouped_native_slab_to_outputs(args, grouped_batch.c_host_slab, batch_c, "grouped finite-u8 export");
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
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
  if (args.sparse_a_4_to_2) {
    fill_sparse_a_4_to_2_finite_u8_input(rng, args, A);
  } else {
    fill_finite_u8_input(rng, args, A);
  }
  fill_finite_u8_input(rng, args, B);

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_amdgpu_builtin_dense_variant_backend_metadata(args, result);
  capture_packing_and_lowering_info(args, plan, result);
  apply_finite_native_a_reuse_b_backend_metadata(args, result);
  apply_finite_sparse_a_4_to_2_backend_metadata(args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = finite_native_a_reuse_b_path(args, result);
  const bool use_sparse_a_4_to_2 = finite_sparse_a_4_to_2_path(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  capture_workspace_arena_info(workspace, result);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_sparse_matrix* sparse_a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_result_cache* result_cache = nullptr;
  std::vector<rns8_dirty_region> result_cache_dirty_regions;
  DeviceBuffer native_a;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes = use_native_a_reuse_b ? checked_bytes(A.size(), sizeof(uint8_t), "native A") : 0;
  if (use_sparse_a_4_to_2) {
    rns8_sparse_matrix_desc sparse_desc{};
    sparse_desc.struct_size = sizeof(sparse_desc);
    sparse_desc.abi_version = RNS8_ABI_VERSION;
    sparse_desc.semantics = c_semantics(args.semantics);
    sparse_desc.bound_kind = RNS8_BOUND_NONE;
    sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
    sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
    sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
    sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
    sparse_desc.rows = args.m;
    sparse_desc.expanded_k = args.k;
    sparse_desc.group_size = 4;
    sparse_desc.nonzeros_per_group = 2;
    sparse_desc.finite_modulus = args.finite_modulus;
    status = rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_sparse_matrix(A)", status);
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(finite native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const bool use_result_cache_candidate =
      incremental_result_cache_runtime_candidate_requested(args, selected_backend) && !use_native_a_reuse_b;
  if (use_result_cache_candidate) {
    result_cache = create_benchmark_result_cache(ctx, plan);
    result_cache_dirty_regions.push_back(benchmark_incremental_result_cache_dirty_region(args));
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_sparse_a_4_to_2) {
      status = rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, sparse_a_matrix, A.data(), args.k, source_version);
      if (status != RNS8_SUCCESS) fail_status("rns8_pack_sparse_a_4_to_2_matrix_u8(A)", status);
      return;
    }
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
    const PackOperandTiming setup_pack_timing = pack_preused_inputs_timed(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    record_prepack_setup_timing(result, elapsed_us(prepack_start, prepack_end), setup_pack_timing, 0);
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    PackOperandTiming pack_operand_timing{};
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      const auto pack_a_start = std::chrono::steady_clock::now();
      status = run_timed_status_operation("finite_native_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(finite native A)", status);
      const auto pack_a_end = std::chrono::steady_clock::now();
      pack_operand_timing.a_us = elapsed_us(pack_a_start, pack_a_end);
      if (collect_gpu_events) {
        collect_finite_native_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_operand_timing = pack_per_repeat_inputs_timed(args, source_version, pack_a_input, pack_b_input);
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
      if (use_sparse_a_4_to_2) {
        status = rns8_gemm_finite_u8_sparse_a(
            ctx, plan, args.finite_modulus, sparse_a_matrix, b_matrix, c_matrix, workspace);
      } else if (result_cache) {
        status = rns8_gemm_finite_u8_incremental(
            ctx,
            plan,
            args.finite_modulus,
            a_matrix,
            b_matrix,
            c_matrix,
            workspace,
            result_cache,
            result_cache_dirty_regions.data(),
            static_cast<uint32_t>(result_cache_dirty_regions.size()));
      } else {
        status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, a_matrix, b_matrix, c_matrix, workspace);
      }
      if (status != RNS8_SUCCESS) {
        fail_status(
            use_sparse_a_4_to_2 ? "rns8_gemm_finite_u8_sparse_a"
                                : result_cache ? "rns8_gemm_finite_u8_incremental" : "rns8_gemm_finite_u8",
            status);
      }
      if (collect_gpu_events) {
        if (result_cache) {
          collect_incremental_result_cache_gemm_gpu_events(args, result.gpu_events);
        } else {
          collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
        }
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
      record_pack_operand_timing(samples, pack_operand_timing);
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_full_finite_pack_export_requested(args)) {
    run_hip_graph_replay_finite_pack_gemm_export(args, result, a_matrix, b_matrix, c_matrix, A, B, C);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
    record_allocation_after_repeats(result);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  if (result_cache) {
    capture_incremental_result_cache_info(result, result_cache, result_cache_dirty_regions, true);
    pack_a_input(static_cast<uint64_t>(args.warmups) + static_cast<uint64_t>(args.repeats) + 1001u);
    const rns8_status stale_status = rns8_gemm_finite_u8_incremental(
        ctx, plan, args.finite_modulus, a_matrix, b_matrix, c_matrix, workspace, result_cache, nullptr, 0);
    if (stale_status == RNS8_SUCCESS) {
      fail_status("rns8_gemm_finite_u8_incremental(expected stale rejection)", RNS8_INVALID_ARGUMENT);
    }
    capture_incremental_result_cache_stale_rejection(result, result_cache);
  }

  if (result_cache) {
    status = rns8_destroy_result_cache(result_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_result_cache", status);
  }
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  if (sparse_a_matrix) {
    rns8_destroy_sparse_matrix(sparse_a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}
