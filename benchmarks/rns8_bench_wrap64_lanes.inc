BenchmarkResult run_wrap_u64_rocwmma_candidate(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("rocwmma-wrap64-candidate requires a HIP-enabled benchmark build");
#else
  (void)ctx;
  (void)bound;
  if (args.k > kWrap64RocwmmaCandidateMaxK) {
    usage_error("rocwmma-wrap64-candidate currently supports K <= 32768");
  }

  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = rng();
  for (auto& value : B) value = rng();

  BenchmarkResult result{};
  result.target_id = benchmark_target_id_for_context(ctx, RNS8_BACKEND_ROCWMMA);
  const auto plan_start = std::chrono::steady_clock::now();
  fill_wrap64_rocwmma_candidate_schedule(args, result);
  fill_wrap64_rocwmma_candidate_backend_info(args, result, wrap64_rocwmma_candidate_workspace_bytes(args));
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  result.schedule_query_us = 0;
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_ROCWMMA);

  const std::size_t a_limb_bytes = wrap64_compact_limb_bytes(args.m, args.k, "candidate A byte limbs");
  const std::size_t b_limb_bytes = wrap64_compact_limb_bytes(args.k, args.n, "candidate B byte limbs");
  const std::size_t c_limb_bytes = wrap64_compact_limb_bytes(args.m, args.n, "candidate C byte limbs");
  DeviceBuffer a_limbs;
  DeviceBuffer b_limbs;
  DeviceBuffer c_limbs;
  DeviceBuffer upload_buffer;
  DeviceBuffer export_buffer;
  upload_buffer.device_id = args.device_id;
  export_buffer.device_id = args.device_id;
  const auto alloc_start = std::chrono::steady_clock::now();
  a_limbs.allocate(args.device_id, a_limb_bytes, "hip_direct_allocate(wrap64 candidate A limbs)");
  b_limbs.allocate(args.device_id, b_limb_bytes, "hip_direct_allocate(wrap64 candidate B limbs)");
  c_limbs.allocate(args.device_id, c_limb_bytes, "hip_direct_allocate(wrap64 candidate C limbs)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  rns8_status status = RNS8_SUCCESS;
  const auto pack_a_input = [&](uint64_t) {
    status = rns8::detail::wrap64_hip_pack_u64_device(
        args.device_id, A.data(), &upload_buffer.ptr, &upload_buffer.bytes, a_limbs.ptr, args.m, args.k, args.k);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_pack_u64_device(A)", status);
  };
  const auto pack_b_input = [&](uint64_t) {
    status = rns8::detail::wrap64_hip_pack_u64_device(
        args.device_id, B.data(), &upload_buffer.ptr, &upload_buffer.bytes, b_limbs.ptr, args.k, args.n, args.n);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_pack_u64_device(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    PackOperandTiming pack_operand_timing{};
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, RNS8_BACKEND_ROCWMMA, result.gpu_events);
      }
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_operand_timing = pack_per_repeat_inputs_timed(args, 1, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, RNS8_BACKEND_ROCWMMA, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8::detail::rocwmma_wrap64_gemm_byte_limbs_candidate_device(
        args.device_id, a_limbs.ptr, b_limbs.ptr, c_limbs.ptr, args.m, args.n, args.k);
    if (status != RNS8_SUCCESS) fail_status("rocwmma_wrap64_gemm_byte_limbs_candidate_device", status);
    if (collect_gpu_events) {
      collect_wrap64_gemm_gpu_events(args, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8::detail::wrap64_hip_export_u64_device(
        args.device_id, c_limbs.ptr, &export_buffer.ptr, &export_buffer.bytes, args.m, args.n, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_export_u64_device(C)", status);
    if (collect_gpu_events) {
      collect_wrap64_export_gpu_events(result.gpu_events);
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
#endif
}

BenchmarkResult run_wrap_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.wrap64_rocwmma_candidate) {
    return run_wrap_u64_rocwmma_candidate(ctx, args, bound);
  }
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = rng();
  for (auto& value : B) value = rng();

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
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  capture_workspace_arena_info(workspace, result);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  if (hip_graph_replay_full_wrap64_pack_export_requested(args)) {
    run_hip_graph_replay_wrap64_pack_gemm_export(args, result, A, B, C, ldc);
    result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    return result;
  }

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
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
    PackOperandTiming pack_operand_timing{};
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
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
    status = rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_wrap_u64", status);
    if (collect_gpu_events) {
      collect_wrap64_gemm_gpu_events(args, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_wrap_u64", status);
    if (collect_gpu_events) {
      collect_wrap64_export_gpu_events(result.gpu_events);
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

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  record_allocation_after_repeats(result);
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}
