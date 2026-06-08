BenchmarkResult run_vector_alu_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("hip-vector-alu-int64-baseline requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
  std::vector<int64_t> compact_C(ldc == args.n ? 0 : checked_elements(args.m, args.n, "compact C"));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(int64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(int64_t), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "device C"), sizeof(int64_t), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  DeviceBuffer d_status;
  const auto alloc_start = std::chrono::steady_clock::now();
  d_a.allocate(args.device_id, a_bytes, "hip_direct_allocate(vector A)");
  d_b.allocate(args.device_id, b_bytes, "hip_direct_allocate(vector B)");
  d_c.allocate(args.device_id, c_bytes, "hip_direct_allocate(vector C)");
  d_status.allocate(args.device_id, status_bytes, "hip_direct_allocate(vector status)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  fill_vector_alu_backend_info(args, result, vector_alu_workspace_bytes<int64_t>(args));
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    uint64_t pack_a_us = 0;
    uint64_t pack_b_us = 0;
    begin_gpu_event_phase(collect_gpu_events);
    const auto pack_a_start = std::chrono::steady_clock::now();
    rns8_status status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_a.ptr, A.data(), a_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector A)", status);
    const auto pack_a_end = std::chrono::steady_clock::now();
    pack_a_us = elapsed_us(pack_a_start, pack_a_end);
    const auto pack_b_start = std::chrono::steady_clock::now();
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_b.ptr, B.data(), b_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector B)", status);
    const auto pack_b_end = std::chrono::steady_clock::now();
    pack_b_us = elapsed_us(pack_b_start, pack_b_end);
    if (collect_gpu_events) {
      collect_pack_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_status_memset", [&]() {
      return rns8::detail::hip_direct_zero(args.device_id, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_zero(vector status)", status);
    const char* kernel_label = vector_to_rns_chain_kernel_label(args);
    const int kernel_status = rns8::detail::run_timed_device_code(kernel_label, [&]() {
      return rns8_bench_vector_i64_gemm_device(
          args.device_id,
          static_cast<const int64_t*>(d_a.ptr),
          static_cast<const int64_t*>(d_b.ptr),
          static_cast<int64_t*>(d_c.ptr),
          static_cast<uint32_t*>(d_status.ptr),
          args.m,
          args.n,
          args.k);
    });
    if (kernel_status != 0) fail_hip_runtime("rns8_bench_vector_i64_gemm_device", kernel_status);
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    uint32_t range_status = 0;
    status = run_timed_status_operation("vector_alu_status_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, &range_status, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector status)", status);
    if (range_status != 0) {
      fail_status("hip-vector-alu-int64 range check", RNS8_RANGE_ERROR);
    }
    status = run_timed_status_operation("vector_alu_output_d2h", [&]() {
      void* destination = ldc == args.n ? static_cast<void*>(C.data()) : static_cast<void*>(compact_C.data());
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, destination, d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (ldc != args.n) {
      copy_compact_to_output(compact_C, C, args.m, args.n, ldc, "vector C");
    }
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
      record_pack_operand_timing(samples, pack_a_us, pack_b_us);
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

BenchmarkResult run_vector_alu_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("hip-vector-alu-int64-baseline requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  std::vector<uint64_t> compact_C(ldc == args.n ? 0 : checked_elements(args.m, args.n, "compact C"));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(uint64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(uint64_t), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "device C"), sizeof(uint64_t), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  DeviceBuffer d_status;
  const auto alloc_start = std::chrono::steady_clock::now();
  d_a.allocate(args.device_id, a_bytes, "hip_direct_allocate(vector A)");
  d_b.allocate(args.device_id, b_bytes, "hip_direct_allocate(vector B)");
  d_c.allocate(args.device_id, c_bytes, "hip_direct_allocate(vector C)");
  d_status.allocate(args.device_id, status_bytes, "hip_direct_allocate(vector status)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  fill_vector_alu_backend_info(args, result, vector_alu_workspace_bytes<uint64_t>(args));
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    uint64_t pack_a_us = 0;
    uint64_t pack_b_us = 0;
    begin_gpu_event_phase(collect_gpu_events);
    const auto pack_a_start = std::chrono::steady_clock::now();
    rns8_status status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_a.ptr, A.data(), a_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector A)", status);
    const auto pack_a_end = std::chrono::steady_clock::now();
    pack_a_us = elapsed_us(pack_a_start, pack_a_end);
    const auto pack_b_start = std::chrono::steady_clock::now();
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_b.ptr, B.data(), b_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector B)", status);
    const auto pack_b_end = std::chrono::steady_clock::now();
    pack_b_us = elapsed_us(pack_b_start, pack_b_end);
    if (collect_gpu_events) {
      collect_pack_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_status_memset", [&]() {
      return rns8::detail::hip_direct_zero(args.device_id, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_zero(vector status)", status);
    const char* kernel_label = vector_to_rns_chain_kernel_label(args);
    const int kernel_status = rns8::detail::run_timed_device_code(kernel_label, [&]() {
      return rns8_bench_vector_u64_gemm_device(
          args.device_id,
          static_cast<const uint64_t*>(d_a.ptr),
          static_cast<const uint64_t*>(d_b.ptr),
          static_cast<uint64_t*>(d_c.ptr),
          static_cast<uint32_t*>(d_status.ptr),
          args.m,
          args.n,
          args.k);
    });
    if (kernel_status != 0) fail_hip_runtime("rns8_bench_vector_u64_gemm_device", kernel_status);
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    uint32_t range_status = 0;
    status = run_timed_status_operation("vector_alu_status_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, &range_status, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector status)", status);
    if (range_status != 0) {
      fail_status("hip-vector-alu-int64 range check", RNS8_RANGE_ERROR);
    }
    status = run_timed_status_operation("vector_alu_output_d2h", [&]() {
      void* destination = ldc == args.n ? static_cast<void*>(C.data()) : static_cast<void*>(compact_C.data());
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, destination, d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (ldc != args.n) {
      copy_compact_to_output(compact_C, C, args.m, args.n, ldc, "vector C");
    }
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
      record_pack_operand_timing(samples, pack_a_us, pack_b_us);
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
