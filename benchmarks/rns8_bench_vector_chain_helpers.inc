template <typename T>
void fill_vector_to_rns_chain_inputs(
    const Args& args,
    std::vector<T>& A,
    std::vector<T>& B,
    std::vector<T>& D,
    std::mt19937_64& rng) {
  if constexpr (std::is_same_v<T, int64_t>) {
    fill_bounded_i64_inputs(args, A, B, rng);
    std::uniform_int_distribution<int64_t> dist(-16, 16);
    for (auto& value : D) {
      value = dist(rng);
    }
  } else {
    fill_bounded_u64_inputs(args, A, B, rng);
    std::uniform_int_distribution<uint64_t> dist(0, 16);
    for (auto& value : D) {
      value = dist(rng);
    }
  }
}

template <typename T>
rns8_status pack_vector_to_rns_chain_operand(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const T* data,
    int64_t ld,
    uint64_t source_version) {
  if constexpr (std::is_same_v<T, int64_t>) {
    return rns8_pack_i64(ctx, matrix, data, ld, source_version);
  } else {
    return rns8_pack_u64(ctx, matrix, data, ld, source_version);
  }
}

template <typename T>
rns8_status export_vector_to_rns_chain_output(
    rns8_context* ctx,
    rns8_plan* plan,
    rns8_matrix* matrix,
    T* data,
    int64_t ld) {
  if constexpr (std::is_same_v<T, int64_t>) {
    return rns8_export_i64(ctx, plan, matrix, data, ld);
  } else {
    return rns8_export_u64(ctx, plan, matrix, data, ld);
  }
}

template <typename T>
BenchmarkResult run_vector_to_rns_chain(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t final_bound,
    const char* label) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)vector_ctx;
  (void)direct_ctx;
  (void)args;
  (void)final_bound;
  (void)label;
  usage_error("--vector-to-rns-chain requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<T> A(checked_elements(args.m, args.k, "vector chain A"));
  std::vector<T> B(checked_elements(args.k, args.n, "vector chain B"));
  std::vector<T> D(checked_elements(args.n, args.n, "vector chain direct B"));
  std::vector<T> produced_host(checked_elements(args.m, args.n, "vector chain produced host"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<T> C(output_elements(args, "vector chain C"), static_cast<T>(0x5a));
  fill_vector_to_rns_chain_inputs(args, A, B, D, rng);

  Args vector_args = args;
  vector_args.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  vector_args.vector_to_rns_chain = false;
  Args direct_args = args;
  direct_args.backend = RNS8_BACKEND_HIP_DIRECT;
  direct_args.k = args.n;
  direct_args.vector_to_rns_chain = false;

  const uint64_t producer_bound = vector_to_rns_chain_producer_bound(args);
  auto vector_desc = gemm_desc(vector_args, producer_bound);
  auto direct_desc = gemm_desc(direct_args, final_bound);

  BenchmarkResult result{};
  result.static_bound = final_bound;
  result.effective_bound = final_bound;
  result.effective_bound_available = true;

  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* vector_plan = nullptr;
  rns8_status status = rns8_create_plan(vector_ctx, &vector_desc, &vector_plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-to-RNS producer)", status);
  rns8_plan* direct_plan = nullptr;
  status = rns8_create_plan(direct_ctx, &direct_desc, &direct_plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-to-RNS consumer)", status);
  capture_schedule_info(direct_plan, result);
  capture_backend_info(direct_plan, result);
  capture_export_plan_info(direct_args, direct_plan, result);
  enforce_per_tile_capture_contract(direct_args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* vector_workspace = nullptr;
  status = rns8_create_workspace(vector_ctx, vector_plan, &vector_workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(vector-to-RNS producer)", status);
  rns8_workspace* direct_workspace = nullptr;
  status = rns8_create_workspace(direct_ctx, direct_plan, &direct_workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(vector-to-RNS consumer)", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* vector_a = nullptr;
  rns8_matrix* vector_b = nullptr;
  rns8_matrix* vector_c = nullptr;
  rns8_matrix* direct_a = nullptr;
  rns8_matrix* direct_b = nullptr;
  rns8_matrix* direct_c = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t direct_matrix_prefix = selected_execution_prefix(args, result);
  auto vector_a_desc = matrix_desc(args.m, args.k, vector_args);
  auto vector_b_desc = matrix_desc(args.k, args.n, vector_args);
  auto vector_c_desc = matrix_desc(args.m, args.n, vector_args);
  auto direct_a_desc = matrix_desc(args.m, args.n, direct_args, direct_matrix_prefix);
  auto direct_b_desc = matrix_desc(args.n, args.n, direct_args, direct_matrix_prefix);
  auto direct_c_desc = matrix_desc(args.m, args.n, direct_args, direct_matrix_prefix);
  status = rns8_create_matrix(vector_ctx, &vector_a_desc, &vector_a);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer A)", status);
  status = rns8_create_matrix(vector_ctx, &vector_b_desc, &vector_b);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer B)", status);
  status = rns8_create_matrix(vector_ctx, &vector_c_desc, &vector_c);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer C)", status);
  status = rns8_create_matrix(direct_ctx, &direct_a_desc, &direct_a);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct A)", status);
  status = rns8_create_matrix(direct_ctx, &direct_b_desc, &direct_b);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct B)", status);
  status = rns8_create_matrix(direct_ctx, &direct_c_desc, &direct_c);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  if (args.reuse_packed_b) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    status = pack_vector_to_rns_chain_operand(direct_ctx, direct_b, D.data(), args.n, 1);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain reused direct B)", status);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return pack_vector_to_rns_chain_operand(vector_ctx, vector_a, A.data(), args.k, source_version);
    });
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain producer A)", status);
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return pack_vector_to_rns_chain_operand(vector_ctx, vector_b, B.data(), args.n, source_version);
    });
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain producer B)", status);
    if (!args.reuse_packed_b) {
      status = pack_vector_to_rns_chain_operand(direct_ctx, direct_b, D.data(), args.n, source_version);
      if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain direct B)", status);
    }
    if (collect_gpu_events) {
      collect_vector_to_rns_chain_pack_gpu_events(args, result.gpu_events, args.reuse_packed_b);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_gemm_rns(vector_ctx, vector_plan, vector_a, vector_b, vector_c, vector_workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(vector chain producer)", status);
    if (vector_to_rns_chain_host_repack_control_requested(args)) {
      status = export_vector_to_rns_chain_output(vector_ctx, vector_plan, vector_c, produced_host.data(), args.n);
      if (status != RNS8_SUCCESS) fail_status("rns8_export(vector chain produced host control)", status);
      status = run_timed_status_operation("vector_to_rns_host_repack_a", [&]() {
        return pack_vector_to_rns_chain_operand(
            direct_ctx, direct_a, produced_host.data(), args.n, source_version);
      });
      if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain host-repacked direct A)", status);
    } else {
      status = rns8::detail::materialize_native_matrix_as_direct_rns(direct_ctx, direct_plan, vector_c, direct_a);
      if (status != RNS8_SUCCESS) fail_status("materialize_native_matrix_as_direct_rns(vector chain)", status);
    }
    status = rns8_gemm_rns(direct_ctx, direct_plan, direct_a, direct_b, direct_c, direct_workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(vector chain consumer)", status);
    if (collect_gpu_events) {
      if (vector_to_rns_chain_host_repack_control_requested(args)) {
        collect_vector_to_rns_chain_host_repack_control_gemm_gpu_events(args, result.gpu_events);
      } else {
        collect_vector_to_rns_chain_gemm_gpu_events(args, result.gpu_events);
      }
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = export_vector_to_rns_chain_output(direct_ctx, direct_plan, direct_c, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export(vector chain direct output)", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_DIRECT, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, label);

  rns8_destroy_matrix(direct_c);
  rns8_destroy_matrix(direct_b);
  rns8_destroy_matrix(direct_a);
  rns8_destroy_matrix(vector_c);
  rns8_destroy_matrix(vector_b);
  rns8_destroy_matrix(vector_a);
  rns8_destroy_workspace(direct_workspace);
  rns8_destroy_workspace(vector_workspace);
  rns8_destroy_plan(direct_plan);
  rns8_destroy_plan(vector_plan);
  return result;
#endif
}

BenchmarkResult run_vector_to_rns_chain_i64(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t bound) {
  return run_vector_to_rns_chain<int64_t>(vector_ctx, direct_ctx, args, bound, "vector chain C");
}

BenchmarkResult run_vector_to_rns_chain_u64(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t bound) {
  return run_vector_to_rns_chain<uint64_t>(vector_ctx, direct_ctx, args, bound, "vector chain C");
}

BenchmarkResult initialize_bounded_oneshot_result(
    rns8_context* ctx,
    const Args& args,
    uint64_t bound,
    const rns8_gemm_desc& desc,
    BenchmarkResult result = {}) {
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(oneshot metadata)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(args, plan, result);
  apply_bounded_oneshot_backend_metadata(args, result);
  apply_finite_oneshot_backend_metadata(args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  status = rns8_destroy_plan(plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_destroy_plan(oneshot metadata)", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  result.matrix_alloc_us = 0;
  (void)bound;
  return result;
}

