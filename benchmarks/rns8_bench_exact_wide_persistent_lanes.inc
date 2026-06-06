BenchmarkResult run_exact_wide_signed(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  if (grouped_task_executor_requested(args)) {
    return run_exact_wide_signed_host_api_batch(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(
      output_limb_elements(args, args.exact_wide_limb_count, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
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
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (should_probe_reusable_b_prepack_cache(args) &&
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
      auto phase_end = std::chrono::steady_clock::now();
      pack_total_us += elapsed_us(phase_start, phase_end);

      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        phase_start = std::chrono::steady_clock::now();
        status = run_rns_gemm_with_optional_b_cache(ctx, plan, a_matrix, b_matrix, nullptr, c_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_gemm_rns(exact-wide signed independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        gemm_total_us += elapsed_us(phase_start, phase_end);

        phase_start = std::chrono::steady_clock::now();
        status = rns8_export_exact_wide_signed_limbs(
            ctx, plan, c_matrix, C.data(), ldc, args.exact_wide_limb_count);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_export_exact_wide_signed_limbs(independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        export_total_us += elapsed_us(phase_start, phase_end);

        if (chain_index + 1 < args.residue_chain_length) {
          phase_start = std::chrono::steady_clock::now();
          status = pack_exact_wide_limbs_as_rns(
              ctx, a_matrix, C.data(), ldc, args.exact_wide_limb_count, true, source_version + chain_index + 2);
          if (status != RNS8_SUCCESS) {
            fail_status("pack_exact_wide_signed_limbs_as_rns(intermediate chain output)", status);
          }
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
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
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
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
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
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_exact_wide_signed_limbs(
          ctx, plan, final_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
    status = rns8_export_exact_wide_signed_limbs(
        ctx, plan, latest_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_unsigned(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  if (grouped_task_executor_requested(args)) {
    return run_exact_wide_unsigned_host_api_batch(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(
      output_limb_elements(args, args.exact_wide_limb_count, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
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
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
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
    if (should_probe_reusable_b_prepack_cache(args) &&
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
      auto phase_end = std::chrono::steady_clock::now();
      pack_total_us += elapsed_us(phase_start, phase_end);

      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        phase_start = std::chrono::steady_clock::now();
        status = run_rns_gemm_with_optional_b_cache(ctx, plan, a_matrix, b_matrix, nullptr, c_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_gemm_rns(exact-wide unsigned independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        gemm_total_us += elapsed_us(phase_start, phase_end);

        phase_start = std::chrono::steady_clock::now();
        status = rns8_export_exact_wide_unsigned_limbs(
            ctx, plan, c_matrix, C.data(), ldc, args.exact_wide_limb_count);
        if (status != RNS8_SUCCESS) {
          fail_status("rns8_export_exact_wide_unsigned_limbs(independent final-output chain)", status);
        }
        phase_end = std::chrono::steady_clock::now();
        export_total_us += elapsed_us(phase_start, phase_end);

        if (chain_index + 1 < args.residue_chain_length) {
          phase_start = std::chrono::steady_clock::now();
          status = pack_exact_wide_limbs_as_rns(
              ctx, a_matrix, C.data(), ldc, args.exact_wide_limb_count, false, source_version + chain_index + 2);
          if (status != RNS8_SUCCESS) {
            fail_status("pack_exact_wide_unsigned_limbs_as_rns(intermediate chain output)", status);
          }
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
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
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
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
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
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_exact_wide_unsigned_limbs(
          ctx, plan, final_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
    status = rns8_export_exact_wide_unsigned_limbs(
        ctx, plan, latest_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}
