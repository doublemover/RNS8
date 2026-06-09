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
  const bool compact_d2h_export_candidate =
      args.export_variant == "compact-d2h-export-candidate" && args.output_ld_padding > 0;
  const bool tree_crt_export_candidate =
      args.export_variant == "tree-crt-export-candidate" && args.reconstruction_variant == "tree_crt_candidate";
  std::vector<uint64_t> compact_export_C(
      compact_d2h_export_candidate
          ? checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "compact exact-wide signed export")
          : 0,
      UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);
  if (args.sparse_a_4_to_2) {
    fill_sparse_a_4_to_2_bounded_i64_input(args, A, rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_amdgpu_builtin_dense_variant_backend_metadata(args, result);
  apply_rns_sparse_a_4_to_2_backend_metadata(args, result, 0);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_sparse_a_4_to_2 = rns_sparse_a_4_to_2_path(args, result);
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
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  rns8_result_cache* result_cache = nullptr;
  std::vector<rns8_dirty_region> result_cache_dirty_regions;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  if (use_sparse_a_4_to_2) {
    rns8_sparse_matrix_desc sparse_desc = sparse_a_4_to_2_rns_desc(args, matrix_prefix);
    status = rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_sparse_matrix(A)", status);
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  if (incremental_result_cache_runtime_candidate_requested(args, selected_backend)) {
    result_cache = create_benchmark_result_cache(ctx, plan);
    result_cache_dirty_regions.push_back(benchmark_incremental_result_cache_dirty_region(args));
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_sparse_a_4_to_2) {
      const std::vector<uint8_t> sparse_planes = make_sparse_a_4_to_2_rns_planes(args, matrix_prefix, A);
      status = rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, sparse_a_matrix, sparse_planes.data(), args.k, source_version);
      if (status != RNS8_SUCCESS) fail_status("rns8_pack_sparse_a_4_to_2_matrix_u8(A)", status);
      return;
    }
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
    const PackOperandTiming setup_pack_timing = pack_preused_inputs_timed(args, 1, pack_a_input, pack_b_input);
    uint64_t setup_runtime_cache_us = 0;
    if (should_probe_reusable_b_prepack_cache(args, selected_backend)) {
      const auto cache_start = std::chrono::steady_clock::now();
      if (maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache, result)) {
        result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
      }
      const auto cache_end = std::chrono::steady_clock::now();
      setup_runtime_cache_us = elapsed_us(cache_start, cache_end);
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    record_prepack_setup_timing(
        result, elapsed_us(prepack_start, prepack_end), setup_pack_timing, setup_runtime_cache_us);
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);
  const auto export_signed_limbs = [&](rns8_matrix* matrix, const char* label) {
    if (tree_crt_export_candidate && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      status = rns8::detail::hip_direct_export_exact_wide_signed_tree_crt_limbs_device(
          matrix->hip_device_id,
          matrix->hip_residues,
          &matrix->hip_export_buffer,
          &matrix->hip_export_bytes,
          &matrix->hip_status_buffer,
          &matrix->hip_status_bytes,
          args.m,
          args.n,
          matrix->prefix,
          C.data(),
          ldc,
          args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status(label, status);
      return;
    }
    if (compact_d2h_export_candidate) {
      status = rns8_export_exact_wide_signed_limbs(
          ctx, plan, matrix, compact_export_C.data(), args.n, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status(label, status);
      copy_compact_limb_output(
          compact_export_C, C, args.m, args.n, ldc, args.exact_wide_limb_count, "compact exact-wide signed export");
      return;
    }
    status = rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status(label, status);
  };

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    if (residue_chain_independent_final_export_requested(args)) {
      uint64_t pack_total_us = 0;
      PackOperandTiming pack_operand_timing{};
      uint64_t gemm_total_us = 0;
      uint64_t export_total_us = 0;
      begin_gpu_event_phase(collect_gpu_events);

      auto phase_start = std::chrono::steady_clock::now();
      const PackOperandTiming initial_pack_timing =
          pack_per_repeat_inputs_timed(args, source_version, pack_a_input, pack_b_input);
      pack_operand_timing.a_us += initial_pack_timing.a_us;
      pack_operand_timing.b_us += initial_pack_timing.b_us;
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
        export_signed_limbs(c_matrix, "rns8_export_exact_wide_signed_limbs(independent final-output chain)");
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
          const uint64_t intermediate_pack_us = elapsed_us(phase_start, phase_end);
          pack_operand_timing.a_us += intermediate_pack_us;
          pack_total_us += intermediate_pack_us;
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
        record_pack_operand_timing(samples, pack_operand_timing);
        samples->gemm_us.push_back(gemm_total_us);
        samples->export_us.push_back(export_total_us);
        samples->end_to_end_us.push_back(elapsed_us(repeat_start, repeat_end));
      }
      return;
    }
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
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
      const rns8_dirty_region* dirty_regions =
          result_cache ? result_cache_dirty_regions.data() : nullptr;
      const uint32_t dirty_region_count =
          result_cache ? static_cast<uint32_t>(result_cache_dirty_regions.size()) : 0;
      if (use_sparse_a_4_to_2) {
        status = rns8_gemm_rns_sparse_a(ctx, plan, sparse_a_matrix, b_matrix, out_matrix, workspace);
      } else {
        status = run_rns_gemm_with_optional_b_cache(
            ctx,
            plan,
            lhs_matrix,
            b_matrix,
            b_prepack_cache,
            out_matrix,
            workspace,
            result_cache,
            dirty_regions,
            dirty_region_count);
      }
      if (status != RNS8_SUCCESS) {
        fail_status(
            use_sparse_a_4_to_2
                ? "rns8_gemm_rns_sparse_a"
                : result_cache ? "rns8_gemm_rns_incremental"
                               : (b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns"),
            status);
      }
      final_output_matrix = out_matrix;
      lhs_matrix = out_matrix;
      out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
    }
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      if (result_cache) {
        collect_incremental_result_cache_gemm_gpu_events(args, result.gpu_events);
      } else {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      export_signed_limbs(final_output_matrix, "rns8_export_exact_wide_signed_limbs");
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
      record_pack_operand_timing(samples, pack_operand_timing);
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_full_exact_wide_pack_export_requested(args)) {
    run_hip_graph_replay_exact_wide_pack_gemm_export(
        args, result, plan, a_matrix, b_matrix, c_matrix, A, B, C, latest_output_matrix);
  } else if (hip_graph_replay_requested(args)) {
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
    record_allocation_after_repeats(result);
  }
  if (chain_residue_output) {
    export_signed_limbs(latest_output_matrix, "rns8_export_exact_wide_signed_limbs(final chain checksum)");
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");
  if (result_cache) {
    capture_incremental_result_cache_info(result, result_cache, result_cache_dirty_regions, true);
    pack_a_input(static_cast<uint64_t>(args.warmups) + static_cast<uint64_t>(args.repeats) + 1001u);
    const rns8_status stale_status =
        rns8_gemm_rns_incremental(ctx, plan, a_matrix, b_matrix, c_matrix, workspace, result_cache, nullptr, 0);
    if (stale_status == RNS8_SUCCESS) {
      fail_status("rns8_gemm_rns_incremental(expected stale rejection)", RNS8_INVALID_ARGUMENT);
    }
    capture_incremental_result_cache_stale_rejection(result, result_cache);
  }

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (result_cache) {
    status = rns8_destroy_result_cache(result_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_result_cache", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
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
  const bool compact_d2h_export_candidate =
      args.export_variant == "compact-d2h-export-candidate" && args.output_ld_padding > 0;
  const bool tree_crt_export_candidate =
      args.export_variant == "tree-crt-export-candidate" && args.reconstruction_variant == "tree_crt_candidate";
  std::vector<uint64_t> compact_export_C(
      compact_d2h_export_candidate
          ? checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "compact exact-wide unsigned export")
          : 0,
      UINT64_C(0x5a5a5a5a5a5a5a5a));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);
  if (args.sparse_a_4_to_2) {
    fill_sparse_a_4_to_2_bounded_u64_input(args, A, rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_amdgpu_builtin_dense_variant_backend_metadata(args, result);
  apply_rns_sparse_a_4_to_2_backend_metadata(args, result, 0);
  capture_packing_and_lowering_info(args, plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_sparse_a_4_to_2 = rns_sparse_a_4_to_2_path(args, result);
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
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  rns8_result_cache* result_cache = nullptr;
  std::vector<rns8_dirty_region> result_cache_dirty_regions;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  if (use_sparse_a_4_to_2) {
    rns8_sparse_matrix_desc sparse_desc = sparse_a_4_to_2_rns_desc(args, matrix_prefix);
    status = rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_sparse_matrix(A)", status);
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  if (incremental_result_cache_runtime_candidate_requested(args, selected_backend)) {
    result_cache = create_benchmark_result_cache(ctx, plan);
    result_cache_dirty_regions.push_back(benchmark_incremental_result_cache_dirty_region(args));
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_sparse_a_4_to_2) {
      const std::vector<uint8_t> sparse_planes = make_sparse_a_4_to_2_rns_planes(args, matrix_prefix, A);
      status = rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, sparse_a_matrix, sparse_planes.data(), args.k, source_version);
      if (status != RNS8_SUCCESS) fail_status("rns8_pack_sparse_a_4_to_2_matrix_u8(A)", status);
      return;
    }
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
    const PackOperandTiming setup_pack_timing = pack_preused_inputs_timed(args, 1, pack_a_input, pack_b_input);
    uint64_t setup_runtime_cache_us = 0;
    if (should_probe_reusable_b_prepack_cache(args, selected_backend)) {
      const auto cache_start = std::chrono::steady_clock::now();
      if (maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache, result)) {
        result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
      }
      const auto cache_end = std::chrono::steady_clock::now();
      setup_runtime_cache_us = elapsed_us(cache_start, cache_end);
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    record_prepack_setup_timing(
        result, elapsed_us(prepack_start, prepack_end), setup_pack_timing, setup_runtime_cache_us);
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);
  const auto export_unsigned_limbs = [&](rns8_matrix* matrix, const char* label) {
    if (tree_crt_export_candidate && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      status = rns8::detail::hip_direct_export_exact_wide_unsigned_tree_crt_limbs_device(
          matrix->hip_device_id,
          matrix->hip_residues,
          &matrix->hip_export_buffer,
          &matrix->hip_export_bytes,
          &matrix->hip_status_buffer,
          &matrix->hip_status_bytes,
          args.m,
          args.n,
          matrix->prefix,
          C.data(),
          ldc,
          args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status(label, status);
      return;
    }
    if (compact_d2h_export_candidate) {
      status = rns8_export_exact_wide_unsigned_limbs(
          ctx, plan, matrix, compact_export_C.data(), args.n, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status(label, status);
      copy_compact_limb_output(
          compact_export_C, C, args.m, args.n, ldc, args.exact_wide_limb_count, "compact exact-wide unsigned export");
      return;
    }
    status = rns8_export_exact_wide_unsigned_limbs(ctx, plan, matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status(label, status);
  };

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    if (residue_chain_independent_final_export_requested(args)) {
      uint64_t pack_total_us = 0;
      PackOperandTiming pack_operand_timing{};
      uint64_t gemm_total_us = 0;
      uint64_t export_total_us = 0;
      begin_gpu_event_phase(collect_gpu_events);

      auto phase_start = std::chrono::steady_clock::now();
      const PackOperandTiming initial_pack_timing =
          pack_per_repeat_inputs_timed(args, source_version, pack_a_input, pack_b_input);
      pack_operand_timing.a_us += initial_pack_timing.a_us;
      pack_operand_timing.b_us += initial_pack_timing.b_us;
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
        export_unsigned_limbs(c_matrix, "rns8_export_exact_wide_unsigned_limbs(independent final-output chain)");
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
          const uint64_t intermediate_pack_us = elapsed_us(phase_start, phase_end);
          pack_operand_timing.a_us += intermediate_pack_us;
          pack_total_us += intermediate_pack_us;
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
        record_pack_operand_timing(samples, pack_operand_timing);
        samples->gemm_us.push_back(gemm_total_us);
        samples->export_us.push_back(export_total_us);
        samples->end_to_end_us.push_back(elapsed_us(repeat_start, repeat_end));
      }
      return;
    }
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
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
      const rns8_dirty_region* dirty_regions =
          result_cache ? result_cache_dirty_regions.data() : nullptr;
      const uint32_t dirty_region_count =
          result_cache ? static_cast<uint32_t>(result_cache_dirty_regions.size()) : 0;
      if (use_sparse_a_4_to_2) {
        status = rns8_gemm_rns_sparse_a(ctx, plan, sparse_a_matrix, b_matrix, out_matrix, workspace);
      } else {
        status = run_rns_gemm_with_optional_b_cache(
            ctx,
            plan,
            lhs_matrix,
            b_matrix,
            b_prepack_cache,
            out_matrix,
            workspace,
            result_cache,
            dirty_regions,
            dirty_region_count);
      }
      if (status != RNS8_SUCCESS) {
        fail_status(
            use_sparse_a_4_to_2
                ? "rns8_gemm_rns_sparse_a"
                : result_cache ? "rns8_gemm_rns_incremental"
                               : (b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns"),
            status);
      }
      final_output_matrix = out_matrix;
      lhs_matrix = out_matrix;
      out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
    }
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      if (result_cache) {
        collect_incremental_result_cache_gemm_gpu_events(args, result.gpu_events);
      } else {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      export_unsigned_limbs(final_output_matrix, "rns8_export_exact_wide_unsigned_limbs");
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
      record_pack_operand_timing(samples, pack_operand_timing);
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_full_exact_wide_pack_export_requested(args)) {
    run_hip_graph_replay_exact_wide_pack_gemm_export(
        args, result, plan, a_matrix, b_matrix, c_matrix, A, B, C, latest_output_matrix);
  } else if (hip_graph_replay_requested(args)) {
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
    record_allocation_after_repeats(result);
  }
  if (chain_residue_output) {
    export_unsigned_limbs(latest_output_matrix, "rns8_export_exact_wide_unsigned_limbs(final chain checksum)");
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");
  if (result_cache) {
    capture_incremental_result_cache_info(result, result_cache, result_cache_dirty_regions, true);
    pack_a_input(static_cast<uint64_t>(args.warmups) + static_cast<uint64_t>(args.repeats) + 1001u);
    const rns8_status stale_status =
        rns8_gemm_rns_incremental(ctx, plan, a_matrix, b_matrix, c_matrix, workspace, result_cache, nullptr, 0);
    if (stale_status == RNS8_SUCCESS) {
      fail_status("rns8_gemm_rns_incremental(expected stale rejection)", RNS8_INVALID_ARGUMENT);
    }
    capture_incremental_result_cache_stale_rejection(result, result_cache);
  }

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (result_cache) {
    status = rns8_destroy_result_cache(result_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_result_cache", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
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
