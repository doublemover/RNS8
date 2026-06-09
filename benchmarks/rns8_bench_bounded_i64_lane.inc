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
  if (args.sparse_a_4_to_2) {
    fill_sparse_a_4_to_2_bounded_i64_input(args, A, rng);
  }

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
  apply_amdgpu_builtin_dense_variant_backend_metadata(args, result);
  capture_packing_and_lowering_info(args, plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_native_b_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_transient_backend_metadata(args, result, bound);
  apply_bounded_direct_hip_skinny_gemv_n1_backend_metadata(args, result, bound);
  apply_bounded_direct_hip_skinny_gemv_small_n_backend_metadata(args, result, bound);
  apply_rns_sparse_a_4_to_2_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_sparse_a_4_to_2 = rns_sparse_a_4_to_2_path(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_native_b_reuse_a = bounded_native_b_reuse_a_large_colpair_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_a = bounded_uniform_small_i8_ab_reuse_a_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  const bool use_uniform_small_i8_ab_transient = bounded_uniform_small_i8_ab_transient_path(args, result);
  const bool use_uniform_small_i8_ab_reuse = use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_reuse_b;
  const bool use_uniform_small_i8_ab_device_inputs =
      use_uniform_small_i8_ab_reuse || use_uniform_small_i8_ab_transient;
  const bool use_direct_hip_skinny_gemv_n1 = bounded_direct_hip_skinny_gemv_n1_path(args, result);
  const bool use_direct_hip_skinny_gemv_small_n =
      bounded_direct_hip_skinny_gemv_small_n_path(args, result);
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
  capture_workspace_arena_info(workspace, result);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_sparse_matrix* sparse_a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* a_stream_matrix = nullptr;
  rns8_matrix* c_stream_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  rns8_result_cache* result_cache = nullptr;
  std::vector<rns8_dirty_region> result_cache_dirty_regions;
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
          ? checked_bytes(A.size(), sizeof(int64_t), "native A")
          : 0;
  const std::size_t native_b_bytes =
      use_native_b_reuse_a ? checked_bytes(B.size(), sizeof(int64_t), "native B") : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A")
          : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B")
          : 0;
  if (use_sparse_a_4_to_2) {
    rns8_sparse_matrix_desc sparse_desc = sparse_a_4_to_2_rns_desc(args, matrix_prefix);
    status = rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_sparse_matrix(A)", status);
  } else if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded i64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded i64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded i64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (use_native_b_reuse_a) {
    native_b.allocate(args.device_id, native_b_bytes, "hip_direct_allocate(bounded i64 native B)");
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
  const bool skip_all_zero_direct_hip_input_pack =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend);
  const bool streaming_overlap_path =
      streaming_overlap_supported(args, result, selected_backend, ldc) && !use_native_a_reuse_b &&
      !use_native_b_reuse_a && !use_uniform_small_i8_ab_transient && !use_uniform_small_i8_ab_reuse_a &&
      !use_uniform_small_i8_ab_reuse_b && !skip_all_zero_direct_hip_input_pack;
  if (streaming_overlap_path) {
    status = rns8_create_matrix(ctx, &a_desc, &a_stream_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A streaming buffer)", status);
    status = rns8_create_matrix(ctx, &c_desc, &c_stream_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C streaming buffer)", status);
  }
  const bool use_result_cache_candidate =
      incremental_result_cache_runtime_candidate_requested(args, selected_backend) &&
      !use_sparse_a_4_to_2 && !use_native_a_reuse_b && !use_uniform_small_i8_ab_transient &&
      !use_native_b_reuse_a && !use_uniform_small_i8_ab_reuse_a && !use_uniform_small_i8_ab_reuse_b &&
      !streaming_overlap_path && !skip_all_zero_direct_hip_input_pack;
  if (use_result_cache_candidate) {
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
    if (use_native_b_reuse_a) {
      fail_status("rns8_pack_i64(B native-B reuse-A path)", RNS8_INVALID_ARGUMENT);
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
    const PackOperandTiming setup_pack_timing = pack_preused_inputs_timed(args, 1, pack_a_input, pack_b_input);
    uint64_t setup_runtime_cache_us = 0;
    if (!use_native_a_reuse_b && b_matrix && should_probe_reusable_b_prepack_cache(args, selected_backend)) {
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
    if (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_uniform_small_i8_ab_transient) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_operand_timing = pack_per_repeat_inputs_timed(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_uniform_small_i8_ab_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      const auto pack_b_start = std::chrono::steady_clock::now();
      pack_b_input(source_version);
      const auto pack_b_end = std::chrono::steady_clock::now();
      pack_operand_timing.b_us = elapsed_us(pack_b_start, pack_b_end);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_b_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      const auto pack_b_start = std::chrono::steady_clock::now();
      status = run_timed_status_operation(bounded_native_b_reuse_a_large_colpair_pack_h2d_label(), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_b.ptr, B.data(), native_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 native B)", status);
      const auto pack_b_end = std::chrono::steady_clock::now();
      pack_operand_timing.b_us = elapsed_us(pack_b_start, pack_b_end);
      if (collect_gpu_events) {
        collect_bounded_native_b_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      const auto pack_a_start = std::chrono::steady_clock::now();
      status = run_timed_status_operation(bounded_native_a_reuse_b_pack_h2d_label(args), [&]() {
        return use_uniform_small_i8_ab_reuse_b
            ? rns8::detail::hip_direct_copy_host_to_device(
                  args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes)
            : rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 native A)", status);
      const auto pack_a_end = std::chrono::steady_clock::now();
      pack_operand_timing.a_us = elapsed_us(pack_a_start, pack_a_end);
      if (collect_gpu_events) {
        collect_bounded_native_a_pack_gpu_events(args, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_operand_timing = pack_per_repeat_inputs_timed(args, source_version, pack_a_input, pack_b_input);
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
      } else if (use_native_a_reuse_b && bounded_native_a_reuse_b_large_colpair(args)) {
        status = rns8::detail::hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_matrix(
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
          fail_status("hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_matrix", status);
        }
      } else if (use_native_b_reuse_a) {
        status = rns8::detail::hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix(
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
          fail_status("hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix", status);
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
        } else if (use_native_b_reuse_a) {
          collect_bounded_native_b_reuse_a_gemm_gpu_events(result.gpu_events);
        } else {
          collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
        }
      }
    } else if (use_direct_hip_skinny_gemv_n1) {
      status = rns8::detail::hip_direct_gemv_n1_rns_prefix9_matrix(
          args.device_id,
          a_matrix,
          b_matrix,
          c_matrix,
          args.m,
          args.k,
          source_version);
      if (status != RNS8_SUCCESS) fail_status("hip_direct_gemv_n1_rns_prefix9_matrix", status);
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_bounded_direct_hip_skinny_gemv_n1_gpu_events(result.gpu_events);
      }
    } else if (use_direct_hip_skinny_gemv_small_n) {
      status = rns8::detail::hip_direct_gemv_small_n_rns_prefix9_matrix(
          args.device_id,
          a_matrix,
          b_matrix,
          c_matrix,
          args.m,
          args.n,
          args.k,
          source_version);
      if (status != RNS8_SUCCESS) fail_status("hip_direct_gemv_small_n_rns_prefix9_matrix", status);
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_bounded_direct_hip_skinny_gemv_small_n_gpu_events(result.gpu_events);
      }
    } else {
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
      if (collect_gpu_events) {
        if (result_cache) {
          collect_incremental_result_cache_gemm_gpu_events(args, result.gpu_events);
        } else {
          collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
        }
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
      record_pack_operand_timing(samples, pack_operand_timing);
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_full_bounded_pack_export_requested(args)) {
    run_hip_graph_replay_bounded_pack_gemm_export(
        args, result, plan, a_matrix, b_matrix, c_matrix, A, B, C, bound, latest_output_matrix);
  } else if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else if (streaming_overlap_path) {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_bounded_streaming_overlap_pipeline(
          args,
          result,
          plan,
          a_matrix,
          a_stream_matrix,
          b_matrix,
          c_matrix,
          c_stream_matrix,
          A,
          C,
          bound,
          static_cast<uint64_t>(r) + 1,
          1,
          false);
    }
    record_allocation_after_warmups(result);
    run_bounded_streaming_overlap_pipeline(
        args,
        result,
        plan,
        a_matrix,
        a_stream_matrix,
        b_matrix,
        c_matrix,
        c_stream_matrix,
        A,
        C,
        bound,
        static_cast<uint64_t>(args.warmups) + 1,
        args.repeats,
        result.gpu_events.requested);
    record_allocation_after_repeats(result);
    latest_output_matrix = c_matrix;
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
    status = rns8_export_i64(ctx, plan, latest_output_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(final chain checksum)", status);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
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
  if (c_stream_matrix) {
    rns8_destroy_matrix(c_stream_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  if (b_matrix) {
    rns8_destroy_matrix(b_matrix);
  }
  if (a_stream_matrix) {
    rns8_destroy_matrix(a_stream_matrix);
  }
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

