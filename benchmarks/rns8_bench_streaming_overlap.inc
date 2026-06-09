#if RNS8_CONFIGURED_HIP_ENABLED
extern "C" int rns8_hip_direct_pack_i64_device_on_stream(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix,
    void* stream);
extern "C" int rns8_hip_direct_pack_u64_device_on_stream(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix,
    void* stream);
extern "C" int rns8_hip_direct_export_i64_device_on_stream(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status,
    void* stream);
extern "C" int rns8_hip_direct_export_u64_device_on_stream(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status,
    void* stream);
extern "C" int rns8_hip_direct_export_exact_wide_signed_limbs_device_on_stream(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status,
    void* stream);
extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status,
    void* stream);
extern "C" int rns8_hip_direct_pack_u8_modulus_device_on_stream(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus,
    void* stream);
extern "C" int rns8_hip_direct_finite_ring_gemm_i8_device_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block,
    void* stream);
extern "C" int rns8_hip_direct_export_u8_modulus_device_on_stream(
    const int8_t* d_residues,
    uint8_t* d_dst,
    int rows,
    int cols,
    int ld,
    int modulus,
    void* stream);
extern "C" int rns8_wrap64_hip_pack_u64_device_on_stream(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    void* stream);
extern "C" int rns8_wrap64_hip_gemm_byte_limbs_device_on_stream(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k,
    void* stream);
extern "C" int rns8_wrap64_hip_export_u64_device_on_stream(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols,
    void* stream);
#endif

bool streaming_overlap_supported(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    int64_t output_ld) {
  return args.streaming_overlap && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && args.bound_mode == BoundMode::Global &&
         args.reuse_packed_inputs && args.reuse_packed_b && !args.reuse_packed_a &&
         args.residue_chain_length == 1 && !args.oneshot && !args.native_to_rns_bridge &&
         !args.vector_to_rns_chain && !args.residue_channel_fusion && output_ld == args.n &&
         result.schedule_info_available && result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active && !result.schedule_info.adaptive_skip_active;
}

#if RNS8_CONFIGURED_HIP_ENABLED
void fail_streaming_hip(const char* label, hipError_t status) {
  if (status != hipSuccess) {
    std::cerr << hip_graph_error_text(label, static_cast<int>(status)) << "\n";
    fail_status(label, RNS8_BACKEND_FAILURE);
  }
}

template <typename Fn>
hipError_t enqueue_timed_stream_operation(const char* label, hipStream_t stream, Fn&& fn) {
  if (!rns8::detail::hip_direct_timing_enabled()) {
    return fn();
  }
  rns8::detail::hip_unique_event_pair events;
  hipError_t status = events.create();
  if (status != hipSuccess) {
    return fn();
  }
  status = events.record_start(stream);
  if (status != hipSuccess) {
    return fn();
  }
  const hipError_t op_status = fn();
  if (op_status != hipSuccess) {
    return op_status;
  }
  status = events.record_stop(stream);
  if (status != hipSuccess) {
    return status;
  }
  rns8::detail::hip_direct_timing_record_pending_event(label, events.release_start(), events.release_stop());
  return hipSuccess;
}

template <typename HostT>
hipError_t enqueue_streaming_pack_kernel(
    const HostT* device_src,
    rns8_matrix* matrix,
    const Args& args,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    hipStream_t stream) {
  if constexpr (std::is_same_v<HostT, int64_t>) {
    const int code = rns8_hip_direct_pack_i64_device_on_stream(
        device_src,
        static_cast<int8_t*>(matrix->hip_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix),
        stream);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  } else {
    const int code = rns8_hip_direct_pack_u64_device_on_stream(
        device_src,
        static_cast<int8_t*>(matrix->hip_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix),
        stream);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  }
}

template <typename HostT>
hipError_t enqueue_streaming_export_kernel(
    const rns8_matrix* matrix,
    HostT* device_dst,
    int* device_status,
    const Args& args,
    uint32_t prefix,
    uint64_t bound,
    hipStream_t stream) {
  if constexpr (std::is_same_v<HostT, int64_t>) {
    const int code = rns8_hip_direct_export_i64_device_on_stream(
        static_cast<const int8_t*>(matrix->hip_residues),
        device_dst,
        static_cast<int>(args.m),
        static_cast<int>(args.n),
        static_cast<int>(prefix),
        bound,
        device_status,
        stream);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  } else {
    const int code = rns8_hip_direct_export_u64_device_on_stream(
        static_cast<const int8_t*>(matrix->hip_residues),
        device_dst,
        static_cast<int>(args.m),
        static_cast<int>(args.n),
        static_cast<int>(prefix),
        bound,
        device_status,
        stream);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  }
}

template <typename HostT>
void enqueue_streaming_pack(
    const Args& args,
    rns8_matrix* matrix,
    const HostT* pinned_src,
    std::size_t source_bytes,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t source_version,
    hipStream_t stream) {
  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, source_bytes, &matrix->hip_upload_buffer, &matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(streaming A)", status);
  }
  fail_streaming_hip(
      "streaming pack_h2d",
      enqueue_timed_stream_operation("pack_h2d", stream, [&]() {
        return hipMemcpyAsync(matrix->hip_upload_buffer, pinned_src, source_bytes, hipMemcpyHostToDevice, stream);
      }));
  fail_streaming_hip(
      "streaming pack_kernel",
      enqueue_timed_stream_operation("pack_kernel", stream, [&]() {
        return enqueue_streaming_pack_kernel(
            static_cast<const HostT*>(matrix->hip_upload_buffer), matrix, args, rows, cols, ld, prefix, stream);
      }));
  rns8::detail::api::mark_device_residues_current(*matrix);
  matrix->source_version = source_version;
}

template <typename HostT>
void enqueue_streaming_export(
    const Args& args,
    rns8_matrix* matrix,
    HostT* pinned_dst,
    int* pinned_status,
    std::size_t output_bytes,
    uint32_t prefix,
    uint64_t bound,
    hipStream_t stream) {
  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, output_bytes, &matrix->hip_export_buffer, &matrix->hip_export_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(streaming export)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, sizeof(int), &matrix->hip_status_buffer, &matrix->hip_status_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(streaming status)", status);
  }
  fail_streaming_hip(
      "streaming crt_export_status_memset",
      enqueue_timed_stream_operation("crt_export_status_memset", stream, [&]() {
        return hipMemsetAsync(matrix->hip_status_buffer, 0, sizeof(int), stream);
      }));
  fail_streaming_hip(
      "streaming crt_export_kernel",
      enqueue_timed_stream_operation("crt_export_kernel", stream, [&]() {
        return enqueue_streaming_export_kernel(
            matrix,
            static_cast<HostT*>(matrix->hip_export_buffer),
            static_cast<int*>(matrix->hip_status_buffer),
            args,
            prefix,
            bound,
            stream);
      }));
  fail_streaming_hip(
      "streaming crt_export_status_d2h",
      enqueue_timed_stream_operation("crt_export_status_d2h", stream, [&]() {
        return hipMemcpyAsync(pinned_status, matrix->hip_status_buffer, sizeof(int), hipMemcpyDeviceToHost, stream);
      }));
  fail_streaming_hip(
      "streaming crt_export_d2h",
      enqueue_timed_stream_operation("crt_export_d2h", stream, [&]() {
        return hipMemcpyAsync(pinned_dst, matrix->hip_export_buffer, output_bytes, hipMemcpyDeviceToHost, stream);
      }));
}

std::vector<double> streaming_event_values(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label,
    uint32_t expected_count,
    GpuEventSamples& events) {
  std::vector<double> values = event_label_values(samples, label);
  if (values.size() != expected_count) {
    add_unavailable_reason(
        events,
        std::string("streaming overlap expected ") + std::to_string(expected_count) + " event samples for " + label +
            " but saw " + std::to_string(values.size()));
  }
  return values;
}

uint64_t rounded_event_us(double value) {
  return value <= 0.0 ? 0 : static_cast<uint64_t>(value + 0.5);
}

void fill_streaming_overlap_event_results(BenchmarkResult& result, uint32_t repeats) {
  rns8::detail::hip_direct_timing_flush_pending_events();
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  auto& events = result.gpu_events;
  const auto pack_h2d = streaming_event_values(samples, "pack_h2d", repeats, events);
  const auto pack_kernel = streaming_event_values(samples, "pack_kernel", repeats, events);
  const auto gemm = streaming_event_values(samples, "rns_gemm_kernel_group", repeats, events);
  const auto status_memset = streaming_event_values(samples, "crt_export_status_memset", repeats, events);
  const auto export_kernel = streaming_event_values(samples, "crt_export_kernel", repeats, events);
  const auto status_d2h = streaming_event_values(samples, "crt_export_status_d2h", repeats, events);
  const auto output_d2h = streaming_event_values(samples, "crt_export_d2h", repeats, events);
  if (!events.complete) {
    return;
  }
  for (uint32_t index = 0; index < repeats; ++index) {
    const double pack_total = pack_h2d[index] + pack_kernel[index];
    const double export_total = status_memset[index] + export_kernel[index] + status_d2h[index] + output_d2h[index];
    events.timings_us["pack_h2d"].push_back(pack_h2d[index]);
    events.timings_us["pack_kernel"].push_back(pack_kernel[index]);
    events.timings_us["pack"].push_back(pack_total);
    events.timings_us["rns_gemm_kernel_group"].push_back(gemm[index]);
    events.timings_us["rns_gemm"].push_back(gemm[index]);
    events.timings_us["crt_export_status_memset"].push_back(status_memset[index]);
    events.timings_us["crt_export_kernel"].push_back(export_kernel[index]);
    events.timings_us["crt_export_status_d2h"].push_back(status_d2h[index]);
    events.timings_us["crt_export_d2h"].push_back(output_d2h[index]);
    events.timings_us["crt_export"].push_back(export_total);
    result.samples.pack_us.push_back(rounded_event_us(pack_total));
    result.samples.gemm_us.push_back(rounded_event_us(gemm[index]));
    result.samples.export_us.push_back(rounded_event_us(export_total));
    result.samples.end_to_end_us.push_back(result.streaming_overlap_per_repeat_pipeline_us);
  }
}

template <typename HostT>
void run_bounded_streaming_overlap_pipeline(
    const Args& args,
    BenchmarkResult& result,
    const rns8_plan* plan,
    rns8_matrix* a0,
    rns8_matrix* a1,
    rns8_matrix* b,
    rns8_matrix* c0,
    rns8_matrix* c1,
    const std::vector<HostT>& A,
    std::vector<HostT>& C,
    uint64_t bound,
    uint64_t first_source_version,
    uint32_t repeat_count,
    bool collect_gpu_events) {
  if (repeat_count == 0) {
    return;
  }
  const std::size_t source_bytes = checked_bytes(A.size(), sizeof(HostT), "streaming A input");
  const std::size_t output_bytes =
      checked_bytes(checked_elements(args.m, args.n, "streaming C output"), sizeof(HostT), "streaming C output");
  rns8_matrix* a_buffers[2] = {a0, a1};
  rns8_matrix* c_buffers[2] = {c0, c1};
  rns8::detail::hip_pinned_host_buffer pinned_a[2];
  rns8::detail::hip_pinned_host_buffer pinned_c[2];
  std::vector<rns8::detail::hip_pinned_host_buffer> pinned_status(repeat_count);
  for (uint32_t buffer = 0; buffer < 2; ++buffer) {
    fail_streaming_hip("hipHostMalloc(streaming A)", pinned_a[buffer].ensure(args.device_id, source_bytes));
    fail_streaming_hip("hipHostMalloc(streaming C)", pinned_c[buffer].ensure(args.device_id, output_bytes));
    std::memcpy(pinned_a[buffer].get(), A.data(), source_bytes);
  }
  for (uint32_t repeat = 0; repeat < repeat_count; ++repeat) {
    fail_streaming_hip("hipHostMalloc(streaming status)", pinned_status[repeat].ensure(args.device_id, sizeof(int)));
    *static_cast<int*>(pinned_status[repeat].get()) = 0;
  }

  rns8::detail::hip_unique_stream pack_stream;
  rns8::detail::hip_unique_stream gemm_stream;
  rns8::detail::hip_unique_stream export_stream;
  fail_streaming_hip("hipStreamCreate(pack)", pack_stream.create_non_blocking());
  fail_streaming_hip("hipStreamCreate(gemm)", gemm_stream.create_non_blocking());
  fail_streaming_hip("hipStreamCreate(export)", export_stream.create_non_blocking());

  rns8::detail::hip_unique_event pack_done[2];
  rns8::detail::hip_unique_event gemm_done[2];
  rns8::detail::hip_unique_event export_done[2];
  for (uint32_t buffer = 0; buffer < 2; ++buffer) {
    fail_streaming_hip("hipEventCreate(pack_done)", pack_done[buffer].create());
    fail_streaming_hip("hipEventCreate(gemm_done)", gemm_done[buffer].create());
    fail_streaming_hip("hipEventCreate(export_done)", export_done[buffer].create());
  }

  if (collect_gpu_events) {
    begin_gpu_event_phase(true);
  }
  const auto batch_start = std::chrono::steady_clock::now();
  const uint32_t prefix = selected_execution_prefix(args, result);
  for (uint32_t repeat = 0; repeat < repeat_count; ++repeat) {
    const uint32_t buffer = repeat & 1u;
    if (repeat >= 2) {
      fail_streaming_hip(
          "hipStreamWaitEvent(pack waits previous gemm)",
          hipStreamWaitEvent(pack_stream.get(), gemm_done[buffer].get(), 0));
      fail_streaming_hip(
          "hipStreamWaitEvent(gemm waits previous export)",
          hipStreamWaitEvent(gemm_stream.get(), export_done[buffer].get(), 0));
    }

    enqueue_streaming_pack(
        args,
        a_buffers[buffer],
        static_cast<const HostT*>(pinned_a[buffer].get()),
        source_bytes,
        args.m,
        args.k,
        args.k,
        prefix,
        first_source_version + repeat,
        pack_stream.get());
    fail_streaming_hip("hipEventRecord(pack_done)", hipEventRecord(pack_done[buffer].get(), pack_stream.get()));

    fail_streaming_hip(
        "hipStreamWaitEvent(gemm waits pack)",
        hipStreamWaitEvent(gemm_stream.get(), pack_done[buffer].get(), 0));
    fail_streaming_hip(
        "streaming rns_gemm_kernel_group",
        enqueue_timed_stream_operation("rns_gemm_kernel_group", gemm_stream.get(), [&]() {
          const rns8_status status = rns8::detail::hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
              plan, a_buffers[buffer], b, c_buffers[buffer], gemm_stream.get());
          return status == RNS8_SUCCESS ? hipSuccess : hipErrorUnknown;
        }));
    fail_streaming_hip("hipEventRecord(gemm_done)", hipEventRecord(gemm_done[buffer].get(), gemm_stream.get()));

    fail_streaming_hip(
        "hipStreamWaitEvent(export waits gemm)",
        hipStreamWaitEvent(export_stream.get(), gemm_done[buffer].get(), 0));
    enqueue_streaming_export(
        args,
        c_buffers[buffer],
        static_cast<HostT*>(pinned_c[buffer].get()),
        static_cast<int*>(pinned_status[repeat].get()),
        output_bytes,
        prefix,
        bound,
        export_stream.get());
    fail_streaming_hip("hipEventRecord(export_done)", hipEventRecord(export_done[buffer].get(), export_stream.get()));
  }
  fail_streaming_hip("hipStreamSynchronize(export)", hipStreamSynchronize(export_stream.get()));
  fail_streaming_hip("hipStreamSynchronize(gemm)", hipStreamSynchronize(gemm_stream.get()));
  fail_streaming_hip("hipStreamSynchronize(pack)", hipStreamSynchronize(pack_stream.get()));
  const auto batch_end = std::chrono::steady_clock::now();

  for (uint32_t repeat = 0; repeat < repeat_count; ++repeat) {
    const int host_status = *static_cast<int*>(pinned_status[repeat].get());
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      fail_status("streaming overlap export status", static_cast<rns8_status>(host_status));
    }
  }
  const uint32_t final_buffer = (repeat_count - 1u) & 1u;
  std::memcpy(C.data(), pinned_c[final_buffer].get(), output_bytes);

  result.streaming_overlap_executed = true;
  result.streaming_overlap_stream_count = 3;
  result.streaming_overlap_buffer_count = 2;
  result.streaming_overlap_measured_repeat_count = repeat_count;
  result.streaming_overlap_batch_wall_us = elapsed_us(batch_start, batch_end);
  result.streaming_overlap_per_repeat_pipeline_us =
      repeat_count == 0 ? 0 : result.streaming_overlap_batch_wall_us / static_cast<uint64_t>(repeat_count);

  if (collect_gpu_events) {
    fill_streaming_overlap_event_results(result, repeat_count);
    end_gpu_event_phase(true);
  }
}

bool hip_graph_replay_full_bounded_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs &&
         args.next_op_hint != NextOpHint::RnsGemm && !args.oneshot &&
         !grouped_task_executor_requested(args) && !args.vector_alu_baseline &&
         args.backend == RNS8_BACKEND_HIP_DIRECT && !args.transient_uniform_small_inputs &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.wrap64_rocwmma_candidate && !args.residue_channel_fusion &&
         !args.streaming_overlap && args.bound_mode == BoundMode::Global &&
         args.bound_source == BoundSource::StaticProfile;
}

bool hip_graph_replay_full_finite_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && finite_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs &&
         args.next_op_hint != NextOpHint::RnsGemm && !args.oneshot &&
         !grouped_task_executor_requested(args) && !args.vector_alu_baseline &&
         args.backend == RNS8_BACKEND_HIP_DIRECT && !args.transient_uniform_small_inputs &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.wrap64_rocwmma_candidate && !args.residue_channel_fusion &&
         !args.streaming_overlap && args.bound_mode == BoundMode::Global &&
         args.bound_source == BoundSource::StaticProfile;
}

bool hip_graph_replay_full_exact_wide_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && exact_wide_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs &&
         args.next_op_hint != NextOpHint::RnsGemm && !args.oneshot &&
         !grouped_task_executor_requested(args) && !args.vector_alu_baseline &&
         args.backend == RNS8_BACKEND_HIP_DIRECT && !args.transient_uniform_small_inputs &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.wrap64_rocwmma_candidate && !args.residue_channel_fusion &&
         !args.streaming_overlap && args.bound_mode == BoundMode::Global &&
         args.bound_source == BoundSource::StaticProfile && args.output_ld_padding == 0 &&
         args.exact_wide_limb_count == 4 && args.export_variant == "default" &&
         args.reconstruction_variant == "default_garner" && !args.sparse_a_4_to_2;
}

bool hip_graph_replay_full_wrap64_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && args.semantics == BenchSemantics::WrapU64Mod2_64 &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs &&
         args.next_op_hint != NextOpHint::RnsGemm && !args.oneshot &&
         !grouped_task_executor_requested(args) && !args.vector_alu_baseline &&
         args.backend == RNS8_BACKEND_HIP_DIRECT && !args.transient_uniform_small_inputs &&
         !args.native_to_rns_bridge && !args.vector_to_rns_chain &&
         !args.wrap64_rocwmma_candidate && !args.residue_channel_fusion &&
         !args.streaming_overlap && args.bound_mode == BoundMode::Global &&
         args.bound_source == BoundSource::StaticProfile && args.output_ld_padding == 0;
}

bool hip_graph_replay_full_pack_export_used(const BenchmarkResult& result) {
  return hip_graph_replay_full_bounded_pack_export_used(result) ||
         result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export" ||
         result.hip_graph_replay_scope == "direct_hip_exact_wide_pack_gemm_export" ||
         result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export";
}

template <typename HostT>
void run_hip_graph_replay_bounded_pack_gemm_export(
    const Args& args,
    BenchmarkResult& result,
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    const std::vector<HostT>& A,
    const std::vector<HostT>& B,
    std::vector<HostT>& C,
    uint64_t bound,
    rns8_matrix*& latest_output_matrix) {
  if (!a_matrix || !b_matrix || !c_matrix) {
    fail_status("hip graph bounded pack/GEMM/export requires resident A/B/C matrices", RNS8_INVALID_ARGUMENT);
  }

  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_bounded_pack_gemm_export";
  result.hip_graph_replay_caveat =
      "captures Direct-HIP bounded pack, one RNS GEMM, export status, export kernel, and output D2H inside the graph";

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(HostT), "hip graph A input");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(HostT), "hip graph B input");
  const std::size_t output_bytes =
      checked_bytes(checked_elements(args.m, args.n, "hip graph C output"), sizeof(HostT), "hip graph C output");

  rns8::detail::hip_pinned_host_buffer pinned_a;
  rns8::detail::hip_pinned_host_buffer pinned_b;
  rns8::detail::hip_pinned_host_buffer pinned_c;
  rns8::detail::hip_pinned_host_buffer pinned_status;
  fail_streaming_hip("hipHostMalloc(hip graph A)", pinned_a.ensure(args.device_id, a_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph B)", pinned_b.ensure(args.device_id, b_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph C)", pinned_c.ensure(args.device_id, output_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph status)", pinned_status.ensure(args.device_id, sizeof(int)));
  std::memcpy(pinned_a.get(), A.data(), a_bytes);
  std::memcpy(pinned_b.get(), B.data(), b_bytes);
  std::memset(pinned_c.get(), 0, output_bytes);
  *static_cast<int*>(pinned_status.get()) = 0;

  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, a_bytes, &a_matrix->hip_upload_buffer, &a_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph A)", status);
  }
  status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, b_bytes, &b_matrix->hip_upload_buffer, &b_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph B)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, output_bytes, &c_matrix->hip_export_buffer, &c_matrix->hip_export_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph export)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, sizeof(int), &c_matrix->hip_status_buffer, &c_matrix->hip_status_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph status)", status);
  }

  const uint32_t prefix = selected_execution_prefix(args, result);
  HipGraphReplayState graph_state{};
  std::string graph_error;
  status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream_ptr) {
        auto* stream = static_cast<hipStream_t>(stream_ptr);
        enqueue_streaming_pack(
            args,
            a_matrix,
            static_cast<const HostT*>(pinned_a.get()),
            a_bytes,
            args.m,
            args.k,
            args.k,
            prefix,
            1,
            stream);
        enqueue_streaming_pack(
            args,
            b_matrix,
            static_cast<const HostT*>(pinned_b.get()),
            b_bytes,
            args.k,
            args.n,
            args.n,
            prefix,
            1,
            stream);
        const rns8_status gemm_status = rns8::detail::hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
            plan, a_matrix, b_matrix, c_matrix, stream);
        if (gemm_status != RNS8_SUCCESS) {
          return gemm_status;
        }
        enqueue_streaming_export(
            args,
            c_matrix,
            static_cast<HostT*>(pinned_c.get()),
            static_cast<int*>(pinned_status.get()),
            output_bytes,
            prefix,
            bound,
            stream);
        latest_output_matrix = c_matrix;
        return RNS8_SUCCESS;
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph bounded pack/GEMM/export capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph bounded pack/GEMM/export warmup launch", status);
    }
    const int host_status = *static_cast<int*>(pinned_status.get());
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      fail_status("hip graph bounded pack/GEMM/export warmup export status", static_cast<rns8_status>(host_status));
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    *static_cast<int*>(pinned_status.get()) = 0;
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph bounded pack/GEMM/export measured launch", status);
    }
    const int host_status = *static_cast<int*>(pinned_status.get());
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      fail_status("hip graph bounded pack/GEMM/export measured export status", static_cast<rns8_status>(host_status));
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  record_allocation_after_repeats(result);
  std::memcpy(C.data(), pinned_c.get(), output_bytes);
  latest_output_matrix = c_matrix;
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
}

template <typename HostT>
void enqueue_exact_wide_graph_export(
    const Args& args,
    rns8_matrix* matrix,
    uint64_t* pinned_dst,
    int* pinned_status,
    std::size_t output_bytes,
    uint32_t prefix,
    hipStream_t stream) {
  if (!matrix || !matrix->hip_export_buffer || matrix->hip_export_bytes < output_bytes || !matrix->hip_status_buffer ||
      matrix->hip_status_bytes < sizeof(int)) {
    fail_status("hip graph exact-wide export buffers", RNS8_INVALID_ARGUMENT);
  }
  fail_streaming_hip(
      "hip graph exact_wide_export_status_memset",
      enqueue_timed_stream_operation("exact_wide_export_status_memset", stream, [&]() {
        return hipMemsetAsync(matrix->hip_status_buffer, 0, sizeof(int), stream);
      }));
  fail_streaming_hip(
      "hip graph exact_wide_export_kernel",
      enqueue_timed_stream_operation("exact_wide_export_kernel", stream, [&]() {
        const int code = std::is_same_v<HostT, int64_t>
                             ? rns8_hip_direct_export_exact_wide_signed_limbs_device_on_stream(
                                   static_cast<const int8_t*>(matrix->hip_residues),
                                   static_cast<uint64_t*>(matrix->hip_export_buffer),
                                   static_cast<int>(args.m),
                                   static_cast<int>(args.n),
                                   static_cast<int>(prefix),
                                   static_cast<int>(args.exact_wide_limb_count),
                                   static_cast<int*>(matrix->hip_status_buffer),
                                   stream)
                             : rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(
                                   static_cast<const int8_t*>(matrix->hip_residues),
                                   static_cast<uint64_t*>(matrix->hip_export_buffer),
                                   static_cast<int>(args.m),
                                   static_cast<int>(args.n),
                                   static_cast<int>(prefix),
                                   static_cast<int>(args.exact_wide_limb_count),
                                   static_cast<int*>(matrix->hip_status_buffer),
                                   stream);
        return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
      }));
  fail_streaming_hip(
      "hip graph exact_wide_export_status_d2h",
      enqueue_timed_stream_operation("exact_wide_export_status_d2h", stream, [&]() {
        return hipMemcpyAsync(pinned_status, matrix->hip_status_buffer, sizeof(int), hipMemcpyDeviceToHost, stream);
      }));
  fail_streaming_hip(
      "hip graph exact_wide_export_d2h",
      enqueue_timed_stream_operation("exact_wide_export_d2h", stream, [&]() {
        return hipMemcpyAsync(pinned_dst, matrix->hip_export_buffer, output_bytes, hipMemcpyDeviceToHost, stream);
      }));
}

template <typename HostT>
void run_hip_graph_replay_exact_wide_pack_gemm_export(
    const Args& args,
    BenchmarkResult& result,
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    const std::vector<HostT>& A,
    const std::vector<HostT>& B,
    std::vector<uint64_t>& C,
    rns8_matrix*& latest_output_matrix) {
  if (!a_matrix || !b_matrix || !c_matrix || args.exact_wide_limb_count != 4 || args.output_ld_padding != 0) {
    fail_status("hip graph exact-wide pack/GEMM/export requires contiguous 4-limb A/B/C matrices", RNS8_INVALID_ARGUMENT);
  }

  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_exact_wide_pack_gemm_export";
  result.hip_graph_replay_caveat =
      "captures Direct-HIP exact-wide pack, one RNS GEMM, fixed-width exact limb export, status D2H, and output D2H inside the graph";

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(HostT), "hip graph exact-wide A input");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(HostT), "hip graph exact-wide B input");
  const std::size_t output_bytes = checked_bytes(
      checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "hip graph exact-wide C output"),
      sizeof(uint64_t),
      "hip graph exact-wide C output");

  rns8::detail::hip_pinned_host_buffer pinned_a;
  rns8::detail::hip_pinned_host_buffer pinned_b;
  rns8::detail::hip_pinned_host_buffer pinned_c;
  rns8::detail::hip_pinned_host_buffer pinned_status;
  fail_streaming_hip("hipHostMalloc(hip graph exact-wide A)", pinned_a.ensure(args.device_id, a_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph exact-wide B)", pinned_b.ensure(args.device_id, b_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph exact-wide C)", pinned_c.ensure(args.device_id, output_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph exact-wide status)", pinned_status.ensure(args.device_id, sizeof(int)));
  std::memcpy(pinned_a.get(), A.data(), a_bytes);
  std::memcpy(pinned_b.get(), B.data(), b_bytes);
  std::memset(pinned_c.get(), 0, output_bytes);
  *static_cast<int*>(pinned_status.get()) = 0;

  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, a_bytes, &a_matrix->hip_upload_buffer, &a_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph exact-wide A)", status);
  }
  status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, b_bytes, &b_matrix->hip_upload_buffer, &b_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph exact-wide B)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, output_bytes, &c_matrix->hip_export_buffer, &c_matrix->hip_export_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph exact-wide export)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, sizeof(int), &c_matrix->hip_status_buffer, &c_matrix->hip_status_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph exact-wide status)", status);
  }

  const uint32_t prefix = selected_execution_prefix(args, result);
  HipGraphReplayState graph_state{};
  std::string graph_error;
  status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream_ptr) {
        auto* stream = static_cast<hipStream_t>(stream_ptr);
        enqueue_streaming_pack(
            args,
            a_matrix,
            static_cast<const HostT*>(pinned_a.get()),
            a_bytes,
            args.m,
            args.k,
            args.k,
            prefix,
            1,
            stream);
        enqueue_streaming_pack(
            args,
            b_matrix,
            static_cast<const HostT*>(pinned_b.get()),
            b_bytes,
            args.k,
            args.n,
            args.n,
            prefix,
            1,
            stream);
        const rns8_status gemm_status = rns8::detail::hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
            plan, a_matrix, b_matrix, c_matrix, stream);
        if (gemm_status != RNS8_SUCCESS) {
          return gemm_status;
        }
        enqueue_exact_wide_graph_export<HostT>(
            args,
            c_matrix,
            static_cast<uint64_t*>(pinned_c.get()),
            static_cast<int*>(pinned_status.get()),
            output_bytes,
            prefix,
            stream);
        latest_output_matrix = c_matrix;
        return RNS8_SUCCESS;
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph exact-wide pack/GEMM/export capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph exact-wide pack/GEMM/export warmup launch", status);
    }
    const int host_status = *static_cast<int*>(pinned_status.get());
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      fail_status("hip graph exact-wide pack/GEMM/export warmup export status", static_cast<rns8_status>(host_status));
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    *static_cast<int*>(pinned_status.get()) = 0;
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph exact-wide pack/GEMM/export measured launch", status);
    }
    const int host_status = *static_cast<int*>(pinned_status.get());
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      fail_status("hip graph exact-wide pack/GEMM/export measured export status", static_cast<rns8_status>(host_status));
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  record_allocation_after_repeats(result);
  std::memcpy(C.data(), pinned_c.get(), output_bytes);
  latest_output_matrix = c_matrix;
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
}

uint32_t benchmark_finite_modulus_reciprocal(uint16_t modulus) {
  return static_cast<uint32_t>((UINT64_C(1) << 32u) / static_cast<uint32_t>(modulus));
}

int benchmark_finite_safe_k_block(const Args& args, const BenchmarkResult& result) {
  uint64_t k_block = result.backend_info_available && result.backend_info.accumulator_k_block_size > 0
                         ? result.backend_info.accumulator_k_block_size
                         : static_cast<uint64_t>(args.k);
  if (k_block == 0 || k_block > UINT64_C(65536)) {
    k_block = UINT64_C(65536);
  }
  if (static_cast<uint64_t>(args.k) < k_block) {
    k_block = static_cast<uint64_t>(args.k);
  }
  if (k_block == 0 || k_block > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    fail_status("HIP Graph finite safe K-block", RNS8_INVALID_ARGUMENT);
  }
  return static_cast<int>(k_block);
}

void enqueue_finite_graph_pack(
    const Args& args,
    rns8_matrix* matrix,
    const uint8_t* pinned_src,
    std::size_t source_bytes,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint64_t source_version,
    hipStream_t stream) {
  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, source_bytes, &matrix->hip_upload_buffer, &matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph finite pack)", status);
  }
  fail_streaming_hip(
      "hip graph finite_pack_h2d",
      enqueue_timed_stream_operation("finite_pack_h2d", stream, [&]() {
        return hipMemcpyAsync(matrix->hip_upload_buffer, pinned_src, source_bytes, hipMemcpyHostToDevice, stream);
      }));
  fail_streaming_hip(
      "hip graph finite_pack_kernel",
      enqueue_timed_stream_operation("finite_pack_kernel", stream, [&]() {
        const int code = rns8_hip_direct_pack_u8_modulus_device_on_stream(
            static_cast<const uint8_t*>(matrix->hip_upload_buffer),
            static_cast<int8_t*>(matrix->hip_residues),
            static_cast<int>(rows),
            static_cast<int>(cols),
            static_cast<int>(ld),
            static_cast<int>(args.finite_modulus),
            stream);
        return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
      }));
  rns8::detail::api::mark_device_residues_current(*matrix);
  matrix->finite_modulus = args.finite_modulus;
  matrix->source_version = source_version;
}

void enqueue_finite_graph_export(
    const Args& args,
    rns8_matrix* matrix,
    uint8_t* pinned_dst,
    std::size_t output_bytes,
    int64_t ld,
    hipStream_t stream) {
  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, output_bytes, &matrix->hip_export_buffer, &matrix->hip_export_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph finite export)", status);
  }
  fail_streaming_hip(
      "hip graph finite_export_kernel",
      enqueue_timed_stream_operation("finite_export_kernel", stream, [&]() {
        const int code = rns8_hip_direct_export_u8_modulus_device_on_stream(
            static_cast<const int8_t*>(matrix->hip_residues),
            static_cast<uint8_t*>(matrix->hip_export_buffer),
            static_cast<int>(args.m),
            static_cast<int>(args.n),
            static_cast<int>(ld),
            static_cast<int>(args.finite_modulus),
            stream);
        return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
      }));
  fail_streaming_hip(
      "hip graph finite_export_d2h",
      enqueue_timed_stream_operation("finite_export_d2h", stream, [&]() {
        return hipMemcpyAsync(pinned_dst, matrix->hip_export_buffer, output_bytes, hipMemcpyDeviceToHost, stream);
      }));
}

void run_hip_graph_replay_finite_pack_gemm_export(
    const Args& args,
    BenchmarkResult& result,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    const std::vector<uint8_t>& A,
    const std::vector<uint8_t>& B,
    std::vector<uint8_t>& C) {
  if (!a_matrix || !b_matrix || !c_matrix) {
    fail_status("hip graph finite pack/GEMM/export requires resident A/B/C matrices", RNS8_INVALID_ARGUMENT);
  }

  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_finite_u8_pack_gemm_export";
  result.hip_graph_replay_caveat =
      "captures Direct-HIP finite-u8 pack, one modular GEMM, canonical finite export, and output D2H inside the graph";

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(uint8_t), "hip graph finite A input");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(uint8_t), "hip graph finite B input");
  const std::size_t output_bytes =
      checked_bytes(checked_elements(args.m, args.n, "hip graph finite C output"), sizeof(uint8_t), "hip graph finite C output");

  rns8::detail::hip_pinned_host_buffer pinned_a;
  rns8::detail::hip_pinned_host_buffer pinned_b;
  rns8::detail::hip_pinned_host_buffer pinned_c;
  fail_streaming_hip("hipHostMalloc(hip graph finite A)", pinned_a.ensure(args.device_id, a_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph finite B)", pinned_b.ensure(args.device_id, b_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph finite C)", pinned_c.ensure(args.device_id, output_bytes));
  std::memcpy(pinned_a.get(), A.data(), a_bytes);
  std::memcpy(pinned_b.get(), B.data(), b_bytes);
  std::memset(pinned_c.get(), 0, output_bytes);

  rns8_status status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, a_bytes, &a_matrix->hip_upload_buffer, &a_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph finite A)", status);
  }
  status =
      rns8::detail::hip_direct_ensure_upload_buffer(args.device_id, b_bytes, &b_matrix->hip_upload_buffer, &b_matrix->hip_upload_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph finite B)", status);
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      args.device_id, output_bytes, &c_matrix->hip_export_buffer, &c_matrix->hip_export_bytes);
  if (status != RNS8_SUCCESS) {
    fail_status("hip_direct_ensure_upload_buffer(hip graph finite export)", status);
  }

  HipGraphReplayState graph_state{};
  std::string graph_error;
  status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream_ptr) {
        auto* stream = static_cast<hipStream_t>(stream_ptr);
        enqueue_finite_graph_pack(
            args,
            a_matrix,
            static_cast<const uint8_t*>(pinned_a.get()),
            a_bytes,
            args.m,
            args.k,
            args.k,
            1,
            stream);
        enqueue_finite_graph_pack(
            args,
            b_matrix,
            static_cast<const uint8_t*>(pinned_b.get()),
            b_bytes,
            args.k,
            args.n,
            args.n,
            1,
            stream);
        const int code = rns8_hip_direct_finite_ring_gemm_i8_device_on_stream(
            static_cast<const int8_t*>(a_matrix->hip_residues),
            static_cast<const int8_t*>(b_matrix->hip_residues),
            static_cast<int8_t*>(c_matrix->hip_residues),
            static_cast<int>(args.m),
            static_cast<int>(args.n),
            static_cast<int>(args.k),
            static_cast<int>(args.k),
            static_cast<int>(args.n),
            static_cast<int>(args.n),
            static_cast<int>(args.finite_modulus),
            benchmark_finite_modulus_reciprocal(args.finite_modulus),
            benchmark_finite_safe_k_block(args, result),
            stream);
        if (code != static_cast<int>(hipSuccess)) {
          return RNS8_BACKEND_FAILURE;
        }
        rns8::detail::api::mark_output_device_residues_current(*c_matrix);
        c_matrix->finite_modulus = args.finite_modulus;
        c_matrix->source_version =
            rns8::detail::api::gemm_output_source_version_values(a_matrix->source_version, b_matrix->source_version);
        enqueue_finite_graph_export(
            args,
            c_matrix,
            static_cast<uint8_t*>(pinned_c.get()),
            output_bytes,
            args.n,
            stream);
        return RNS8_SUCCESS;
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph finite pack/GEMM/export capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph finite pack/GEMM/export warmup launch", status);
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph finite pack/GEMM/export measured launch", status);
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  record_allocation_after_repeats(result);
  std::memcpy(C.data(), pinned_c.get(), output_bytes);
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
}

void run_hip_graph_replay_wrap64_pack_gemm_export(
    const Args& args,
    BenchmarkResult& result,
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B,
    std::vector<uint64_t>& C,
    int64_t ldc) {
  if (ldc != args.n) {
    fail_status("HIP Graph wrap64 pack/GEMM/export requires contiguous output", RNS8_INVALID_ARGUMENT);
  }

  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_wrap64_pack_gemm_export";
  result.hip_graph_replay_caveat =
      "captures Direct-HIP wrap64 byte-limb pack, byte-GEMM36, low64 export, and output D2H inside the graph";

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(uint64_t), "hip graph wrap64 A input");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(uint64_t), "hip graph wrap64 B input");
  const std::size_t output_bytes =
      checked_bytes(checked_elements(args.m, args.n, "hip graph wrap64 C output"), sizeof(uint64_t), "hip graph wrap64 C output");
  const std::size_t a_limb_bytes = wrap64_compact_limb_bytes(args.m, args.k, "hip graph wrap64 A byte limbs");
  const std::size_t b_limb_bytes = wrap64_compact_limb_bytes(args.k, args.n, "hip graph wrap64 B byte limbs");
  const std::size_t c_limb_bytes = wrap64_compact_limb_bytes(args.m, args.n, "hip graph wrap64 C byte limbs");

  rns8::detail::hip_pinned_host_buffer pinned_a;
  rns8::detail::hip_pinned_host_buffer pinned_b;
  rns8::detail::hip_pinned_host_buffer pinned_c;
  fail_streaming_hip("hipHostMalloc(hip graph wrap64 A)", pinned_a.ensure(args.device_id, a_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph wrap64 B)", pinned_b.ensure(args.device_id, b_bytes));
  fail_streaming_hip("hipHostMalloc(hip graph wrap64 C)", pinned_c.ensure(args.device_id, output_bytes));
  std::memcpy(pinned_a.get(), A.data(), a_bytes);
  std::memcpy(pinned_b.get(), B.data(), b_bytes);
  std::memset(pinned_c.get(), 0, output_bytes);

  DeviceBuffer a_upload;
  DeviceBuffer b_upload;
  DeviceBuffer a_limbs;
  DeviceBuffer b_limbs;
  DeviceBuffer c_limbs;
  DeviceBuffer export_buffer;
  const auto alloc_start = std::chrono::steady_clock::now();
  a_upload.allocate(args.device_id, a_bytes, "hip_direct_allocate(hip graph wrap64 A upload)");
  b_upload.allocate(args.device_id, b_bytes, "hip_direct_allocate(hip graph wrap64 B upload)");
  a_limbs.allocate(args.device_id, a_limb_bytes, "hip_direct_allocate(hip graph wrap64 A limbs)");
  b_limbs.allocate(args.device_id, b_limb_bytes, "hip_direct_allocate(hip graph wrap64 B limbs)");
  c_limbs.allocate(args.device_id, c_limb_bytes, "hip_direct_allocate(hip graph wrap64 C limbs)");
  export_buffer.allocate(args.device_id, output_bytes, "hip_direct_allocate(hip graph wrap64 export)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  HipGraphReplayState graph_state{};
  std::string graph_error;
  rns8_status status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream_ptr) {
        auto* stream = static_cast<hipStream_t>(stream_ptr);
        fail_streaming_hip(
            "hip graph wrap64_pack_a_h2d",
            enqueue_timed_stream_operation("pack_h2d", stream, [&]() {
              return hipMemcpyAsync(a_upload.ptr, pinned_a.get(), a_bytes, hipMemcpyHostToDevice, stream);
            }));
        fail_streaming_hip(
            "hip graph wrap64_pack_a_kernel",
            enqueue_timed_stream_operation("pack_kernel", stream, [&]() {
              const int code = rns8_wrap64_hip_pack_u64_device_on_stream(
                  static_cast<const uint64_t*>(a_upload.ptr),
                  static_cast<uint8_t*>(a_limbs.ptr),
                  args.m,
                  args.k,
                  args.k,
                  stream);
              return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
            }));
        fail_streaming_hip(
            "hip graph wrap64_pack_b_h2d",
            enqueue_timed_stream_operation("pack_h2d", stream, [&]() {
              return hipMemcpyAsync(b_upload.ptr, pinned_b.get(), b_bytes, hipMemcpyHostToDevice, stream);
            }));
        fail_streaming_hip(
            "hip graph wrap64_pack_b_kernel",
            enqueue_timed_stream_operation("pack_kernel", stream, [&]() {
              const int code = rns8_wrap64_hip_pack_u64_device_on_stream(
                  static_cast<const uint64_t*>(b_upload.ptr),
                  static_cast<uint8_t*>(b_limbs.ptr),
                  args.k,
                  args.n,
                  args.n,
                  stream);
              return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
            }));
        fail_streaming_hip(
            "hip graph wrap64_gemm",
            enqueue_timed_stream_operation(wrap64_direct_hip_gemm_event_label(args), stream, [&]() {
              const int code = rns8_wrap64_hip_gemm_byte_limbs_device_on_stream(
                  static_cast<const uint8_t*>(a_limbs.ptr),
                  static_cast<const uint8_t*>(b_limbs.ptr),
                  static_cast<uint8_t*>(c_limbs.ptr),
                  args.m,
                  args.n,
                  args.k,
                  stream);
              return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
            }));
        fail_streaming_hip(
            "hip graph wrap64_export_kernel",
            enqueue_timed_stream_operation("wrap64_export_kernel", stream, [&]() {
              const int code = rns8_wrap64_hip_export_u64_device_on_stream(
                  static_cast<const uint8_t*>(c_limbs.ptr),
                  static_cast<uint64_t*>(export_buffer.ptr),
                  args.m,
                  args.n,
                  stream);
              return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
            }));
        fail_streaming_hip(
            "hip graph wrap64_export_d2h",
            enqueue_timed_stream_operation("wrap64_export_d2h", stream, [&]() {
              return hipMemcpyAsync(pinned_c.get(), export_buffer.ptr, output_bytes, hipMemcpyDeviceToHost, stream);
            }));
        return RNS8_SUCCESS;
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph wrap64 pack/GEMM/export capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph wrap64 pack/GEMM/export warmup launch", status);
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph wrap64 pack/GEMM/export measured launch", status);
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  record_allocation_after_repeats(result);
  std::memcpy(C.data(), pinned_c.get(), output_bytes);
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
}
#else
template <typename HostT>
void run_bounded_streaming_overlap_pipeline(
    const Args&,
    BenchmarkResult&,
    const rns8_plan*,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    const std::vector<HostT>&,
    std::vector<HostT>&,
    uint64_t,
    uint64_t,
    uint32_t,
    bool) {
  fail_status("streaming overlap requires a HIP-enabled benchmark build", RNS8_UNSUPPORTED_BACKEND);
}

bool hip_graph_replay_full_bounded_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs;
}

bool hip_graph_replay_full_finite_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && finite_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs;
}

bool hip_graph_replay_full_exact_wide_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && exact_wide_benchmark_semantics(args.semantics) &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs && args.output_ld_padding == 0 &&
         args.exact_wide_limb_count == 4;
}

bool hip_graph_replay_full_wrap64_pack_export_requested(const Args& args) {
  return hip_graph_replay_requested(args) && args.semantics == BenchSemantics::WrapU64Mod2_64 &&
         args.residue_chain_length == 1 && !args.reuse_packed_inputs;
}

bool hip_graph_replay_full_pack_export_used(const BenchmarkResult& result) {
  return hip_graph_replay_full_bounded_pack_export_used(result) ||
         result.hip_graph_replay_scope == "direct_hip_finite_u8_pack_gemm_export" ||
         result.hip_graph_replay_scope == "direct_hip_exact_wide_pack_gemm_export" ||
         result.hip_graph_replay_scope == "direct_hip_wrap64_pack_gemm_export";
}

template <typename HostT>
void run_hip_graph_replay_bounded_pack_gemm_export(
    const Args&,
    BenchmarkResult&,
    const rns8_plan*,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    const std::vector<HostT>&,
    const std::vector<HostT>&,
    std::vector<HostT>&,
    uint64_t,
    rns8_matrix*&) {
  fail_status("HIP Graph bounded pack/GEMM/export requires a HIP-enabled benchmark build", RNS8_UNSUPPORTED_BACKEND);
}

void run_hip_graph_replay_finite_pack_gemm_export(
    const Args&,
    BenchmarkResult&,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    const std::vector<uint8_t>&,
    const std::vector<uint8_t>&,
    std::vector<uint8_t>&) {
  fail_status("HIP Graph finite-u8 pack/GEMM/export requires a HIP-enabled benchmark build", RNS8_UNSUPPORTED_BACKEND);
}

template <typename HostT>
void run_hip_graph_replay_exact_wide_pack_gemm_export(
    const Args&,
    BenchmarkResult&,
    const rns8_plan*,
    rns8_matrix*,
    rns8_matrix*,
    rns8_matrix*,
    const std::vector<HostT>&,
    const std::vector<HostT>&,
    std::vector<uint64_t>&,
    rns8_matrix*&) {
  fail_status("HIP Graph exact-wide pack/GEMM/export requires a HIP-enabled benchmark build", RNS8_UNSUPPORTED_BACKEND);
}

void run_hip_graph_replay_wrap64_pack_gemm_export(
    const Args&,
    BenchmarkResult&,
    const std::vector<uint64_t>&,
    const std::vector<uint64_t>&,
    std::vector<uint64_t>&,
    int64_t) {
  fail_status("HIP Graph wrap64 pack/GEMM/export requires a HIP-enabled benchmark build", RNS8_UNSUPPORTED_BACKEND);
}
#endif
