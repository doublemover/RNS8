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
    uint32_t prefix,
    hipStream_t stream) {
  if constexpr (std::is_same_v<HostT, int64_t>) {
    const int code = rns8_hip_direct_pack_i64_device_on_stream(
        device_src,
        static_cast<int8_t*>(matrix->hip_residues),
        static_cast<int>(args.m),
        static_cast<int>(args.k),
        static_cast<int>(args.k),
        static_cast<int>(prefix),
        stream);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  } else {
    const int code = rns8_hip_direct_pack_u64_device_on_stream(
        device_src,
        static_cast<int8_t*>(matrix->hip_residues),
        static_cast<int>(args.m),
        static_cast<int>(args.k),
        static_cast<int>(args.k),
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
            static_cast<const HostT*>(matrix->hip_upload_buffer), matrix, args, prefix, stream);
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
#endif
