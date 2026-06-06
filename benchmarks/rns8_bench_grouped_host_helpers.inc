bool schedule_uses_adaptive_work(const BenchmarkResult& result) {
  return result.schedule_info.adaptive_prefix_active != 0 || result.schedule_info.adaptive_skip_active != 0;
}

void enforce_per_tile_capture_contract(const Args& args, const BenchmarkResult& result) {
  if (args.bound_mode != BoundMode::PerTile) {
    return;
  }
  if (result.tile_bounds.empty()) {
    usage_error("per-tile benchmark capture has no generated tile bounds");
  }
  if (args.require_adaptive_execution && !schedule_uses_adaptive_work(result)) {
    usage_error("--require-adaptive-execution was requested but the plan is fixed-prefix");
  }
  if (!fixed_requested_prefix_policy(args) && !schedule_uses_adaptive_work(result)) {
    usage_error(
        "per-tile benchmark capture did not produce adaptive prefix grouping or prefix skipping; adjust shape, seed, or inputs");
  }
}

using HostApiBatchTask = rns8::detail::hip_direct_grouped_gemm_task;

void destroy_host_api_batch_tasks(std::vector<HostApiBatchTask>& tasks) {
  for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
    if (it->c) {
      rns8_destroy_matrix(it->c);
      it->c = nullptr;
    }
    if (it->b) {
      rns8_destroy_matrix(it->b);
      it->b = nullptr;
    }
    if (it->a) {
      rns8_destroy_matrix(it->a);
      it->a = nullptr;
    }
    if (it->workspace) {
      rns8_destroy_workspace(it->workspace);
      it->workspace = nullptr;
    }
  }
}

std::vector<HostApiBatchTask> create_host_api_batch_tasks(
    rns8_context* ctx,
    rns8_plan* plan,
    const rns8_matrix_desc& a_desc,
    const rns8_matrix_desc& b_desc,
    const rns8_matrix_desc& c_desc,
    uint32_t batch_size) {
  std::vector<HostApiBatchTask> tasks(batch_size);
  for (uint32_t task_index = 0; task_index < batch_size; ++task_index) {
    rns8_status status = rns8_create_workspace(ctx, plan, &tasks[task_index].workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(host API batch)", status);
    status = rns8_create_matrix(ctx, &a_desc, &tasks[task_index].a);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch A)", status);
    status = rns8_create_matrix(ctx, &b_desc, &tasks[task_index].b);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch B)", status);
    status = rns8_create_matrix(ctx, &c_desc, &tasks[task_index].c);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch C)", status);
  }
  return tasks;
}

rns8::detail::hip_direct_grouped_gemm_descriptor make_grouped_dispatch_descriptor(
    const Args& args,
    const rns8_plan* plan,
    const std::vector<HostApiBatchTask>& tasks,
    rns8_semantics semantics,
    uint32_t matrix_prefix) {
  rns8::detail::hip_direct_grouped_gemm_descriptor descriptor{};
  descriptor.plan = plan;
  descriptor.tasks = tasks.data();
  descriptor.task_count = static_cast<uint32_t>(tasks.size());
  descriptor.semantics = semantics;
  descriptor.m = args.m;
  descriptor.n = args.n;
  descriptor.k = args.k;
  descriptor.prefix = matrix_prefix;
  return descriptor;
}

void enforce_grouped_dispatch_task_setup_contract(
    const Args& args,
    const rns8_plan* plan,
    const std::vector<HostApiBatchTask>& tasks,
    rns8_semantics semantics,
    uint32_t matrix_prefix,
    const char* label) {
  if (!grouped_dispatch_requested(args)) {
    return;
  }
  const uint32_t task_count = measured_task_count(args);
  if (tasks.size() != task_count) {
    usage_error(std::string(label) + " grouped descriptor task count mismatch");
  }
  const rns8_status status = rns8::detail::hip_direct_validate_grouped_gemm_descriptor_setup(
      make_grouped_dispatch_descriptor(args, plan, tasks, semantics, matrix_prefix));
  if (status != RNS8_SUCCESS) {
    usage_error(std::string(label) + " violates grouped descriptor plan/shape contract");
  }
}

uint64_t host_api_batch_source_version(uint64_t repeat_source_version, uint32_t task_index) {
  return repeat_source_version * UINT64_C(104729) + static_cast<uint64_t>(task_index) + UINT64_C(1);
}

void enforce_grouped_dispatch_pack_source_versions(
    const Args& args,
    const rns8_plan* plan,
    const std::vector<HostApiBatchTask>& tasks,
    rns8_semantics semantics,
    uint32_t matrix_prefix,
    uint64_t repeat_source_version,
    const char* label) {
  if (!grouped_dispatch_requested(args)) {
    return;
  }
  const uint32_t task_count = measured_task_count(args);
  if (tasks.size() != task_count) {
    usage_error(std::string(label) + " grouped source-version task count mismatch");
  }
  const uint64_t first_source_version = host_api_batch_source_version(repeat_source_version, 0);
  const rns8_status status = rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_pack(
      make_grouped_dispatch_descriptor(args, plan, tasks, semantics, matrix_prefix), first_source_version);
  if (status != RNS8_SUCCESS) {
    usage_error(std::string(label) + " violates grouped per-task source-version repack contract");
  }
}

void enforce_grouped_dispatch_gemm_output_contract(
    const Args& args,
    const rns8_plan* plan,
    const std::vector<HostApiBatchTask>& tasks,
    rns8_semantics semantics,
    uint32_t matrix_prefix,
    const char* label) {
  if (!grouped_dispatch_requested(args)) {
    return;
  }
  const uint32_t task_count = measured_task_count(args);
  if (tasks.size() != task_count) {
    usage_error(std::string(label) + " grouped output task count mismatch");
  }
  const rns8_status status = rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(
      make_grouped_dispatch_descriptor(args, plan, tasks, semantics, matrix_prefix));
  if (status != RNS8_SUCCESS) {
    usage_error(std::string(label) + " violates grouped device-current output contract");
  }
}

template <typename PackTask, typename GemmTask, typename ExportTask>
void run_host_api_batch_iteration(
    const Args& args,
    BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    const rns8_plan* plan,
    std::vector<HostApiBatchTask>& tasks,
    rns8_semantics semantics,
    uint32_t matrix_prefix,
    uint64_t repeat_source_version,
    TimingSamples* samples,
    const PackTask& pack_task,
    const GemmTask& gemm_task,
    const ExportTask& export_task) {
  const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
  const uint32_t task_count = measured_task_count(args);
  const auto repeat_start = std::chrono::steady_clock::now();

  const auto pack_start = repeat_start;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    pack_task(tasks[task_index], task_index, host_api_batch_source_version(repeat_source_version, task_index));
  }
  enforce_grouped_dispatch_pack_source_versions(
      args, plan, tasks, semantics, matrix_prefix, repeat_source_version, "grouped dispatch pack phase");
  if (collect_gpu_events) {
    collect_pack_gpu_events(args, selected_backend, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto pack_end = std::chrono::steady_clock::now();

  const auto gemm_start = pack_end;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    gemm_task(tasks[task_index], task_index);
  }
  enforce_grouped_dispatch_gemm_output_contract(
      args, plan, tasks, semantics, matrix_prefix, "grouped dispatch rns_gemm phase");
  if (collect_gpu_events) {
    collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto gemm_end = std::chrono::steady_clock::now();

  const auto export_start = gemm_end;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    export_task(tasks[task_index], task_index);
  }
  if (collect_gpu_events) {
    collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto export_end = std::chrono::steady_clock::now();

  if (samples) {
    samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
    samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
    samples->export_us.push_back(elapsed_us(export_start, export_end));
    samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
  }
}

std::size_t checked_task_slab_limb_elements(
    std::size_t task_count,
    std::size_t compact_limb_elements,
    const char* label) {
  if (task_count != 0 && compact_limb_elements > std::numeric_limits<std::size_t>::max() / task_count) {
    usage_error(std::string("limb output size overflows size_t for ") + label);
  }
  return task_count * compact_limb_elements;
}

std::size_t checked_task_slab_elements(
    std::size_t task_count,
    std::size_t compact_elements,
    const char* label) {
  if (task_count != 0 && compact_elements > std::numeric_limits<std::size_t>::max() / task_count) {
    usage_error(std::string("task slab size overflows size_t for ") + label);
  }
  return task_count * compact_elements;
}

template <typename T>
std::vector<T> flatten_grouped_pack_slab(
    const std::vector<std::vector<T>>& batch,
    std::size_t elements_per_task,
    const char* label) {
  const std::size_t total_elements = checked_task_slab_elements(batch.size(), elements_per_task, label);
  std::vector<T> slab(total_elements);
  for (std::size_t task_index = 0; task_index < batch.size(); ++task_index) {
    if (batch[task_index].size() != elements_per_task) {
      usage_error(std::string("host batch input size mismatch for ") + label);
    }
    std::copy(
        batch[task_index].begin(),
        batch[task_index].end(),
        slab.begin() + static_cast<std::ptrdiff_t>(task_index * elements_per_task));
  }
  return slab;
}

int64_t checked_int64_extent(std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    usage_error(std::string("extent overflows int64_t for ") + label);
  }
  return static_cast<int64_t>(value);
}

void prepare_grouped_matrix_residue_pointer_table(
    const std::vector<rns8_matrix*>& matrices,
    rns8_semantics expected_semantics,
    DeviceBuffer& device_residue_ptrs,
    uint32_t expected_prefix,
    const char* operation_label,
    const char* metadata_label) {
  int pointer_device_id = -1;
  uint32_t pointer_prefix = 0;
  const rns8_status status = rns8::detail::hip_direct_prepare_grouped_matrix_residue_pointers(
      matrices.data(),
      static_cast<uint32_t>(matrices.size()),
      expected_semantics,
      device_residue_ptrs.ptr,
      device_residue_ptrs.bytes,
      &pointer_device_id,
      &pointer_prefix);
  if (status != RNS8_SUCCESS) {
    fail_status(operation_label, status);
  }
  if (pointer_device_id != device_residue_ptrs.device_id || pointer_prefix != expected_prefix) {
    usage_error(std::string(metadata_label) + " metadata does not match benchmark storage");
  }
}

template <typename NativeT>
struct GroupedNativeRnsBatch {
  DeviceBuffer a_device_slab;
  DeviceBuffer b_device_slab;
  DeviceBuffer c_device_slab;
  DeviceBuffer status_buffer;
  DeviceBuffer a_residue_ptrs;
  DeviceBuffer b_residue_ptrs;
  DeviceBuffer c_residue_ptrs;
  std::vector<NativeT> a_host_slab;
  std::vector<NativeT> b_host_slab;
  std::vector<NativeT> c_host_slab;
  std::vector<rns8_matrix*> a_matrices;
  std::vector<rns8_matrix*> b_matrices;
  std::vector<rns8_matrix*> c_matrices;
};

template <typename NativeT>
rns8_status hip_direct_pack_grouped_native_matrices_device(
    int device_id,
    const NativeT* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t first_source_version) {
  if constexpr (std::is_same_v<NativeT, int64_t>) {
    return rns8::detail::hip_direct_pack_i64_grouped_matrices_device(
        device_id,
        src_slab,
        device_src_slab,
        device_src_slab_bytes,
        matrices,
        task_count,
        expected_semantics,
        device_residue_ptrs,
        rows,
        cols,
        ld,
        prefix,
        first_source_version);
  } else if constexpr (std::is_same_v<NativeT, uint64_t>) {
    return rns8::detail::hip_direct_pack_u64_grouped_matrices_device(
        device_id,
        src_slab,
        device_src_slab,
        device_src_slab_bytes,
        matrices,
        task_count,
        expected_semantics,
        device_residue_ptrs,
        rows,
        cols,
        ld,
        prefix,
        first_source_version);
  } else {
    return rns8::detail::hip_direct_pack_finite_u8_grouped_matrices_device(
        device_id,
        src_slab,
        device_src_slab,
        device_src_slab_bytes,
        matrices,
        task_count,
        expected_semantics,
        device_residue_ptrs,
        rows,
        cols,
        ld,
        static_cast<uint16_t>(prefix),
        first_source_version);
  }
}

template <typename NativeT>
void initialize_grouped_native_rns_batch(
    const Args& args,
    rns8_semantics semantics,
    uint32_t matrix_prefix,
    const std::vector<HostApiBatchTask>& tasks,
    const std::vector<std::vector<NativeT>>& batch_a,
    const std::vector<std::vector<NativeT>>& batch_b,
    std::size_t a_elements,
    std::size_t b_elements,
    GroupedNativeRnsBatch<NativeT>& grouped,
    const char* label) {
  const uint32_t task_count = measured_task_count(args);
  if (tasks.size() != task_count || batch_a.size() != task_count || batch_b.size() != task_count) {
    usage_error(std::string(label) + " grouped batch task count mismatch");
  }
  const std::size_t total_a_elements = checked_task_slab_elements(task_count, a_elements, label);
  const std::size_t total_b_elements = checked_task_slab_elements(task_count, b_elements, label);
  const std::size_t c_elements = checked_elements(args.m, args.n, label);
  const std::size_t total_c_elements = checked_task_slab_elements(task_count, c_elements, label);
  const std::size_t a_slab_bytes = checked_bytes(total_a_elements, sizeof(NativeT), label);
  const std::size_t b_slab_bytes = checked_bytes(total_b_elements, sizeof(NativeT), label);
  const std::size_t c_slab_bytes = checked_bytes(total_c_elements, sizeof(NativeT), label);
  const std::size_t pointer_bytes = checked_bytes(task_count, sizeof(const void*), label);
  const int device_id = args.device_id < 0 ? 0 : args.device_id;

  const std::string a_slab_label = std::string("hip_direct_allocate(") + label + " A slab)";
  const std::string b_slab_label = std::string("hip_direct_allocate(") + label + " B slab)";
  const std::string c_slab_label = std::string("hip_direct_allocate(") + label + " C output slab)";
  const std::string status_label = std::string("hip_direct_allocate(") + label + " export status)";
  const std::string a_ptr_label = std::string("hip_direct_allocate(") + label + " A ptrs)";
  const std::string b_ptr_label = std::string("hip_direct_allocate(") + label + " B ptrs)";
  const std::string c_ptr_label = std::string("hip_direct_allocate(") + label + " C ptrs)";
  grouped.a_device_slab.allocate(device_id, a_slab_bytes, a_slab_label.c_str());
  grouped.b_device_slab.allocate(device_id, b_slab_bytes, b_slab_label.c_str());
  grouped.c_device_slab.allocate(device_id, c_slab_bytes, c_slab_label.c_str());
  grouped.status_buffer.allocate(device_id, sizeof(int), status_label.c_str());
  grouped.a_residue_ptrs.allocate(device_id, pointer_bytes, a_ptr_label.c_str());
  grouped.b_residue_ptrs.allocate(device_id, pointer_bytes, b_ptr_label.c_str());
  grouped.c_residue_ptrs.allocate(device_id, pointer_bytes, c_ptr_label.c_str());

  grouped.a_matrices.reserve(task_count);
  grouped.b_matrices.reserve(task_count);
  grouped.c_matrices.reserve(task_count);
  for (const auto& task : tasks) {
    grouped.a_matrices.push_back(task.a);
    grouped.b_matrices.push_back(task.b);
    grouped.c_matrices.push_back(task.c);
  }
  const std::string a_prepare_label = std::string("hip_direct_prepare_grouped_matrix_residue_pointers(") + label + " A)";
  const std::string b_prepare_label = std::string("hip_direct_prepare_grouped_matrix_residue_pointers(") + label + " B)";
  const std::string c_prepare_label = std::string("hip_direct_prepare_grouped_matrix_residue_pointers(") + label + " C)";
  const std::string a_metadata_label = std::string(label) + " A residue pointer table";
  const std::string b_metadata_label = std::string(label) + " B residue pointer table";
  const std::string c_metadata_label = std::string(label) + " C residue pointer table";
  prepare_grouped_matrix_residue_pointer_table(
      grouped.a_matrices,
      semantics,
      grouped.a_residue_ptrs,
      matrix_prefix,
      a_prepare_label.c_str(),
      a_metadata_label.c_str());
  prepare_grouped_matrix_residue_pointer_table(
      grouped.b_matrices,
      semantics,
      grouped.b_residue_ptrs,
      matrix_prefix,
      b_prepare_label.c_str(),
      b_metadata_label.c_str());
  prepare_grouped_matrix_residue_pointer_table(
      grouped.c_matrices,
      semantics,
      grouped.c_residue_ptrs,
      matrix_prefix,
      c_prepare_label.c_str(),
      c_metadata_label.c_str());

  grouped.a_host_slab = flatten_grouped_pack_slab(batch_a, a_elements, label);
  grouped.b_host_slab = flatten_grouped_pack_slab(batch_b, b_elements, label);
  grouped.c_host_slab.assign(total_c_elements, NativeT{});
}

template <typename NativeT>
void scatter_grouped_native_slab_to_outputs(
    const Args& args,
    const std::vector<NativeT>& host_slab,
    std::vector<std::vector<NativeT>>& batch_c,
    const char* label) {
  const uint32_t task_count = measured_task_count(args);
  const int64_t ldc = output_logical_ld(args);
  const std::size_t compact_elements = checked_elements(args.m, args.n, label);
  const std::size_t padded_elements = checked_elements(args.m, ldc, label);
  const std::size_t total_elements = checked_task_slab_elements(task_count, compact_elements, label);
  if (host_slab.size() != total_elements || batch_c.size() != task_count) {
    usage_error(std::string(label) + " grouped native output slab is not initialized for checksum materialization");
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    const NativeT* task_src = host_slab.data() + static_cast<std::size_t>(task_index) * compact_elements;
    std::vector<NativeT>& task_dst = batch_c[task_index];
    if (task_dst.size() < padded_elements) {
      usage_error(std::string(label) + " grouped native output destination is too small");
    }
    if (ldc == args.n) {
      std::copy(task_src, task_src + compact_elements, task_dst.begin());
      continue;
    }
    for (int64_t row = 0; row < args.m; ++row) {
      const std::size_t src_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(args.n);
      const std::size_t dst_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(ldc);
      std::copy(task_src + src_offset, task_src + src_offset + static_cast<std::size_t>(args.n),
                task_dst.begin() + dst_offset);
    }
  }
}

template <typename NativeT>
void export_bounded_grouped_dispatch_slab(
    const Args& args,
    rns8_semantics semantics,
    uint64_t bound,
    GroupedNativeRnsBatch<NativeT>& grouped,
    const char* label) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_elements = checked_elements(args.m, args.n, label);
  const std::size_t total_elements = checked_task_slab_elements(task_count, compact_elements, label);
  if (!grouped.c_device_slab.ptr || !grouped.status_buffer.ptr || !grouped.c_residue_ptrs.ptr ||
      grouped.c_matrices.size() != task_count || grouped.c_host_slab.size() != total_elements) {
    usage_error(std::string(label) + " grouped bounded export slab is not initialized for the current task count");
  }

  rns8_status export_status = RNS8_INVALID_ARGUMENT;
  if constexpr (std::is_same_v<NativeT, int64_t>) {
    if (semantics != RNS8_BOUNDED_I64) {
      usage_error(std::string(label) + " expected bounded-i64 export semantics");
    }
    export_status = rns8::detail::hip_direct_export_i64_grouped_matrices_to_device(
        grouped.c_matrices.data(),
        task_count,
        grouped.c_residue_ptrs.ptr,
        grouped.c_device_slab.ptr,
        grouped.status_buffer.ptr,
        args.m,
        args.n,
        bound);
  } else {
    if (semantics != RNS8_BOUNDED_U64) {
      usage_error(std::string(label) + " expected bounded-u64 export semantics");
    }
    export_status = rns8::detail::hip_direct_export_u64_grouped_matrices_to_device(
        grouped.c_matrices.data(),
        task_count,
        grouped.c_residue_ptrs.ptr,
        grouped.c_device_slab.ptr,
        grouped.status_buffer.ptr,
        args.m,
        args.n,
        bound);
  }
  if (export_status != RNS8_SUCCESS) {
    fail_status(label, export_status);
  }

  const int64_t total_extent = checked_int64_extent(total_elements, label);
  const rns8_status copy_status = rns8::detail::hip_direct_copy_compact_matrix_device_to_host(
      grouped.c_device_slab.device_id,
      "crt_export_d2h",
      grouped.c_host_slab.data(),
      total_extent,
      grouped.c_device_slab.ptr,
      1,
      total_extent,
      sizeof(NativeT),
      false);
  if (copy_status != RNS8_SUCCESS) {
    fail_status("hip_direct_copy_compact_matrix_device_to_host(grouped bounded export slab)", copy_status);
  }
}

void export_finite_grouped_dispatch_slab(
    const Args& args,
    rns8_semantics semantics,
    GroupedNativeRnsBatch<uint8_t>& grouped,
    const char* label) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_elements = checked_elements(args.m, args.n, label);
  const std::size_t total_elements = checked_task_slab_elements(task_count, compact_elements, label);
  if (!grouped.c_device_slab.ptr || !grouped.c_residue_ptrs.ptr ||
      grouped.c_matrices.size() != task_count || grouped.c_host_slab.size() != total_elements) {
    usage_error(std::string(label) + " grouped finite export slab is not initialized for the current task count");
  }
  if (semantics != RNS8_FINITE_RING_U8 && semantics != RNS8_FINITE_FIELD_U8) {
    usage_error(std::string(label) + " expected finite-u8 export semantics");
  }

  const rns8_status export_status = rns8::detail::hip_direct_export_finite_u8_grouped_matrices_to_device(
      grouped.c_matrices.data(),
      task_count,
      grouped.c_residue_ptrs.ptr,
      grouped.c_device_slab.ptr,
      args.m,
      args.n,
      static_cast<uint16_t>(args.finite_modulus));
  if (export_status != RNS8_SUCCESS) {
    fail_status(label, export_status);
  }

  const int64_t total_extent = checked_int64_extent(total_elements, label);
  const rns8_status copy_status = rns8::detail::hip_direct_copy_compact_matrix_device_to_host(
      grouped.c_device_slab.device_id,
      "finite_export_d2h",
      grouped.c_host_slab.data(),
      total_extent,
      grouped.c_device_slab.ptr,
      1,
      total_extent,
      sizeof(uint8_t),
      false);
  if (copy_status != RNS8_SUCCESS) {
    fail_status("hip_direct_copy_compact_matrix_device_to_host(grouped finite export slab)", copy_status);
  }
}

void scatter_compact_limb_output(
    const uint64_t* compact,
    std::vector<uint64_t>& dst,
    const Args& args,
    const char* label) {
  const int64_t ldc = output_logical_ld(args);
  const std::size_t compact_row_elements =
      checked_limb_elements(1, args.n, args.exact_wide_limb_count, label);
  const std::size_t dst_row_elements =
      checked_limb_elements(1, ldc, args.exact_wide_limb_count, label);
  if (ldc == args.n) {
    const std::size_t total = checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, label);
    std::copy(compact, compact + total, dst.begin());
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const std::size_t src_offset = static_cast<std::size_t>(row) * compact_row_elements;
    const std::size_t dst_offset = static_cast<std::size_t>(row) * dst_row_elements;
    std::copy(compact + src_offset, compact + src_offset + compact_row_elements, dst.begin() + dst_offset);
  }
}

void export_exact_wide_grouped_dispatch_slab(
    const Args& args,
    const std::vector<rns8_matrix*>& c_matrices,
    DeviceBuffer& device_slab,
    const DeviceBuffer& device_residue_ptrs,
    std::vector<uint64_t>& host_slab,
    bool signed_output) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_limb_elements =
      checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped export compact task");
  const std::size_t total_limb_elements =
      checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped export slab");
  if (!device_slab.ptr || !device_residue_ptrs.ptr || c_matrices.size() != task_count ||
      host_slab.size() != total_limb_elements) {
    usage_error("grouped exact-wide export slab is not initialized for the current task count");
  }

  const rns8_status export_status =
      signed_output
          ? rns8::detail::hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
                c_matrices.data(),
                task_count,
                device_residue_ptrs.ptr,
                device_slab.ptr,
                args.m,
                args.n,
                args.exact_wide_limb_count)
          : rns8::detail::hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
                c_matrices.data(),
                task_count,
                device_residue_ptrs.ptr,
                device_slab.ptr,
                args.m,
                args.n,
                args.exact_wide_limb_count);
  if (export_status != RNS8_SUCCESS) {
    fail_status(
        signed_output ? "hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device"
                      : "hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device",
        export_status);
  }

  const rns8_status copy_status = rns8::detail::hip_direct_copy_compact_matrix_device_to_host(
      device_slab.device_id,
      "exact_wide_export_d2h",
      host_slab.data(),
      checked_int64_extent(total_limb_elements, "grouped exact-wide export slab D2H"),
      device_slab.ptr,
      1,
      checked_int64_extent(total_limb_elements, "grouped exact-wide export slab D2H"),
      sizeof(uint64_t),
      false);
  if (copy_status != RNS8_SUCCESS) {
    fail_status("hip_direct_copy_compact_matrix_device_to_host(grouped exact-wide export slab)", copy_status);
  }
}

void scatter_grouped_limb_slab_to_outputs(
    const Args& args,
    const std::vector<uint64_t>& host_slab,
    std::vector<std::vector<uint64_t>>& batch_c) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_limb_elements =
      checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped export compact task");
  const std::size_t total_limb_elements =
      checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped export slab");
  if (host_slab.size() != total_limb_elements || batch_c.size() != task_count) {
    usage_error("grouped exact-wide host slab is not initialized for checksum materialization");
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    const uint64_t* task_src = host_slab.data() + static_cast<std::size_t>(task_index) * compact_limb_elements;
    scatter_compact_limb_output(task_src, batch_c[task_index], args, "grouped exact-wide host output");
  }
}

uint64_t combine_host_api_batch_checksums(const std::vector<uint64_t>& task_checksums) {
  uint64_t checksum = 1469598103934665603ull;
  mix_checksum(checksum, static_cast<uint64_t>(task_checksums.size()));
  for (std::size_t index = 0; index < task_checksums.size(); ++index) {
    mix_checksum(checksum, static_cast<uint64_t>(index));
    mix_checksum(checksum, task_checksums[index]);
  }
  return checksum;
}

template <typename Fn>
rns8_status run_timed_status_operation(const char* label, Fn&& fn) {
  rns8_status status = RNS8_SUCCESS;
  const int code = rns8::detail::run_timed_device_code(label, [&]() {
    status = fn();
    return status == RNS8_SUCCESS ? 0 : 3;
  });
  if (code != 0 && status == RNS8_SUCCESS) {
    return RNS8_BACKEND_FAILURE;
  }
  return status;
}

void capture_vector_alu_schedule(rns8_context* ctx, const Args& args, uint64_t bound, BenchmarkResult& result) {
  result.target_id = benchmark_target_id_for_context(ctx, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-alu schedule)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  rns8_destroy_plan(plan);
}

