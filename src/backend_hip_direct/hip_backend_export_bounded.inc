rns8_status hip_direct_export_i64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const bool status_structurally_elided = (prefix >= 9);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (!status_structurally_elided) {
    status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
    if (status != RNS8_SUCCESS) {
      return status;
    }
  }
  hipError_t err = hipSuccess;
  if (!status_structurally_elided) {
    err = timed_hip_operation("crt_export_status_memset", [&]() {
      return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
    });
    if (err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_i64_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<int64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        status_structurally_elided ? nullptr : static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (!status_structurally_elided) {
    int host_status = 0;
    err = timed_hip_operation("crt_export_status_d2h", [&]() {
      return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
    });
    if (err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
    if (host_status != static_cast<int>(RNS8_SUCCESS)) {
      return static_cast<rns8_status>(host_status);
    }
  }
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)bound;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_i64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residue_ptrs || !device_dst || !device_status || task_count == 0 ||
      task_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      task_count > static_cast<uint32_t>(std::numeric_limits<std::size_t>::max() / sizeof(int64_t)) ||
      !checked_output_bytes(rows, cols, static_cast<std::size_t>(task_count) * sizeof(int64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  uint32_t prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      matrices,
      task_count,
      RNS8_BOUNDED_I64,
      rows,
      cols,
      true,
      &device_id,
      &prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const rns8_status device_status_result = set_hip_device(device_id);
  if (device_status_result != RNS8_SUCCESS) {
    return device_status_result;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(device_status, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_i64_grouped_device(
        static_cast<const int8_t* const*>(device_residue_ptrs),
        static_cast<int64_t*>(device_dst),
        static_cast<int>(task_count),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        static_cast<int*>(device_status));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, device_status, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  return host_status == static_cast<int>(RNS8_SUCCESS)
             ? RNS8_SUCCESS
             : static_cast<rns8_status>(host_status);
#else
  (void)matrices;
  (void)task_count;
  (void)device_residue_ptrs;
  (void)device_dst;
  (void)device_status;
  (void)rows;
  (void)cols;
  (void)bound;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_i64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !dst || ld < cols ||
      !checked_matrix_elements_i32(rows, cols) || !checked_output_bytes(rows, cols, sizeof(int64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  if (!all_zero_output_tiles) {
    if (!status_buffer || !status_bytes || !device_entries || !device_bounds || entry_count == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
        max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
    if (blocks_per_tile == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (all_zero_output_tiles) {
    const hipError_t zero_err = timed_hip_operation("crt_export_kernel", [&]() {
      const hipError_t memset_status = hipMemsetAsync(*export_buffer, 0, output_bytes, nullptr);
      return memset_status == hipSuccess ? hipDeviceSynchronize() : memset_status;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
    const hipError_t copy_err = copy_compact_matrix_device_to_host(
        device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
    return copy_err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_i64_scheduled_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<int64_t*>(*export_buffer),
        static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
        static_cast<const uint64_t*>(device_bounds),
        static_cast<const uint8_t*>(zero_a_rows),
        static_cast<const uint8_t*>(zero_b_cols),
        static_cast<int>(entry_count),
        static_cast<int>(max_tile_elements),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)device_entries;
  (void)device_bounds;
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)entry_count;
  (void)max_tile_elements;
  (void)all_zero_output_tiles;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_u64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u64_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)bound;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_u64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residue_ptrs || !device_dst || !device_status || task_count == 0 ||
      task_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      task_count > static_cast<uint32_t>(std::numeric_limits<std::size_t>::max() / sizeof(uint64_t)) ||
      !checked_output_bytes(rows, cols, static_cast<std::size_t>(task_count) * sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  uint32_t prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      matrices,
      task_count,
      RNS8_BOUNDED_U64,
      rows,
      cols,
      true,
      &device_id,
      &prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const rns8_status device_status_result = set_hip_device(device_id);
  if (device_status_result != RNS8_SUCCESS) {
    return device_status_result;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(device_status, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u64_grouped_device(
        static_cast<const int8_t* const*>(device_residue_ptrs),
        static_cast<uint64_t*>(device_dst),
        static_cast<int>(task_count),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        static_cast<int*>(device_status));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, device_status, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  return host_status == static_cast<int>(RNS8_SUCCESS)
             ? RNS8_SUCCESS
             : static_cast<rns8_status>(host_status);
#else
  (void)matrices;
  (void)task_count;
  (void)device_residue_ptrs;
  (void)device_dst;
  (void)device_status;
  (void)rows;
  (void)cols;
  (void)bound;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_u64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !dst || ld < cols ||
      !checked_matrix_elements_i32(rows, cols) || !checked_output_bytes(rows, cols, sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  if (!all_zero_output_tiles) {
    if (!status_buffer || !status_bytes || !device_entries || !device_bounds || entry_count == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
        max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
    if (blocks_per_tile == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (all_zero_output_tiles) {
    const hipError_t zero_err = timed_hip_operation("crt_export_kernel", [&]() {
      const hipError_t memset_status = hipMemsetAsync(*export_buffer, 0, output_bytes, nullptr);
      return memset_status == hipSuccess ? hipDeviceSynchronize() : memset_status;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
    const hipError_t copy_err = copy_compact_matrix_device_to_host(
        device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
    return copy_err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u64_scheduled_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
        static_cast<const uint64_t*>(device_bounds),
        static_cast<const uint8_t*>(zero_a_rows),
        static_cast<const uint8_t*>(zero_b_cols),
        static_cast<int>(entry_count),
        static_cast<int>(max_tile_elements),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)device_entries;
  (void)device_bounds;
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)entry_count;
  (void)max_tile_elements;
  (void)all_zero_output_tiles;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

