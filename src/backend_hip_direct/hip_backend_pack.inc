
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_pack_i64_persistent_small_device(
    const int64_t* src, int8_t* residues, int rows, int cols, int ld, int prefix);
extern "C" int rns8_hip_direct_pack_i64_4wide_coalesced_device(
    const int64_t* src, int8_t* residues, int rows, int cols, int prefix);
#endif

rns8_status hip_direct_pack_i64_device(
    int device_id,
    const int64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err =
      timed_hip_operation("pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_i64_device(
        static_cast<const int64_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err =
      timed_hip_operation("pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u64_device(
        static_cast<const uint64_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_i64_grouped_matrices_device(
    int device_id,
    const int64_t* src_slab,
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
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src_slab || !device_src_slab || !device_residue_ptrs || !checked_i32_shape(rows, cols, ld, prefix) ||
      !checked_grouped_source_versions(first_source_version, task_count) ||
      (expected_semantics != RNS8_BOUNDED_I64 && expected_semantics != RNS8_EXACT_WIDE_SIGNED)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t source_bytes = 0;
  int64_t source_task_stride = 0;
  if (!checked_grouped_source_bytes(task_count, rows, ld, sizeof(int64_t), &source_bytes, &source_task_stride) ||
      device_src_slab_bytes < source_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  int matrix_device_id = -1;
  uint32_t matrix_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      matrices,
      task_count,
      expected_semantics,
      rows,
      cols,
      false,
      &matrix_device_id,
      &matrix_prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (matrix_device_id != device_id || matrix_prefix != prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  hipError_t err = timed_hip_operation("pack_h2d", [&]() {
    return hipMemcpy(device_src_slab, src_slab, source_bytes, hipMemcpyHostToDevice);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_i64_grouped_device(
        static_cast<const int64_t*>(device_src_slab),
        static_cast<int8_t* const*>(const_cast<void*>(device_residue_ptrs)),
        static_cast<int>(task_count),
        source_task_stride,
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = matrices[index];
    mark_device_residue_output_current(matrix);
    matrix->source_version = first_source_version + static_cast<uint64_t>(index);
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)src_slab;
  (void)device_src_slab;
  (void)device_src_slab_bytes;
  (void)matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_residue_ptrs;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  (void)first_source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_u64_grouped_matrices_device(
    int device_id,
    const uint64_t* src_slab,
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
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src_slab || !device_src_slab || !device_residue_ptrs || !checked_i32_shape(rows, cols, ld, prefix) ||
      !checked_grouped_source_versions(first_source_version, task_count) ||
      (expected_semantics != RNS8_BOUNDED_U64 && expected_semantics != RNS8_EXACT_WIDE_UNSIGNED)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t source_bytes = 0;
  int64_t source_task_stride = 0;
  if (!checked_grouped_source_bytes(task_count, rows, ld, sizeof(uint64_t), &source_bytes, &source_task_stride) ||
      device_src_slab_bytes < source_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  int matrix_device_id = -1;
  uint32_t matrix_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      matrices,
      task_count,
      expected_semantics,
      rows,
      cols,
      false,
      &matrix_device_id,
      &matrix_prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (matrix_device_id != device_id || matrix_prefix != prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  hipError_t err = timed_hip_operation("pack_h2d", [&]() {
    return hipMemcpy(device_src_slab, src_slab, source_bytes, hipMemcpyHostToDevice);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u64_grouped_device(
        static_cast<const uint64_t*>(device_src_slab),
        static_cast<int8_t* const*>(const_cast<void*>(device_residue_ptrs)),
        static_cast<int>(task_count),
        source_task_stride,
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = matrices[index];
    mark_device_residue_output_current(matrix);
    matrix->source_version = first_source_version + static_cast<uint64_t>(index);
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)src_slab;
  (void)device_src_slab;
  (void)device_src_slab_bytes;
  (void)matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_residue_ptrs;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  (void)first_source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_native_i64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_native || !device_residues || !checked_i32_shape(rows, cols, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("native_i64_to_rns_kernel", [&]() {
    const int launch_status = rns8_hip_direct_pack_i64_device(
        static_cast<const int64_t*>(device_native),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        static_cast<int>(prefix));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_native;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_native_u64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_native || !device_residues || !checked_i32_shape(rows, cols, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("native_u64_to_rns_kernel", [&]() {
    const int launch_status = rns8_hip_direct_pack_u64_device(
        static_cast<const uint64_t*>(device_native),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        static_cast<int>(prefix));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_native;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_finite_u8_device(
    int device_id,
    const uint8_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || modulus < 2 || modulus > 256 ||
      rows <= 0 || cols <= 0 || ld < cols || rows > std::numeric_limits<int>::max() ||
      cols > std::numeric_limits<int>::max() || ld > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_output_bytes(rows, ld, sizeof(uint8_t)) || !checked_output_bytes(rows, cols, sizeof(int8_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(uint8_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation(
      "finite_pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("finite_pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u8_modulus_device(
        static_cast<const uint8_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(modulus));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_finite_u8_grouped_matrices_device(
    int device_id,
    const uint8_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus,
    uint64_t first_source_version) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src_slab || !device_src_slab || !device_residue_ptrs || !checked_i32_shape(rows, cols, ld, 1) ||
      !checked_grouped_source_versions(first_source_version, task_count) ||
      !valid_finite_modulus_for_semantics(expected_semantics, modulus)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t source_bytes = 0;
  int64_t source_task_stride = 0;
  if (!checked_grouped_source_bytes(task_count, rows, ld, sizeof(uint8_t), &source_bytes, &source_task_stride) ||
      device_src_slab_bytes < source_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  int matrix_device_id = -1;
  uint32_t matrix_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      matrices,
      task_count,
      expected_semantics,
      rows,
      cols,
      false,
      &matrix_device_id,
      &matrix_prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (matrix_device_id != device_id || matrix_prefix != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  hipError_t err = timed_hip_operation("finite_pack_h2d", [&]() {
    return hipMemcpy(device_src_slab, src_slab, source_bytes, hipMemcpyHostToDevice);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("finite_pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u8_grouped_modulus_device(
        static_cast<const uint8_t*>(device_src_slab),
        static_cast<int8_t* const*>(const_cast<void*>(device_residue_ptrs)),
        static_cast<int>(task_count),
        source_task_stride,
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(modulus));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = matrices[index];
    mark_device_residue_output_current(matrix);
    matrix->finite_modulus = modulus;
    matrix->source_version = first_source_version + static_cast<uint64_t>(index);
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)src_slab;
  (void)device_src_slab;
  (void)device_src_slab_bytes;
  (void)matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_residue_ptrs;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)modulus;
  (void)first_source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

