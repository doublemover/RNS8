bool hip_direct_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return true;
#else
  return false;
#endif
}

rns8_status hip_direct_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess || count <= 0) {
    copy_c_string(out.detail, sizeof(out.detail), hipGetErrorString(err));
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id >= count) {
    return RNS8_INVALID_ARGUMENT;
  }

  hipDeviceProp_t prop{};
  err = hipGetDeviceProperties(&prop, device_id);
  if (err != hipSuccess) {
    copy_c_string(out.detail, sizeof(out.detail), hipGetErrorString(err));
    return RNS8_BACKEND_FAILURE;
  }

  int runtime_version = 0;
  int driver_version = 0;
  (void)hipRuntimeGetVersion(&runtime_version);
  (void)hipDriverGetVersion(&driver_version);

  out.backend = RNS8_BACKEND_HIP_DIRECT;
  out.device_id = device_id;
  out.hip_available = 1;
  out.hip_runtime_version = static_cast<uint32_t>(runtime_version);
  out.hip_driver_version = static_cast<uint32_t>(driver_version);
  out.global_mem_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
  copy_c_string(out.name, sizeof(out.name), prop.name);
  copy_c_string(out.gcn_arch, sizeof(out.gcn_arch), prop.gcnArchName);
  copy_c_string(out.detail, sizeof(out.detail), "direct HIP runtime detected");
  return RNS8_SUCCESS;
#else
  (void)device_id;
  copy_c_string(out.detail, sizeof(out.detail), "RNS8 was built without the direct HIP backend");
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_allocate(int device_id, std::size_t bytes, void** out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!out || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out = nullptr;
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  void* ptr = nullptr;
  const hipError_t err = hipMalloc(&ptr, bytes);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_allocate_calls.fetch_add(1, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
  *out = ptr;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)bytes;
  (void)out;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_free(int device_id, void* ptr) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!ptr) {
    return RNS8_SUCCESS;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipFree(ptr);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_free_calls.fetch_add(1, std::memory_order_relaxed);
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)ptr;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_zero(int device_id, void* ptr, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!ptr || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipMemset(ptr, 0, bytes);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)ptr;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      timed_hip_operation("residue_d2h_sync", [&]() { return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost); });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_host_to_device_labeled(
    int device_id,
    const char* timing_label,
    void* dst,
    const void* src,
    std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!timing_label || timing_label[0] == '\0' || !dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      timed_hip_operation(timing_label, [&]() { return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice); });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)timing_label;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src, std::size_t bytes) {
  return hip_direct_copy_host_to_device_labeled(device_id, "residue_h2d_sync", dst, src, bytes);
}

rns8_status hip_direct_copy_device_to_device(int device_id, void* dst, const void* src, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      timed_hip_operation("residue_d2d_sync", [&]() { return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice); });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_compact_matrix_device_to_host(
    int device_id,
    const char* timing_label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!timing_label) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      copy_compact_matrix_device_to_host(
          device_id, timing_label, dst, dst_ld, src, rows, cols, cell_bytes, default_padded_staging);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)timing_label;
  (void)dst;
  (void)dst_ld;
  (void)src;
  (void)rows;
  (void)cols;
  (void)cell_bytes;
  (void)default_padded_staging;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!buffer || !capacity || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (*buffer && *capacity >= bytes) {
    return RNS8_SUCCESS;
  }
  if (*buffer) {
    const rns8_status free_status = hip_direct_free(device_id, *buffer);
    if (free_status != RNS8_SUCCESS) {
      return free_status;
    }
    *buffer = nullptr;
    *capacity = 0;
  }
  void* ptr = nullptr;
  const rns8_status alloc_status = hip_direct_allocate(device_id, bytes, &ptr);
  if (alloc_status != RNS8_SUCCESS) {
    return alloc_status;
  }
  *buffer = ptr;
  *capacity = bytes;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)bytes;
  (void)buffer;
  (void)capacity;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

