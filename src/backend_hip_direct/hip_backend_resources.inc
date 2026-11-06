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
  AllocationObservationLedger::Transaction observation(hip_direct_allocation_ledger(device_id));
  const hipError_t err = hipMalloc(&ptr, bytes);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_allocate_calls.fetch_add(1, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
  observation.allocated(ptr, bytes, AllocationObservationDomain::device);
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
  AllocationObservationLedger::Transaction observation(hip_direct_allocation_ledger(device_id));
  const hipError_t err = hipFree(ptr);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_free_calls.fetch_add(1, std::memory_order_relaxed);
  observation.released(ptr, AllocationObservationDomain::device);
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
  // hipMemset may return before its default-stream write completes. A caller
  // can immediately hand this storage to a nonblocking stream, which has no
  // implicit dependency on that write. Finish initialization before publishing
  // the allocation; synchronizing only its stream avoids a device-wide fence.
  const hipError_t err = hipMemset(ptr, 0, bytes);
  if (err != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  return hipStreamSynchronize(nullptr) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)ptr;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src,
                                           std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("residue_d2h_sync", [&]() {
    if (bytes <= HostReadbackSlot::capacity)
      return tracked_status_copy(device_id, dst, src, bytes);
    return tracked_sync_copy(dst, src, bytes, hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
namespace {
template <typename Transfer>
rns8_status copy_selected_owned_host_payload(int device_id, std::shared_ptr<void> payload,
                                             std::size_t bytes, const char* label,
                                             Transfer&& transfer) {
  HostTransferPayload retained(hip_direct_defer_buffer_cleanup);
  const auto admitted = retained.begin(device_id, std::move(payload), bytes);
  if (admitted != RNS8_SUCCESS)
    return admitted;
  const auto copied = timed_hip_operation(label, [&]() { return transfer(retained.data()); });
  if (copied != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  retained.complete();
  return RNS8_SUCCESS;
}

template <typename Transfer>
rns8_status copy_owned_host_payload(int device_id, std::shared_ptr<void> payload, std::size_t bytes,
                                    const char* label, Transfer&& transfer) {
  const auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS)
    return selected;
  return copy_selected_owned_host_payload(device_id, std::move(payload), bytes, label,
                                          std::forward<Transfer>(transfer));
}

rns8_status copy_borrowed_host_to_device(int device_id, const char* label, void* destination,
                                         const void* source, const HostUploadPlan& plan) {
  const auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS)
    return selected;
  const bool record_staging = hip_direct_timing_enabled();
  const auto start =
      record_staging ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const auto staged = g_pageable_upload_staging.snapshot(source, plan);
  if (staged != RNS8_SUCCESS)
    return staged;
  rns8_status copied;
  try {
    if (record_staging)
      record_host_staging_copy_sample("upload_host_staging_copy", start,
                                      std::chrono::steady_clock::now());
    copied = copy_selected_owned_host_payload(
        device_id, g_pageable_upload_staging.owner(), plan.device_bytes, label, [&](void* host) {
          return tracked_sync_copy(destination, host, plan.device_bytes, hipMemcpyHostToDevice);
        });
  } catch (...) {
    g_pageable_upload_staging.discard();
    throw;
  }
  if (copied != RNS8_SUCCESS)
    g_pageable_upload_staging.discard();
  else
    g_pageable_upload_staging.trim(kPageableTransferStagingMaxCachedBytes);
  return copied;
}
} // namespace
#endif

rns8_status hip_direct_copy_device_to_owned_host(int device_id, std::shared_ptr<void> destination,
                                                 const void* source, std::size_t bytes) {
  if (!destination || !source || !bytes)
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return copy_owned_host_payload(
      device_id, std::move(destination), bytes, "residue_d2h_sync",
      [&](void* host) { return tracked_sync_copy(host, source, bytes, hipMemcpyDeviceToHost); });
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_owned_host_to_device(int device_id, void* destination,
                                                 std::shared_ptr<void> source, std::size_t bytes,
                                                 const char* timing_label) {
  if (!destination || !source || !bytes || !timing_label || !*timing_label)
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return copy_owned_host_payload(
      device_id, std::move(source), bytes, timing_label, [&](void* host) {
        return tracked_sync_copy(destination, host, bytes, hipMemcpyHostToDevice);
      });
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_host_to_device_labeled(int device_id, const char* timing_label,
                                                   void* dst, const void* src, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!timing_label || timing_label[0] == '\0' || !dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const HostUploadPlan plan{1, bytes, bytes, bytes, bytes, 1};
  if (!valid_host_upload_plan(plan))
    return RNS8_INVALID_ARGUMENT;
  return copy_borrowed_host_to_device(device_id, timing_label, dst, src, plan);
#else
  (void)device_id;
  (void)timing_label;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src,
                                           std::size_t bytes) {
  return hip_direct_copy_host_to_device_labeled(device_id, "residue_h2d_sync", dst, src, bytes);
}

rns8_status hip_direct_upload_host_matrix(int device_id, const char* timing_label, void* dst,
                                          std::size_t capacity, const void* src,
                                          const HostUploadPlan& plan) {
  if (!timing_label || !*timing_label || !dst || !src || !valid_host_upload_plan(plan) ||
      capacity < plan.device_bytes)
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return copy_borrowed_host_to_device(device_id, timing_label, dst, src, plan);
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_device_to_device(int device_id, void* dst, const void* src,
                                             std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("residue_d2d_sync", [&]() {
    return tracked_sync_copy(dst, src, bytes, hipMemcpyDeviceToDevice);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_compact_matrix_device_to_host(int device_id, const char* timing_label,
                                                          void* dst, int64_t dst_ld,
                                                          const void* src, int64_t rows,
                                                          int64_t cols, std::size_t cell_bytes,
                                                          bool default_padded_staging) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!timing_label) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = copy_compact_matrix_device_to_host(
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

rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer,
                                            std::size_t* capacity) {
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
