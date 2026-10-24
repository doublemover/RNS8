#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_queue(const NativeWmmaTask*, const NativeWmmaWork*, uint32_t, uint32_t*, uint32_t, uint32_t, bool);
#endif
rns8_status hip_direct_gemm_native_wmma_queue(
    int device_id, const std::vector<NativeWmmaTask>& tasks, uint32_t workers,
    uint64_t max_scratch_bytes, bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  auto status = build_native_wmma_queue(tasks, workers, max_scratch_bytes, work, bytes);
  if (status != RNS8_SUCCESS) return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto uniform_group = native_wmma_uniform_group(tasks);
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer storage;
  status = storage.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* device_tasks = static_cast<NativeWmmaTask*>(storage.get());
  auto* device_work = reinterpret_cast<NativeWmmaWork*>(device_tasks + tasks.size());
  auto* cursor = reinterpret_cast<uint32_t*>(device_work + work.size());
  auto error = timed_hip_operation("native_queue_capability", [&]() {
    auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_capability(reinterpret_cast<int*>(cursor)));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, cursor, sizeof(supported));
  if (status != RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  error = timed_hip_operation("native_queue_descriptor_upload", [&]() {
    auto upload = hipMemcpy(device_tasks, tasks.data(), tasks.size() * sizeof(NativeWmmaTask), hipMemcpyHostToDevice);
    if (upload == hipSuccess) upload = hipMemcpy(device_work, work.data(), work.size() * sizeof(NativeWmmaWork), hipMemcpyHostToDevice);
    if (upload == hipSuccess) upload = hipMemset(cursor, 0, sizeof(uint32_t));
    return upload == hipSuccess ? hipDeviceSynchronize() : upload;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  outputs_may_be_written = true;
  error = timed_hip_operation("native_queue_persistent_wmma", [&]() {
    auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_queue(device_tasks, device_work,
        static_cast<uint32_t>(work.size()), cursor, std::min(workers, static_cast<uint32_t>(work.size())),
        uniform_group, tasks.front().unsigned_inputs != 0));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
