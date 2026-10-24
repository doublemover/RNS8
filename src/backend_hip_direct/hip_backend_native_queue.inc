#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_queue(const NativeWmmaTask*, const NativeWmmaWork*, uint32_t, uint32_t*, uint32_t, uint32_t, bool);
#endif
rns8_status NativeWmmaPreparedQueue::prepare(int device_id,
    const std::vector<NativeWmmaTask>& tasks, uint32_t workers, uint64_t max_scratch_bytes) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  // Transactional replacement keeps the old allocation until upload succeeds.
  // Charge both allocations against the preparation budget at that peak.
  const auto retained = static_cast<uint64_t>(storage_.bytes());
  const auto available = max_scratch_bytes >= retained ? max_scratch_bytes - retained : 0;
  auto status = build_native_wmma_queue(tasks, workers, available, work, bytes);
  if (status != RNS8_SUCCESS) return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  // Allocate the ordinary-launch control descriptors before replacing any state.
  // These are metadata only; operand contents are never retained on the host.
  auto host_tasks = tasks;
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer storage;
  status = storage.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* device_tasks = static_cast<NativeWmmaTask*>(storage.get());
  auto* device_work = reinterpret_cast<NativeWmmaWork*>(device_tasks + tasks.size());
  auto* cursor = reinterpret_cast<uint32_t*>(device_work + work.size());
  auto error = timed_hip_operation("native_queue_prepare_capability", [&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_capability(reinterpret_cast<int*>(cursor)));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, cursor, sizeof(supported));
  if (status != RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  error = timed_hip_operation("native_queue_prepare_upload", [&]() {
    auto upload = hipMemcpy(device_tasks, tasks.data(), tasks.size() * sizeof(NativeWmmaTask), hipMemcpyHostToDevice);
    if (upload == hipSuccess) upload = hipMemcpy(device_work, work.data(), work.size() * sizeof(NativeWmmaWork), hipMemcpyHostToDevice);
    return upload == hipSuccess ? hipDeviceSynchronize() : upload;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  storage_ = std::move(storage);
  host_tasks_.swap(host_tasks);
  device_id_ = device_id;
  tasks_ = device_tasks; work_ = device_work; cursor_ = cursor;
  count_ = static_cast<uint32_t>(work.size());
  workers_ = std::min(workers, count_);
  uniform_group_ = native_wmma_uniform_group(tasks);
  unsigned_inputs_ = tasks.front().unsigned_inputs != 0;
  poisoned_ = false;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status NativeWmmaPreparedQueue::execute(bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  if (!storage_) return RNS8_INVALID_ARGUMENT;
  if (poisoned_) return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id_);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("native_queue_prepared_execute", [&]() {
    auto launch = hipMemsetAsync(cursor_, 0, sizeof(uint32_t), nullptr);
    if (launch == hipSuccess) {
      outputs_may_be_written = true;
      launch = static_cast<hipError_t>(rns8_hip_native_wmma_queue(
          tasks_, work_, count_, cursor_, workers_, uniform_group_, unsigned_inputs_));
      // Drain the submitted cursor reset even when launch validation fails.
      // A failed synchronization still poisons the owner below.
      const auto completion = hipDeviceSynchronize();
      return launch == hipSuccess ? completion : launch;
    }
    return launch;
  });
  if (error != hipSuccess) { poisoned_ = true; return RNS8_BACKEND_FAILURE; }
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status NativeWmmaPreparedQueue::execute_ordinary(bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  if (!storage_) return RNS8_INVALID_ARGUMENT;
  if (poisoned_) return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id_);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("native_queue_prepared_ordinary_grids", [&]() {
    auto launch = hipSuccess;
    for (const auto& task : host_tasks_) {
      outputs_may_be_written = true;
      launch = static_cast<hipError_t>(rns8_hip_native_wmma_grouped(
          task.a, task.b, task.c, static_cast<int>(task.m), static_cast<int>(task.n),
          static_cast<int>(task.k), static_cast<int>(task.lda), static_cast<int>(task.ldb),
          static_cast<int>(task.ldc), task.prefix, task.group, task.unsigned_inputs != 0));
      if (launch != hipSuccess) break;
    }
    // Drain earlier grids even if a later launch failed. The whole batch is
    // poisoned on either launch or completion failure; no partial publication.
    const auto completion = hipDeviceSynchronize();
    return launch == hipSuccess ? completion : launch;
  });
  if (error != hipSuccess) { poisoned_ = true; return RNS8_BACKEND_FAILURE; }
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

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
