bool hipblaslt_compiled() {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  return true;
#else
  return false;
#endif
}

rns8_status hipblaslt_create_context(int device_id, rns8_device_info& out, void** handle,
                                     std::string& version) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle) {
    return RNS8_INVALID_ARGUMENT;
  }
  *handle = nullptr;
  version.clear();
  const rns8_status probe_status = hip_direct_probe(device_id, out);
  if (probe_status != RNS8_SUCCESS) {
    return probe_status;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  auto owner = std::make_unique<HipblasLtContext>();
  auto status =
      owner->handle.create([](hipblasLtHandle_t* out) noexcept { return hipblasLtCreate(out); });
  if (status != HIPBLAS_STATUS_SUCCESS)
    return status_from_hipblas(status);
  int raw_version = 0;
  status = hipblasLtGetVersion(owner->handle.get(), &raw_version);
  if (status != HIPBLAS_STATUS_SUCCESS)
    return status_from_hipblas(status);
  auto& environment = owner->environment;
  environment.device_id = device_id;
  environment.library_version = uint32_t(raw_version);
  environment.runtime_version = out.hip_runtime_version;
  environment.driver_version = out.hip_driver_version;
  uint64_t target = 14695981039346656037ull;
  for (const unsigned char* text = reinterpret_cast<const unsigned char*>(out.gcn_arch); *text;
       ++text) {
    target ^= *text;
    target *= 1099511628211ull;
  }
  environment.target_identity = target;
  // Isolate selections across independently configured library contexts and
  // pointer reuse. Configuration changes require a newly created context.
  static std::atomic<uint64_t> next_configuration{1};
  auto configuration = next_configuration.load(std::memory_order_relaxed);
  while (configuration != UINT64_MAX &&
         !next_configuration.compare_exchange_weak(configuration, configuration + 1,
                                                   std::memory_order_relaxed)) {
  }
  if (configuration == UINT64_MAX)
    return RNS8_RANGE_ERROR;
  environment.context_configuration = configuration;
  std::string prepared_version = "hipBLASLt " + std::to_string(raw_version);
  out.backend = RNS8_BACKEND_HIPBLASLT;
  copy_c_string(out.detail, sizeof(out.detail), "hipBLASLt baseline accelerator detected");
  version.swap(prepared_version);
  *handle = owner.release();
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)out;
  (void)handle;
  (void)version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hipblaslt_destroy_context(int device_id, void* handle) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle) {
    return RNS8_SUCCESS;
  }
  auto* owner = static_cast<HipblasLtContext*>(handle);
  if (owner->environment.device_id != device_id)
    return RNS8_INVALID_ARGUMENT;
  const auto status = status_from_hipblas(owner->handle.reset());
  if (status == RNS8_SUCCESS)
    delete owner;
  return status;
#else
  (void)device_id;
  (void)handle;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
