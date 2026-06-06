bool hipblaslt_compiled() {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  return true;
#else
  return false;
#endif
}

rns8_status hipblaslt_create_context(int device_id, rns8_device_info& out, void** handle, std::string& version) {
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
  hipblasLtHandle_t hipblaslt_handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&hipblaslt_handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status_from_hipblas(status);
  }
  int raw_version = 0;
  status = hipblasLtGetVersion(hipblaslt_handle, &raw_version);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    (void)hipblasLtDestroy(hipblaslt_handle);
    return status_from_hipblas(status);
  }
  out.backend = RNS8_BACKEND_HIPBLASLT;
  copy_c_string(out.detail, sizeof(out.detail), "hipBLASLt baseline accelerator detected");
  version = "hipBLASLt ";
  version += std::to_string(raw_version);
  *handle = hipblaslt_handle;
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
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipblasStatus_t status = hipblasLtDestroy(static_cast<hipblasLtHandle_t>(handle));
  return status_from_hipblas(status);
#else
  (void)device_id;
  (void)handle;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

