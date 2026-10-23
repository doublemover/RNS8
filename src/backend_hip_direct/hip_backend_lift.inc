rns8_status hip_direct_lift_exact_suffix(
    int device_id, void* residues, int64_t cells, uint32_t source_prefix, uint32_t target_prefix,
    bool signed_value, const uint64_t* bound, bool& commit_started) {
  commit_started = false;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!residues || !bound || cells <= 0 || cells > INT32_MAX - 256 || source_prefix == 0 ||
      target_prefix > RNS8_MAX_SUPPORTED_PREFIX || source_prefix >= target_prefix) return RNS8_INVALID_ARGUMENT;
  const auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS) return selected;
  const std::size_t bytes = static_cast<std::size_t>(cells) * (target_prefix - source_prefix);
  hip_direct_device_buffer suffix, device_status;
  auto status = suffix.allocate(device_id, bytes);
  if (status != RNS8_SUCCESS) return status;
  status = device_status.allocate(device_id, sizeof(int));
  if (status != RNS8_SUCCESS) return status;
  int host_status = 0;
  if (hipMemset(device_status.get(), 0, sizeof(int)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  const auto error = timed_hip_operation("exact_lift_suffix_kernel", [&]() {
    const int code = rns8_hip_direct_lift_suffix_device(
        static_cast<const int8_t*>(residues), static_cast<int8_t*>(suffix.get()), static_cast<int>(cells),
        static_cast<int>(source_prefix), static_cast<int>(target_prefix), signed_value,
        bound[0], bound[1], bound[2], static_cast<int*>(device_status.get()));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  if (error != hipSuccess || hipMemcpy(&host_status, device_status.get(), sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  if (host_status) return host_status == 2 ? RNS8_RANGE_ERROR : RNS8_INVALID_ARGUMENT;
  commit_started = true;
  auto* destination = static_cast<int8_t*>(residues) + static_cast<std::size_t>(cells) * source_prefix;
  return hip_direct_copy_device_to_device(device_id, destination, suffix.get(), bytes);
#else
  (void)device_id; (void)residues; (void)cells; (void)source_prefix; (void)target_prefix;
  (void)signed_value; (void)bound;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
