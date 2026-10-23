#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_convert_residue_layout(const int8_t*, ResidueLayout, int8_t*, ResidueLayout);
#endif

rns8_status hip_direct_convert_residue_layout(
    int device_id, const void* src, const ResidueLayout& source, void* dst, const ResidueLayout& destination) {
  if (!compatible_residue_layouts(source, destination) ||
      !nonoverlapping_byte_ranges(src, source.bytes, dst, destination.bytes)) return RNS8_INVALID_ARGUMENT;
  if (source.bytes > INT32_MAX || destination.bytes > INT32_MAX) return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("residue_layout_convert_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_convert_residue_layout(
        static_cast<const int8_t*>(src), source, static_cast<int8_t*>(dst), destination));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
