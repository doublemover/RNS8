#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_materialize_ragged(
    const int8_t*, const RaggedResidueTile*, int, int, int, bool, int8_t*, ResidueLayout, int*);
#endif
rns8_status hip_direct_materialize_ragged_residues(
    int device_id, const void* source, const RaggedResidueLayout& layout,
    void* destination, const ResidueLayout& target, uint64_t instance, uint64_t generation,
    uint64_t max_scratch_bytes, bool& commit_started) {
  commit_started = false;
  if (!compatible_ragged_destination(layout, target, instance, generation) ||
      !nonoverlapping_byte_ranges(source, layout.bytes, destination, target.bytes)) return RNS8_INVALID_ARGUMENT;
  if (layout.bytes > INT32_MAX || target.bytes > INT32_MAX ||
      layout.tiles.size() > INT32_MAX / sizeof(RaggedResidueTile)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t metadata_bytes = layout.tiles.size() * sizeof(RaggedResidueTile);
  if (target.bytes + metadata_bytes + sizeof(int) > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer staged, directory, device_status;
  status = staged.allocate(device_id, target.bytes);
  if (status != RNS8_SUCCESS) return status;
  status = directory.allocate(device_id, metadata_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = device_status.allocate(device_id, sizeof(int));
  if (status != RNS8_SUCCESS) return status;
  status = hip_direct_copy_host_to_device(device_id, directory.get(), layout.tiles.data(), metadata_bytes);
  if (status != RNS8_SUCCESS) return status;
  // Preserve every destination padding byte in the staged publication.
  status = hip_direct_copy_device_to_device(device_id, staged.get(), destination, target.bytes);
  if (status != RNS8_SUCCESS) return status;
  if (hipMemset(device_status.get(), 0, sizeof(int)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  const auto error = timed_hip_operation("ragged_exact_materialization_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_materialize_ragged(
        static_cast<const int8_t*>(source), static_cast<const RaggedResidueTile*>(directory.get()),
        static_cast<int>(layout.tile_rows), static_cast<int>(layout.tile_cols), static_cast<int>(layout.grid_cols),
        layout.signed_value, static_cast<int8_t*>(staged.get()), target, static_cast<int*>(device_status.get())));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  int failure = 0;
  status = hip_direct_copy_device_to_host(device_id, &failure, device_status.get(), sizeof(failure));
  if (status != RNS8_SUCCESS) return status;
  if (failure) return failure == 2 ? RNS8_RANGE_ERROR : RNS8_INVALID_ARGUMENT;
  commit_started = true;
  return hip_direct_copy_device_to_device(device_id, destination, staged.get(), target.bytes);
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
