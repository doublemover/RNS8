#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int
rns8_hip_export_canonical_balanced_crt(const int8_t *, uint64_t *, int,
                                       const balanced_crt::Plan *, ResidueLayout);
#endif
rns8_status hip_direct_export_canonical_balanced_crt_device(
    int device_id, const void *residues, void *limbs, int64_t cells,
    uint32_t prefix) {
  if (!residues || !limbs || cells <= 0 || !prefix || prefix > 20)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(1, cells, cells, prefix, 24))
    return RNS8_UNSUPPORTED_BACKEND;
  ResidueLayout source;
  if (!make_residue_layout(ResidueLayoutKind::modulus_major, 1,
                           static_cast<uint64_t>(cells), static_cast<uint64_t>(cells),
                           prefix, 0, 0, source)) return RNS8_INVALID_ARGUMENT;
  return hip_direct_export_canonical_balanced_crt_layout_device(device_id, residues, source, limbs);
}

rns8_status hip_direct_export_canonical_balanced_crt_layout_device(
    int device_id, const void* residues, const ResidueLayout& source, void* limbs) {
  if (!residues || !limbs || !valid_residue_layout(source)) return RNS8_INVALID_ARGUMENT;
  // A valid descriptor proves rows*cols <= bytes. Keep the existing conservative
  // output domain so the launch and three-limb cell indexing cannot overflow.
  const uint64_t cells = source.rows * source.cols;
  if (cells > INT32_MAX || !qualified_i32_matrix_address(1, cells, cells, source.prefix, 24))
    return RNS8_UNSUPPORTED_BACKEND;
  if (!nonoverlapping_byte_ranges(residues, source.bytes, limbs, cells * 24))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  try {
    const auto &plan = balanced_crt::plan(source.prefix);
    const auto status = set_hip_device(device_id);
    if (status != RNS8_SUCCESS)
      return status;
    const auto error =
        timed_hip_operation(source.kind == ResidueLayoutKind::tile_major
            ? "balanced_crt_tile_major_export_kernel" : "balanced_crt_canonical_export_kernel", [&]() {
          const auto launch =
              static_cast<hipError_t>(rns8_hip_export_canonical_balanced_crt(
                  static_cast<const int8_t *>(residues),
                  static_cast<uint64_t *>(limbs), static_cast<int>(cells),
                  &plan, source));
          return launch == hipSuccess ? hipDeviceSynchronize() : launch;
        });
    return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  } catch (const std::exception &) {
    return RNS8_INTERNAL_ERROR;
  }
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
