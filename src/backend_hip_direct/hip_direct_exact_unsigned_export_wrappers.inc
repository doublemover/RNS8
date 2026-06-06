extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
      case 1:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 1>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 2:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 2>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 3:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 3>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 4:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 4>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 8:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 8>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 16:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 16>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 32:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 32>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      default:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_limbs_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            limb_count,
            d_status);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_unsigned_limbs_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        limb_count,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count) {
  if (!d_residue_ptrs || !d_dst || task_count <= 0 || rows <= 0 || cols <= 0 ||
      prefix <= 0 || prefix > kRns8MaxSupportedPrefix || limb_count <= 0 || limb_count > 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
#define RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(LimbCount) \
      case LimbCount:                                       \
        hipLaunchKernelGGL(                                 \
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_grouped_fixed_prefix_fixed_limbs_kernel< \
                            kRns8MaxSupportedPrefix, LimbCount>), \
            dim3(blocks, task_count),                       \
            dim3(threads),                                  \
            0,                                              \
            nullptr,                                        \
            d_residue_ptrs,                                 \
            d_dst,                                          \
            task_count,                                     \
            rows,                                           \
            cols);                                          \
        break
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(1);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(2);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(3);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(4);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(8);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(16);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(32);
#undef RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS
      default:
        hipLaunchKernelGGL(
            rns8_export_exact_wide_unsigned_grouped_limbs_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_residue_ptrs,
            d_dst,
            task_count,
            rows,
            cols,
            prefix,
            limb_count);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_unsigned_grouped_limbs_kernel,
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        prefix,
        limb_count);
  }
  return static_cast<int>(hipGetLastError());
}

