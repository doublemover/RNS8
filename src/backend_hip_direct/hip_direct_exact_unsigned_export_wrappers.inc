namespace {

constexpr int kRns8ExactWideUnsignedExportPrefix18 = 18;

void launch_rns8_exact_wide_unsigned_generic(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status,
    int blocks,
    int threads) {
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

template <int Prefix>
int launch_rns8_exact_wide_unsigned_fixed_prefix(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int limb_count,
    int* d_status,
    int blocks,
    int quad_blocks,
    int threads) {
  switch (limb_count) {
    case 1:
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 1>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 2>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 3>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 4>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 8>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 16>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 32>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_limbs_fixed_prefix_kernel<Prefix>),
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
  return static_cast<int>(hipGetLastError());
}

template <int Prefix>
int launch_rns8_exact_wide_unsigned_tree_crt_fixed_prefix(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int limb_count,
    int* d_status,
    int blocks,
    int quad_blocks,
    int threads) {
  switch (limb_count) {
    case 1:
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 1>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 2>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 3>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 4>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 8>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 16>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_fixed_limbs_quad_kernel<Prefix, 32>),
          dim3(quad_blocks),
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
          HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_tree_crt_fixed_prefix_kernel<Prefix>),
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
  return static_cast<int>(hipGetLastError());
}

void launch_rns8_exact_wide_unsigned_grouped_generic(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int blocks,
    int threads) {
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

template <int Prefix>
int launch_rns8_exact_wide_unsigned_grouped_fixed_prefix(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int limb_count,
    int blocks,
    int quad_blocks,
    int threads) {
  switch (limb_count) {
#define RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(LimbCount)                                    \
    case LimbCount:                                                                             \
      hipLaunchKernelGGL(                                                                       \
          HIP_KERNEL_NAME(                                                                      \
              rns8_export_exact_wide_unsigned_grouped_fixed_prefix_fixed_limbs_quad_kernel<Prefix, LimbCount>), \
          dim3(quad_blocks, task_count),                                                        \
          dim3(threads),                                                                        \
          0,                                                                                    \
          nullptr,                                                                              \
          d_residue_ptrs,                                                                       \
          d_dst,                                                                                \
          task_count,                                                                           \
          rows,                                                                                 \
          cols);                                                                                \
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
      launch_rns8_exact_wide_unsigned_grouped_generic(
          d_residue_ptrs, d_dst, task_count, rows, cols, Prefix, limb_count, blocks, threads);
      break;
  }
  return static_cast<int>(hipGetLastError());
}

}  // namespace

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
  const int quad_blocks = ((elements + 3) / 4 + threads - 1) / threads;
  if (prefix == kRns8ExactWideUnsignedExportPrefix18) {
    return launch_rns8_exact_wide_unsigned_fixed_prefix<kRns8ExactWideUnsignedExportPrefix18>(
        d_residues, d_dst, rows, cols, limb_count, d_status, blocks, quad_blocks, threads);
  }
  if (prefix == kRns8MaxSupportedPrefix) {
    return launch_rns8_exact_wide_unsigned_fixed_prefix<kRns8MaxSupportedPrefix>(
        d_residues, d_dst, rows, cols, limb_count, d_status, blocks, quad_blocks, threads);
  }
  launch_rns8_exact_wide_unsigned_generic(
      d_residues, d_dst, rows, cols, prefix, limb_count, d_status, blocks, threads);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status,
    void* stream) {
  if (!d_residues || !d_dst || rows <= 0 || cols <= 0 || limb_count != 4 ||
      (prefix != kRns8ExactWideUnsignedExportPrefix18 && prefix != kRns8MaxSupportedPrefix)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  const int quad_blocks = ((elements + 3) / 4 + threads - 1) / threads;
  const hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (prefix == kRns8ExactWideUnsignedExportPrefix18) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<
                        kRns8ExactWideUnsignedExportPrefix18,
                        4>),
        dim3(quad_blocks),
        dim3(threads),
        0,
        hip_stream,
        d_residues,
        d_dst,
        rows,
        cols,
        d_status);
    return static_cast<int>(hipGetLastError());
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_quad_kernel<kRns8MaxSupportedPrefix, 4>),
      dim3(quad_blocks),
      dim3(threads),
      0,
      hip_stream,
      d_residues,
      d_dst,
      rows,
      cols,
      d_status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_tree_crt_limbs_device(
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
  const int quad_blocks = ((elements + 3) / 4 + threads - 1) / threads;
  if (prefix == kRns8ExactWideUnsignedExportPrefix18) {
    return launch_rns8_exact_wide_unsigned_tree_crt_fixed_prefix<kRns8ExactWideUnsignedExportPrefix18>(
        d_residues, d_dst, rows, cols, limb_count, d_status, blocks, quad_blocks, threads);
  }
  if (prefix == kRns8MaxSupportedPrefix) {
    return launch_rns8_exact_wide_unsigned_tree_crt_fixed_prefix<kRns8MaxSupportedPrefix>(
        d_residues, d_dst, rows, cols, limb_count, d_status, blocks, quad_blocks, threads);
  }
  return static_cast<int>(hipErrorInvalidValue);
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
  const int quad_blocks = ((elements + 3) / 4 + threads - 1) / threads;
  if (prefix == kRns8ExactWideUnsignedExportPrefix18) {
    return launch_rns8_exact_wide_unsigned_grouped_fixed_prefix<kRns8ExactWideUnsignedExportPrefix18>(
        d_residue_ptrs, d_dst, task_count, rows, cols, limb_count, blocks, quad_blocks, threads);
  }
  if (prefix == kRns8MaxSupportedPrefix) {
    return launch_rns8_exact_wide_unsigned_grouped_fixed_prefix<kRns8MaxSupportedPrefix>(
        d_residue_ptrs, d_dst, task_count, rows, cols, limb_count, blocks, quad_blocks, threads);
  }
  launch_rns8_exact_wide_unsigned_grouped_generic(
      d_residue_ptrs, d_dst, task_count, rows, cols, prefix, limb_count, blocks, threads);
  return static_cast<int>(hipGetLastError());
}
