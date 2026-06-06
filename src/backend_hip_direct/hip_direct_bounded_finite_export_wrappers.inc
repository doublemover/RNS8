extern "C" int rns8_hip_direct_export_u8_modulus_device(
    const int8_t* d_residues,
    uint8_t* d_dst,
    int rows,
    int cols,
    int ld,
    int modulus) {
  if (!d_residues || !d_dst || rows <= 0 || cols <= 0 || ld < cols || modulus < 2 || modulus > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (modulus == 256) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<256>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 255) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<255>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 251) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<251>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  hipLaunchKernelGGL(
      rns8_export_u8_modulus_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      rows,
      cols,
      ld,
      modulus);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_i64_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_i64_grouped_device(
    const int8_t* const* d_residue_ptrs,
    int64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  if (!d_residue_ptrs || !d_dst || !d_status || task_count <= 0 || rows <= 0 || cols <= 0 ||
      prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_grouped_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_grouped_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_i64_grouped_kernel,
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
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_i64_scheduled_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status) {
  if (!d_residues || !d_dst || !d_schedule || !d_bounds || !d_status || entry_count <= 0 ||
      max_tile_elements <= 0 || rows <= 0 || cols <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = (max_tile_elements + threads - 1) / threads;
  if (entry_count > kRns8MaxInt / blocks) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int total_blocks = entry_count * blocks;
  hipLaunchKernelGGL(
      rns8_export_i64_scheduled_kernel,
      dim3(total_blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      d_schedule,
      d_bounds,
      d_zero_a_rows,
      d_zero_b_cols,
      entry_count,
      max_tile_elements,
      rows,
      cols,
      d_status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_u64_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_u64_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_u64_grouped_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  if (!d_residue_ptrs || !d_dst || !d_status || task_count <= 0 || rows <= 0 || cols <= 0 ||
      prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_grouped_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_grouped_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_u64_grouped_kernel,
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
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_u64_scheduled_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status) {
  if (!d_residues || !d_dst || !d_schedule || !d_bounds || !d_status || entry_count <= 0 ||
      max_tile_elements <= 0 || rows <= 0 || cols <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = (max_tile_elements + threads - 1) / threads;
  if (entry_count > kRns8MaxInt / blocks) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int total_blocks = entry_count * blocks;
  hipLaunchKernelGGL(
      rns8_export_u64_scheduled_kernel,
      dim3(total_blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      d_schedule,
      d_bounds,
      d_zero_a_rows,
      d_zero_b_cols,
      entry_count,
      max_tile_elements,
      rows,
      cols,
      d_status);
  return static_cast<int>(hipGetLastError());
}

