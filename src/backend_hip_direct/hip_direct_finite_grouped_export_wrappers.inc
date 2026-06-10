// Host wrapper for grouped finite-u8 export kernel (Gap 103)
// Kernel rns8_export_u8_grouped_modulus_kernel is defined in hip_direct_export_kernels.cuh

extern "C" int rns8_hip_direct_export_u8_grouped_device(
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols,
    int modulus) {
  dim3 block(256);
  dim3 grid(
      (static_cast<unsigned int>(rows) * static_cast<unsigned int>(cols) + block.x - 1) / block.x,
      static_cast<unsigned int>(task_count));
  rns8_export_u8_grouped_modulus_kernel<<<grid, block>>>(
      residue_ptrs, dst, task_count, rows, cols, modulus);
  return static_cast<int>(hipGetLastError());
}
