extern "C" int rns8_rocwmma_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    unsigned int modulus) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !checked_i32_shape(m) ||
      !checked_i32_shape(n) || !checked_i32_shape(k) || !checked_i32_shape(lda) || !checked_i32_shape(ldb) ||
      !checked_i32_shape(ldc)) {
    return rocwmma_invalid_argument;
  }
  if (!rocwmma_static_modulus_supported(modulus)) {
    return rocwmma_unsupported_backend;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!rocwmma_workspace_requirements_u64(m, n, k, &a_bytes, &b_bytes)) {
    return rocwmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return rocwmma_invalid_argument;
  }
  auto* a_pack = static_cast<int8_t*>(workspace);
  auto* b_pack = a_pack + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? static_cast<int>(k) : kRns8SignedSafeInt32KBlock;
  const int status = launch_pack_and_gemm(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_pack,
      b_pack,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(modulus),
      modulus_reciprocal_u32(static_cast<uint16_t>(modulus)),
      safe_k_block,
      -1);
  if (status != rocwmma_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? rocwmma_success : rocwmma_backend_failure;
}

extern "C" int rns8_rocwmma_wrap64_gemm_byte_limbs_candidate_device(
    int device_id,
    const void* device_a_limbs,
    const void* device_b_limbs,
    void* device_c_limbs,
    long long m,
    long long n,
    long long k) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_a_limbs || !device_b_limbs || !device_c_limbs || m <= 0 || n <= 0 || k <= 0 ||
      !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) || k > 32768) {
    return rocwmma_invalid_argument;
  }
  const dim3 grid(
      static_cast<unsigned int>((n + kRns8WmmaTileN - 1) / kRns8WmmaTileN),
      static_cast<unsigned int>((m + kRns8WmmaTileM - 1) / kRns8WmmaTileM),
      1);
  rocwmma_wrap64_byte_gemm36_candidate_kernel<<<grid, kRns8WmmaWaveThreads>>>(
      static_cast<const uint8_t*>(device_a_limbs),
      static_cast<const uint8_t*>(device_b_limbs),
      static_cast<uint8_t*>(device_c_limbs),
      m,
      n,
      k);
  if (hipGetLastError() != hipSuccess) {
    return rocwmma_backend_failure;
  }
  return hipDeviceSynchronize() == hipSuccess ? rocwmma_success : rocwmma_backend_failure;
}
