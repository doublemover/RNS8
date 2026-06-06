rns8_status hipblaslt_gemm_finite_u8_device(
    int device_id,
    void* handle,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* int32_scratch,
    std::size_t int32_scratch_bytes,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle || !device_a_residues || !device_b_residues || !device_c_residues || !int32_scratch ||
      !workspace || !checked_common_shape(m, n, k, lda, ldb, ldc) ||
      !checked_workspace(m, n, k, int32_scratch_bytes, workspace_bytes) || modulus < 2 || modulus > 256) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const rns8_status status = gemm_one_plane(
      device_id,
      static_cast<hipblasLtHandle_t>(handle),
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      static_cast<int32_t*>(int32_scratch),
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      nullptr,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const hipError_t sync_status = hipDeviceSynchronize();
  return sync_status == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)handle;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)int32_scratch;
  (void)int32_scratch_bytes;
  (void)workspace;
  (void)workspace_bytes;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
