rns8_status hipblaslt_gemm_rns_device(
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
    uint32_t prefix,
    rns8_workspace* workspace_state,
    uint64_t a_source_version,
    uint64_t b_source_version) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle || !device_a_residues || !device_b_residues || !device_c_residues || !int32_scratch ||
      !workspace || !checked_common_shape(m, n, k, lda, ldb, ldc) ||
      !checked_workspace(m, n, k, int32_scratch_bytes, workspace_bytes) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  auto* scratch = static_cast<int32_t*>(int32_scratch);
  const auto typed_handle = static_cast<hipblasLtHandle_t>(handle);
  const int8_t* cached_a_base = nullptr;
  std::size_t cached_a_plane_bytes = 0;
  const int8_t* cached_b_base = nullptr;
  std::size_t cached_b_plane_bytes = 0;
  if (workspace_state && k <= static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK)) {
    padded_block_shape cache_shape;
    if (!padded_block_shape_for(m, n, k, cache_shape)) {
      return RNS8_RANGE_ERROR;
    }
    cached_a_plane_bytes = packed_a_plane_bytes_for(cache_shape);
    cached_b_plane_bytes = packed_b_plane_bytes_for(cache_shape);
    rns8_status cache_status =
        ensure_a_prepack_cache(device_id, a_base, *workspace_state, a_source_version, m, k, lda, prefix, cache_shape);
    if (cache_status != RNS8_SUCCESS) {
      return cache_status;
    }
    cache_status =
        ensure_b_prepack_cache(device_id, b_base, *workspace_state, b_source_version, k, n, ldb, prefix, cache_shape);
    if (cache_status != RNS8_SUCCESS) {
      return cache_status;
    }
    cached_a_base = static_cast<const int8_t*>(workspace_state->hipblaslt_a_prepack_cache);
    cached_b_base = static_cast<const int8_t*>(workspace_state->hipblaslt_b_prepack_cache);
  }
  for (uint32_t p = 0; p < prefix; ++p) {
    const std::size_t a_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(m) * static_cast<std::size_t>(lda);
    const std::size_t b_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb);
    const std::size_t c_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc);
    const int8_t* cached_a_for_plane =
        cached_a_base ? cached_a_base + static_cast<std::size_t>(p) * cached_a_plane_bytes : nullptr;
    const int8_t* cached_b_for_plane =
        cached_b_base ? cached_b_base + static_cast<std::size_t>(p) * cached_b_plane_bytes : nullptr;
    const rns8_status status = gemm_one_plane(
        device_id,
        typed_handle,
        a_base + a_offset,
        b_base + b_offset,
        c_base + c_offset,
        scratch,
        workspace,
        workspace_bytes,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        kHipblasLtDefaultModuli[p],
        cached_a_for_plane,
        cached_b_for_plane);
    if (status != RNS8_SUCCESS) {
      return status;
    }
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
  (void)prefix;
  (void)workspace_state;
  (void)a_source_version;
  (void)b_source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

