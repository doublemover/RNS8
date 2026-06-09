extern "C" int rns8_rocwmma_gemm_rns_device(
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
    unsigned int prefix) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc)) {
    return rocwmma_invalid_argument;
  }
  return run_rns_prefix(
      device_id,
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      static_cast<int8_t*>(workspace),
      workspace_bytes,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      prefix);
}

extern "C" int rns8_rocwmma_prepack_b_rns_device(
    int device_id,
    const void* device_b_residues,
    void* device_b_prepack,
    unsigned long long device_b_prepack_bytes,
    long long k,
    long long n,
    long long ldb,
    unsigned int prefix) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_b_residues || !device_b_prepack || k <= 0 || n <= 0 || ldb < n ||
      !checked_i32_shape(k) || !checked_i32_shape(n) || !checked_i32_shape(ldb)) {
    return rocwmma_invalid_argument;
  }
  return prepack_b_rns_prefix(
      device_id,
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_b_prepack),
      device_b_prepack_bytes,
      static_cast<int>(k),
      static_cast<int>(n),
      static_cast<int>(ldb),
      prefix);
}

extern "C" int rns8_rocwmma_prepack_b_native_i64_device(
    int device_id,
    const void* device_b_native,
    void* device_b_prepack,
    unsigned long long device_b_prepack_bytes,
    long long k,
    long long n,
    long long ldb,
    unsigned int prefix) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_b_native || !device_b_prepack || k <= 0 || n <= 0 || ldb < n ||
      !checked_i32_shape(k) || !checked_i32_shape(n) || !checked_i32_shape(ldb)) {
    return rocwmma_invalid_argument;
  }
  return prepack_b_native_rns_prefix(
      device_id,
      static_cast<const int64_t*>(device_b_native),
      static_cast<int8_t*>(device_b_prepack),
      device_b_prepack_bytes,
      static_cast<int>(k),
      static_cast<int>(n),
      static_cast<int>(ldb),
      prefix);
}

extern "C" int rns8_rocwmma_prepack_b_native_u64_device(
    int device_id,
    const void* device_b_native,
    void* device_b_prepack,
    unsigned long long device_b_prepack_bytes,
    long long k,
    long long n,
    long long ldb,
    unsigned int prefix) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_b_native || !device_b_prepack || k <= 0 || n <= 0 || ldb < n ||
      !checked_i32_shape(k) || !checked_i32_shape(n) || !checked_i32_shape(ldb)) {
    return rocwmma_invalid_argument;
  }
  return prepack_b_native_rns_prefix(
      device_id,
      static_cast<const uint64_t*>(device_b_native),
      static_cast<int8_t*>(device_b_prepack),
      device_b_prepack_bytes,
      static_cast<int>(k),
      static_cast<int>(n),
      static_cast<int>(ldb),
      prefix);
}

extern "C" int rns8_rocwmma_gemm_rns_prepacked_b_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_prepack,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldc,
    unsigned int prefix) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_a_residues || !device_b_prepack || !device_c_residues || !workspace || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldc < n || !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldc)) {
    return rocwmma_invalid_argument;
  }
  return run_rns_prefix_prepacked_b(
      device_id,
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_prepack),
      static_cast<int8_t*>(device_c_residues),
      static_cast<int8_t*>(workspace),
      workspace_bytes,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldc),
      prefix);
}

extern "C" int rns8_rocwmma_gemm_rns_tiled_device(
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
    const rns8_plan_tile_schedule_entry* entries,
    unsigned long long entry_count) {
  if (hipSetDevice(device_id) != hipSuccess) {
    return rocwmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || !entries || entry_count == 0 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !checked_i32_shape(m) ||
      !checked_i32_shape(n) || !checked_i32_shape(k) || !checked_i32_shape(lda) || !checked_i32_shape(ldb) ||
      !checked_i32_shape(ldc)) {
    return rocwmma_invalid_argument;
  }
  auto* a_pack = static_cast<int8_t*>(workspace);
  const auto* a = static_cast<const int8_t*>(device_a_residues);
  const auto* b = static_cast<const int8_t*>(device_b_residues);
  auto* c = static_cast<int8_t*>(device_c_residues);
  int max_rows = 0;
  int max_cols = 0;
  for (unsigned long long entry_index = 0; entry_index < entry_count; ++entry_index) {
    const auto& entry = entries[entry_index];
    if ((entry.flags & ~kRns8SupportedTileScheduleFlags) != 0 || entry.row_extent <= 0 || entry.col_extent <= 0 ||
        entry.row_offset < 0 || entry.col_offset < 0 || entry.row_offset + entry.row_extent > m ||
        entry.col_offset + entry.col_extent > n || entry.selected_prefix == 0 ||
        entry.selected_prefix > kRns8MaxSupportedPrefix) {
      return rocwmma_invalid_argument;
    }
    if (entry.row_extent > max_rows) {
      max_rows = static_cast<int>(entry.row_extent);
    }
    if (entry.col_extent > max_cols) {
      max_cols = static_cast<int>(entry.col_extent);
    }
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!rocwmma_workspace_requirements_u64(max_rows, max_cols, k, &a_bytes, &b_bytes)) {
    return rocwmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return rocwmma_invalid_argument;
  }
  auto* b_pack = a_pack + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? static_cast<int>(k) : kRns8SignedSafeInt32KBlock;
  for (unsigned long long entry_index = 0; entry_index < entry_count; ++entry_index) {
    const auto& entry = entries[entry_index];
    if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
      const int status = zero_tile_residues(
          c,
          m,
          ldc,
          entry.row_offset,
          entry.col_offset,
          entry.row_extent,
          entry.col_extent,
          entry.selected_prefix);
      if (status != rocwmma_success) {
        return status;
      }
      continue;
    }
    for (uint32_t modulus_index = 0; modulus_index < entry.selected_prefix; ++modulus_index) {
      const uint16_t modulus = kDefaultModuli[modulus_index];
      const auto plane_a = a + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * lda +
                           static_cast<int64_t>(entry.row_offset) * lda;
      const auto plane_b = b + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * ldb +
                           static_cast<int64_t>(entry.col_offset);
      auto plane_c = c + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * ldc +
                     static_cast<int64_t>(entry.row_offset) * ldc + static_cast<int64_t>(entry.col_offset);
      const int status = launch_pack_and_gemm(
          plane_a,
          plane_b,
          plane_c,
          a_pack,
          b_pack,
          static_cast<int>(entry.row_extent),
          static_cast<int>(entry.col_extent),
          static_cast<int>(k),
          static_cast<int>(lda),
          static_cast<int>(ldb),
          static_cast<int>(ldc),
          static_cast<int>(modulus),
          modulus_reciprocal_u32(modulus),
          safe_k_block,
          static_cast<int>(modulus_index));
      if (status != rocwmma_success) {
        return status;
      }
    }
  }
  return hipDeviceSynchronize() == hipSuccess ? rocwmma_success : rocwmma_backend_failure;
}

