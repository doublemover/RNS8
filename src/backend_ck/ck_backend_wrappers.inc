extern "C" int rns8_ck_gemm_rns_device(
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
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix) {
    return ck_invalid_argument;
  }
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int32_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      m,
      n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  status = run_ck_entry(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_workspace,
      b_workspace,
      temp_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      0,
      0,
      m,
      n,
      prefix);
  if (status != ck_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}

extern "C" int rns8_ck_gemm_rns_tiled_device(
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
  if (!entries || entry_count == 0 || entry_count > static_cast<unsigned long long>(INT32_MAX)) {
    return ck_invalid_argument;
  }
  int64_t max_m = 0;
  int64_t max_n = 0;
  for (uint64_t i = 0; i < entry_count; ++i) {
    if (!checked_tile_entry(entries[i], m, n)) {
      return ck_invalid_argument;
    }
    if (entries[i].row_extent > max_m) {
      max_m = entries[i].row_extent;
    }
    if (entries[i].col_extent > max_n) {
      max_n = entries[i].col_extent;
    }
  }
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int32_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      max_m,
      max_n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  for (uint64_t i = 0; i < entry_count; ++i) {
    const auto& entry = entries[i];
    if ((entry.flags & kRns8TileScheduleZeroOutput) != 0) {
      status = zero_tile_residues(
          static_cast<int8_t*>(device_c_residues),
          m,
          ldc,
          entry.row_offset,
          entry.col_offset,
          entry.row_extent,
          entry.col_extent,
          entry.selected_prefix);
      if (status != ck_success) {
        return status;
      }
      continue;
    }
    status = run_ck_entry(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        a_workspace,
        b_workspace,
        temp_c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        entry.row_offset,
        entry.col_offset,
        entry.row_extent,
        entry.col_extent,
        entry.selected_prefix);
    if (status != ck_success) {
      return status;
    }
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}

extern "C" int rns8_ck_gemm_finite_u8_device(
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
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int32_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      m,
      n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  status = run_ck_finite(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_workspace,
      b_workspace,
      temp_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      static_cast<uint16_t>(modulus));
  if (status != ck_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}
