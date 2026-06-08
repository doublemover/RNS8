bool checked_tile_entry(const rns8_plan_tile_schedule_entry& entry, int64_t rows, int64_t cols) {
  if (entry.struct_size != sizeof(rns8_plan_tile_schedule_entry) || entry.abi_version != kRns8AbiVersion ||
      (entry.flags & ~kRns8SupportedTileScheduleFlags) != 0 || entry.row_offset < 0 || entry.col_offset < 0 ||
      entry.row_extent <= 0 || entry.col_extent <= 0 || entry.required_prefix == 0 || entry.selected_prefix == 0 ||
      entry.required_prefix > entry.selected_prefix || entry.selected_prefix > kRns8MaxSupportedPrefix ||
      entry.row_offset > rows || entry.col_offset > cols || entry.row_extent > rows - entry.row_offset ||
      entry.col_extent > cols - entry.col_offset) {
    return false;
  }
  return true;
}

int run_ck_entry(
    const int8_t* a_base,
    const int8_t* b_base,
    int8_t* c_base,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int32_t* temp_c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t row_offset,
    int64_t col_offset,
    int64_t row_extent,
    int64_t col_extent,
    uint32_t selected_prefix) {
  if (!checked_i32_shape(row_extent) || !checked_i32_shape(col_extent) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc) ||
      selected_prefix == 0 || selected_prefix > kRns8MaxSupportedPrefix) {
    return ck_invalid_argument;
  }
  for (uint32_t p = 0; p < selected_prefix; ++p) {
    const uint16_t modulus = kDefaultModuli[p];
    const uint32_t reciprocal = modulus_reciprocal_u32(modulus);
    const uint64_t a_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(lda);
    const uint64_t b_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(k) * static_cast<uint64_t>(ldb);
    const uint64_t c_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(ldc);
    const int result = run_ck_tile_modulus(
        a_base + a_plane_offset + static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(lda),
        b_base + b_plane_offset + static_cast<uint64_t>(col_offset),
        c_base + c_plane_offset + static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(ldc) +
            static_cast<uint64_t>(col_offset),
        a_workspace,
        b_workspace,
        temp_c,
        static_cast<int>(row_extent),
        static_cast<int>(col_extent),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        reciprocal,
        static_cast<int>(p));
    if (result != ck_success) {
      return result;
    }
  }
  return ck_success;
}

int zero_tile_residues(
    int8_t* c_base,
    int64_t m,
    int64_t ldc,
    int64_t row_offset,
    int64_t col_offset,
    int64_t row_extent,
    int64_t col_extent,
    uint32_t selected_prefix) {
  for (uint32_t p = 0; p < selected_prefix; ++p) {
    auto* tile_c = c_base + static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(ldc) +
                   static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(ldc) +
                   static_cast<uint64_t>(col_offset);
    const hipError_t status = timed_hip_launch("ck_zero_output_tile_memset", nullptr, [&]() {
      return hipMemset2D(
          tile_c,
          static_cast<std::size_t>(ldc) * sizeof(int8_t),
          0,
          static_cast<std::size_t>(col_extent) * sizeof(int8_t),
          static_cast<std::size_t>(row_extent));
    });
    if (status != hipSuccess) {
      return ck_backend_failure;
    }
  }
  return ck_success;
}

int run_ck_finite(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int32_t* temp_c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
  if (modulus < 2 || modulus > 256) {
    return ck_invalid_argument;
  }
  return run_ck_tile_modulus(
      a,
      b,
      c,
      a_workspace,
      b_workspace,
      temp_c,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(modulus),
      modulus_reciprocal_u32(modulus),
      -1);
}

int checked_common(
    int device_id,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    uint64_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_m,
    int64_t max_n,
    int8_t** a_workspace,
    int8_t** b_workspace,
    int32_t** temp_c) {
  if (!a || !b || !c || !workspace || !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc) || lda < k || ldb < n ||
      ldc < n) {
    return ck_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  uint64_t temp_bytes = 0;
  if (!ck_workspace_requirements_u64(max_m, max_n, k, &a_bytes, &b_bytes, &temp_bytes) ||
      a_bytes > UINT64_MAX - b_bytes || a_bytes + b_bytes > UINT64_MAX - temp_bytes ||
      workspace_bytes < a_bytes + b_bytes + temp_bytes) {
    return ck_invalid_argument;
  }
  if (device_id < 0) {
    return ck_invalid_argument;
  }
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return ck_backend_failure;
  }
  *a_workspace = static_cast<int8_t*>(workspace);
  *b_workspace = static_cast<int8_t*>(workspace) + a_bytes;
  *temp_c = temp_bytes == 0 ? nullptr
                            : reinterpret_cast<int32_t*>(static_cast<int8_t*>(workspace) + a_bytes + b_bytes);
  return ck_success;
}

}  // namespace

