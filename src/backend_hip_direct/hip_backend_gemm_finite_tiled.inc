rns8_status hip_direct_gemm_finite_u8_resident_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_resident_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_i8_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
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

rns8_status hip_direct_gemm_finite_u8_region_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || region_row_offset < 0 ||
      region_col_offset < 0 || region_rows <= 0 || region_cols <= 0 || region_row_offset > m ||
      region_col_offset > n || region_rows > m - region_row_offset || region_cols > n - region_col_offset ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      region_row_offset > std::numeric_limits<int>::max() ||
      region_col_offset > std::numeric_limits<int>::max() ||
      region_rows > std::numeric_limits<int>::max() || region_cols > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_resident_gemm_region_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_i8_region_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(region_row_offset),
        static_cast<int>(region_col_offset),
        static_cast<int>(region_rows),
        static_cast<int>(region_cols),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)region_row_offset;
  (void)region_col_offset;
  (void)region_rows;
  (void)region_cols;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_finite_u8_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!a_matrices || !b_matrices || !c_matrices || task_count == 0 || !device_a_residue_ptrs ||
      !device_b_residue_ptrs || !device_c_residue_ptrs ||
      !valid_finite_modulus_for_semantics(expected_semantics, modulus) ||
      m <= 0 || n <= 0 || k <= 0 || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }

  int a_device_id = -1;
  int b_device_id = -1;
  int c_device_id = -1;
  uint32_t a_prefix = 0;
  uint32_t b_prefix = 0;
  uint32_t c_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      a_matrices, task_count, expected_semantics, m, k, true, &a_device_id, &a_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      b_matrices, task_count, expected_semantics, k, n, true, &b_device_id, &b_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      c_matrices, task_count, expected_semantics, m, n, false, &c_device_id, &c_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (a_device_id != b_device_id || a_device_id != c_device_id || a_prefix != 0 || b_prefix != 0 || c_prefix != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    device_id = a_device_id;
  } else if (device_id != a_device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    if (a_matrices[index]->finite_modulus != modulus || b_matrices[index]->finite_modulus != modulus) {
      return RNS8_INVALID_ARGUMENT;
    }
  }

  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_ptrs = static_cast<const int8_t* const*>(device_a_residue_ptrs);
  const auto* b_ptrs = static_cast<const int8_t* const*>(device_b_residue_ptrs);
  auto* c_ptrs = static_cast<int8_t* const*>(const_cast<void*>(device_c_residue_ptrs));
  const hipError_t err = timed_hip_operation("finite_resident_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_i8_grouped_device(
        a_ptrs,
        b_ptrs,
        c_ptrs,
        static_cast<int>(task_count),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(k),
        static_cast<int>(n),
        static_cast<int>(n),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = c_matrices[index];
    mark_device_residue_output_current(matrix);
    matrix->finite_modulus = modulus;
    matrix->source_version = api::gemm_output_source_version_values(
        a_matrices[index]->source_version,
        b_matrices[index]->source_version);
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)a_matrices;
  (void)b_matrices;
  (void)c_matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_a_residue_ptrs;
  (void)device_b_residue_ptrs;
  (void)device_c_residue_ptrs;
  (void)m;
  (void)n;
  (void)k;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_finite_u8_native_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_native_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_u8_native_device(
        static_cast<const uint8_t*>(device_a_native),
        static_cast<const uint8_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
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

rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_native_a_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_u8_native_a_i8_b_device(
        static_cast<const uint8_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
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

rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint16_t modulus,
    uint64_t source_version) {
  if (!B || !C || B->finite_modulus != modulus || !B->device_residues_current ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_finite_u8_native_a_resident_b_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld,
      modulus);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = modulus;
  C->source_version = source_version;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || !entries || entry_count == 0 || m <= 0 ||
      n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  checked_tile_schedule schedule{};
  if (!checked_tile_schedule_contract(entries, entry_count, m, n, &schedule)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    for (const uint32_t selected_prefix : schedule.selected_prefix_groups) {
      for (uint32_t p = 0; p < selected_prefix; ++p) {
        const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(lda);
        const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(k) *
                                     static_cast<std::size_t>(ldb);
        const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(ldc);
        for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
          const auto& entry = entries[static_cast<std::size_t>(entry_index)];
          if (entry.selected_prefix != selected_prefix) {
            continue;
          }
          if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
            auto* tile_c =
                c_base + c_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(ldc) +
                static_cast<std::size_t>(entry.col_offset);
            const hipError_t zero_status = hipMemset2D(
                tile_c,
                static_cast<std::size_t>(ldc) * sizeof(int8_t),
                0,
                static_cast<std::size_t>(entry.col_extent) * sizeof(int8_t),
                static_cast<std::size_t>(entry.row_extent));
            if (zero_status != hipSuccess) {
              return zero_status;
            }
            continue;
          }
          const hipError_t launch_status = launch_rns_modulus_gemm({
              a_base + a_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(lda),
              b_base + b_offset + static_cast<std::size_t>(entry.col_offset),
              c_base + c_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(ldc) +
                  static_cast<std::size_t>(entry.col_offset),
              entry.row_extent,
              entry.col_extent,
              k,
              lda,
              ldb,
              ldc,
              kDefaultModuli[p],
              p,
              entry.selected_prefix});
          if (launch_status != hipSuccess) {
            return launch_status;
          }
        }
      }
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)entries;
  (void)entry_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_tiled_device_schedule(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* host_entries,
    const void* device_entries,
    const void* active_device_entries,
    const void* zero_a_rows,
    const void* zero_b_cols,
    const uint64_t* active_offsets,
    const uint64_t* active_counts,
    uint32_t active_prefix_count,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || !host_entries ||
      !active_offsets || !active_counts || active_prefix_count == 0 || entry_count == 0 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  checked_tile_schedule schedule{};
  if (!checked_tile_schedule_contract(host_entries, entry_count, m, n, &schedule)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int max_tile_row_blocks = 0;
  int max_tile_col_blocks = 0;
  if (!scheduled_tile_block_shape(host_entries, entry_count, &max_tile_row_blocks, &max_tile_col_blocks)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int max_tile_elements = 0;
  if (!scheduled_tile_max_elements(host_entries, entry_count, &max_tile_elements)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const auto* active_schedule_base = static_cast<const rns8_plan_tile_schedule_entry*>(active_device_entries);
  const auto* zero_a_rows_base = static_cast<const uint8_t*>(zero_a_rows);
  const auto* zero_b_cols_base = static_cast<const uint8_t*>(zero_b_cols);
  const uint32_t max_selected_prefix = schedule.selected_prefix_groups.back();
  if (active_prefix_count != max_selected_prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  uint64_t running_offset = 0;
  for (uint32_t p = 0; p < max_selected_prefix; ++p) {
    uint64_t expected_count = 0;
    for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
      const auto& entry = host_entries[static_cast<std::size_t>(entry_index)];
      if (entry.selected_prefix > p && (entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
        ++expected_count;
      }
    }
    if (active_offsets[p] != running_offset || active_counts[p] != expected_count) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (expected_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    running_offset += expected_count;
  }
  if (running_offset != 0 && !active_device_entries) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool zero_output_tiles = schedule_has_zero_output_tiles(host_entries, entry_count);
  const bool zero_row_col_products = schedule_has_zero_row_col_products(host_entries, entry_count);
  if (zero_row_col_products && (!zero_a_rows || !zero_b_cols)) {
    return RNS8_INVALID_ARGUMENT;
  }
  uint32_t uniform_zero_selected_prefix = 0;
  const bool uniform_all_zero_output_tiles =
      schedule_all_zero_output_tiles_uniform_prefix(host_entries, entry_count, uniform_zero_selected_prefix);
  if (zero_output_tiles && !uniform_all_zero_output_tiles && !device_entries) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (zero_output_tiles) {
    const bool synchronize_zero_fill = running_offset == 0;
    const hipError_t zero_err = timed_hip_operation("direct_hip_zero_output_tile_memset", [&]() {
      if (uniform_all_zero_output_tiles) {
        const auto rows_u = static_cast<uint64_t>(m);
        const auto ldc_u = static_cast<uint64_t>(ldc);
        const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
        if (ldc_u != 0 && rows_u > max_bytes / ldc_u) {
          return hipErrorInvalidValue;
        }
        const uint64_t plane_stride = rows_u * ldc_u;
        if (plane_stride != 0 && uniform_zero_selected_prefix > max_bytes / plane_stride) {
          return hipErrorInvalidValue;
        }
        const auto zero_bytes = static_cast<std::size_t>(uniform_zero_selected_prefix * plane_stride);
        const hipError_t memset_status = hipMemsetAsync(c_base, 0, zero_bytes, nullptr);
        return memset_status == hipSuccess && synchronize_zero_fill ? hipDeviceSynchronize() : memset_status;
      }
      const int code = rns8_hip_direct_zero_scheduled_residue_tiles_device(
          c_base,
          static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
          static_cast<int>(entry_count),
          max_tile_elements,
          static_cast<int>(m),
          static_cast<int>(ldc));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
      return synchronize_zero_fill ? hipDeviceSynchronize() : hipSuccess;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  if (running_offset == 0) {
    return RNS8_SUCCESS;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    const hipError_t launch_status = launch_rns_grouped_active_schedule_gemm({
        a_base,
        b_base,
        c_base,
        active_schedule_base,
        zero_a_rows_base,
        zero_b_cols_base,
        running_offset,
        max_tile_row_blocks,
        max_tile_col_blocks,
        m,
        k,
        lda,
        ldb,
        ldc});
    return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)host_entries;
  (void)device_entries;
  (void)active_device_entries;
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)active_offsets;
  (void)active_counts;
  (void)active_prefix_count;
  (void)entry_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}



// === Gap 103: Grouped finite-u8 GEMM dispatch ===

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_export_u8_grouped_device(
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols,
    int modulus);
#endif

rns8_status hip_direct_gemm_finite_u8_grouped(
    int device_id,
    const int8_t* const* residue_ptrs,
    uint8_t* dst,
    int task_count,
    int rows,
    int cols,
    int modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !residue_ptrs || residue_ptrs == nullptr || task_count <= 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;

  // Same-shape grouped finite GEMM: all tasks share dimensions and modulus
  dim3 block(256);
  dim3 grid(
      (static_cast<unsigned int>(rows) * static_cast<unsigned int>(cols) + block.x - 1) / block.x,
      static_cast<unsigned int>(task_count));

  // Allocate device-side residue pointer table
  int8_t** d_residue_ptrs = nullptr;
  hipError_t err = hipMalloc(&d_residue_ptrs, sizeof(int8_t*) * task_count);
  if (err != hipSuccess) return RNS8_BACKEND_FAILURE;
  err = hipMemcpy(d_residue_ptrs, residue_ptrs, sizeof(int8_t*) * task_count, hipMemcpyHostToDevice);
  if (err != hipSuccess) { hipFree(d_residue_ptrs); return RNS8_BACKEND_FAILURE; }

  err = timed_hip_operation("finite_rns_gemm_kernel_group", [&]() {
    int code = rns8_hip_direct_export_u8_grouped_device(d_residue_ptrs, dst, task_count, rows, cols, modulus);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });

  hipFree(d_residue_ptrs);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)residue_ptrs; (void)dst;
  (void)task_count; (void)rows; (void)cols; (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}


// === Gap 87: Finite-u8 full-path graph replay ===

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
namespace {

struct finite_graph_capture_state {
  hipGraphExec_t graph_exec = nullptr;
  bool captured = false;
  hipStream_t capture_stream = nullptr;
};

thread_local finite_graph_capture_state g_finite_graph;

}  // namespace

rns8_status hip_direct_finite_graph_capture_begin(hipStream_t stream) {
  g_finite_graph.capture_stream = stream;
  hipError_t err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status hip_direct_finite_graph_capture_end() {
  if (!g_finite_graph.capture_stream) return RNS8_BACKEND_FAILURE;
  hipGraph_t graph = nullptr;
  hipError_t err = hipStreamEndCapture(g_finite_graph.capture_stream, &graph);
  if (err != hipSuccess) return RNS8_BACKEND_FAILURE;
  err = hipGraphInstantiate(&g_finite_graph.graph_exec, graph, nullptr, nullptr, 0);
  hipGraphDestroy(graph);
  g_finite_graph.captured = (err == hipSuccess);
  return g_finite_graph.captured ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status hip_direct_finite_graph_launch() {
  if (!g_finite_graph.captured || !g_finite_graph.graph_exec) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  hipError_t err = hipGraphLaunch(g_finite_graph.graph_exec, g_finite_graph.capture_stream);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

void hip_direct_finite_graph_destroy() {
  if (g_finite_graph.graph_exec) {
    hipGraphExecDestroy(g_finite_graph.graph_exec);
    g_finite_graph.graph_exec = nullptr;
  }
  g_finite_graph.captured = false;
  g_finite_graph.capture_stream = nullptr;
}
#endif  // RNS8_ENABLE_HIP

