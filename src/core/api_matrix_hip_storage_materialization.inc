rns8_status free_hip_matrix_storage(rns8_matrix& matrix) {
  rns8_status status = RNS8_SUCCESS;
  if (matrix.hip_upload_buffer) {
    status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_upload_buffer);
    matrix.hip_upload_buffer = nullptr;
    matrix.hip_upload_bytes = 0;
  }
  if (matrix.hip_status_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_status_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_status_buffer = nullptr;
    matrix.hip_status_bytes = 0;
  }
  if (matrix.hip_export_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_buffer = nullptr;
    matrix.hip_export_bytes = 0;
  }
  if (matrix.hip_export_tile_bounds) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_tile_bounds);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_tile_bounds = nullptr;
    matrix.hip_export_tile_bounds_bytes = 0;
    matrix.hip_export_tile_bounds_count = 0;
  }
  if (matrix.hip_export_tile_schedule) {
    const rns8_status free_status =
        rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_tile_schedule);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_tile_schedule = nullptr;
    matrix.hip_export_tile_schedule_bytes = 0;
    matrix.hip_export_tile_schedule_count = 0;
  }
  if (matrix.hip_export_zero_a_rows) {
    const rns8_status free_status =
        rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_zero_a_rows);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_zero_a_rows = nullptr;
    matrix.hip_export_zero_a_rows_bytes = 0;
  }
  if (matrix.hip_export_zero_b_cols) {
    const rns8_status free_status =
        rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_zero_b_cols);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_zero_b_cols = nullptr;
    matrix.hip_export_zero_b_cols_bytes = 0;
  }
  if (matrix.hip_residues) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_residues);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_residues = nullptr;
    matrix.hip_residue_bytes = 0;
  }
  if (matrix.hip_byte_limbs) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_byte_limbs);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_byte_limbs = nullptr;
    matrix.hip_byte_limb_bytes = 0;
  }
  if (matrix.hip_native_i64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_i64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_i64 = nullptr;
    matrix.hip_native_i64_bytes = 0;
  }
  if (matrix.hip_native_u64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_u64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
  }
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_max_elements = 0;
  matrix.finite_modulus = 0;
  clear_device_residues_current(matrix);
  clear_device_byte_limb_current(matrix);
  clear_native_current(matrix);
  return status;
}

rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix) {
  matrix.hip_device_id = ctx.device_id;
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    if (matrix.byte_limbs.empty()) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_byte_limb_bytes = matrix.byte_limbs.size() * sizeof(uint8_t);
    rns8_status status =
        rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_byte_limb_bytes, &matrix.hip_byte_limbs);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_byte_limbs, matrix.hip_byte_limb_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    clear_residue_current(matrix);
    clear_byte_limb_current(matrix);
    return RNS8_SUCCESS;
  }
  if (matrix.residues.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_residue_bytes = matrix.residues.size() * sizeof(int8_t);
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_residue_bytes, &matrix.hip_residues);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_residues, matrix.hip_residue_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  clear_residue_current(matrix);
  clear_byte_limb_current(matrix);
  return RNS8_SUCCESS;
}

rns8_status ensure_hip_native_storage(rns8_context& ctx, rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (matrix.hip_device_id != -1 && matrix.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_device_id = ctx.device_id;
  std::size_t expected_bytes = 0;
  if (!native_matrix_bytes(matrix.desc.rows, matrix.desc.cols, sizeof(uint64_t), expected_bytes)) {
    return RNS8_RANGE_ERROR;
  }
  if (matrix.desc.semantics == RNS8_BOUNDED_I64) {
    if (matrix.hip_native_i64 && matrix.hip_native_i64_bytes == expected_bytes) {
      return RNS8_SUCCESS;
    }
    if (matrix.hip_native_i64 || matrix.hip_native_i64_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_native_i64_bytes = expected_bytes;
    rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_i64);
    if (status != RNS8_SUCCESS) {
      matrix.hip_native_i64 = nullptr;
      matrix.hip_native_i64_bytes = 0;
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_i64, matrix.hip_native_i64_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 && matrix.hip_native_u64_bytes == expected_bytes) {
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 || matrix.hip_native_u64_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_native_u64_bytes = expected_bytes;
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_u64);
  if (status != RNS8_SUCCESS) {
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_u64, matrix.hip_native_u64_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  return RNS8_SUCCESS;
}

rns8_status upload_native_i64(rns8_context& ctx, rns8_matrix& matrix, const int64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_I64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<int64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_i64, staged.data(), staged.size() * sizeof(int64_t));
  if (status != RNS8_SUCCESS) {
    clear_device_native_current(matrix);
    return status;
  }
  mark_device_native_current(matrix);
  return RNS8_SUCCESS;
}

rns8_status upload_native_u64(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_U64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<uint64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_u64, staged.data(), staged.size() * sizeof(uint64_t));
  if (status != RNS8_SUCCESS) {
    clear_device_native_current(matrix);
    return status;
  }
  mark_device_native_current(matrix);
  return RNS8_SUCCESS;
}

bool should_populate_native_on_pack(const rns8_context& ctx, const rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  return native_vector_backend(ctx.backend) ||
         (ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT && matrix.backend == RNS8_BACKEND_HIP_DIRECT);
}

rns8_status ensure_bounded_native_residues_current_for_rns_plan(
    rns8_context& ctx,
    const rns8_plan& plan,
    rns8_matrix& matrix) {
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !hip_resident_rns_backend(plan.backend) || native_vector_backend(plan.backend) ||
      (plan.desc.semantics != RNS8_BOUNDED_I64 && plan.desc.semantics != RNS8_BOUNDED_U64)) {
    return RNS8_SUCCESS;
  }
  if (rns_residue_state_current_for_backend(matrix, plan.backend)) {
    return RNS8_SUCCESS;
  }
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT || matrix.hip_device_id != ctx.device_id ||
      matrix.desc.semantics != plan.desc.semantics ||
      !rns_matrix_storage_matches(
          matrix, plan.backend, matrix.desc.rows, matrix.desc.cols, rns_storage_prefix_for_plan(plan)) ||
      !bounded_native_storage_matches(matrix, plan.desc.semantics, matrix.desc.rows, matrix.desc.cols) ||
      !bounded_native_state_current(matrix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  rns8_status status = RNS8_SUCCESS;
  if (plan.desc.semantics == RNS8_BOUNDED_I64) {
    status = rns8::detail::hip_direct_native_i64_to_rns_device(
        ctx.device_id,
        matrix.hip_native_i64,
        matrix.hip_residues,
        matrix.desc.rows,
        matrix.desc.cols,
        storage_prefix);
  } else {
    status = rns8::detail::hip_direct_native_u64_to_rns_device(
        ctx.device_id,
        matrix.hip_native_u64,
        matrix.hip_residues,
        matrix.desc.rows,
        matrix.desc.cols,
        storage_prefix);
  }
  if (status != RNS8_SUCCESS) {
    clear_device_residues_current(matrix);
    return status;
  }
  mark_device_residues_current(matrix);
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail::api
