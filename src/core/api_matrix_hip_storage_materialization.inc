rns8_status free_hip_matrix_storage(rns8_matrix& matrix) {
  const auto status =
      rns8::detail::release_matrix_allocations(matrix, rns8::detail::hip_direct_free);
  matrix.finite_modulus = 0;
  clear_residue_current(matrix);
  clear_byte_limb_current(matrix);
  clear_native_current(matrix);
  return status;
}

rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix) {
  matrix.hip_device_id = ctx.device_id;
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    if (!wrap_byte_limb_bytes(matrix.desc.rows, matrix.desc.cols, matrix.hip_byte_limb_bytes))
      return RNS8_RANGE_ERROR;
    rns8_status status = rns8::detail::hip_direct_allocate(
        ctx.device_id, matrix.hip_byte_limb_bytes, &matrix.hip_byte_limbs);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_byte_limbs,
                                           matrix.hip_byte_limb_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    clear_residue_current(matrix);
    clear_byte_limb_current(matrix);
    return RNS8_SUCCESS;
  }
  const bool valid_capacity =
      uses_finite_storage(matrix.desc.semantics)
          ? matrix_cell_count(matrix.desc.rows, matrix.desc.cols, matrix.hip_residue_bytes)
          : rns_residue_count(matrix.desc.rows, matrix.desc.cols, matrix.prefix,
                              matrix.hip_residue_bytes);
  if (!valid_capacity)
    return RNS8_RANGE_ERROR;
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_residue_bytes,
                                                         &matrix.hip_residues);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status =
      rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_residues, matrix.hip_residue_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  clear_residue_current(matrix);
  clear_byte_limb_current(matrix);
  return RNS8_SUCCESS;
}

rns8_status ensure_hip_native_storage(rns8_context& ctx, rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64 &&
      !(matrix.desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT)) {
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
  if (matrix.desc.semantics == RNS8_BOUNDED_I64 ||
      matrix.desc.semantics == RNS8_EXACT_WIDE_SIGNED) {
    if (matrix.hip_native_i64 && matrix.hip_native_i64_bytes == expected_bytes) {
      return RNS8_SUCCESS;
    }
    if (matrix.hip_native_i64 || matrix.hip_native_i64_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_native_i64_bytes = expected_bytes;
    rns8_status status =
        rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_i64);
    if (status != RNS8_SUCCESS) {
      matrix.hip_native_i64 = nullptr;
      matrix.hip_native_i64_bytes = 0;
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_i64,
                                           matrix.hip_native_i64_bytes);
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
  rns8_status status =
      rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_u64);
  if (status != RNS8_SUCCESS) {
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_u64,
                                         matrix.hip_native_u64_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  return RNS8_SUCCESS;
}

rns8_status upload_native_i64(rns8_context& ctx, rns8_matrix& matrix, const int64_t* src,
                              int64_t ld) {
  if (!src ||
      (matrix.desc.semantics != RNS8_BOUNDED_I64 &&
       !(matrix.desc.semantics == RNS8_EXACT_WIDE_SIGNED &&
         (matrix.desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT))) ||
      !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  rns8::detail::HostUploadPlan upload;
  if (!rns8::detail::make_host_upload_plan(matrix.desc.rows, matrix.desc.cols, ld, sizeof(*src),
                                           upload))
    return RNS8_INVALID_ARGUMENT;
  status = rns8::detail::hip_direct_upload_host_matrix(ctx.device_id, "residue_h2d_sync",
                                                       matrix.hip_native_i64,
                                                       matrix.hip_native_i64_bytes, src, upload);
  if (status != RNS8_SUCCESS) {
    clear_device_native_current(matrix);
    return status;
  }
  mark_device_native_current(matrix);
  return RNS8_SUCCESS;
}

rns8_status upload_native_u64(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src,
                              int64_t ld) {
  if (!src ||
      (matrix.desc.semantics != RNS8_BOUNDED_U64 &&
       !(matrix.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED &&
         (matrix.desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT))) ||
      !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  rns8::detail::HostUploadPlan upload;
  if (!rns8::detail::make_host_upload_plan(matrix.desc.rows, matrix.desc.cols, ld, sizeof(*src),
                                           upload))
    return RNS8_INVALID_ARGUMENT;
  status = rns8::detail::hip_direct_upload_host_matrix(ctx.device_id, "residue_h2d_sync",
                                                       matrix.hip_native_u64,
                                                       matrix.hip_native_u64_bytes, src, upload);
  if (status != RNS8_SUCCESS) {
    clear_device_native_current(matrix);
    return status;
  }
  mark_device_native_current(matrix);
  return RNS8_SUCCESS;
}

bool should_populate_native_on_pack(const rns8_context& ctx, const rns8_matrix& matrix) {
  if (matrix.desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT)
    return ctx.backend == RNS8_BACKEND_HIP_DIRECT && matrix.backend == RNS8_BACKEND_HIP_DIRECT;
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  return native_vector_backend(ctx.backend) ||
         (ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT &&
          matrix.backend == RNS8_BACKEND_HIP_DIRECT);
}

bool same_source_device_rns_import_current(const rns8_context& ctx, const rns8_matrix& matrix,
                                           uint64_t version) {
  if (!version || !matrix.matrix_instance_id || !matrix.content_generation ||
      !matrix.imported_source || matrix.generation_exhausted || matrix.source_version != version ||
      ctx.hip_failure_poisoned || matrix.in_flight_writer || matrix.in_flight_readers ||
      ctx.backend != matrix.backend || !hip_resident_rns_backend(ctx.backend) ||
      !uses_rns_storage(matrix.desc.semantics) || !matrix.prefix ||
      matrix.prefix > RNS8_MAX_SUPPORTED_PREFIX || matrix.hip_device_id != ctx.device_id ||
      !matrix.device_residues_current || matrix.materialized_rns_prefix != matrix.prefix ||
      !matrix_descriptor_matches(matrix, matrix.desc.semantics, matrix.desc.bound_kind,
                                 matrix.desc.rows, matrix.desc.cols, matrix.prefix,
                                 matrix.desc.tile_m, matrix.desc.tile_n) ||
      !rns_matrix_storage_matches(matrix, ctx.backend, matrix.desc.rows, matrix.desc.cols,
                                  matrix.prefix))
    return false;
  if (rns8::detail::exact_wide_semantics(matrix.desc.semantics) &&
      (matrix.exact_range_prefix != matrix.prefix || matrix.exact_max_magnitude < 0 ||
       !rns8::detail::exact_range_fits(matrix.desc.semantics, matrix.exact_max_magnitude,
                                       rns8::detail::modulus_product(matrix.prefix))))
    return false;
  if (!should_populate_native_on_pack(ctx, matrix))
    return true;
  std::size_t bytes = 0;
  if (!matrix.device_native_current ||
      !native_matrix_bytes(matrix.desc.rows, matrix.desc.cols, 8, bytes))
    return false;
  const bool sign =
      matrix.desc.semantics == RNS8_BOUNDED_I64 || matrix.desc.semantics == RNS8_EXACT_WIDE_SIGNED;
  return sign ? matrix.hip_native_i64 && matrix.hip_native_i64_bytes == bytes
              : matrix.hip_native_u64 && matrix.hip_native_u64_bytes == bytes;
}

rns8_status ensure_bounded_native_residues_current_for_rns_plan(rns8_context& ctx,
                                                                const rns8_plan& plan,
                                                                rns8_matrix& matrix) {
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !hip_resident_rns_backend(plan.backend) || native_vector_backend(plan.backend) ||
      (plan.desc.semantics != RNS8_BOUNDED_I64 && plan.desc.semantics != RNS8_BOUNDED_U64)) {
    return RNS8_SUCCESS;
  }
  if (rns_residue_state_current_for_backend(matrix, plan.backend) &&
      materialized_rns_prefix_covers(matrix, rns_storage_prefix_for_plan(plan))) {
    return RNS8_SUCCESS;
  }
  if (rns_residue_state_current_for_backend(matrix, plan.backend) &&
      !bounded_native_state_current(matrix))
    return RNS8_RANGE_ERROR;
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT || matrix.hip_device_id != ctx.device_id ||
      matrix.desc.semantics != plan.desc.semantics ||
      !rns_matrix_storage_matches(matrix, plan.backend, matrix.desc.rows, matrix.desc.cols,
                                  rns_storage_prefix_for_plan(plan)) ||
      !bounded_native_storage_matches(matrix, plan.desc.semantics, matrix.desc.rows,
                                      matrix.desc.cols) ||
      !bounded_native_state_current(matrix) || matrix.source_version == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  size_t token = 0;
  const auto reserved = begin_synchronous_output_lease(ctx, matrix, token);
  if (reserved != RNS8_SUCCESS)
    return reserved;
  SynchronousLease lease(ctx, token);
  // Physical cache writes do not replace the logical native value/generation.
  return lease.finish([&]() -> rns8_status {
    const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
    rns8_status status = synchronous_dispatch_checkpoint(ctx, "native_to_rns");
    if (status != RNS8_SUCCESS)
      return status;
    if (plan.desc.semantics == RNS8_BOUNDED_I64) {
      status = rns8::detail::hip_direct_native_i64_to_rns_device(
          ctx.device_id, matrix.hip_native_i64, matrix.hip_residues, matrix.desc.rows,
          matrix.desc.cols, storage_prefix);
    } else {
      status = rns8::detail::hip_direct_native_u64_to_rns_device(
          ctx.device_id, matrix.hip_native_u64, matrix.hip_residues, matrix.desc.rows,
          matrix.desc.cols, storage_prefix);
    }
    if (status != RNS8_SUCCESS) {
      clear_device_residues_current(matrix);
      return status;
    }
    mark_device_residues_current(matrix, storage_prefix);
    return RNS8_SUCCESS;
  }());
}

} // namespace rns8::detail::api
