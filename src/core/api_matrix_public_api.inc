namespace {
void discard_failed_matrix(rns8_context& ctx, rns8_matrix* matrix) {
  if (free_hip_matrix_storage(*matrix) == RNS8_SUCCESS) {
    delete matrix;
    return;
  }
  // Creation still returns a null output; its context owns the failed cleanup.
  matrix->failed_cleanup_next = ctx.failed_matrix_cleanup;
  ctx.failed_matrix_cleanup = matrix;
  ctx.hip_failure_poisoned = true;
}
} // namespace

rns8_status assign_matrix_instance_id(rns8_matrix& matrix) {
  static std::atomic<uint64_t> next_matrix_instance_id{1};
  const uint64_t id = next_matrix_instance_id.fetch_add(1, std::memory_order_relaxed);
  if (id == 0) {
    return RNS8_INTERNAL_ERROR;
  }
  matrix.matrix_instance_id = id;
  return RNS8_SUCCESS;
}

rns8_status rns8_create_matrix(rns8_context* ctx, const rns8_matrix_desc* desc, rns8_matrix** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_matrix_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }
    if (desc->flags && ctx->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    if (!backend_supports_semantics(ctx->backend, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE && ctx->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB &&
        !rns8::detail::qualified_i32_matrix_address(desc->rows, desc->cols, desc->cols,
                                                  std::max(prefix, 1u), 8)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    const auto discard = [ctx](rns8_matrix* matrix) noexcept {
      if (matrix) discard_failed_matrix(*ctx, matrix);
    };
    std::unique_ptr<rns8_matrix, decltype(discard)> owner(new (std::nothrow) rns8_matrix(), discard);
    auto* matrix = owner.get();
    if (!matrix) {
      return RNS8_INTERNAL_ERROR;
    }
    const rns8_status id_status = assign_matrix_instance_id(*matrix);
    if (id_status != RNS8_SUCCESS) {
      return id_status;
    }
    matrix->desc = *desc;
    matrix->backend = ctx->backend;
    matrix->desc.max_prefix = prefix;
    if (matrix->desc.logical_ld == 0) {
      matrix->desc.logical_ld = matrix->desc.cols;
    }
    if (matrix->desc.tile_m == 0) {
      matrix->desc.tile_m = 128;
    }
    if (matrix->desc.tile_n == 0) {
      matrix->desc.tile_n = 128;
    }
    matrix->prefix = prefix;
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics == RNS8_BOUNDED_I64) {
        matrix->native_i64.clear();
      } else if (matrix->desc.semantics == RNS8_BOUNDED_U64) {
        matrix->native_u64.clear();
      } else {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      invalidate_output_currentness(*matrix);
      const rns8_status status = ensure_hip_native_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
    } else if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      std::size_t bytes = 0;
      if (!wrap_byte_limb_bytes(desc->rows, desc->cols, bytes)) {
        return RNS8_RANGE_ERROR;
      }
      if (!hip_resident_rns_backend(ctx->backend)) {
        matrix->byte_limbs.assign(bytes, 0);
        mark_host_byte_limbs_current(*matrix);
      }
    } else if (uses_finite_storage(matrix->desc.semantics)) {
      std::size_t cells = 0;
      if (!matrix_cell_count(desc->rows, desc->cols, cells)) {
        return RNS8_RANGE_ERROR;
      }
      if (!hip_resident_rns_backend(ctx->backend)) matrix->residues.assign(cells, 0);
      matrix->finite_modulus = 0;
      clear_residue_current(*matrix);
      clear_byte_limb_current(*matrix);
    } else {
      std::size_t elements = 0;
      if (!rns_residue_count(desc->rows, desc->cols, prefix, elements)) {
        return RNS8_RANGE_ERROR;
      }
      if (!hip_resident_rns_backend(ctx->backend)) matrix->residues.assign(elements, 0);
      if (rns8::detail::exact_wide_semantics(matrix->desc.semantics) &&
          ctx->backend == RNS8_BACKEND_CPU_REFERENCE) {
        matrix->exact_range_prefix = prefix;  // The freshly created host matrix is zero.
      }
      clear_byte_limb_current(*matrix);
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = allocate_hip_matrix_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      if (ctx->auto_backend_selection &&
          (matrix->desc.semantics == RNS8_BOUNDED_I64 || matrix->desc.semantics == RNS8_BOUNDED_U64)) {
        const rns8_status native_status = ensure_hip_native_storage(*ctx, *matrix);
        if (native_status != RNS8_SUCCESS) {
          return native_status;
        }
      }
    }
    *out = owner.release();
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_matrix(rns8_matrix* matrix) {
  if (matrix && (matrix->prepared_references || matrix->in_flight_readers || matrix->in_flight_writer))
    return RNS8_BUSY;
  if (matrix) {
    const rns8_status status = free_hip_matrix_storage(*matrix);
    if (status != RNS8_SUCCESS) {
      matrix->generation_exhausted = true; // Failed destruction permits cleanup retry only.
      return status;
    }
    delete matrix;
    return status;
  }
  delete matrix;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_matrix_storage_info(const rns8_matrix* matrix, rns8_matrix_storage_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = matrix->backend;
    out->semantics = matrix->desc.semantics;
    out->logical_layout = matrix->desc.logical_layout;
    out->bound_kind = matrix->desc.bound_kind;
    out->rows = matrix->desc.rows;
    out->cols = matrix->desc.cols;
    out->logical_ld = matrix->desc.logical_ld;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->matrix_instance_id = matrix->matrix_instance_id;
    out->source_version = matrix->source_version;
    out->host_residues_current = matrix->host_residues_current ? 1u : 0u;
    out->device_residues_current = matrix->device_residues_current ? 1u : 0u;
    out->host_byte_limbs_current = matrix->host_byte_limbs_current ? 1u : 0u;
    out->device_byte_limbs_current = matrix->device_byte_limbs_current ? 1u : 0u;
    out->host_native_current = matrix->host_native_current ? 1u : 0u;
    out->device_native_current = matrix->device_native_current ? 1u : 0u;
    out->uses_residue_storage = (!matrix->residues.empty() || matrix->hip_residues != nullptr) ? 1u : 0u;
    out->uses_byte_limb_storage = (!matrix->byte_limbs.empty() || matrix->hip_byte_limbs != nullptr) ? 1u : 0u;
    out->uses_native_storage = matrix_has_native_storage(*matrix) ? 1u : 0u;
    out->hip_device_id = static_cast<int32_t>(matrix->hip_device_id);
    out->flags = 0;
    out->host_residue_bytes = static_cast<uint64_t>(matrix->residues.size() * sizeof(int8_t));
    out->device_residue_bytes = static_cast<uint64_t>(matrix->hip_residue_bytes);
    out->host_byte_limb_bytes = static_cast<uint64_t>(matrix->byte_limbs.size() * sizeof(uint8_t));
    out->device_byte_limb_bytes = static_cast<uint64_t>(matrix->hip_byte_limb_bytes);
    out->host_native_bytes = static_cast<uint64_t>(
        matrix->native_i64.size() * sizeof(int64_t) + matrix->native_u64.size() * sizeof(uint64_t));
    out->device_native_bytes =
        static_cast<uint64_t>(matrix->hip_native_i64_bytes + matrix->hip_native_u64_bytes);
    set_text(
        out->layout_version,
        sizeof(out->layout_version),
        native_vector_backend(matrix->backend) ? native_layout_version_for_semantics(matrix->desc.semantics)
                                               : persistent_layout_version_for_semantics(matrix->desc.semantics));
    set_text(out->storage_scope, sizeof(out->storage_scope), storage_scope_for_matrix(*matrix));
    set_text(out->detail, sizeof(out->detail), storage_detail_for_matrix(*matrix));
    return RNS8_SUCCESS;
  });
}


rns8_status rns8_get_matrix_allocation_info(const rns8_matrix* matrix, rns8_matrix_allocation_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!matrix || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))
      return RNS8_INVALID_ARGUMENT;
    rns8_matrix_allocation_info info{};
    info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
    const auto add=[](uint64_t& total, uint64_t value) {
      if (value>UINT64_MAX-total) return false;
      total+=value; return true;
    };
    std::size_t logical=0;
    const bool valid = native_vector_backend(matrix->backend)
      ? native_matrix_bytes(matrix->desc.rows,matrix->desc.cols,8,logical)
      : matrix->desc.semantics==RNS8_WRAP_U64_MOD_2_64
        ? wrap_byte_limb_bytes(matrix->desc.rows,matrix->desc.cols,logical)
        : uses_finite_storage(matrix->desc.semantics)
          ? matrix_cell_count(matrix->desc.rows,matrix->desc.cols,logical)
          : rns_residue_count(matrix->desc.rows,matrix->desc.cols,matrix->prefix,logical);
    if (!valid) return RNS8_RANGE_ERROR;
    info.logical_primary_bytes=logical;
    const auto host=[&](const auto& vector, bool current) {
      const uint64_t width=sizeof(typename std::decay_t<decltype(vector)>::value_type);
      if (vector.capacity()>UINT64_MAX/width || vector.size()>UINT64_MAX/width) return false;
      return add(info.host_payload_capacity_bytes,vector.capacity()*width) &&
             (!current || add(info.host_current_bytes,vector.size()*width));
    };
    if (!host(matrix->residues,matrix->host_residues_current) ||
        !host(matrix->byte_limbs,matrix->host_byte_limbs_current) ||
        !host(matrix->native_i64,matrix->host_native_current) ||
        !host(matrix->native_u64,matrix->host_native_current)) return RNS8_RANGE_ERROR;
    const auto device=[&](std::size_t bytes,bool current) {
      return add(info.device_payload_capacity_bytes,bytes) &&
             (!current || add(info.device_current_bytes,bytes));
    };
    if (!device(matrix->hip_residue_bytes,matrix->device_residues_current) ||
        !device(matrix->hip_byte_limb_bytes,matrix->device_byte_limbs_current) ||
        !device(matrix->hip_native_i64_bytes,matrix->device_native_current) ||
        !device(matrix->hip_native_u64_bytes,matrix->device_native_current)) return RNS8_RANGE_ERROR;
    info.device_upload_staging_bytes=matrix->hip_upload_bytes;
    info.device_export_staging_bytes=matrix->hip_export_bytes;
    for (auto bytes : {matrix->hip_status_bytes,matrix->hip_export_tile_schedule_bytes,
        matrix->hip_export_tile_bounds_bytes,matrix->hip_export_zero_a_rows_bytes,
        matrix->hip_export_zero_b_cols_bytes})
      if (!add(info.device_status_and_schedule_bytes,bytes)) return RNS8_RANGE_ERROR;
    for (auto bytes : {info.device_payload_capacity_bytes,info.device_upload_staging_bytes,
        info.device_export_staging_bytes,info.device_status_and_schedule_bytes})
      if (!add(info.device_total_capacity_bytes,bytes)) return RNS8_RANGE_ERROR;
    *out=info;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_trim_matrix_staging(rns8_matrix* matrix) {
  return guard_api([&]() -> rns8_status {
    if (!matrix) return RNS8_INVALID_ARGUMENT;
    if (matrix->prepared_references || matrix->in_flight_readers || matrix->in_flight_writer)
      return RNS8_BUSY;
    const auto release=[&](void*& pointer,std::size_t& bytes) {
      if (!pointer) { bytes=0; return RNS8_SUCCESS; }
      const auto status=rns8::detail::hip_direct_free(matrix->hip_device_id,pointer);
      if (status==RNS8_SUCCESS) { pointer=nullptr; bytes=0; }
      return status;
    };
    // Invalidate the descriptor cache before any partial release, so a retry or
    // later export must rebuild it even if one of the frees reports a failure.
    matrix->hip_export_schedule_fingerprint=0;
    matrix->hip_export_tile_max_elements=0;
    const auto release_schedule=[&](void*& pointer,std::size_t& bytes,uint64_t& count) {
      const auto status=release(pointer,bytes);
      if (status==RNS8_SUCCESS) count=0;
      return status;
    };
    rns8_status status;
    if ((status=release(matrix->hip_upload_buffer,matrix->hip_upload_bytes))!=RNS8_SUCCESS ||
        (status=release(matrix->hip_export_buffer,matrix->hip_export_bytes))!=RNS8_SUCCESS ||
        (status=release(matrix->hip_status_buffer,matrix->hip_status_bytes))!=RNS8_SUCCESS ||
        (status=release_schedule(matrix->hip_export_tile_schedule,matrix->hip_export_tile_schedule_bytes,
                                 matrix->hip_export_tile_schedule_count))!=RNS8_SUCCESS ||
        (status=release_schedule(matrix->hip_export_tile_bounds,matrix->hip_export_tile_bounds_bytes,
                                 matrix->hip_export_tile_bounds_count))!=RNS8_SUCCESS ||
        (status=release(matrix->hip_export_zero_a_rows,matrix->hip_export_zero_a_rows_bytes))!=RNS8_SUCCESS ||
        (status=release(matrix->hip_export_zero_b_cols,matrix->hip_export_zero_b_cols_bytes))!=RNS8_SUCCESS)
      return status;
    return RNS8_SUCCESS;
  });
}
