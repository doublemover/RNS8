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
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_matrix_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }
    if (!backend_supports_semantics(ctx->backend, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* matrix = new (std::nothrow) rns8_matrix();
    if (!matrix) {
      return RNS8_INTERNAL_ERROR;
    }
    const rns8_status id_status = assign_matrix_instance_id(*matrix);
    if (id_status != RNS8_SUCCESS) {
      delete matrix;
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
        delete matrix;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      invalidate_output_currentness(*matrix);
      const rns8_status status = ensure_hip_native_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        (void)free_hip_matrix_storage(*matrix);
        delete matrix;
        return status;
      }
    } else if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      std::size_t bytes = 0;
      if (!wrap_byte_limb_bytes(desc->rows, desc->cols, bytes)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->byte_limbs.assign(bytes, 0);
      mark_host_byte_limbs_current(*matrix);
    } else if (uses_finite_storage(matrix->desc.semantics)) {
      std::size_t cells = 0;
      if (!matrix_cell_count(desc->rows, desc->cols, cells)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(cells, 0);
      matrix->finite_modulus = 0;
      clear_residue_current(*matrix);
      clear_byte_limb_current(*matrix);
    } else {
      std::size_t elements = 0;
      if (!rns_residue_count(desc->rows, desc->cols, prefix, elements)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(elements, 0);
      clear_byte_limb_current(*matrix);
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = allocate_hip_matrix_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        delete matrix;
        return status;
      }
      if (ctx->auto_backend_selection &&
          (matrix->desc.semantics == RNS8_BOUNDED_I64 || matrix->desc.semantics == RNS8_BOUNDED_U64)) {
        const rns8_status native_status = ensure_hip_native_storage(*ctx, *matrix);
        if (native_status != RNS8_SUCCESS) {
          (void)free_hip_matrix_storage(*matrix);
          delete matrix;
          return native_status;
        }
      }
    }
    *out = matrix;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_matrix(rns8_matrix* matrix) {
  if (matrix) {
    const rns8_status status = free_hip_matrix_storage(*matrix);
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

