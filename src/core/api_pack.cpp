#include "core/api_internal.hpp"

using namespace rns8::detail::api;

rns8_status rns8_pack_i64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_I64 && matrix->desc.semantics != RNS8_EXACT_WIDE_SIGNED) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_I64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_I64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = upload_native_i64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->source_version = source_version;
      return RNS8_SUCCESS;
    }
    if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend) && matrix->hip_device_id != ctx->device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = rns8::detail::hip_direct_pack_i64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_i64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_i64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else {
      matrix->host_native_current = false;
      matrix->device_native_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_u64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_U64 && matrix->desc.semantics != RNS8_EXACT_WIDE_UNSIGNED &&
        matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_U64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_U64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = upload_native_u64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->source_version = source_version;
      return RNS8_SUCCESS;
    }
    if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      if (!wrap_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (ctx->backend == RNS8_BACKEND_HIP_DIRECT && matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (ctx->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
        rns8::detail::pack_wrap_u64_matrix(*matrix, src, ld);
        matrix->host_residues_current = false;
        matrix->device_residues_current = false;
        matrix->host_byte_limbs_current = true;
        matrix->device_byte_limbs_current = false;
      } else if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
        const rns8_status status = rns8::detail::wrap64_hip_pack_u64_device(
            ctx->device_id,
            src,
            &matrix->hip_upload_buffer,
            &matrix->hip_upload_bytes,
            matrix->hip_byte_limbs,
            matrix->desc.rows,
            matrix->desc.cols,
            ld);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        matrix->host_residues_current = false;
        matrix->device_residues_current = false;
        matrix->host_byte_limbs_current = false;
        matrix->device_byte_limbs_current = true;
      } else {
        return RNS8_UNSUPPORTED_BACKEND;
      }
    } else if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    } else if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = rns8::detail::hip_direct_pack_u64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_u64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_u64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else if (matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      matrix->host_native_current = false;
      matrix->device_native_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_finite_u8(
    rns8_context* ctx,
    rns8_matrix* matrix,
    uint16_t modulus,
    const uint8_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld) ||
        !rns8::detail::valid_finite_modulus_for_semantics(matrix->desc.semantics, modulus)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!uses_finite_storage(matrix->desc.semantics) ||
        !finite_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = rns8::detail::hip_direct_pack_finite_u8_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = true;
    } else if (ctx->backend == RNS8_BACKEND_CPU_REFERENCE) {
      rns8::detail::pack_finite_u8_matrix(*matrix, src, ld, modulus);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    } else {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    matrix->host_byte_limbs_current = false;
    matrix->device_byte_limbs_current = false;
    matrix->finite_modulus = modulus;
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}
