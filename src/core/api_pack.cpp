#include "core/api_internal.hpp"
#include <optional>

using namespace rns8::detail::api;

namespace {

bool same_source_device_residue_pack_current(const rns8_context& ctx, const rns8_matrix& matrix, uint64_t source_version) {
  return source_version != 0 && matrix.imported_source && !matrix.generation_exhausted && matrix.source_version == source_version && hip_resident_rns_backend(ctx.backend) &&
         matrix.device_residues_current && !should_populate_native_on_pack(ctx, matrix);
}

bool same_source_device_byte_limb_pack_current(
    const rns8_context& ctx,
    const rns8_matrix& matrix,
    uint64_t source_version) {
  return source_version != 0 && matrix.imported_source && !matrix.generation_exhausted && matrix.source_version == source_version && ctx.backend == RNS8_BACKEND_HIP_DIRECT &&
         matrix.device_byte_limbs_current;
}

bool same_source_finite_device_residue_pack_current(
    const rns8_context& ctx,
    const rns8_matrix& matrix,
    uint16_t modulus,
    uint64_t source_version) {
  return source_version != 0 && matrix.imported_source && !matrix.generation_exhausted && matrix.source_version == source_version && matrix.finite_modulus == modulus &&
         hip_resident_rns_backend(ctx.backend) && matrix.device_residues_current;
}

}  // namespace

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
    require_matrix_writable(*matrix);
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    // Reject unsupported source strides before scanning exact bounds or beginning
    // a resident write transaction. Rejection must preserve the previous value.
    if (hip_resident_rns_backend(ctx->backend) &&
        !rns8::detail::qualified_i32_matrix_address(matrix->desc.rows, matrix->desc.cols, ld, 1, sizeof(*src)))
      return RNS8_RANGE_ERROR;
    std::optional<MatrixWriteGuard> write;
    boost::multiprecision::cpp_int exact_bound = 0;
    rns8_exact_matrix_ranges exact_ranges;
    if (matrix->desc.semantics == RNS8_EXACT_WIDE_SIGNED) {
      if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix) ||
          (hip_resident_rns_backend(ctx->backend) && matrix->hip_device_id != ctx->device_id)) {
        return RNS8_INVALID_ARGUMENT;
      }
      // Stable source versions preserve the original packed value and proof.
      if (same_source_device_residue_pack_current(*ctx, *matrix, source_version)) {
        exact_bound = matrix->exact_max_magnitude;
      } else {
        exact_ranges =
            rns8::detail::native_exact_ranges(src, matrix->desc.rows, matrix->desc.cols, ld, exact_bound);
      }
      if (!rns8::detail::exact_range_fits(matrix->desc.semantics, exact_bound,
                                        rns8::detail::modulus_product(matrix->prefix))) {
        return RNS8_RANGE_ERROR;
      }
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_I64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_I64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      write.emplace(*matrix);
      const rns8_status status = upload_native_i64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      clear_residue_current(*matrix);
      clear_byte_limb_current(*matrix);
      publish_matrix_value(*matrix, true, source_version);
      write->committed = true;
      return RNS8_SUCCESS;
    }
    if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend) && matrix->hip_device_id != ctx->device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      if (same_source_device_residue_pack_current(*ctx, *matrix, source_version)) {
        return RNS8_SUCCESS;
      }
      write.emplace(*matrix);
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
      mark_device_residues_current(*matrix);
    } else {
      write.emplace(*matrix);
      rns8::detail::pack_i64_matrix(*matrix, src, ld);
      mark_host_residues_current(*matrix);
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_i64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else {
      clear_native_current(*matrix);
    }
    if (matrix->desc.semantics == RNS8_EXACT_WIDE_SIGNED) {
      matrix->exact_max_magnitude.swap(exact_bound);
      matrix->exact_axis_ranges.rows.swap(exact_ranges.rows);
      matrix->exact_axis_ranges.cols.swap(exact_ranges.cols);
      matrix->exact_range_prefix = matrix->prefix;
    }
    publish_matrix_value(*matrix, true, source_version);
    write->committed = true;
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
    require_matrix_writable(*matrix);
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    // Reject unsupported source strides before scanning exact bounds or beginning
    // a resident write transaction. Rejection must preserve the previous value.
    if (hip_resident_rns_backend(ctx->backend) &&
        !rns8::detail::qualified_i32_matrix_address(matrix->desc.rows, matrix->desc.cols, ld, 1, sizeof(*src)))
      return RNS8_RANGE_ERROR;
    std::optional<MatrixWriteGuard> write;
    boost::multiprecision::cpp_int exact_bound = 0;
    rns8_exact_matrix_ranges exact_ranges;
    if (matrix->desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix) ||
          (hip_resident_rns_backend(ctx->backend) && matrix->hip_device_id != ctx->device_id)) {
        return RNS8_INVALID_ARGUMENT;
      }
      // Stable source versions preserve the original packed value and proof.
      if (same_source_device_residue_pack_current(*ctx, *matrix, source_version)) {
        exact_bound = matrix->exact_max_magnitude;
      } else {
        exact_ranges =
            rns8::detail::native_exact_ranges(src, matrix->desc.rows, matrix->desc.cols, ld, exact_bound);
      }
      if (!rns8::detail::exact_range_fits(matrix->desc.semantics, exact_bound,
                                        rns8::detail::modulus_product(matrix->prefix))) {
        return RNS8_RANGE_ERROR;
      }
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_U64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_U64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      write.emplace(*matrix);
      const rns8_status status = upload_native_u64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      clear_residue_current(*matrix);
      clear_byte_limb_current(*matrix);
      publish_matrix_value(*matrix, true, source_version);
      write->committed = true;
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
      write.emplace(*matrix);
        rns8::detail::pack_wrap_u64_matrix(*matrix, src, ld);
        mark_host_byte_limbs_current(*matrix);
      } else if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
        if (same_source_device_byte_limb_pack_current(*ctx, *matrix, source_version)) {
          return RNS8_SUCCESS;
        }
      write.emplace(*matrix);
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
        mark_device_byte_limbs_current(*matrix);
      } else {
        return RNS8_UNSUPPORTED_BACKEND;
      }
    } else if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    } else if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (same_source_device_residue_pack_current(*ctx, *matrix, source_version)) {
        return RNS8_SUCCESS;
      }
      write.emplace(*matrix);
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
      mark_device_residues_current(*matrix);
    } else {
      write.emplace(*matrix);
      rns8::detail::pack_u64_matrix(*matrix, src, ld);
      mark_host_residues_current(*matrix);
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_u64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else if (matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      clear_native_current(*matrix);
    }
    if (matrix->desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      matrix->exact_max_magnitude.swap(exact_bound);
      matrix->exact_axis_ranges.rows.swap(exact_ranges.rows);
      matrix->exact_axis_ranges.cols.swap(exact_ranges.cols);
      matrix->exact_range_prefix = matrix->prefix;
    }
    publish_matrix_value(*matrix, true, source_version);
    write->committed = true;
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
    // Reject unsupported source strides before scanning exact bounds or beginning
    // a resident write transaction. Rejection must preserve the previous value.
    if (hip_resident_rns_backend(ctx->backend) &&
        !rns8::detail::qualified_i32_matrix_address(matrix->desc.rows, matrix->desc.cols, ld, 1, sizeof(*src)))
      return RNS8_RANGE_ERROR;
    if (!uses_finite_storage(matrix->desc.semantics) ||
        !finite_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
    std::optional<MatrixWriteGuard> write;
    if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (same_source_finite_device_residue_pack_current(*ctx, *matrix, modulus, source_version)) {
        return RNS8_SUCCESS;
      }
      write.emplace(*matrix);
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
      mark_device_residues_current(*matrix);
    } else if (ctx->backend == RNS8_BACKEND_CPU_REFERENCE) {
      write.emplace(*matrix);
      rns8::detail::pack_finite_u8_matrix(*matrix, src, ld, modulus);
      mark_host_residues_current(*matrix);
    } else {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    matrix->finite_modulus = modulus;
    publish_matrix_value(*matrix, true, source_version);
    write->committed = true;
    return RNS8_SUCCESS;
  });
}

// Shared checked arbitrary-width import. Decode and prove before invalidating;
// after begin-write, any copy/allocation failure leaves every domain unavailable.
rns8_status rns8::detail::api::import_exact_wide_limbs(
    rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src,
    int64_t ld, uint32_t limb_count, bool signed_values, uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    const auto expected = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
    require_matrix_writable(matrix);
    if (!src || limb_count == 0 || limb_count > 32 || matrix.desc.semantics != expected ||
        ctx.backend != matrix.backend || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld) ||
        !rns_matrix_storage_matches(matrix, ctx.backend, matrix.desc.rows, matrix.desc.cols, matrix.prefix) ||
        (hip_resident_rns_backend(ctx.backend) && matrix.hip_device_id != ctx.device_id))
      return RNS8_INVALID_ARGUMENT;
    std::size_t cells = 0;
    if (!matrix_cell_count(matrix.desc.rows, matrix.desc.cols, cells) ||
        static_cast<uint64_t>(ld) > std::numeric_limits<std::size_t>::max() / limb_count / matrix.desc.rows)
      return RNS8_INVALID_ARGUMENT;
    std::vector<boost::multiprecision::cpp_int> values(cells);
    for (int64_t row = 0; row < matrix.desc.rows; ++row) {
      for (int64_t col = 0; col < matrix.desc.cols; ++col) {
        const auto* limbs = src + (static_cast<std::size_t>(row) * ld + col) * limb_count;
        auto& value = values[static_cast<std::size_t>(row) * matrix.desc.cols + col];
        for (uint32_t i = limb_count; i != 0; --i) { value <<= 64; value += limbs[i - 1]; }
        if (signed_values && (limbs[limb_count - 1] >> 63))
          value -= boost::multiprecision::cpp_int(1) << (64 * limb_count);
      }
    }
    boost::multiprecision::cpp_int bound;
    auto ranges = rns8::detail::native_exact_ranges(values.data(), matrix.desc.rows, matrix.desc.cols,
                                                    matrix.desc.cols, bound);
    if (!rns8::detail::exact_range_fits(expected, bound, rns8::detail::modulus_product(matrix.prefix)))
      return RNS8_RANGE_ERROR;
    MatrixWriteGuard write(matrix);
    for (uint32_t p = 0; p < matrix.prefix; ++p)
      for (std::size_t cell = 0; cell < cells; ++cell)
        matrix.residues[p * cells + cell] =
            rns8::detail::centered_residue(values[cell], rns8::detail::kDefaultModuli[p]);
    if (matrix.hip_residues) {
      const auto status = run_timed_api_status("pack_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(ctx.device_id, matrix.hip_residues,
                                                          matrix.residues.data(), matrix.hip_residue_bytes);
      });
      if (status != RNS8_SUCCESS) return status;
      mark_device_residues_current(matrix);
    } else mark_host_residues_current(matrix);
    matrix.exact_max_magnitude.swap(bound);
    matrix.exact_axis_ranges.rows.swap(ranges.rows);
    matrix.exact_axis_ranges.cols.swap(ranges.cols);
    matrix.exact_range_prefix = matrix.prefix;
    publish_matrix_value(matrix, true, source_version);
    write.committed = true;
    return RNS8_SUCCESS;
  });
}
