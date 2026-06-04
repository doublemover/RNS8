#include "core/api_internal.hpp"

namespace rns8::detail::api {

rns8_status validate_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix) {
  if (!plan_schedule_contract_matches(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend) ||
      plan.desc.semantics != semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) && C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = uses_rns_storage(semantics) ? rns_storage_prefix_for_plan(plan) : prefix;
  const rns8_bound_kind storage_bound_kind =
      uses_rns_storage(semantics) ? storage_bound_kind_for_plan(plan) : bound_kind;
  if (!matrix_descriptor_matches(
          C, semantics, storage_bound_kind, plan.desc.m, plan.desc.n, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (native_vector_backend(plan.backend)) {
    return bounded_native_storage_matches(C, semantics, plan.desc.m, plan.desc.n) &&
                   bounded_native_state_current(C)
               ? RNS8_SUCCESS
               : RNS8_INVALID_ARGUMENT;
  }
  if (uses_rns_storage(semantics) &&
      (!rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, storage_prefix) ||
       !rns_residue_state_current_for_backend(C, plan.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (semantics == RNS8_WRAP_U64_MOD_2_64 &&
      (!wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
       !wrap_byte_limb_state_current_for_backend(C, plan.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_finite_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& C) {
  if (!plan_schedule_contract_matches(plan) || !uses_finite_storage(plan.desc.semantics) ||
      !rns8::detail::valid_finite_modulus_for_semantics(plan.desc.semantics, modulus) ||
      plan.desc.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) && C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          C, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!finite_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
      !rns_residue_state_current_for_backend(C, plan.backend) || C.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}
bool signed_value_within_bound(int64_t value, uint64_t bound) {
  if (value < 0) {
    const uint64_t magnitude = value == std::numeric_limits<int64_t>::min()
                                   ? (uint64_t{1} << 63u)
                                   : static_cast<uint64_t>(-value);
    return magnitude <= bound;
  }
  return static_cast<uint64_t>(value) <= bound;
}

rns8_status export_native_i64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    int64_t* dst,
    int64_t ld) {
  std::vector<int64_t> staged(static_cast<std::size_t>(plan.desc.m) * static_cast<std::size_t>(plan.desc.n), 0);
  rns8_status status = run_timed_api_status("vector_alu_output_d2h", [&]() {
    return rns8::detail::hip_direct_copy_device_to_host(
        ctx.device_id, staged.data(), C.hip_native_i64, staged.size() * sizeof(int64_t));
  });
  if (status != RNS8_SUCCESS) {
    return status;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const int64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      if (!signed_value_within_bound(value, bound_for_cell(plan, row, col))) {
        return RNS8_RANGE_ERROR;
      }
    }
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const int64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      dst[row * ld + col] = value;
    }
  }
  return RNS8_SUCCESS;
}

rns8_status export_native_u64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    uint64_t* dst,
    int64_t ld) {
  std::vector<uint64_t> staged(static_cast<std::size_t>(plan.desc.m) * static_cast<std::size_t>(plan.desc.n), 0);
  rns8_status status = run_timed_api_status("vector_alu_output_d2h", [&]() {
    return rns8::detail::hip_direct_copy_device_to_host(
        ctx.device_id, staged.data(), C.hip_native_u64, staged.size() * sizeof(uint64_t));
  });
  if (status != RNS8_SUCCESS) {
    return status;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const uint64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      if (value > bound_for_cell(plan, row, col)) {
        return RNS8_RANGE_ERROR;
      }
    }
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const uint64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      dst[row * ld + col] = value;
    }
  }
  return RNS8_SUCCESS;
}

rns8_status ensure_hip_export_tile_metadata(rns8_context& ctx, const rns8_plan& plan, rns8_matrix& matrix) {
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, matrix, plan.backend) ||
      !hip_resident_rns_backend(plan.backend) || plan.tile_schedule.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.tile_bounds.size() != plan.tile_schedule.size() ||
      plan.tile_schedule.size() > std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry) ||
      plan.tile_bounds.size() > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t)) {
    return RNS8_INTERNAL_ERROR;
  }
  uint64_t max_tile_elements = 0;
  for (const auto& entry : plan.tile_schedule) {
    if (entry.row_extent <= 0 || entry.col_extent <= 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (col_extent != 0 && row_extent > std::numeric_limits<uint64_t>::max() / col_extent) {
      return RNS8_RANGE_ERROR;
    }
    max_tile_elements = std::max(max_tile_elements, row_extent * col_extent);
  }
  if (max_tile_elements == 0 || max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_RANGE_ERROR;
  }
  const uint64_t fingerprint = plan_workspace_fingerprint(plan);
  const uint64_t entry_count = static_cast<uint64_t>(plan.tile_schedule.size());
  const std::size_t schedule_bytes = plan.tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry);
  const std::size_t bounds_bytes = plan.tile_bounds.size() * sizeof(uint64_t);
  const bool metadata_current =
      matrix.hip_export_schedule_fingerprint == fingerprint &&
      matrix.hip_export_tile_schedule != nullptr &&
      matrix.hip_export_tile_schedule_bytes >= schedule_bytes &&
      matrix.hip_export_tile_schedule_count == entry_count &&
      matrix.hip_export_tile_bounds != nullptr &&
      matrix.hip_export_tile_bounds_bytes >= bounds_bytes &&
      matrix.hip_export_tile_bounds_count == entry_count &&
      matrix.hip_export_tile_max_elements == max_tile_elements;
  if (metadata_current) {
    return RNS8_SUCCESS;
  }
  rns8_status status = rns8::detail::hip_direct_ensure_upload_buffer(
      ctx.device_id, schedule_bytes, &matrix.hip_export_tile_schedule, &matrix.hip_export_tile_schedule_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      ctx.device_id, bounds_bytes, &matrix.hip_export_tile_bounds, &matrix.hip_export_tile_bounds_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_export_tile_schedule, plan.tile_schedule.data(), schedule_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_export_tile_bounds, plan.tile_bounds.data(), bounds_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  matrix.hip_export_tile_schedule_count = entry_count;
  matrix.hip_export_tile_bounds_count = entry_count;
  matrix.hip_export_schedule_fingerprint = fingerprint;
  matrix.hip_export_tile_max_elements = max_tile_elements;
  return RNS8_SUCCESS;
}

bool schedule_all_zero_output_tiles(const rns8_plan& plan) {
  if (plan.tile_schedule.empty()) {
    return false;
  }
  for (const auto& entry : plan.tile_schedule) {
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
      return false;
    }
  }
  return true;
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

rns8_status rns8_export_i64(rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* C, int64_t* dst, int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_BOUNDED_I64, plan->desc.bound_kind, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (native_vector_backend(plan->backend)) {
      return export_native_i64(*ctx, *plan, *C, dst, ld);
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        auto* mutable_c = const_cast<rns8_matrix*>(C);
        const rns8_status metadata_status = ensure_hip_export_tile_metadata(*ctx, *plan, *mutable_c);
        if (metadata_status != RNS8_SUCCESS) {
          return metadata_status;
        }
        return rns8::detail::hip_direct_export_i64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &mutable_c->hip_export_buffer,
            &mutable_c->hip_export_bytes,
            &mutable_c->hip_status_buffer,
            &mutable_c->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            mutable_c->hip_export_tile_schedule,
            mutable_c->hip_export_tile_bounds,
            mutable_c->hip_export_tile_schedule_count,
            mutable_c->hip_export_tile_max_elements,
            schedule_all_zero_output_tiles(*plan),
            dst,
            ld);
      }
      return rns8::detail::hip_direct_export_i64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
    }
    std::vector<int64_t> staged(static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n), 0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const auto* entry = tile_schedule_entry_for_cell(*plan, row, col);
        if (entry && (entry->flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
          staged[static_cast<std::size_t>(row * plan->desc.n + col)] = 0;
          continue;
        }
        int64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_signed(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        staged[static_cast<std::size_t>(row * plan->desc.n + col)] = value;
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        dst[row * ld + col] = staged[static_cast<std::size_t>(row * plan->desc.n + col)];
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_BOUNDED_U64, plan->desc.bound_kind, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (native_vector_backend(plan->backend)) {
      return export_native_u64(*ctx, *plan, *C, dst, ld);
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        auto* mutable_c = const_cast<rns8_matrix*>(C);
        const rns8_status metadata_status = ensure_hip_export_tile_metadata(*ctx, *plan, *mutable_c);
        if (metadata_status != RNS8_SUCCESS) {
          return metadata_status;
        }
        return rns8::detail::hip_direct_export_u64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &mutable_c->hip_export_buffer,
            &mutable_c->hip_export_bytes,
            &mutable_c->hip_status_buffer,
            &mutable_c->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            mutable_c->hip_export_tile_schedule,
            mutable_c->hip_export_tile_bounds,
            mutable_c->hip_export_tile_schedule_count,
            mutable_c->hip_export_tile_max_elements,
            schedule_all_zero_output_tiles(*plan),
            dst,
            ld);
      }
      return rns8::detail::hip_direct_export_u64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
    }
    std::vector<uint64_t> staged(static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n), 0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const auto* entry = tile_schedule_entry_for_cell(*plan, row, col);
        if (entry && (entry->flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
          staged[static_cast<std::size_t>(row * plan->desc.n + col)] = 0;
          continue;
        }
        uint64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_unsigned(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        staged[static_cast<std::size_t>(row * plan->desc.n + col)] = value;
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        dst[row * ld + col] = staged[static_cast<std::size_t>(row * plan->desc.n + col)];
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, 0);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (!wrap_matrix_storage_matches(*C, plan->backend, plan->desc.m, plan->desc.n)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      if (!C->host_byte_limbs_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      for (int64_t row = 0; row < plan->desc.m; ++row) {
        for (int64_t col = 0; col < plan->desc.n; ++col) {
          dst[row * ld + col] = rns8::detail::wrap_u64_matrix_cell(*C, row, col);
        }
      }
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      return rns8::detail::wrap64_hip_export_u64_device(
          ctx->device_id,
          C->hip_byte_limbs,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          plan->desc.m,
          plan->desc.n,
          dst,
          ld);
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_export_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* C,
    uint8_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status = validate_finite_export_matrix(*ctx, *plan, modulus, *C);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      auto* mutable_c = const_cast<rns8_matrix*>(C);
      return rns8::detail::hip_direct_export_finite_u8_device(
          ctx->device_id,
          C->hip_residues,
          &mutable_c->hip_export_buffer,
          &mutable_c->hip_export_bytes,
          plan->desc.m,
          plan->desc.n,
          modulus,
          dst,
          ld);
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      rns8::detail::export_finite_u8_matrix(*C, dst, ld, modulus);
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_limb_export_access(plan->desc.m, plan->desc.n, ld, limb_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!C->device_residues_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      return rns8::detail::hip_direct_export_exact_wide_signed_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    std::vector<uint64_t> staged(
        static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n) *
            static_cast<std::size_t>(limb_count),
        0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        uint64_t* cell_dst =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        const rns8_status status = rns8::detail::export_exact_wide_signed_limbs(
            residues, plan->prefix, cell_dst, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const uint64_t* cell_src =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        uint64_t* cell_dst = dst + static_cast<std::size_t>((row * ld + col) * limb_count);
        std::copy(cell_src, cell_src + limb_count, cell_dst);
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_exact_wide_unsigned_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_limb_export_access(plan->desc.m, plan->desc.n, ld, limb_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!C->device_residues_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      return rns8::detail::hip_direct_export_exact_wide_unsigned_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    std::vector<uint64_t> staged(
        static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n) *
            static_cast<std::size_t>(limb_count),
        0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        uint64_t* cell_dst =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        const rns8_status status = rns8::detail::export_exact_wide_unsigned_limbs(
            residues, plan->prefix, cell_dst, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const uint64_t* cell_src =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        uint64_t* cell_dst = dst + static_cast<std::size_t>((row * ld + col) * limb_count);
        std::copy(cell_src, cell_src + limb_count, cell_dst);
      }
    }
    return RNS8_SUCCESS;
  });
}
