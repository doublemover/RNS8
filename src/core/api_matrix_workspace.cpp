#include "core/api_internal.hpp"

namespace rns8::detail::api {

bool matrix_descriptor_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t tile_m,
    uint32_t tile_n) {
  if (matrix.desc.semantics != semantics || matrix.desc.bound_kind != bound_kind || matrix.desc.rows != rows ||
      matrix.desc.cols != cols || matrix.desc.logical_layout != RNS8_LAYOUT_ROW_MAJOR ||
      matrix.desc.logical_ld < matrix.desc.cols || matrix.desc.flags != 0 || matrix.prefix != prefix ||
      matrix.desc.max_prefix != prefix) {
    return false;
  }
  if (semantics == RNS8_WRAP_U64_MOD_2_64 || is_per_tile_bound_kind(bound_kind)) {
    return matrix.desc.tile_m == tile_m && matrix.desc.tile_n == tile_n;
  }
  return true;
}

bool configured_tile_size_valid(uint32_t value) {
  return value >= 64 && value <= 512 && (value & (value - 1u)) == 0;
}

bool wrap_byte_limb_bytes(int64_t rows, int64_t cols, std::size_t& bytes) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  constexpr uint64_t limbs_per_cell = 8;
  const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_bytes / u_cols / limbs_per_cell) {
    return false;
  }
  bytes = static_cast<std::size_t>(u_rows * u_cols * limbs_per_cell);
  return true;
}

bool matrix_cell_count(int64_t rows, int64_t cols, std::size_t& cells) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  const uint64_t max_cells = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_cells / u_cols) {
    return false;
  }
  cells = static_cast<std::size_t>(u_rows * u_cols);
  return true;
}

bool rns_residue_count(int64_t rows, int64_t cols, uint32_t prefix, std::size_t& residues) {
  std::size_t cells = 0;
  if (prefix == 0 || !matrix_cell_count(rows, cols, cells)) {
    return false;
  }
  if (cells > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(prefix)) {
    return false;
  }
  residues = cells * static_cast<std::size_t>(prefix);
  return true;
}

bool rns_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols, uint32_t prefix) {
  std::size_t expected_residues = 0;
  if (!rns_residue_count(rows, cols, prefix, expected_residues)) {
    return false;
  }
  if (matrix.residues.size() != expected_residues || !matrix.byte_limbs.empty() || matrix.hip_byte_limbs ||
      matrix.hip_byte_limb_bytes != 0 || matrix.host_byte_limbs_current || matrix.device_byte_limbs_current) {
    return false;
  }
  const std::size_t expected_bytes = expected_residues * sizeof(int8_t);
  if (hip_resident_rns_backend(backend)) {
    return matrix.hip_residues != nullptr && matrix.hip_residue_bytes == expected_bytes;
  }
  return matrix.hip_residues == nullptr && matrix.hip_residue_bytes == 0 && !matrix.device_residues_current;
}

bool finite_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols) {
  std::size_t expected_cells = 0;
  if (!matrix_cell_count(rows, cols, expected_cells)) {
    return false;
  }
  if (matrix.prefix != 0 || matrix.residues.size() != expected_cells || !matrix.byte_limbs.empty() ||
      matrix.hip_byte_limbs || matrix.hip_byte_limb_bytes != 0 || matrix.host_byte_limbs_current ||
      matrix.device_byte_limbs_current) {
    return false;
  }
  const std::size_t expected_bytes = expected_cells * sizeof(int8_t);
  if (hip_resident_rns_backend(backend)) {
    return matrix.hip_residues != nullptr && matrix.hip_residue_bytes == expected_bytes;
  }
  return matrix.hip_residues == nullptr && matrix.hip_residue_bytes == 0 && !matrix.device_residues_current;
}

bool rns_residue_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend) {
  if (hip_resident_rns_backend(backend)) {
    return matrix.device_residues_current;
  }
  return matrix.host_residues_current && !matrix.device_residues_current;
}

bool plan_schedule_contract_matches(const rns8_plan& plan) {
  if (!backend_supports_semantics(plan.backend, plan.desc.semantics) ||
      !configured_tile_size_valid(plan.desc.tile_m) || !configured_tile_size_valid(plan.desc.tile_n) ||
      plan.desc.flags != 0 || plan.prefix != plan.desc.max_prefix) {
    return false;
  }
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(plan.backend, capability);
  if (plan.backend_selected_kernel != selected_kernel_for_plan(plan) ||
      plan.backend_library != capability.library_name ||
      !backend_library_version_matches_plan(plan, capability) ||
      plan.backend_capability_status != capability.status ||
      plan.backend_epilogue_mode != epilogue_mode_for_plan(plan) ||
      plan.backend_workspace_mode != workspace_mode_for_plan(plan) ||
      plan.backend_isa_evidence != isa_evidence_for_plan(plan) ||
      plan.backend_workspace_required_bytes != workspace_required_bytes_for_plan(plan) ||
      plan.backend_autotune_key != build_autotune_key(plan)) {
    return false;
  }
  const uint64_t expected_tile_rows = ceil_div_i64_u32(plan.desc.m, plan.desc.tile_m);
  const uint64_t expected_tile_cols = ceil_div_i64_u32(plan.desc.n, plan.desc.tile_n);
  if (plan.schedule_tile_rows != expected_tile_rows || plan.schedule_tile_cols != expected_tile_cols) {
    return false;
  }
  if (expected_tile_cols != 0 && expected_tile_rows > std::numeric_limits<uint64_t>::max() / expected_tile_cols) {
    return false;
  }
  if (plan.schedule_tile_count != expected_tile_rows * expected_tile_cols || plan.schedule_tile_count == 0) {
    return false;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return plan.desc.bound_kind == RNS8_BOUND_NONE && plan.desc.bound == 0 && plan.prefix == 0 &&
           plan.modulus_product == 0 && plan.tile_bounds.empty() && plan.tile_schedule.empty() &&
           plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.schedule_flags == 0;
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return plan.desc.bound_kind == RNS8_BOUND_NONE && plan.desc.bound == 0 && plan.prefix == 0 &&
           plan.desc.max_prefix == 0 && plan.modulus_product == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.schedule_flags == 0;
  }
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0 || plan.prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (plan.modulus_product != rns8::detail::modulus_product(plan.prefix)) {
    return false;
  }
  if ((plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) &&
      (plan.desc.bound_kind != RNS8_BOUND_NONE || plan.desc.bound != 0 || plan.desc.tile_bounds != nullptr ||
       plan.desc.tile_bounds_count != 0)) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_I64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS && plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_ABS) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_U64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED &&
      plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_UNSIGNED) {
    return false;
  }
  const bool per_tile = is_per_tile_bound_kind(plan.desc.bound_kind);
  if (!per_tile) {
    return plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.schedule_min_required_prefix > 0 &&
           plan.schedule_max_required_prefix == plan.schedule_min_required_prefix &&
           plan.schedule_min_selected_prefix == plan.prefix && plan.schedule_max_selected_prefix == plan.prefix &&
           plan.schedule_prefix_group_count == 1 && plan.schedule_adaptive_prefix_active == 0 &&
           plan.schedule_adaptive_skip_active == 0 && plan.schedule_flags == 0;
  }
  if (plan.desc.bound != 0 || plan.desc.tile_bounds_count != static_cast<uint64_t>(plan.tile_bounds.size()) ||
      plan.desc.tile_bounds_count != plan.schedule_tile_count || plan.desc.tile_bounds != nullptr ||
      plan.tile_schedule.size() != plan.tile_bounds.size() || plan.schedule_prefix_group_count == 0 ||
      plan.schedule_min_required_prefix == 0 || plan.schedule_min_selected_prefix == 0 ||
      plan.schedule_max_required_prefix > plan.prefix || plan.schedule_max_selected_prefix > plan.prefix ||
      plan.schedule_min_required_prefix > plan.schedule_max_required_prefix ||
      plan.schedule_min_selected_prefix > plan.schedule_max_selected_prefix || plan.schedule_flags != 0) {
    return false;
  }
  for (uint64_t index = 0; index < plan.schedule_tile_count; ++index) {
    const auto& entry = plan.tile_schedule[static_cast<std::size_t>(index)];
    if (!rns8::detail::valid_abi(entry.struct_size, entry.abi_version, sizeof(entry)) || entry.flags != 0 ||
        entry.tile_row != index / plan.schedule_tile_cols || entry.tile_col != index % plan.schedule_tile_cols ||
        entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 || entry.col_extent <= 0 ||
        entry.row_offset >= plan.desc.m || entry.col_offset >= plan.desc.n ||
        entry.row_extent > plan.desc.m - entry.row_offset || entry.col_extent > plan.desc.n - entry.col_offset ||
        entry.required_prefix == 0 || entry.selected_prefix == 0 || entry.required_prefix > entry.selected_prefix ||
        entry.selected_prefix > plan.prefix || entry.group_index >= plan.schedule_prefix_group_count) {
      return false;
    }
  }
  return true;
}

bool wrap_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols) {
  std::size_t expected_bytes = 0;
  if (!wrap_byte_limb_bytes(rows, cols, expected_bytes)) {
    return false;
  }
  if (!matrix.residues.empty() || matrix.hip_residues || matrix.hip_residue_bytes != 0 ||
      matrix.byte_limbs.size() != expected_bytes || matrix.host_residues_current ||
      matrix.device_residues_current) {
    return false;
  }
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return !matrix.host_byte_limbs_current && matrix.hip_byte_limbs != nullptr &&
           matrix.hip_byte_limb_bytes == expected_bytes;
  }
  return matrix.hip_byte_limbs == nullptr && matrix.hip_byte_limb_bytes == 0 &&
         matrix.hip_upload_buffer == nullptr && matrix.hip_upload_bytes == 0 &&
         matrix.hip_export_buffer == nullptr && matrix.hip_export_bytes == 0 &&
         matrix.hip_status_buffer == nullptr && matrix.hip_status_bytes == 0 &&
         !matrix.device_byte_limbs_current;
}

bool wrap_byte_limb_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend) {
  if (matrix.host_residues_current || matrix.device_residues_current) {
    return false;
  }
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return matrix.device_byte_limbs_current;
  }
  return matrix.host_byte_limbs_current && !matrix.device_byte_limbs_current;
}

rns8_status validate_plan_context_workspace(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace) {
  if (!plan_schedule_contract_matches(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) || workspace.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.semantics != plan.desc.semantics || workspace.bound_kind != plan.desc.bound_kind) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.m != plan.desc.m || workspace.n != plan.desc.n || workspace.k != plan.desc.k ||
      workspace.prefix != plan.prefix) {
    return RNS8_WORKSPACE_TOO_SMALL;
  }
  if (workspace.bound != plan.desc.bound || workspace.finite_modulus != plan.desc.finite_modulus ||
      workspace.tile_m != plan.desc.tile_m || workspace.tile_n != plan.desc.tile_n ||
      workspace.schedule_tile_rows != plan.schedule_tile_rows ||
      workspace.schedule_tile_cols != plan.schedule_tile_cols ||
      workspace.schedule_tile_count != plan.schedule_tile_count ||
      workspace.schedule_min_required_prefix != plan.schedule_min_required_prefix ||
      workspace.schedule_max_required_prefix != plan.schedule_max_required_prefix ||
      workspace.schedule_min_selected_prefix != plan.schedule_min_selected_prefix ||
      workspace.schedule_max_selected_prefix != plan.schedule_max_selected_prefix ||
      workspace.schedule_prefix_group_count != plan.schedule_prefix_group_count ||
      workspace.schedule_range_bit_length != plan.schedule_range_bit_length ||
      workspace.schedule_adaptive_prefix_active != plan.schedule_adaptive_prefix_active ||
      workspace.schedule_adaptive_skip_active != plan.schedule_adaptive_skip_active ||
      workspace.schedule_flags != plan.schedule_flags ||
      workspace.schedule_fingerprint != plan_workspace_fingerprint(plan) ||
      workspace.backend_workspace_required_bytes != plan.backend_workspace_required_bytes ||
      workspace.backend_selected_kernel != plan.backend_selected_kernel ||
      workspace.backend_library != plan.backend_library ||
      workspace.backend_library_version != plan.backend_library_version ||
      workspace.backend_capability_status != plan.backend_capability_status ||
      workspace.backend_epilogue_mode != plan.backend_epilogue_mode ||
      workspace.backend_workspace_mode != plan.backend_workspace_mode ||
      workspace.backend_isa_evidence != plan.backend_isa_evidence ||
      workspace.backend_autotune_key != plan.backend_autotune_key ||
      workspace.backend_performance_validated != plan.backend_performance_validated) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) && workspace.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    const bool scheduled = !plan.tile_schedule.empty();
    if (scheduled && plan.tile_schedule.size() >
                         std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_schedule_bytes =
        scheduled ? plan.tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry) : 0;
    if (scheduled) {
      if (!workspace.hip_tile_schedule ||
          workspace.hip_tile_schedule_bytes != expected_schedule_bytes ||
          workspace.hip_tile_schedule_count != static_cast<uint64_t>(plan.tile_schedule.size())) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
               workspace.hip_tile_schedule_count != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
             workspace.hip_tile_schedule_count != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    std::size_t expected_scratch_bytes = 0;
    std::size_t expected_workspace_bytes = 0;
    if (!hipblaslt_scratch_bytes_for_plan(plan, expected_scratch_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!hipblaslt_workspace_bytes_for_plan(plan, expected_workspace_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!workspace.hipblaslt_int32_scratch ||
        workspace.hipblaslt_int32_scratch_bytes != expected_scratch_bytes ||
        !workspace.hipblaslt_workspace ||
        workspace.hipblaslt_workspace_bytes != expected_workspace_bytes) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hipblaslt_int32_scratch || workspace.hipblaslt_int32_scratch_bytes != 0 ||
             workspace.hipblaslt_workspace || workspace.hipblaslt_workspace_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rns8::detail::accelerator_backend_kind(plan.backend)) {
    if (!rns8::detail::accelerator_backend_compiled(plan.backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_workspace_bytes =
        plan.backend_workspace_required_bytes > std::numeric_limits<std::size_t>::max()
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(plan.backend_workspace_required_bytes);
    if (expected_workspace_bytes != 0) {
      if (!workspace.accelerator_workspace ||
          workspace.accelerator_workspace_bytes != expected_workspace_bytes) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0 ||
             workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

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
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_max_elements = 0;
  matrix.finite_modulus = 0;
  matrix.device_residues_current = false;
  matrix.device_byte_limbs_current = false;
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
    matrix.host_residues_current = false;
    matrix.device_residues_current = false;
    matrix.host_byte_limbs_current = false;
    matrix.device_byte_limbs_current = false;
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
  matrix.host_residues_current = false;
  matrix.device_residues_current = false;
  matrix.host_byte_limbs_current = false;
  matrix.device_byte_limbs_current = false;
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

rns8_status rns8_create_workspace(rns8_context* ctx, const rns8_plan* plan, rns8_workspace** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!context_accepts_backend(*ctx, plan->backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace = new (std::nothrow) rns8_workspace();
    if (!workspace) {
      return RNS8_INTERNAL_ERROR;
    }
    workspace->semantics = plan->desc.semantics;
    workspace->bound_kind = plan->desc.bound_kind;
    workspace->m = plan->desc.m;
    workspace->n = plan->desc.n;
    workspace->k = plan->desc.k;
    workspace->bound = plan->desc.bound;
    workspace->finite_modulus = plan->desc.finite_modulus;
    workspace->tile_m = plan->desc.tile_m;
    workspace->tile_n = plan->desc.tile_n;
    workspace->prefix = plan->prefix;
    workspace->schedule_tile_rows = plan->schedule_tile_rows;
    workspace->schedule_tile_cols = plan->schedule_tile_cols;
    workspace->schedule_tile_count = plan->schedule_tile_count;
    workspace->schedule_min_required_prefix = plan->schedule_min_required_prefix;
    workspace->schedule_max_required_prefix = plan->schedule_max_required_prefix;
    workspace->schedule_min_selected_prefix = plan->schedule_min_selected_prefix;
    workspace->schedule_max_selected_prefix = plan->schedule_max_selected_prefix;
    workspace->schedule_prefix_group_count = plan->schedule_prefix_group_count;
    workspace->schedule_range_bit_length = plan->schedule_range_bit_length;
    workspace->schedule_adaptive_prefix_active = plan->schedule_adaptive_prefix_active;
    workspace->schedule_adaptive_skip_active = plan->schedule_adaptive_skip_active;
    workspace->schedule_flags = plan->schedule_flags;
    workspace->schedule_fingerprint = plan_workspace_fingerprint(*plan);
    workspace->backend_workspace_required_bytes = plan->backend_workspace_required_bytes;
    workspace->backend_selected_kernel = plan->backend_selected_kernel;
    workspace->backend_library = plan->backend_library;
    workspace->backend_library_version = plan->backend_library_version;
    workspace->backend_capability_status = plan->backend_capability_status;
    workspace->backend_epilogue_mode = plan->backend_epilogue_mode;
    workspace->backend_workspace_mode = plan->backend_workspace_mode;
    workspace->backend_isa_evidence = plan->backend_isa_evidence;
    workspace->backend_autotune_key = plan->backend_autotune_key;
    workspace->backend_performance_validated = plan->backend_performance_validated;
    workspace->backend = plan->backend;
    workspace->hip_device_id = hip_resident_rns_backend(plan->backend) ? ctx->device_id : -1;
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT && !plan->tile_schedule.empty()) {
      if (plan->tile_schedule.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t schedule_bytes = plan->tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry);
      rns8_status status =
          rns8::detail::hip_direct_allocate(ctx->device_id, schedule_bytes, &workspace->hip_tile_schedule);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hip_tile_schedule_bytes = schedule_bytes;
      workspace->hip_tile_schedule_count = static_cast<uint64_t>(plan->tile_schedule.size());
      status = rns8::detail::hip_direct_copy_host_to_device(
          ctx->device_id, workspace->hip_tile_schedule, plan->tile_schedule.data(), schedule_bytes);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule);
        delete workspace;
        return status;
      }
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
      if (!ctx->hipblaslt_handle) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      std::size_t scratch_bytes = 0;
      std::size_t workspace_bytes = 0;
      if (!hipblaslt_scratch_bytes_for_plan(*plan, scratch_bytes) ||
          !hipblaslt_workspace_bytes_for_plan(*plan, workspace_bytes)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      rns8_status status =
          rns8::detail::hip_direct_allocate(ctx->device_id, scratch_bytes, &workspace->hipblaslt_int32_scratch);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hipblaslt_int32_scratch_bytes = scratch_bytes;
      status = rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->hipblaslt_workspace);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hipblaslt_int32_scratch);
        delete workspace;
        return status;
      }
      workspace->hipblaslt_workspace_bytes = workspace_bytes;
    }
    if (rns8::detail::accelerator_backend_kind(plan->backend)) {
      if (!rns8::detail::accelerator_backend_compiled(plan->backend)) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      if (plan->backend_workspace_required_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t workspace_bytes = static_cast<std::size_t>(plan->backend_workspace_required_bytes);
      if (workspace_bytes != 0) {
        rns8_status status =
            rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->accelerator_workspace);
        if (status != RNS8_SUCCESS) {
          delete workspace;
          return status;
        }
        workspace->accelerator_workspace_bytes = workspace_bytes;
      }
    }
    *out = workspace;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_workspace(rns8_workspace* workspace) {
  if (workspace) {
    rns8_status status = RNS8_SUCCESS;
    if (workspace->hip_tile_schedule) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hip_tile_schedule);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hip_tile_schedule = nullptr;
      workspace->hip_tile_schedule_bytes = 0;
      workspace->hip_tile_schedule_count = 0;
    }
    if (workspace->hipblaslt_int32_scratch) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hipblaslt_int32_scratch);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_int32_scratch = nullptr;
      workspace->hipblaslt_int32_scratch_bytes = 0;
    }
    if (workspace->hipblaslt_workspace) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hipblaslt_workspace);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_workspace = nullptr;
      workspace->hipblaslt_workspace_bytes = 0;
    }
    if (workspace->accelerator_auxiliary) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->accelerator_auxiliary);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->accelerator_auxiliary = nullptr;
      workspace->accelerator_auxiliary_bytes = 0;
    }
    if (workspace->accelerator_workspace) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->accelerator_workspace);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->accelerator_workspace = nullptr;
      workspace->accelerator_workspace_bytes = 0;
    }
    delete workspace;
    return status;
  }
  delete workspace;
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
    if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      std::size_t bytes = 0;
      if (!wrap_byte_limb_bytes(desc->rows, desc->cols, bytes)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->byte_limbs.assign(bytes, 0);
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = true;
      matrix->device_byte_limbs_current = false;
    } else if (uses_finite_storage(matrix->desc.semantics)) {
      std::size_t cells = 0;
      if (!matrix_cell_count(desc->rows, desc->cols, cells)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(cells, 0);
      matrix->finite_modulus = 0;
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
    } else {
      std::size_t elements = 0;
      if (!rns_residue_count(desc->rows, desc->cols, prefix, elements)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(elements, 0);
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = allocate_hip_matrix_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        delete matrix;
        return status;
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
