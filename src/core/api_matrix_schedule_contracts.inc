static uint64_t count_set_mask_values(const std::vector<uint8_t>& values) {
  uint64_t count = 0;
  for (const uint8_t value : values) {
    count += value != 0 ? 1u : 0u;
  }
  return count;
}

static bool mask_range_has_set_value(const std::vector<uint8_t>& values, int64_t begin, int64_t end) {
  if (begin < 0 || end < begin || static_cast<uint64_t>(end) > static_cast<uint64_t>(values.size())) {
    return false;
  }
  for (int64_t index = begin; index < end; ++index) {
    if (values[static_cast<std::size_t>(index)] != 0) {
      return true;
    }
  }
  return false;
}

bool zero_row_col_contract_matches(const rns8_plan& plan) {
  const bool requested = (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0;
  if (!requested) {
    return plan.desc.zero_a_rows == nullptr && plan.desc.zero_a_rows_count == 0 &&
           plan.desc.zero_b_cols == nullptr && plan.desc.zero_b_cols_count == 0 &&
           plan.zero_a_rows.empty() && plan.zero_b_cols.empty() &&
           plan.zero_a_row_count == 0 && plan.zero_b_col_count == 0 &&
           plan.zero_row_col_product_count == 0;
  }
  if (!is_per_tile_bound_kind(plan.desc.bound_kind) || plan.desc.zero_a_rows != nullptr ||
      plan.desc.zero_b_cols != nullptr ||
      plan.desc.zero_a_rows_count != static_cast<uint64_t>(plan.zero_a_rows.size()) ||
      plan.desc.zero_b_cols_count != static_cast<uint64_t>(plan.zero_b_cols.size()) ||
      plan.zero_a_rows.size() != static_cast<std::size_t>(plan.desc.m) ||
      plan.zero_b_cols.size() != static_cast<std::size_t>(plan.desc.n)) {
    return false;
  }
  for (const uint8_t value : plan.zero_a_rows) {
    if (value > 1u) {
      return false;
    }
  }
  for (const uint8_t value : plan.zero_b_cols) {
    if (value > 1u) {
      return false;
    }
  }
  const uint64_t zero_a_rows = count_set_mask_values(plan.zero_a_rows);
  const uint64_t zero_b_cols = count_set_mask_values(plan.zero_b_cols);
  const uint64_t nonzero_a_rows = static_cast<uint64_t>(plan.desc.m) - zero_a_rows;
  if (zero_a_rows != 0 &&
      static_cast<uint64_t>(plan.desc.n) > std::numeric_limits<uint64_t>::max() / zero_a_rows) {
    return false;
  }
  const uint64_t row_products = zero_a_rows * static_cast<uint64_t>(plan.desc.n);
  if (zero_b_cols != 0 &&
      nonzero_a_rows > std::numeric_limits<uint64_t>::max() / zero_b_cols) {
    return false;
  }
  return plan.zero_a_row_count == zero_a_rows && plan.zero_b_col_count == zero_b_cols &&
         plan.zero_row_col_product_count == row_products + nonzero_a_rows * zero_b_cols;
}

bool plan_schedule_contract_matches(const rns8_plan& plan) {
  constexpr uint32_t allowed_flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS |
                                     RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  constexpr uint32_t known_tile_schedule_flags =
      RNS8_TILE_SCHEDULE_ZERO_OUTPUT | RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;
  if (!backend_supports_semantics(plan.backend, plan.desc.semantics) ||
      !configured_tile_size_valid(plan.desc.tile_m) || !configured_tile_size_valid(plan.desc.tile_n) ||
      (plan.desc.flags & ~allowed_flags) != 0 ||
      !zero_row_col_contract_matches(plan)) {
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
      plan.backend_target_id.empty() ||
      (!hip_device_backend(plan.backend) && plan.backend_target_id != "cpu") ||
      plan.backend_workspace_required_bytes != workspace_required_bytes_for_plan(plan) ||
      plan.backend_accumulator_k_block_size != accumulator_k_block_size_for_plan(plan) ||
      plan.backend_accumulator_k_block_cap != accumulator_k_block_cap_for_plan(plan) ||
      plan.backend_accumulator_type != accumulator_type_for_plan(plan) ||
      plan.backend_accumulator_signedness != accumulator_signedness_for_plan(plan) ||
      plan.backend_accumulator_modulus_policy != accumulator_modulus_policy_for_plan(plan) ||
      plan.backend_accumulator_safety_status != accumulator_safety_status_for_plan(plan) ||
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
           plan.desc.max_prefix == 0 && plan.modulus_product == 0 && plan.tile_bounds.empty() && plan.tile_schedule.empty() &&
           plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.desc.flags == 0 && plan.schedule_flags == 0 &&
           plan.desc.lhs_bound == 0 && plan.desc.rhs_bound == 0 &&
           zero_row_col_contract_matches(plan);
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return plan.desc.bound_kind == RNS8_BOUND_NONE && plan.desc.bound == 0 && plan.prefix == 0 &&
           plan.desc.max_prefix == 0 && plan.modulus_product == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.desc.flags == 0 && plan.schedule_flags == 0 &&
           plan.desc.lhs_bound == 0 && plan.desc.rhs_bound == 0 &&
           zero_row_col_contract_matches(plan);
  }
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0 || plan.prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (plan.desc.max_prefix == 0 || plan.desc.max_prefix > RNS8_MAX_SUPPORTED_PREFIX || plan.prefix > plan.desc.max_prefix) {
    return false;
  }
  if (plan.modulus_product != rns8::detail::modulus_product(plan.prefix)) {
    return false;
  }
  if ((plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) &&
      (plan.desc.bound_kind != RNS8_BOUND_NONE || plan.desc.bound != 0 || plan.desc.tile_bounds != nullptr ||
       plan.desc.tile_bounds_count != 0 || plan.desc.lhs_bound != 0 || plan.desc.rhs_bound != 0 ||
       (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0)) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_I64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS &&
      plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_ABS &&
      plan.desc.bound_kind != RNS8_BOUND_INPUT_RANGE_AND_K) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_U64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED &&
      plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_UNSIGNED &&
      plan.desc.bound_kind != RNS8_BOUND_INPUT_RANGE_AND_K) {
    return false;
  }
  const bool per_tile = is_per_tile_bound_kind(plan.desc.bound_kind);
  const bool input_range = is_input_range_bound_kind(plan.desc.bound_kind);
  if (!input_range && (plan.desc.lhs_bound != 0 || plan.desc.rhs_bound != 0)) {
    return false;
  }
  if (input_range) {
    uint64_t derived_bound = 0;
    if (!rns8::detail::input_range_output_bound(plan.desc, derived_bound) ||
        derived_bound != plan.desc.bound) {
      return false;
    }
  }
  if (!per_tile) {
    const bool fixed_prefix_requested = (plan.desc.flags & RNS8_PLAN_FORCE_FIXED_PREFIX) != 0;
    return plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.schedule_min_required_prefix > 0 &&
           plan.schedule_max_required_prefix == plan.schedule_min_required_prefix &&
           plan.schedule_min_selected_prefix == plan.prefix && plan.schedule_max_selected_prefix == plan.prefix &&
           plan.schedule_prefix_group_count == 1 && plan.schedule_adaptive_prefix_active == 0 &&
           plan.schedule_max_required_prefix <= plan.prefix &&
           plan.schedule_adaptive_skip_active == (plan.prefix < plan.desc.max_prefix ? 1u : 0u) &&
           (!fixed_prefix_requested || plan.prefix == plan.desc.max_prefix) &&
           (plan.desc.flags & (RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS |
                               RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS)) == 0 &&
           plan.schedule_flags == 0;
  }
  if (plan.desc.bound != 0 || plan.desc.tile_bounds_count != static_cast<uint64_t>(plan.tile_bounds.size()) ||
      plan.desc.tile_bounds_count != plan.schedule_tile_count || plan.desc.tile_bounds != nullptr ||
      plan.schedule_prefix_group_count == 0 || plan.schedule_min_required_prefix == 0 ||
      plan.schedule_min_selected_prefix == 0 || plan.schedule_max_required_prefix > plan.prefix ||
      plan.schedule_max_selected_prefix > plan.prefix ||
      plan.schedule_min_required_prefix > plan.schedule_max_required_prefix ||
      plan.schedule_min_selected_prefix > plan.schedule_max_selected_prefix ||
      (plan.schedule_flags & ~known_tile_schedule_flags) != 0) {
    return false;
  }
  if (plan.tile_schedule.empty()) {
    return plan.schedule_prefix_group_count == 1 && plan.schedule_min_required_prefix == plan.prefix &&
           plan.schedule_max_required_prefix == plan.prefix && plan.schedule_min_selected_prefix == plan.prefix &&
           plan.schedule_max_selected_prefix == plan.prefix && plan.schedule_adaptive_prefix_active == 0 &&
           plan.schedule_adaptive_skip_active == 0 && plan.schedule_flags == 0;
  }
  if (plan.tile_schedule.size() != plan.tile_bounds.size()) {
    return false;
  }
  const bool allow_proven_zero_tile_skips =
      (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS) != 0;
  const bool allow_proven_zero_row_col_skips =
      (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0;
  uint32_t aggregate_flags = 0;
  for (uint64_t index = 0; index < plan.schedule_tile_count; ++index) {
    const auto& entry = plan.tile_schedule[static_cast<std::size_t>(index)];
    if (!rns8::detail::valid_abi(entry.struct_size, entry.abi_version, sizeof(entry)) ||
        (entry.flags & ~known_tile_schedule_flags) != 0 ||
        entry.tile_row != index / plan.schedule_tile_cols || entry.tile_col != index % plan.schedule_tile_cols ||
        entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 || entry.col_extent <= 0 ||
        entry.row_offset >= plan.desc.m || entry.col_offset >= plan.desc.n ||
        entry.row_extent > plan.desc.m - entry.row_offset || entry.col_extent > plan.desc.n - entry.col_offset ||
        entry.required_prefix == 0 || entry.selected_prefix == 0 || entry.required_prefix > entry.selected_prefix ||
        entry.selected_prefix > plan.prefix || entry.group_index >= plan.schedule_prefix_group_count) {
      return false;
    }
    const uint64_t tile_bound = plan.tile_bounds[static_cast<std::size_t>(index)];
    uint32_t expected_flags =
        allow_proven_zero_tile_skips && tile_bound == 0 ? RNS8_TILE_SCHEDULE_ZERO_OUTPUT : 0u;
    if (allow_proven_zero_row_col_skips) {
      const int64_t row_begin = entry.row_offset;
      const int64_t col_begin = entry.col_offset;
      const int64_t row_end = entry.row_offset + entry.row_extent;
      const int64_t col_end = entry.col_offset + entry.col_extent;
      if (mask_range_has_set_value(plan.zero_a_rows, row_begin, row_end) ||
          mask_range_has_set_value(plan.zero_b_cols, col_begin, col_end)) {
        expected_flags |= RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;
      }
    }
    if (entry.flags != expected_flags) {
      return false;
    }
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0 &&
        (tile_bound != 0 || entry.range_bit_length != 0)) {
      return false;
    }
    aggregate_flags |= entry.flags;
  }
  return aggregate_flags == plan.schedule_flags;
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
