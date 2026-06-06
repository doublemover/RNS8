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
      matrix.desc.logical_ld < matrix.desc.cols || matrix.desc.flags != 0 || matrix.prefix < prefix ||
      matrix.desc.max_prefix < prefix) {
    return false;
  }
  if (prefix == 0 && (matrix.prefix != 0 || matrix.desc.max_prefix != 0)) {
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

bool build_active_prefix_tile_schedule(
    const rns8_plan& plan,
    std::vector<rns8_plan_tile_schedule_entry>& active_entries,
    uint64_t* active_offsets,
    uint64_t* active_counts,
    uint32_t& active_prefix_count) {
  if (plan.tile_schedule.empty() || plan.schedule_max_selected_prefix == 0 ||
      plan.schedule_max_selected_prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  active_entries.clear();
  for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    active_offsets[p] = 0;
    active_counts[p] = 0;
  }
  active_prefix_count = plan.schedule_max_selected_prefix;
  for (uint32_t p = 0; p < active_prefix_count; ++p) {
    active_offsets[p] = static_cast<uint64_t>(active_entries.size());
    for (const auto& entry : plan.tile_schedule) {
      if (entry.selected_prefix > p && (entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
        active_entries.push_back(entry);
      }
    }
    active_counts[p] = static_cast<uint64_t>(active_entries.size()) - active_offsets[p];
  }
  return true;
}

bool active_prefix_tile_schedule_matches_workspace(const rns8_plan& plan, const rns8_workspace& workspace) {
  if (plan.tile_schedule.empty()) {
    if (workspace.hip_tile_schedule_active_entries != nullptr ||
        workspace.hip_tile_schedule_active_entries_bytes != 0 ||
        workspace.hip_tile_schedule_active_entries_count != 0 ||
        workspace.hip_tile_schedule_active_prefix_count != 0) {
      return false;
    }
    for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
      if (workspace.hip_tile_schedule_active_offsets[p] != 0 ||
          workspace.hip_tile_schedule_active_counts[p] != 0) {
        return false;
      }
    }
    return true;
  }
  if (plan.schedule_max_selected_prefix == 0 ||
      plan.schedule_max_selected_prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      workspace.hip_tile_schedule_active_prefix_count != plan.schedule_max_selected_prefix) {
    return false;
  }
  uint64_t running_offset = 0;
  for (uint32_t p = 0; p < workspace.hip_tile_schedule_active_prefix_count; ++p) {
    uint64_t expected_count = 0;
    for (const auto& entry : plan.tile_schedule) {
      if (entry.selected_prefix > p && (entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
        ++expected_count;
      }
    }
    if (workspace.hip_tile_schedule_active_offsets[p] != running_offset ||
        workspace.hip_tile_schedule_active_counts[p] != expected_count) {
      return false;
    }
    running_offset += expected_count;
  }
  for (uint32_t p = workspace.hip_tile_schedule_active_prefix_count; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    if (workspace.hip_tile_schedule_active_offsets[p] != 0 ||
        workspace.hip_tile_schedule_active_counts[p] != 0) {
      return false;
    }
  }
  if (running_offset > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) /
                           sizeof(rns8_plan_tile_schedule_entry)) {
    return false;
  }
  const std::size_t expected_bytes =
      static_cast<std::size_t>(running_offset) * sizeof(rns8_plan_tile_schedule_entry);
  if (running_offset == 0) {
    return workspace.hip_tile_schedule_active_entries == nullptr &&
           workspace.hip_tile_schedule_active_entries_count == 0 &&
           workspace.hip_tile_schedule_active_entries_bytes == 0;
  }
  return workspace.hip_tile_schedule_active_entries != nullptr &&
         workspace.hip_tile_schedule_active_entries_count == running_offset &&
         workspace.hip_tile_schedule_active_entries_bytes == expected_bytes;
}

bool active_prefix_tile_schedule_workspace_empty(const rns8_workspace& workspace) {
  if (workspace.hip_tile_schedule_active_entries != nullptr ||
      workspace.hip_tile_schedule_active_entries_bytes != 0 ||
      workspace.hip_tile_schedule_active_entries_count != 0 ||
      workspace.hip_tile_schedule_active_prefix_count != 0) {
    return false;
  }
  for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    if (workspace.hip_tile_schedule_active_offsets[p] != 0 ||
        workspace.hip_tile_schedule_active_counts[p] != 0) {
      return false;
    }
  }
  return true;
}

bool native_matrix_bytes(int64_t rows, int64_t cols, std::size_t element_size, std::size_t& bytes) {
  std::size_t cells = 0;
  if (element_size == 0 || !matrix_cell_count(rows, cols, cells)) {
    return false;
  }
  if (cells > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = cells * element_size;
  return true;
}

bool bounded_native_storage_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    int64_t rows,
    int64_t cols) {
  if (semantics != RNS8_BOUNDED_I64 && semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  std::size_t expected_bytes = 0;
  if (!native_matrix_bytes(rows, cols, sizeof(uint64_t), expected_bytes)) {
    return false;
  }
  if (semantics == RNS8_BOUNDED_I64) {
    return matrix.hip_native_i64 != nullptr && matrix.hip_native_i64_bytes == expected_bytes &&
           matrix.hip_native_u64 == nullptr && matrix.hip_native_u64_bytes == 0 && matrix.native_u64.empty();
  }
  return matrix.hip_native_u64 != nullptr && matrix.hip_native_u64_bytes == expected_bytes &&
         matrix.hip_native_i64 == nullptr && matrix.hip_native_i64_bytes == 0 && matrix.native_i64.empty();
}

bool bounded_native_state_current(const rns8_matrix& matrix) {
  return matrix.device_native_current && !matrix.host_native_current;
}

bool matrix_has_native_storage(const rns8_matrix& matrix) {
  return !matrix.native_i64.empty() || !matrix.native_u64.empty() || matrix.hip_native_i64 ||
         matrix.hip_native_u64 || matrix.hip_native_i64_bytes != 0 || matrix.hip_native_u64_bytes != 0;
}

void clear_native_current(rns8_matrix& matrix) {
  matrix.host_native_current = false;
  matrix.device_native_current = false;
}

void clear_device_native_current(rns8_matrix& matrix) {
  matrix.device_native_current = false;
}

void clear_residue_current(rns8_matrix& matrix) {
  matrix.host_residues_current = false;
  matrix.device_residues_current = false;
}

void clear_device_residues_current(rns8_matrix& matrix) {
  matrix.device_residues_current = false;
}

void clear_byte_limb_current(rns8_matrix& matrix) {
  matrix.host_byte_limbs_current = false;
  matrix.device_byte_limbs_current = false;
}

void clear_device_byte_limb_current(rns8_matrix& matrix) {
  matrix.device_byte_limbs_current = false;
}

void invalidate_output_currentness(rns8_matrix& matrix) {
  clear_residue_current(matrix);
  clear_byte_limb_current(matrix);
  clear_native_current(matrix);
}

void mark_host_residues_current(rns8_matrix& matrix) {
  matrix.host_residues_current = true;
  matrix.device_residues_current = false;
  clear_byte_limb_current(matrix);
}

void mark_device_residues_current(rns8_matrix& matrix) {
  matrix.host_residues_current = false;
  matrix.device_residues_current = true;
  clear_byte_limb_current(matrix);
}

void mark_host_byte_limbs_current(rns8_matrix& matrix) {
  clear_residue_current(matrix);
  matrix.host_byte_limbs_current = true;
  matrix.device_byte_limbs_current = false;
  clear_native_current(matrix);
}

void mark_device_byte_limbs_current(rns8_matrix& matrix) {
  clear_residue_current(matrix);
  matrix.host_byte_limbs_current = false;
  matrix.device_byte_limbs_current = true;
  clear_native_current(matrix);
}

void mark_device_native_current(rns8_matrix& matrix) {
  matrix.host_native_current = false;
  matrix.device_native_current = true;
}

void mark_output_host_residues_current(rns8_matrix& matrix) {
  mark_host_residues_current(matrix);
  clear_native_current(matrix);
}

void mark_output_device_residues_current(rns8_matrix& matrix) {
  mark_device_residues_current(matrix);
  clear_native_current(matrix);
}

void mark_output_host_byte_limbs_current(rns8_matrix& matrix) {
  mark_host_byte_limbs_current(matrix);
}

void mark_output_device_byte_limbs_current(rns8_matrix& matrix) {
  mark_device_byte_limbs_current(matrix);
}

void mark_output_device_native_current(rns8_matrix& matrix) {
  clear_residue_current(matrix);
  clear_byte_limb_current(matrix);
  matrix.host_native_current = false;
  matrix.device_native_current = true;
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
  if (matrix.prefix < prefix || !rns_residue_count(rows, cols, matrix.prefix, expected_residues)) {
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

