uint64_t workspace_fingerprint_mix(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= 1099511628211ull;
  return hash;
}

uint64_t workspace_fingerprint_mix_string(uint64_t hash, const std::string& value) {
  for (const unsigned char c : value) {
    hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(c));
  }
  return workspace_fingerprint_mix(hash, static_cast<uint64_t>(value.size()));
}

uint64_t signed_to_fingerprint(int64_t value) {
  return static_cast<uint64_t>(value);
}

uint64_t gemm_output_source_version_values(uint64_t a_source_version, uint64_t b_source_version) {
  uint64_t hash = 1469598103934665603ull;
  hash = workspace_fingerprint_mix(hash, a_source_version);
  hash = workspace_fingerprint_mix(hash, b_source_version);
  return hash == 0 ? 1 : hash;
}

uint64_t gemm_output_source_version(const rns8_matrix& A, const rns8_matrix& B) {
  return gemm_output_source_version_values(A.source_version, B.source_version);
}

uint64_t plan_workspace_fingerprint(const rns8_plan& plan) {
  uint64_t hash = 1469598103934665603ull;
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.desc.semantics));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.desc.bound_kind));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.m));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.n));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.k));
  hash = workspace_fingerprint_mix(hash, plan.desc.bound);
  hash = workspace_fingerprint_mix(hash, plan.desc.lhs_bound);
  hash = workspace_fingerprint_mix(hash, plan.desc.rhs_bound);
  hash = workspace_fingerprint_mix(hash, plan.desc.finite_modulus);
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_m);
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_n);
  hash = workspace_fingerprint_mix(hash, plan.prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_rows);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_cols);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_count);
  hash = workspace_fingerprint_mix(hash, plan.schedule_min_required_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_max_required_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_min_selected_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_max_selected_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_prefix_group_count);
  hash = workspace_fingerprint_mix(hash, plan.schedule_range_bit_length);
  hash = workspace_fingerprint_mix(hash, plan.schedule_adaptive_prefix_active);
  hash = workspace_fingerprint_mix(hash, plan.schedule_adaptive_skip_active);
  hash = workspace_fingerprint_mix(hash, plan.schedule_flags);
  hash = workspace_fingerprint_mix(hash, plan.zero_a_row_count);
  hash = workspace_fingerprint_mix(hash, plan.zero_b_col_count);
  hash = workspace_fingerprint_mix(hash, plan.zero_row_col_product_count);
  hash = workspace_fingerprint_mix(hash, plan.backend_workspace_required_bytes);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_selected_kernel);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library_version);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_capability_status);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_epilogue_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_workspace_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_isa_evidence);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_target_id);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_autotune_key);
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.tile_bounds.size()));
  for (const uint64_t bound : plan.tile_bounds) {
    hash = workspace_fingerprint_mix(hash, bound);
  }
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.zero_a_rows.size()));
  for (const uint8_t value : plan.zero_a_rows) {
    hash = workspace_fingerprint_mix(hash, value);
  }
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.zero_b_cols.size()));
  for (const uint8_t value : plan.zero_b_cols) {
    hash = workspace_fingerprint_mix(hash, value);
  }
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.tile_schedule.size()));
  for (const auto& entry : plan.tile_schedule) {
    hash = workspace_fingerprint_mix(hash, entry.struct_size);
    hash = workspace_fingerprint_mix(hash, entry.abi_version);
    hash = workspace_fingerprint_mix(hash, entry.flags);
    hash = workspace_fingerprint_mix(hash, entry.tile_row);
    hash = workspace_fingerprint_mix(hash, entry.tile_col);
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.row_offset));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.col_offset));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.row_extent));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.col_extent));
    hash = workspace_fingerprint_mix(hash, entry.required_prefix);
    hash = workspace_fingerprint_mix(hash, entry.selected_prefix);
    hash = workspace_fingerprint_mix(hash, entry.group_index);
    hash = workspace_fingerprint_mix(hash, entry.range_bit_length);
  }
  return hash == 0 ? 1 : hash;
}

}  // namespace rns8::detail::api
