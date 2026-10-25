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

static uint64_t compute_plan_workspace_fingerprint(const rns8_plan& plan, std::vector<uint64_t>* key) {
  ++plan_validation_work.fingerprint_traversals;
  const auto mix = [&](uint64_t hash, uint64_t value) {
    if (key) key->push_back(value);
    return workspace_fingerprint_mix(hash, value);
  };
  const auto mix_string = [&](uint64_t hash, const std::string& value) {
    // Include length before bytes in the structural key, so concatenations
    // cannot confuse boundaries. Keep the existing fingerprint algorithm.
    if (key) key->push_back(value.size());
    for (const unsigned char c : value) {
      if (key) key->push_back(c);
      hash = workspace_fingerprint_mix(hash, c);
    }
    return workspace_fingerprint_mix(hash, value.size());
  };
  uint64_t hash = 1469598103934665603ull;
  hash = mix(hash, static_cast<uint64_t>(plan.desc.semantics));
  hash = mix(hash, plan.desc.flags);
  hash = mix(hash, static_cast<uint64_t>(plan.desc.bound_kind));
  hash = mix(hash, signed_to_fingerprint(plan.desc.m));
  hash = mix(hash, signed_to_fingerprint(plan.desc.n));
  hash = mix(hash, signed_to_fingerprint(plan.desc.k));
  hash = mix(hash, plan.desc.bound);
  hash = mix(hash, plan.desc.lhs_bound);
  hash = mix(hash, plan.desc.rhs_bound);
  hash = mix(hash, plan.desc.finite_modulus);
  hash = mix(hash, plan.desc.tile_m);
  hash = mix(hash, plan.desc.tile_n);
  hash = mix(hash, plan.prefix);
  hash = mix(hash, plan.schedule_tile_rows);
  hash = mix(hash, plan.schedule_tile_cols);
  hash = mix(hash, plan.schedule_tile_count);
  hash = mix(hash, plan.schedule_min_required_prefix);
  hash = mix(hash, plan.schedule_max_required_prefix);
  hash = mix(hash, plan.schedule_min_selected_prefix);
  hash = mix(hash, plan.schedule_max_selected_prefix);
  hash = mix(hash, plan.schedule_prefix_group_count);
  hash = mix(hash, plan.schedule_range_bit_length);
  hash = mix(hash, plan.schedule_adaptive_prefix_active);
  hash = mix(hash, plan.schedule_adaptive_skip_active);
  hash = mix(hash, plan.schedule_flags);
  hash = mix(hash, plan.zero_a_row_count);
  hash = mix(hash, plan.zero_b_col_count);
  hash = mix(hash, plan.zero_row_col_product_count);
  hash = mix(hash, plan.backend_workspace_required_bytes);
  hash = mix_string(hash, plan.backend_selected_kernel);
  hash = mix_string(hash, plan.backend_library);
  hash = mix_string(hash, plan.backend_library_version);
  hash = mix_string(hash, plan.backend_capability_status);
  hash = mix_string(hash, plan.backend_epilogue_mode);
  hash = mix_string(hash, plan.backend_workspace_mode);
  hash = mix_string(hash, plan.backend_isa_evidence);
  hash = mix_string(hash, plan.backend_target_id);
  hash = mix_string(hash, plan.backend_autotune_key);
  hash = mix_string(hash, plan.arithmetic_guard ? arithmetic::canonical(plan.arithmetic_guard->certificate) : "missing");
  hash = mix(hash, static_cast<uint64_t>(plan.tile_bounds.size()));
  for (const uint64_t bound : plan.tile_bounds) {
    hash = mix(hash, bound);
  }
  hash = mix(hash, static_cast<uint64_t>(plan.zero_a_rows.size()));
  for (const uint8_t value : plan.zero_a_rows) {
    hash = mix(hash, value);
  }
  hash = mix(hash, static_cast<uint64_t>(plan.zero_b_cols.size()));
  for (const uint8_t value : plan.zero_b_cols) {
    hash = mix(hash, value);
  }
  hash = mix(hash, static_cast<uint64_t>(plan.tile_schedule.size()));
  for (const auto& entry : plan.tile_schedule) {
    hash = mix(hash, entry.struct_size);
    hash = mix(hash, entry.abi_version);
    hash = mix(hash, entry.flags);
    hash = mix(hash, entry.tile_row);
    hash = mix(hash, entry.tile_col);
    hash = mix(hash, signed_to_fingerprint(entry.row_offset));
    hash = mix(hash, signed_to_fingerprint(entry.col_offset));
    hash = mix(hash, signed_to_fingerprint(entry.row_extent));
    hash = mix(hash, signed_to_fingerprint(entry.col_extent));
    hash = mix(hash, entry.required_prefix);
    hash = mix(hash, entry.selected_prefix);
    hash = mix(hash, entry.group_index);
    hash = mix(hash, entry.range_bit_length);
  }
  return hash == 0 ? 1 : hash;
}

uint64_t plan_workspace_fingerprint(const rns8_plan& plan) {
  return plan.validated_contract ? plan.validated_contract->fingerprint
                                 : compute_plan_workspace_fingerprint(plan, nullptr);
}

bool seal_plan_contract(rns8_plan& plan) {
  plan.validated_contract.reset();
  if (!plan.arithmetic_guard || !plan.arithmetic_guard->checked.admitted() || !revalidate_plan_schedule_contract(plan)) return false;
  auto contract = std::make_shared<rns8_validated_plan_contract>();
  contract->arithmetic_guard = plan.arithmetic_guard;
  contract->fingerprint = compute_plan_workspace_fingerprint(plan, &contract->canonical_workspace_key);
  // These facts are separately checked by workspace validation; also preserve
  // them in the complete equivalence key rather than relying on a hash.
  contract->canonical_workspace_key.push_back(static_cast<uint64_t>(plan.backend));
  contract->canonical_workspace_key.push_back(plan.backend_performance_validated);
  auto& launch = contract->launch;
  launch.entry_count = plan.tile_schedule.size();
  bool uniform_zero = !plan.tile_schedule.empty();
  uint32_t first_prefix = uniform_zero ? plan.tile_schedule.front().selected_prefix : 0;
  ++plan_validation_work.backend_schedule_seals;
  for (const auto& entry : plan.tile_schedule) {
    ++plan_validation_work.backend_schedule_sealed_tile_visits;
    const bool zero = (entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0;
    launch.zero_output_tiles |= zero;
    launch.zero_row_col_products |= (entry.flags & RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0;
    uniform_zero &= zero && entry.selected_prefix == first_prefix;
    launch.max_selected_prefix = std::max(launch.max_selected_prefix, entry.selected_prefix);
    const uint64_t rows = static_cast<uint64_t>(entry.row_extent);
    const uint64_t cols = static_cast<uint64_t>(entry.col_extent);
    // CPU plans may be larger than the direct-HIP integer launch domain. Keep
    // the general plan valid, but never narrow an unqualified device launch.
    if (rows > INT32_MAX || cols > INT32_MAX || rows > uint64_t{INT32_MAX} / cols) {
      launch.i32_launch_qualified = false;
    } else {
      launch.max_row_blocks = std::max(launch.max_row_blocks, static_cast<uint32_t>((rows + 15) / 16));
      launch.max_col_blocks = std::max(launch.max_col_blocks, static_cast<uint32_t>((cols + 15) / 16));
      launch.max_elements = std::max(launch.max_elements, static_cast<uint32_t>(rows * cols));
    }
    if (!zero)
      for (uint32_t p = 0; p < entry.selected_prefix; ++p) ++contract->active_counts[p];
  }
  launch.uniform_zero_selected_prefix = uniform_zero ? first_prefix : 0;
  plan.validated_contract = std::move(contract);
  return true;
}

}  // namespace rns8::detail::api
