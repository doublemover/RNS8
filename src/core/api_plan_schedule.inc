
// === Rank 133: Mixed-precision modulus family classification ===
// Classifies modulus planes into families for optimal backend routing:
//   Hot moduli (251,255,256): specialized CK/rocWMMA kernels
//   Prime moduli: generic CK/rocWMMA
//   Composite moduli: Direct HIP (fastest generic path)
// Research placeholder. Full scheduler implementation deferred.

namespace {
bool is_hot_modulus(uint32_t modulus) {
  return modulus == 251 || modulus == 255 || modulus == 256;
}
bool is_prime_modulus(uint32_t modulus) {
  // Simplified: known primes from the default ladder
  return modulus == 251 || modulus == 239 || modulus == 233 || modulus == 229 ||
         modulus == 227 || modulus == 223 || modulus == 211 || modulus == 199 ||
         modulus == 197 || modulus == 193 || modulus == 191 || modulus == 181 ||
         modulus == 179 || modulus == 173 || modulus == 167 || modulus == 163 ||
         modulus == 157 || modulus == 151 || modulus == 149 || modulus == 139 ||
         modulus == 137 || modulus == 131 || modulus == 127;
}
}  // namespace

bool is_per_tile_bound_kind(rns8_bound_kind bound_kind) {
  return bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS || bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
}

bool is_input_range_bound_kind(rns8_bound_kind bound_kind) {
  return bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K;
}

std::string ck_finite_selected_kernel(uint32_t finite_modulus) {
  switch (finite_modulus) {
    case 251:
      return "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2";
    case 255:
      return "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2";
    case 256:
      return "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2";
    default:
      return "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2";
  }
}

std::string rocwmma_finite_selected_kernel(uint32_t finite_modulus) {
  switch (finite_modulus) {
    case 251:
      return "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2";
    case 255:
      return "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2";
    case 256:
      return "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2";
    default:
      return "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1";
  }
}

boost::multiprecision::cpp_int bounded_range_from_bound(rns8_semantics semantics, uint64_t bound) {
  using boost::multiprecision::cpp_int;
  return semantics == RNS8_BOUNDED_I64 ? cpp_int(2) * cpp_int(bound) : cpp_int(bound);
}

rns8_bound_kind storage_bound_kind_for_plan(const rns8_plan& plan) {
  if (plan.desc.bound_kind != RNS8_BOUND_INPUT_RANGE_AND_K) {
    return plan.desc.bound_kind;
  }
  return plan.desc.semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
}

uint32_t rns_storage_prefix_for_plan(const rns8_plan& plan) {
  if (is_per_tile_bound_kind(plan.desc.bound_kind) && plan.schedule_max_selected_prefix > 0) {
    return plan.schedule_max_selected_prefix;
  }
  return plan.prefix;
}

rns8_plan_tile_schedule_entry make_tile_schedule_entry(
    const rns8_plan& plan,
    uint64_t index,
    uint32_t required_prefix,
    uint32_t selected_prefix,
    uint32_t group_index,
    uint32_t range_bit_length,
    uint32_t flags) {
  const uint64_t tile_row = index / plan.schedule_tile_cols;
  const uint64_t tile_col = index % plan.schedule_tile_cols;
  const int64_t row_offset = static_cast<int64_t>(tile_row * static_cast<uint64_t>(plan.desc.tile_m));
  const int64_t col_offset = static_cast<int64_t>(tile_col * static_cast<uint64_t>(plan.desc.tile_n));
  rns8_plan_tile_schedule_entry entry{};
  entry.struct_size = sizeof(entry);
  entry.abi_version = RNS8_ABI_VERSION;
  entry.flags = flags;
  entry.tile_row = tile_row;
  entry.tile_col = tile_col;
  entry.row_offset = row_offset;
  entry.col_offset = col_offset;
  entry.row_extent = std::min<int64_t>(plan.desc.tile_m, plan.desc.m - row_offset);
  entry.col_extent = std::min<int64_t>(plan.desc.tile_n, plan.desc.n - col_offset);
  entry.required_prefix = required_prefix;
  entry.selected_prefix = selected_prefix;
  entry.group_index = group_index;
  entry.range_bit_length = range_bit_length;
  return entry;
}

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

bool plan_has_zero_row_col_proofs(const rns8_plan& plan) {
  return (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0 &&
         plan.zero_a_rows.size() == static_cast<std::size_t>(plan.desc.m) &&
         plan.zero_b_cols.size() == static_cast<std::size_t>(plan.desc.n);
}

rns8_status configure_plan_schedule(rns8_plan& plan) {
  const uint32_t tile_m = plan.desc.tile_m == 0 ? 128u : plan.desc.tile_m;
  const uint32_t tile_n = plan.desc.tile_n == 0 ? 128u : plan.desc.tile_n;
  const uint32_t requested_max_prefix = plan.prefix;
  plan.schedule_tile_rows = ceil_div_i64_u32(plan.desc.m, tile_m);
  plan.schedule_tile_cols = ceil_div_i64_u32(plan.desc.n, tile_n);
  if (plan.schedule_tile_cols != 0 &&
      plan.schedule_tile_rows > std::numeric_limits<uint64_t>::max() / plan.schedule_tile_cols) {
    return RNS8_RANGE_ERROR;
  }
  plan.schedule_tile_count = plan.schedule_tile_rows * plan.schedule_tile_cols;
  plan.tile_bounds.clear();
  plan.tile_schedule.clear();
  plan.zero_a_rows.clear();
  plan.zero_b_cols.clear();
  plan.zero_a_row_count = 0;
  plan.zero_b_col_count = 0;
  plan.zero_row_col_product_count = 0;
  plan.schedule_flags = 0;
  if ((plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS) != 0) {
    if (!plan.desc.zero_a_rows || !plan.desc.zero_b_cols || plan.desc.m <= 0 || plan.desc.n <= 0 ||
        plan.desc.zero_a_rows_count != static_cast<uint64_t>(plan.desc.m) ||
        plan.desc.zero_b_cols_count != static_cast<uint64_t>(plan.desc.n)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (static_cast<uint64_t>(plan.desc.m) > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        static_cast<uint64_t>(plan.desc.n) > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return RNS8_RANGE_ERROR;
    }
    plan.zero_a_rows.assign(plan.desc.zero_a_rows, plan.desc.zero_a_rows + plan.desc.zero_a_rows_count);
    plan.zero_b_cols.assign(plan.desc.zero_b_cols, plan.desc.zero_b_cols + plan.desc.zero_b_cols_count);
    plan.desc.zero_a_rows = nullptr;
    plan.desc.zero_b_cols = nullptr;
    plan.desc.zero_a_rows_count = static_cast<uint64_t>(plan.zero_a_rows.size());
    plan.desc.zero_b_cols_count = static_cast<uint64_t>(plan.zero_b_cols.size());
    plan.zero_a_row_count = count_set_mask_values(plan.zero_a_rows);
    plan.zero_b_col_count = count_set_mask_values(plan.zero_b_cols);
    const uint64_t nonzero_a_rows = static_cast<uint64_t>(plan.desc.m) - plan.zero_a_row_count;
    if (plan.zero_a_row_count != 0 &&
        static_cast<uint64_t>(plan.desc.n) >
            std::numeric_limits<uint64_t>::max() / plan.zero_a_row_count) {
      return RNS8_RANGE_ERROR;
    }
    const uint64_t row_products = plan.zero_a_row_count * static_cast<uint64_t>(plan.desc.n);
    if (plan.zero_b_col_count != 0 &&
        nonzero_a_rows > std::numeric_limits<uint64_t>::max() / plan.zero_b_col_count) {
      return RNS8_RANGE_ERROR;
    }
    plan.zero_row_col_product_count = row_products + nonzero_a_rows * plan.zero_b_col_count;
  } else {
    plan.desc.zero_a_rows = nullptr;
    plan.desc.zero_b_cols = nullptr;
    plan.desc.zero_a_rows_count = 0;
    plan.desc.zero_b_cols_count = 0;
  }

  const boost::multiprecision::cpp_int required_range = schedule_required_range(plan.desc);
  plan.schedule_range_bit_length = rns8::detail::bit_length(required_range);
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 || uses_finite_storage(plan.desc.semantics)) {
    plan.schedule_min_required_prefix = 0;
    plan.schedule_max_required_prefix = 0;
    plan.schedule_min_selected_prefix = 0;
    plan.schedule_max_selected_prefix = 0;
    plan.schedule_prefix_group_count = 0;
    plan.schedule_adaptive_prefix_active = 0;
    plan.schedule_adaptive_skip_active = 0;
    return RNS8_SUCCESS;
  }

  if (is_per_tile_bound_kind(plan.desc.bound_kind)) {
    if (!plan.desc.tile_bounds || plan.desc.tile_bounds_count != plan.schedule_tile_count) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan.schedule_tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return RNS8_RANGE_ERROR;
    }
    plan.tile_bounds.assign(plan.desc.tile_bounds, plan.desc.tile_bounds + plan.desc.tile_bounds_count);
    plan.desc.tile_bounds = nullptr;
    plan.desc.tile_bounds_count = static_cast<uint64_t>(plan.tile_bounds.size());
    plan.tile_schedule.reserve(plan.tile_bounds.size());

    uint32_t min_required = std::numeric_limits<uint32_t>::max();
    uint32_t max_required = 0;
    uint32_t min_selected = std::numeric_limits<uint32_t>::max();
    uint32_t max_selected = 0;
    uint32_t max_range_bits = 0;
    std::vector<uint32_t> groups;
    groups.reserve(plan.tile_bounds.size());
    const bool force_fixed_prefix = (plan.desc.flags & RNS8_PLAN_FORCE_FIXED_PREFIX) != 0;
    const bool allow_proven_zero_tile_skips =
        (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS) != 0;

    for (uint64_t index = 0; index < plan.schedule_tile_count; ++index) {
      const uint64_t bound = plan.tile_bounds[static_cast<std::size_t>(index)];
      if (plan.desc.semantics == RNS8_BOUNDED_I64 && bound > (uint64_t{1} << 63u)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const boost::multiprecision::cpp_int range = bounded_range_from_bound(plan.desc.semantics, bound);
      const uint32_t range_bits = rns8::detail::bit_length(range);
      const uint32_t required_prefix = rns8::detail::required_prefix_for_range(range);
      if (required_prefix == 0 || required_prefix > plan.prefix) {
        return RNS8_RANGE_ERROR;
      }
      const uint32_t selected_prefix = force_fixed_prefix ? plan.prefix : required_prefix;
      min_required = std::min(min_required, required_prefix);
      max_required = std::max(max_required, required_prefix);
      min_selected = std::min(min_selected, selected_prefix);
      max_selected = std::max(max_selected, selected_prefix);
      max_range_bits = std::max(max_range_bits, range_bits);
      if (std::find(groups.begin(), groups.end(), selected_prefix) == groups.end()) {
        groups.push_back(selected_prefix);
      }
      const uint32_t flags =
          allow_proven_zero_tile_skips && bound == 0 ? RNS8_TILE_SCHEDULE_ZERO_OUTPUT : 0u;
      uint32_t entry_flags = flags;
      if (plan_has_zero_row_col_proofs(plan)) {
        const uint64_t tile_row = index / plan.schedule_tile_cols;
        const uint64_t tile_col = index % plan.schedule_tile_cols;
        const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(plan.desc.tile_m));
        const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(plan.desc.tile_n));
        const int64_t row_end = std::min<int64_t>(plan.desc.m, row_begin + static_cast<int64_t>(plan.desc.tile_m));
        const int64_t col_end = std::min<int64_t>(plan.desc.n, col_begin + static_cast<int64_t>(plan.desc.tile_n));
        if (mask_range_has_set_value(plan.zero_a_rows, row_begin, row_end) ||
            mask_range_has_set_value(plan.zero_b_cols, col_begin, col_end)) {
          entry_flags |= RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;
        }
      }
      plan.schedule_flags |= entry_flags;
      plan.tile_schedule.push_back(
          make_tile_schedule_entry(plan, index, required_prefix, selected_prefix, 0, range_bits, entry_flags));
    }

    std::sort(groups.begin(), groups.end());
    for (auto& entry : plan.tile_schedule) {
      entry.group_index = static_cast<uint32_t>(
          std::lower_bound(groups.begin(), groups.end(), entry.selected_prefix) - groups.begin());
    }
    plan.schedule_min_required_prefix = min_required == std::numeric_limits<uint32_t>::max() ? 0 : min_required;
    plan.schedule_max_required_prefix = max_required;
    plan.schedule_min_selected_prefix = min_selected == std::numeric_limits<uint32_t>::max() ? 0 : min_selected;
    plan.schedule_max_selected_prefix = max_selected;
    plan.schedule_prefix_group_count = static_cast<uint32_t>(groups.size());
    plan.schedule_range_bit_length = max_range_bits;
    plan.schedule_adaptive_prefix_active = groups.size() > 1 ? 1u : 0u;
    plan.schedule_adaptive_skip_active = max_selected < plan.prefix ? 1u : 0u;
    if (plan.schedule_prefix_group_count == 1 && plan.schedule_min_required_prefix == plan.prefix &&
        plan.schedule_max_required_prefix == plan.prefix && plan.schedule_min_selected_prefix == plan.prefix &&
        plan.schedule_max_selected_prefix == plan.prefix && plan.schedule_adaptive_prefix_active == 0 &&
        plan.schedule_adaptive_skip_active == 0 && plan.schedule_flags == 0) {
      plan.tile_schedule.clear();
    }
    return RNS8_SUCCESS;
  }

  const uint32_t required_prefix = rns8::detail::required_prefix_for_range(required_range);
  if (required_prefix == 0 || required_prefix > requested_max_prefix) {
    return RNS8_RANGE_ERROR;
  }
  const bool force_fixed_prefix = (plan.desc.flags & RNS8_PLAN_FORCE_FIXED_PREFIX) != 0;
  const uint32_t selected_prefix = force_fixed_prefix ? requested_max_prefix : required_prefix;
  plan.prefix = selected_prefix;
  plan.modulus_product = rns8::detail::modulus_product(selected_prefix);
  plan.schedule_min_required_prefix = required_prefix;
  plan.schedule_max_required_prefix = required_prefix;
  plan.schedule_min_selected_prefix = selected_prefix;
  plan.schedule_max_selected_prefix = selected_prefix;
  plan.schedule_prefix_group_count = 1;
  plan.schedule_adaptive_prefix_active = 0;
  plan.schedule_adaptive_skip_active = selected_prefix < requested_max_prefix ? 1u : 0u;
  return RNS8_SUCCESS;
}

