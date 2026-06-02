#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"

namespace {

template <typename Fn>
rns8_status guard_api(Fn&& fn) {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  } catch (...) {
    return RNS8_INTERNAL_ERROR;
  }
}

rns8_backend_kind effective_backend(rns8_backend_kind requested, rns8_backend_kind default_backend) {
  return requested == RNS8_BACKEND_AUTO ? default_backend : requested;
}

bool backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics) {
  switch (backend) {
    case RNS8_BACKEND_CPU_REFERENCE:
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
             semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
    case RNS8_BACKEND_HIP_DIRECT:
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
             semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
             semantics == RNS8_WRAP_U64_MOD_2_64;
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return semantics == RNS8_WRAP_U64_MOD_2_64;
    case RNS8_BACKEND_AUTO:
    case RNS8_BACKEND_HIPBLASLT:
    case RNS8_BACKEND_CK:
    case RNS8_BACKEND_WMMA:
      return false;
  }
  return false;
}

bool uses_rns_storage(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
         semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
}

rns8_matrix_desc make_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = 128;
  desc.tile_n = 128;
  desc.max_prefix = prefix;
  return desc;
}

bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld) {
  if (rows <= 0 || cols <= 0 || ld < cols) {
    return false;
  }
  return rows <= std::numeric_limits<int64_t>::max() / ld;
}

bool valid_limb_export_access(int64_t rows, int64_t cols, int64_t ld, uint32_t limb_count) {
  if (!valid_matrix_access(rows, cols, ld) || limb_count == 0 || limb_count > 32) {
    return false;
  }
  const auto max = std::numeric_limits<int64_t>::max();
  if (ld > max / static_cast<int64_t>(limb_count)) {
    return false;
  }
  const int64_t limb_ld = ld * static_cast<int64_t>(limb_count);
  return rows <= max / limb_ld;
}

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor) {
  const auto unsigned_value = static_cast<uint64_t>(value);
  return (unsigned_value + static_cast<uint64_t>(divisor) - 1u) / static_cast<uint64_t>(divisor);
}

boost::multiprecision::cpp_int schedule_required_range(const rns8_gemm_desc& desc) {
  using boost::multiprecision::cpp_int;
  switch (desc.semantics) {
    case RNS8_BOUNDED_I64:
      return cpp_int(2) * cpp_int(desc.bound);
    case RNS8_BOUNDED_U64:
      return cpp_int(desc.bound);
    case RNS8_EXACT_WIDE_SIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 127u);
    case RNS8_EXACT_WIDE_UNSIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 128u);
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return 0;
  }
  return 0;
}

bool is_per_tile_bound_kind(rns8_bound_kind bound_kind) {
  return bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS || bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
}

boost::multiprecision::cpp_int bounded_range_from_bound(rns8_semantics semantics, uint64_t bound) {
  using boost::multiprecision::cpp_int;
  return semantics == RNS8_BOUNDED_I64 ? cpp_int(2) * cpp_int(bound) : cpp_int(bound);
}

rns8_plan_tile_schedule_entry make_tile_schedule_entry(
    const rns8_plan& plan,
    uint64_t index,
    uint32_t required_prefix,
    uint32_t selected_prefix,
    uint32_t group_index,
    uint32_t range_bit_length) {
  const uint64_t tile_row = index / plan.schedule_tile_cols;
  const uint64_t tile_col = index % plan.schedule_tile_cols;
  const int64_t row_offset = static_cast<int64_t>(tile_row * static_cast<uint64_t>(plan.desc.tile_m));
  const int64_t col_offset = static_cast<int64_t>(tile_col * static_cast<uint64_t>(plan.desc.tile_n));
  rns8_plan_tile_schedule_entry entry{};
  entry.struct_size = sizeof(entry);
  entry.abi_version = RNS8_ABI_VERSION;
  entry.flags = plan.schedule_flags;
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

rns8_status configure_plan_schedule(rns8_plan& plan) {
  const uint32_t tile_m = plan.desc.tile_m == 0 ? 128u : plan.desc.tile_m;
  const uint32_t tile_n = plan.desc.tile_n == 0 ? 128u : plan.desc.tile_n;
  plan.schedule_tile_rows = ceil_div_i64_u32(plan.desc.m, tile_m);
  plan.schedule_tile_cols = ceil_div_i64_u32(plan.desc.n, tile_n);
  if (plan.schedule_tile_cols != 0 &&
      plan.schedule_tile_rows > std::numeric_limits<uint64_t>::max() / plan.schedule_tile_cols) {
    return RNS8_RANGE_ERROR;
  }
  plan.schedule_tile_count = plan.schedule_tile_rows * plan.schedule_tile_cols;
  plan.tile_bounds.clear();
  plan.tile_schedule.clear();

  const boost::multiprecision::cpp_int required_range = schedule_required_range(plan.desc);
  plan.schedule_range_bit_length = rns8::detail::bit_length(required_range);
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
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
      const uint32_t selected_prefix = required_prefix;
      min_required = std::min(min_required, required_prefix);
      max_required = std::max(max_required, required_prefix);
      min_selected = std::min(min_selected, selected_prefix);
      max_selected = std::max(max_selected, selected_prefix);
      max_range_bits = std::max(max_range_bits, range_bits);
      if (std::find(groups.begin(), groups.end(), selected_prefix) == groups.end()) {
        groups.push_back(selected_prefix);
      }
      plan.tile_schedule.push_back(
          make_tile_schedule_entry(plan, index, required_prefix, selected_prefix, 0, range_bits));
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
    return RNS8_SUCCESS;
  }

  const uint32_t required_prefix = rns8::detail::required_prefix_for_range(required_range);
  if (required_prefix == 0 || required_prefix > plan.prefix) {
    return RNS8_RANGE_ERROR;
  }
  plan.schedule_min_required_prefix = required_prefix;
  plan.schedule_max_required_prefix = required_prefix;
  plan.schedule_min_selected_prefix = plan.prefix;
  plan.schedule_max_selected_prefix = plan.prefix;
  plan.schedule_prefix_group_count = 1;
  plan.schedule_adaptive_prefix_active = 0;
  plan.schedule_adaptive_skip_active = 0;
  return RNS8_SUCCESS;
}

std::vector<int8_t> gather_cell_residues(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = matrix.residues[rns8::detail::residue_index(matrix, p, row, col)];
  }
  return residues;
}

uint64_t tile_index_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  const auto tile_row = static_cast<uint64_t>(row) / static_cast<uint64_t>(plan.desc.tile_m);
  const auto tile_col = static_cast<uint64_t>(col) / static_cast<uint64_t>(plan.desc.tile_n);
  return tile_row * plan.schedule_tile_cols + tile_col;
}

const rns8_plan_tile_schedule_entry* tile_schedule_entry_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  if (plan.tile_schedule.empty()) {
    return nullptr;
  }
  const uint64_t index = tile_index_for_cell(plan, row, col);
  if (index >= static_cast<uint64_t>(plan.tile_schedule.size())) {
    return nullptr;
  }
  return &plan.tile_schedule[static_cast<std::size_t>(index)];
}

uint32_t selected_prefix_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  const rns8_plan_tile_schedule_entry* entry = tile_schedule_entry_for_cell(plan, row, col);
  return entry ? entry->selected_prefix : plan.prefix;
}

uint64_t bound_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  if (plan.tile_bounds.empty()) {
    return plan.desc.bound;
  }
  const uint64_t index = tile_index_for_cell(plan, row, col);
  if (index >= static_cast<uint64_t>(plan.tile_bounds.size())) {
    return plan.desc.bound;
  }
  return plan.tile_bounds[static_cast<std::size_t>(index)];
}

bool matrix_descriptor_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
  return matrix.desc.semantics == semantics && matrix.desc.bound_kind == bound_kind && matrix.desc.rows == rows &&
         matrix.desc.cols == cols && matrix.desc.logical_layout == RNS8_LAYOUT_ROW_MAJOR &&
         matrix.desc.logical_ld >= matrix.desc.cols && matrix.desc.flags == 0 && matrix.prefix == prefix &&
         matrix.desc.max_prefix == prefix;
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
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return matrix.hip_residues != nullptr && matrix.hip_residue_bytes == expected_bytes;
  }
  return matrix.hip_residues == nullptr && matrix.hip_residue_bytes == 0 && !matrix.device_residues_current;
}

bool rns_residue_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend) {
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return matrix.host_residues_current || matrix.device_residues_current;
  }
  return matrix.host_residues_current && !matrix.device_residues_current;
}

bool plan_schedule_contract_matches(const rns8_plan& plan) {
  if (!backend_supports_semantics(plan.backend, plan.desc.semantics) || plan.desc.tile_m == 0 ||
      plan.desc.tile_n == 0 || plan.desc.flags != 0 || plan.prefix != plan.desc.max_prefix) {
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
    return matrix.hip_byte_limbs != nullptr && matrix.hip_byte_limb_bytes == expected_bytes;
  }
  return matrix.hip_byte_limbs == nullptr && matrix.hip_byte_limb_bytes == 0 &&
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
  if (ctx.backend != plan.backend || workspace.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.semantics != plan.desc.semantics || workspace.bound_kind != plan.desc.bound_kind) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.m != plan.desc.m || workspace.n != plan.desc.n || workspace.k != plan.desc.k ||
      workspace.prefix != plan.prefix) {
    return RNS8_WORKSPACE_TOO_SMALL;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && workspace.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_rns_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.backend != plan.backend || B.backend != plan.backend || C.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(A, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.k, plan.prefix) ||
      !matrix_descriptor_matches(B, plan.desc.semantics, plan.desc.bound_kind, plan.desc.k, plan.desc.n, plan.prefix) ||
      !matrix_descriptor_matches(C, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.n, plan.prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, plan.prefix) ||
      !rns_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n, plan.prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, plan.prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_residue_state_current_for_backend(A, plan.backend) ||
      !rns_residue_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
      (plan.desc.semantics == RNS8_BOUNDED_I64 || plan.desc.semantics == RNS8_BOUNDED_U64) &&
      (!A.device_residues_current || !B.device_residues_current)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_wrap_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 || plan.prefix != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.backend != plan.backend || B.backend != plan.backend || C.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(A, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.k, 0) ||
      !matrix_descriptor_matches(B, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0) ||
      !matrix_descriptor_matches(C, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.hip_residues || B.hip_residues || C.hip_residues) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k) ||
      !wrap_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_byte_limb_state_current_for_backend(A, plan.backend) ||
      !wrap_byte_limb_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

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
  if (ctx.backend != plan.backend || C.backend != plan.backend || plan.desc.semantics != semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT && C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(C, semantics, bound_kind, plan.desc.m, plan.desc.n, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (uses_rns_storage(semantics) &&
      (!rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, prefix) ||
       !rns_residue_state_current_for_backend(C, plan.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
      (semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64) &&
      !C.device_residues_current) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (semantics == RNS8_WRAP_U64_MOD_2_64 &&
      (!wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
       !wrap_byte_limb_state_current_for_backend(C, plan.backend))) {
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
    matrix.host_byte_limbs_current = true;
    matrix.device_byte_limbs_current = true;
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
  matrix.host_residues_current = true;
  matrix.device_residues_current = true;
  matrix.host_byte_limbs_current = false;
  matrix.device_byte_limbs_current = false;
  return RNS8_SUCCESS;
}

rns8_status ensure_device_residues_current(rns8_matrix& matrix) {
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT) {
    return RNS8_SUCCESS;
  }
  if (matrix.device_residues_current) {
    return RNS8_SUCCESS;
  }
  if (!matrix.host_residues_current || !matrix.hip_residues || matrix.hip_residue_bytes == 0) {
    return RNS8_INTERNAL_ERROR;
  }
  const rns8_status status = rns8::detail::hip_direct_copy_host_to_device(
      matrix.hip_device_id, matrix.hip_residues, matrix.residues.data(), matrix.hip_residue_bytes);
  if (status == RNS8_SUCCESS) {
    matrix.device_residues_current = true;
  }
  return status;
}

rns8_status ensure_host_residues_current(const rns8_matrix& const_matrix) {
  auto& matrix = const_cast<rns8_matrix&>(const_matrix);
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT) {
    return RNS8_SUCCESS;
  }
  if (matrix.host_residues_current) {
    return RNS8_SUCCESS;
  }
  if (!matrix.device_residues_current || !matrix.hip_residues || matrix.hip_residue_bytes == 0) {
    return RNS8_INTERNAL_ERROR;
  }
  const rns8_status status = rns8::detail::hip_direct_copy_device_to_host(
      matrix.hip_device_id, matrix.residues.data(), matrix.hip_residues, matrix.hip_residue_bytes);
  if (status == RNS8_SUCCESS) {
    matrix.host_residues_current = true;
  }
  return status;
}

}  // namespace

rns8_status rns8_create_context(int device_id, const rns8_context_options* options, rns8_context** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;

    rns8_backend_kind requested = RNS8_BACKEND_CPU_REFERENCE;
    if (options) {
      if (!rns8::detail::valid_abi(options->struct_size, options->abi_version, sizeof(*options))) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (options->flags != 0) {
        return RNS8_INVALID_ARGUMENT;
      }
      requested = effective_backend(options->requested_backend, RNS8_BACKEND_CPU_REFERENCE);
    }

    auto* ctx = new (std::nothrow) rns8_context();
    if (!ctx) {
      return RNS8_INTERNAL_ERROR;
    }

    if (requested == RNS8_BACKEND_CPU_REFERENCE) {
      ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_cpu_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_HIP_DIRECT) {
      ctx->backend = RNS8_BACKEND_HIP_DIRECT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::hip_direct_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      ctx->backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_wrap64_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    delete ctx;
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_destroy_context(rns8_context* ctx) {
  delete ctx;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_device_info(rns8_context* ctx, rns8_device_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = ctx->device_info;
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_plan(rns8_context* ctx, const rns8_gemm_desc* desc, rns8_plan** out) {
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
    const rns8_status validation = rns8::detail::validate_gemm_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }

    const rns8_backend_kind requested = effective_backend(desc->requested_backend, ctx->backend);
    if (requested != ctx->backend) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (requested != RNS8_BACKEND_CPU_REFERENCE && requested != RNS8_BACKEND_HIP_DIRECT &&
        requested != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* plan = new (std::nothrow) rns8_plan();
    if (!plan) {
      return RNS8_INTERNAL_ERROR;
    }
    plan->desc = *desc;
    plan->desc.max_prefix = prefix;
    if (plan->desc.tile_m == 0) {
      plan->desc.tile_m = 128;
    }
    if (plan->desc.tile_n == 0) {
      plan->desc.tile_n = 128;
    }
    plan->prefix = prefix;
    plan->modulus_product = prefix == 0 ? 0 : rns8::detail::modulus_product(prefix);
    plan->backend = requested;
    const rns8_status schedule_status = configure_plan_schedule(*plan);
    if (schedule_status != RNS8_SUCCESS) {
      delete plan;
      return schedule_status;
    }
    *out = plan;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_plan(rns8_plan* plan) {
  delete plan;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_plan_schedule_info(const rns8_plan* plan, rns8_plan_schedule_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->tile_m = plan->desc.tile_m;
    out->tile_n = plan->desc.tile_n;
    out->tile_rows = plan->schedule_tile_rows;
    out->tile_cols = plan->schedule_tile_cols;
    out->tile_count = plan->schedule_tile_count;
    out->min_required_prefix = plan->schedule_min_required_prefix;
    out->max_required_prefix = plan->schedule_max_required_prefix;
    out->min_selected_prefix = plan->schedule_min_selected_prefix;
    out->max_selected_prefix = plan->schedule_max_selected_prefix;
    out->prefix_group_count = plan->schedule_prefix_group_count;
    out->adaptive_prefix_active = plan->schedule_adaptive_prefix_active;
    out->adaptive_skip_active = plan->schedule_adaptive_skip_active;
    out->range_bit_length = plan->schedule_range_bit_length;
    out->flags = plan->schedule_flags;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_plan_tile_schedule(
    const rns8_plan* plan,
    rns8_plan_tile_schedule_entry* entries,
    uint64_t capacity,
    uint64_t* written) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !written) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    *written = plan->schedule_tile_count;
    if (!entries || capacity == 0) {
      return RNS8_SUCCESS;
    }
    if (capacity < plan->schedule_tile_count) {
      return RNS8_WORKSPACE_TOO_SMALL;
    }
    if (!plan->tile_schedule.empty()) {
      std::copy(plan->tile_schedule.begin(), plan->tile_schedule.end(), entries);
      return RNS8_SUCCESS;
    }
    for (uint64_t index = 0; index < plan->schedule_tile_count; ++index) {
      entries[index] = make_tile_schedule_entry(
          *plan,
          index,
          plan->schedule_min_required_prefix,
          plan->schedule_min_selected_prefix,
          0,
          plan->schedule_range_bit_length);
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_workspace(rns8_context* ctx, const rns8_plan* plan, rns8_workspace** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend) {
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
    workspace->prefix = plan->prefix;
    workspace->backend = ctx->backend;
    workspace->hip_device_id = ctx->backend == RNS8_BACKEND_HIP_DIRECT ? ctx->device_id : -1;
    *out = workspace;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_workspace(rns8_workspace* workspace) {
  if (workspace && workspace->hip_scratch) {
    const rns8_status status = rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hip_scratch);
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
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
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
    if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT && matrix->hip_device_id != ctx->device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
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
    } else if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
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
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_rns(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = true;
        C->device_residues_current = false;
        C->host_byte_limbs_current = false;
        C->device_byte_limbs_current = false;
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      rns8_status status = RNS8_SUCCESS;
      if (plan->desc.semantics != RNS8_BOUNDED_I64 && plan->desc.semantics != RNS8_BOUNDED_U64) {
        status = ensure_device_residues_current(const_cast<rns8_matrix&>(*A));
        if (status != RNS8_SUCCESS) {
          return status;
        }
        status = ensure_device_residues_current(const_cast<rns8_matrix&>(*B));
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::hip_direct_gemm_rns_tiled_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::hip_direct_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_wrap_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      const rns8_status status = rns8::detail::cpu_gemm_wrap_u64(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = false;
        C->device_residues_current = false;
        C->host_byte_limbs_current = true;
        C->device_byte_limbs_current = false;
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
          ctx->device_id,
          A->hip_byte_limbs,
          B->hip_byte_limbs,
          C->hip_byte_limbs,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = false;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = true;
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

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
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        return rns8::detail::hip_direct_export_i64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &const_cast<rns8_matrix*>(C)->hip_export_buffer,
            &const_cast<rns8_matrix*>(C)->hip_export_bytes,
            &const_cast<rns8_matrix*>(C)->hip_status_buffer,
            &const_cast<rns8_matrix*>(C)->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->tile_schedule.data(),
            plan->tile_bounds.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()),
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
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        int64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_signed(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        dst[row * ld + col] = value;
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
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        return rns8::detail::hip_direct_export_u64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &const_cast<rns8_matrix*>(C)->hip_export_buffer,
            &const_cast<rns8_matrix*>(C)->hip_export_bytes,
            &const_cast<rns8_matrix*>(C)->hip_status_buffer,
            &const_cast<rns8_matrix*>(C)->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->tile_schedule.data(),
            plan->tile_bounds.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()),
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
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        uint64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_unsigned(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        dst[row * ld + col] = value;
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
        return RNS8_INTERNAL_ERROR;
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
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
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
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
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
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
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
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
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

rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc)) ||
        desc->semantics != RNS8_BOUNDED_I64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!valid_matrix_access(desc->m, desc->k, lda) || !valid_matrix_access(desc->k, desc->n, ldb) ||
        !valid_matrix_access(desc->m, desc->n, ldc)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_plan* plan = nullptr;
    rns8_status status = rns8_create_plan(ctx, desc, &plan);
    if (status != RNS8_SUCCESS) {
      return status;
    }

    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    rns8_workspace* workspace = nullptr;
    const rns8_matrix_desc a_desc =
        make_matrix_desc(desc->m, desc->k, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc b_desc =
        make_matrix_desc(desc->k, desc->n, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc c_desc =
        make_matrix_desc(desc->m, desc->n, desc->semantics, desc->bound_kind, plan->prefix);

    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, plan, &workspace);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, a_matrix, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, b_matrix, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_i64(ctx, plan, c_matrix, C, ldc);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
    return status;
  });
}

rns8_status rns8_gemm_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc)) ||
        desc->semantics != RNS8_BOUNDED_U64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!valid_matrix_access(desc->m, desc->k, lda) || !valid_matrix_access(desc->k, desc->n, ldb) ||
        !valid_matrix_access(desc->m, desc->n, ldc)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_plan* plan = nullptr;
    rns8_status status = rns8_create_plan(ctx, desc, &plan);
    if (status != RNS8_SUCCESS) {
      return status;
    }

    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    rns8_workspace* workspace = nullptr;
    const rns8_matrix_desc a_desc =
        make_matrix_desc(desc->m, desc->k, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc b_desc =
        make_matrix_desc(desc->k, desc->n, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc c_desc =
        make_matrix_desc(desc->m, desc->n, desc->semantics, desc->bound_kind, plan->prefix);

    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, plan, &workspace);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, a_matrix, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, b_matrix, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_u64(ctx, plan, c_matrix, C, ldc);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
    return status;
  });
}

rns8_status rns8_gemm_wrap_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C || !rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (desc->semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_backend_kind requested = effective_backend(desc->requested_backend, ctx->backend);
    if (requested != ctx->backend || !backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!valid_matrix_access(desc->m, desc->k, lda) || !valid_matrix_access(desc->k, desc->n, ldb) ||
        !valid_matrix_access(desc->m, desc->n, ldc)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_plan* plan = nullptr;
    rns8_status status = rns8_create_plan(ctx, desc, &plan);
    if (status != RNS8_SUCCESS) {
      return status;
    }

    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    rns8_workspace* workspace = nullptr;
    const rns8_matrix_desc a_desc =
        make_matrix_desc(desc->m, desc->k, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc b_desc =
        make_matrix_desc(desc->k, desc->n, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc c_desc =
        make_matrix_desc(desc->m, desc->n, desc->semantics, desc->bound_kind, plan->prefix);

    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, plan, &workspace);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, a_matrix, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, b_matrix, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_wrap_u64(ctx, plan, c_matrix, C, ldc);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
    return status;
  });
}
