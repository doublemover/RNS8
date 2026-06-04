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

bool plan_schedule_contract_matches(const rns8_plan& plan) {
  constexpr uint32_t allowed_flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  constexpr uint32_t known_tile_schedule_flags = RNS8_TILE_SCHEDULE_ZERO_OUTPUT;
  if (!backend_supports_semantics(plan.backend, plan.desc.semantics) ||
      !configured_tile_size_valid(plan.desc.tile_m) || !configured_tile_size_valid(plan.desc.tile_n) ||
      (plan.desc.flags & ~allowed_flags) != 0) {
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
           plan.desc.lhs_bound == 0 && plan.desc.rhs_bound == 0;
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
           plan.desc.lhs_bound == 0 && plan.desc.rhs_bound == 0;
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
       plan.desc.tile_bounds_count != 0 || plan.desc.lhs_bound != 0 || plan.desc.rhs_bound != 0)) {
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
           (plan.desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS) == 0 && plan.schedule_flags == 0;
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
    const uint32_t expected_flags =
        allow_proven_zero_tile_skips && tile_bound == 0 ? RNS8_TILE_SCHEDULE_ZERO_OUTPUT : 0u;
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

bool valid_prepack_operand_role(rns8_operand_role operand_role) {
  return operand_role == RNS8_OPERAND_A || operand_role == RNS8_OPERAND_B;
}

const char* operand_role_name(rns8_operand_role operand_role) {
  switch (operand_role) {
    case RNS8_OPERAND_A:
      return "A";
    case RNS8_OPERAND_B:
      return "B";
  }
  return "unknown";
}

bool prepack_operand_shape_for_plan(
    const rns8_plan& plan,
    rns8_operand_role operand_role,
    int64_t& rows,
    int64_t& cols) {
  if (operand_role == RNS8_OPERAND_A) {
    rows = plan.desc.m;
    cols = plan.desc.k;
    return true;
  }
  if (operand_role == RNS8_OPERAND_B) {
    rows = plan.desc.k;
    cols = plan.desc.n;
    return true;
  }
  rows = 0;
  cols = 0;
  return false;
}

bool matrix_backend_can_feed_plan(const rns8_matrix& matrix, const rns8_plan& plan) {
  if (matrix.backend == plan.backend) {
    return true;
  }
  return matrix.backend == RNS8_BACKEND_HIP_DIRECT && hip_resident_rns_backend(plan.backend);
}

const char* prepack_operand_layout_version_for_plan(const rns8_plan& plan, rns8_operand_role operand_role) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return operand_role == RNS8_OPERAND_A ? "hipblaslt_a_transposed_centered_i8_mk16_v1"
                                          : "hipblaslt_b_transposed_centered_i8_nk16_v1";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return operand_role == RNS8_OPERAND_A ? "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1"
                                          : "ck_b_canonical_colmajor_i8_n64_kblock32768_v1";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    return operand_role == RNS8_OPERAND_A ? "rocwmma_a_rowmajor_i8_m16_kblock65536_v1"
                                          : "rns_i8_tile_swizzled_b_v1";
  }
  return persistent_layout_version_for_plan(plan);
}

const char* rocwmma_b_prepack_kernel_variant() {
  return "rocwmma_rns_i8_tile_swizzled_b_prepack_v1";
}

uint64_t prepack_prefix_schedule_fingerprint(const rns8_plan& plan) {
  uint64_t hash = 1469598103934665603ull;
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
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.tile_bounds.size()));
  for (const uint64_t bound : plan.tile_bounds) {
    hash = workspace_fingerprint_mix(hash, bound);
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

uint64_t rocwmma_b_prepack_k_block_cap() {
  return static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK);
}

uint64_t rocwmma_b_prepack_k_block_size(const rns8_plan& plan) {
  uint64_t k_block = static_cast<uint64_t>(plan.desc.k) < rocwmma_b_prepack_k_block_cap()
                         ? static_cast<uint64_t>(plan.desc.k)
                         : rocwmma_b_prepack_k_block_cap();
  if (k_block < 16) {
    k_block = 16;
  }
  return (k_block + 15u) / 16u * 16u;
}

std::string prepack_target_id_for_device(int hip_device_id) {
  if (hip_device_id < 0) {
    return "cpu";
  }
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (rns8::detail::hip_direct_probe(hip_device_id, info) == RNS8_SUCCESS && info.gcn_arch[0] != '\0' &&
      std::string(info.gcn_arch) != "none") {
    return info.gcn_arch;
  }
  return "hip_device_" + std::to_string(hip_device_id);
}

std::string prepack_target_id_for_context(const rns8_context& ctx) {
  if (ctx.device_id < 0) {
    return "cpu";
  }
  if (ctx.device_info.gcn_arch[0] != '\0' && std::string(ctx.device_info.gcn_arch) != "none") {
    return ctx.device_info.gcn_arch;
  }
  return prepack_target_id_for_device(ctx.device_id);
}

bool rocwmma_b_prepack_cache_supported(const rns8_plan& plan);

bool prepack_operand_matrix_compatible(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role) {
  if (!valid_prepack_operand_role(operand_role) || !plan_schedule_contract_matches(plan) ||
      !matrix_backend_can_feed_plan(matrix, plan)) {
    return false;
  }
  int64_t rows = 0;
  int64_t cols = 0;
  if (!prepack_operand_shape_for_plan(plan, operand_role, rows, cols)) {
    return false;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  if (!matrix_descriptor_matches(
          matrix,
          plan.desc.semantics,
          storage_bound_kind_for_plan(plan),
          rows,
          cols,
          storage_prefix,
          plan.desc.tile_m,
          plan.desc.tile_n)) {
    return false;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return wrap_matrix_storage_matches(matrix, plan.backend, rows, cols) &&
           wrap_byte_limb_state_current_for_backend(matrix, plan.backend);
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return finite_matrix_storage_matches(matrix, plan.backend, rows, cols) &&
           matrix.finite_modulus == plan.desc.finite_modulus &&
           rns_residue_state_current_for_backend(matrix, plan.backend);
  }
  return rns_matrix_storage_matches(matrix, plan.backend, rows, cols, storage_prefix) &&
         rns_residue_state_current_for_backend(matrix, plan.backend);
}

uint64_t prepack_cache_key_hash(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version) {
  uint64_t hash = plan_workspace_fingerprint(plan);
  hash = workspace_fingerprint_mix_string(hash, prepack_target_id_for_device(matrix.hip_device_id));
  hash = workspace_fingerprint_mix_string(hash, plan.backend_selected_kernel);
  hash = workspace_fingerprint_mix_string(hash, rocwmma_b_prepack_cache_supported(plan) && operand_role == RNS8_OPERAND_B
                                                    ? rocwmma_b_prepack_kernel_variant()
                                                    : "none");
  hash = workspace_fingerprint_mix(hash, prepack_prefix_schedule_fingerprint(plan));
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_m);
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_n);
  if (plan.backend == RNS8_BACKEND_ROCWMMA && operand_role == RNS8_OPERAND_B) {
    hash = workspace_fingerprint_mix(hash, 16);
    hash = workspace_fingerprint_mix(hash, 16);
    hash = workspace_fingerprint_mix(hash, rocwmma_b_prepack_k_block_size(plan));
    hash = workspace_fingerprint_mix(hash, rocwmma_b_prepack_k_block_cap());
  } else {
    hash = workspace_fingerprint_mix(hash, 0);
    hash = workspace_fingerprint_mix(hash, 0);
    hash = workspace_fingerprint_mix(hash, 0);
    hash = workspace_fingerprint_mix(hash, 0);
  }
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(operand_role));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(matrix.backend));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(matrix.desc.semantics));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.rows));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.cols));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.logical_ld));
  hash = workspace_fingerprint_mix(hash, matrix.prefix);
  hash = workspace_fingerprint_mix(hash, matrix.finite_modulus);
  hash = workspace_fingerprint_mix(hash, matrix.source_version);
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.hip_device_id));
  hash = workspace_fingerprint_mix(hash, matrix.host_residues_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.device_residues_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.host_byte_limbs_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.device_byte_limbs_current ? 1u : 0u);
  hash = workspace_fingerprint_mix_string(hash, matrix_layout_version);
  hash = workspace_fingerprint_mix_string(hash, operand_layout_version);
  return hash == 0 ? 1 : hash;
}

std::string build_prepack_cache_key(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    uint64_t plan_fingerprint,
    uint64_t cache_key_hash,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version) {
  std::string key = "prepack-v2";
  key += ";backend=";
  key += backend_name(plan.backend);
  key += ";target_id=" + prepack_target_id_for_device(matrix.hip_device_id);
  key += ";kernel=" + plan.backend_selected_kernel;
  key += ";prepack_kernel=";
  key += rocwmma_b_prepack_cache_supported(plan) && operand_role == RNS8_OPERAND_B ? rocwmma_b_prepack_kernel_variant()
                                                                                : "none";
  key += ";semantics=";
  key += semantics_name_for_key(plan.desc.semantics);
  key += ";prefix_schedule_hash=" + std::to_string(prepack_prefix_schedule_fingerprint(plan));
  key += ";tile_m=" + std::to_string(plan.desc.tile_m);
  key += ";tile_n=" + std::to_string(plan.desc.tile_n);
  key += ";operand_tile_m=" + std::to_string(
      plan.backend == RNS8_BACKEND_ROCWMMA && operand_role == RNS8_OPERAND_B ? 16 : 0);
  key += ";operand_tile_n=" + std::to_string(
      plan.backend == RNS8_BACKEND_ROCWMMA && operand_role == RNS8_OPERAND_B ? 16 : 0);
  key += ";k_block_size=" + std::to_string(
      plan.backend == RNS8_BACKEND_ROCWMMA && operand_role == RNS8_OPERAND_B ? rocwmma_b_prepack_k_block_size(plan) : 0);
  key += ";k_block_cap=" + std::to_string(
      plan.backend == RNS8_BACKEND_ROCWMMA && operand_role == RNS8_OPERAND_B ? rocwmma_b_prepack_k_block_cap() : 0);
  key += ";operand=";
  key += operand_role_name(operand_role);
  key += ";m=" + std::to_string(plan.desc.m);
  key += ";n=" + std::to_string(plan.desc.n);
  key += ";k=" + std::to_string(plan.desc.k);
  key += ";source_version=" + std::to_string(matrix.source_version);
  key += ";hip_device_id=" + std::to_string(matrix.hip_device_id);
  key += ";matrix_rows=" + std::to_string(matrix.desc.rows);
  key += ";matrix_cols=" + std::to_string(matrix.desc.cols);
  key += ";prefix=" + std::to_string(matrix.prefix);
  key += ";finite_modulus=" + std::to_string(matrix.finite_modulus);
  key += ";matrix_layout=" + matrix_layout_version;
  key += ";operand_layout=" + operand_layout_version;
  key += ";plan_fingerprint=" + std::to_string(plan_fingerprint);
  key += ";hash=" + std::to_string(cache_key_hash);
  return key;
}

bool rocwmma_b_prepack_cache_supported(const rns8_plan& plan) {
  return plan.backend == RNS8_BACKEND_ROCWMMA && !uses_finite_storage(plan.desc.semantics) &&
         plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 && plan.tile_schedule.empty() && plan.desc.k > 0 &&
         plan.desc.k <= static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) && plan_schedule_contract_matches(plan);
}

bool rocwmma_b_prepack_bytes_for_plan(const rns8_plan& plan, std::size_t& b_pack_bytes, std::size_t& total_cache_bytes) {
  b_pack_bytes = 0;
  total_cache_bytes = 0;
  if (!rocwmma_b_prepack_cache_supported(plan)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::rocwmma_workspace_requirements(
          plan.desc.m, plan.desc.n, plan.desc.k, a_bytes, b_pack_bytes, total_workspace)) {
    return false;
  }
  if (plan.prefix != 0 && b_pack_bytes > std::numeric_limits<std::size_t>::max() / plan.prefix) {
    return false;
  }
  total_cache_bytes = b_pack_bytes * static_cast<std::size_t>(plan.prefix);
  return total_cache_bytes != 0;
}

bool cache_key_contains_field(const std::string& key, const std::string& field, const std::string& value) {
  return key.find(";" + field + "=" + value) != std::string::npos;
}

bool prepack_cache_key_self_consistent(const rns8_prepack_cache& cache) {
  return cache.cache_key.rfind("prepack-v2;", 0) == 0 &&
         cache_key_contains_field(cache.cache_key, "target_id", cache.target_id) &&
         cache_key_contains_field(cache.cache_key, "kernel", cache.selected_kernel) &&
         cache_key_contains_field(cache.cache_key, "prepack_kernel", cache.prepack_kernel_variant) &&
         cache_key_contains_field(
             cache.cache_key, "prefix_schedule_hash", std::to_string(cache.prefix_schedule_fingerprint)) &&
         cache_key_contains_field(cache.cache_key, "tile_m", std::to_string(cache.tile_m)) &&
         cache_key_contains_field(cache.cache_key, "tile_n", std::to_string(cache.tile_n)) &&
         cache_key_contains_field(cache.cache_key, "k_block_size", std::to_string(cache.k_block_size)) &&
         cache_key_contains_field(cache.cache_key, "k_block_cap", std::to_string(cache.k_block_cap)) &&
         cache_key_contains_field(cache.cache_key, "source_version", std::to_string(cache.source_version)) &&
         cache_key_contains_field(cache.cache_key, "hip_device_id", std::to_string(cache.hip_device_id)) &&
         cache_key_contains_field(cache.cache_key, "operand_layout", cache.operand_layout_version) &&
         cache_key_contains_field(cache.cache_key, "hash", std::to_string(cache.cache_key_hash));
}

bool prepack_cache_matches_plan(const rns8_prepack_cache& cache, const rns8_plan& plan) {
  std::size_t b_pack_bytes = 0;
  std::size_t total_cache_bytes = 0;
  return rocwmma_b_prepack_bytes_for_plan(plan, b_pack_bytes, total_cache_bytes) &&
         cache.backend == plan.backend && cache.semantics == plan.desc.semantics &&
         cache.operand_role == RNS8_OPERAND_B && cache.rows == plan.desc.k && cache.cols == plan.desc.n &&
         cache.k == plan.desc.k && cache.tile_m == plan.desc.tile_m && cache.tile_n == plan.desc.tile_n &&
         cache.prefix == plan.prefix &&
         cache.finite_modulus == plan.desc.finite_modulus &&
         cache.prefix_schedule_fingerprint == prepack_prefix_schedule_fingerprint(plan) &&
         cache.k_block_size == rocwmma_b_prepack_k_block_size(plan) &&
         cache.k_block_cap == rocwmma_b_prepack_k_block_cap() &&
         cache.plan_fingerprint == plan_workspace_fingerprint(plan) &&
         cache.selected_kernel == plan.backend_selected_kernel &&
         cache.prepack_kernel_variant == rocwmma_b_prepack_kernel_variant() &&
         cache.matrix_layout_version == persistent_layout_version_for_semantics(plan.desc.semantics) &&
         cache.operand_layout_version == prepack_operand_layout_version_for_plan(plan, RNS8_OPERAND_B) &&
         prepack_cache_key_self_consistent(cache) &&
         cache.device_data != nullptr && cache.device_bytes == total_cache_bytes &&
         cache.operand_pack_bytes == b_pack_bytes;
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
      workspace.backend_target_id != plan.backend_target_id ||
      workspace.backend_autotune_key != plan.backend_autotune_key ||
      workspace.backend_performance_validated != plan.backend_performance_validated) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) && workspace.hip_device_id != ctx.device_id) {
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
  if (matrix.hip_native_i64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_i64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_i64 = nullptr;
    matrix.hip_native_i64_bytes = 0;
  }
  if (matrix.hip_native_u64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_u64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
  }
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_max_elements = 0;
  matrix.finite_modulus = 0;
  matrix.device_residues_current = false;
  matrix.device_byte_limbs_current = false;
  matrix.host_native_current = false;
  matrix.device_native_current = false;
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

rns8_status ensure_hip_native_storage(rns8_context& ctx, rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (matrix.hip_device_id != -1 && matrix.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_device_id = ctx.device_id;
  std::size_t expected_bytes = 0;
  if (!native_matrix_bytes(matrix.desc.rows, matrix.desc.cols, sizeof(uint64_t), expected_bytes)) {
    return RNS8_RANGE_ERROR;
  }
  if (matrix.desc.semantics == RNS8_BOUNDED_I64) {
    if (matrix.hip_native_i64 && matrix.hip_native_i64_bytes == expected_bytes) {
      return RNS8_SUCCESS;
    }
    if (matrix.hip_native_i64 || matrix.hip_native_i64_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_native_i64_bytes = expected_bytes;
    rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_i64);
    if (status != RNS8_SUCCESS) {
      matrix.hip_native_i64 = nullptr;
      matrix.hip_native_i64_bytes = 0;
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_i64, matrix.hip_native_i64_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 && matrix.hip_native_u64_bytes == expected_bytes) {
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 || matrix.hip_native_u64_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_native_u64_bytes = expected_bytes;
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_u64);
  if (status != RNS8_SUCCESS) {
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_u64, matrix.hip_native_u64_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  return RNS8_SUCCESS;
}

rns8_status upload_native_i64(rns8_context& ctx, rns8_matrix& matrix, const int64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_I64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<int64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_i64, staged.data(), staged.size() * sizeof(int64_t));
  if (status != RNS8_SUCCESS) {
    matrix.device_native_current = false;
    return status;
  }
  matrix.host_native_current = false;
  matrix.device_native_current = true;
  return RNS8_SUCCESS;
}

rns8_status upload_native_u64(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_U64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<uint64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_u64, staged.data(), staged.size() * sizeof(uint64_t));
  if (status != RNS8_SUCCESS) {
    matrix.device_native_current = false;
    return status;
  }
  matrix.host_native_current = false;
  matrix.device_native_current = true;
  return RNS8_SUCCESS;
}

bool should_populate_native_on_pack(const rns8_context& ctx, const rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  return native_vector_backend(ctx.backend) ||
         (ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT && matrix.backend == RNS8_BACKEND_HIP_DIRECT);
}

rns8_status ensure_bounded_native_residues_current_for_rns_plan(
    rns8_context& ctx,
    const rns8_plan& plan,
    rns8_matrix& matrix) {
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !hip_resident_rns_backend(plan.backend) || native_vector_backend(plan.backend) ||
      (plan.desc.semantics != RNS8_BOUNDED_I64 && plan.desc.semantics != RNS8_BOUNDED_U64)) {
    return RNS8_SUCCESS;
  }
  if (rns_residue_state_current_for_backend(matrix, plan.backend)) {
    return RNS8_SUCCESS;
  }
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT || matrix.hip_device_id != ctx.device_id ||
      matrix.desc.semantics != plan.desc.semantics ||
      !rns_matrix_storage_matches(
          matrix, plan.backend, matrix.desc.rows, matrix.desc.cols, rns_storage_prefix_for_plan(plan)) ||
      !bounded_native_storage_matches(matrix, plan.desc.semantics, matrix.desc.rows, matrix.desc.cols) ||
      !bounded_native_state_current(matrix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  rns8_status status = RNS8_SUCCESS;
  if (plan.desc.semantics == RNS8_BOUNDED_I64) {
    status = rns8::detail::hip_direct_native_i64_to_rns_device(
        ctx.device_id,
        matrix.hip_native_i64,
        matrix.hip_residues,
        matrix.desc.rows,
        matrix.desc.cols,
        storage_prefix);
  } else {
    status = rns8::detail::hip_direct_native_u64_to_rns_device(
        ctx.device_id,
        matrix.hip_native_u64,
        matrix.hip_residues,
        matrix.desc.rows,
        matrix.desc.cols,
        storage_prefix);
  }
  if (status != RNS8_SUCCESS) {
    matrix.device_residues_current = false;
    return status;
  }
  matrix.host_residues_current = false;
  matrix.device_residues_current = true;
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
    workspace->backend_target_id = plan->backend_target_id;
    workspace->backend_autotune_key = plan->backend_autotune_key;
    workspace->backend_performance_validated = plan->backend_performance_validated;
    workspace->backend = plan->backend;
    workspace->hip_device_id = hip_device_backend(plan->backend) ? ctx->device_id : -1;
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
    if (workspace->hipblaslt_a_prepack_cache) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hipblaslt_a_prepack_device_id, workspace->hipblaslt_a_prepack_cache);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_a_prepack_cache = nullptr;
      workspace->hipblaslt_a_prepack_cache_bytes = 0;
      workspace->hipblaslt_a_prepack_current = false;
      workspace->hipblaslt_a_prepack_source_version = 0;
      workspace->hipblaslt_a_prepack_m = 0;
      workspace->hipblaslt_a_prepack_k = 0;
      workspace->hipblaslt_a_prepack_lda = 0;
      workspace->hipblaslt_a_prepack_prefix = 0;
      workspace->hipblaslt_a_prepack_device_id = -1;
    }
    if (workspace->hipblaslt_b_prepack_cache) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hipblaslt_b_prepack_device_id, workspace->hipblaslt_b_prepack_cache);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_b_prepack_cache = nullptr;
      workspace->hipblaslt_b_prepack_cache_bytes = 0;
      workspace->hipblaslt_b_prepack_current = false;
      workspace->hipblaslt_b_prepack_source_version = 0;
      workspace->hipblaslt_b_prepack_k = 0;
      workspace->hipblaslt_b_prepack_n = 0;
      workspace->hipblaslt_b_prepack_ldb = 0;
      workspace->hipblaslt_b_prepack_prefix = 0;
      workspace->hipblaslt_b_prepack_device_id = -1;
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
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics == RNS8_BOUNDED_I64) {
        matrix->native_i64.clear();
      } else if (matrix->desc.semantics == RNS8_BOUNDED_U64) {
        matrix->native_u64.clear();
      } else {
        delete matrix;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->host_native_current = false;
      matrix->device_native_current = false;
      const rns8_status status = ensure_hip_native_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        (void)free_hip_matrix_storage(*matrix);
        delete matrix;
        return status;
      }
    } else if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
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
      if (ctx->auto_backend_selection &&
          (matrix->desc.semantics == RNS8_BOUNDED_I64 || matrix->desc.semantics == RNS8_BOUNDED_U64)) {
        const rns8_status native_status = ensure_hip_native_storage(*ctx, *matrix);
        if (native_status != RNS8_SUCCESS) {
          (void)free_hip_matrix_storage(*matrix);
          delete matrix;
          return native_status;
        }
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

rns8_status rns8_get_matrix_storage_info(const rns8_matrix* matrix, rns8_matrix_storage_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = matrix->backend;
    out->semantics = matrix->desc.semantics;
    out->logical_layout = matrix->desc.logical_layout;
    out->bound_kind = matrix->desc.bound_kind;
    out->rows = matrix->desc.rows;
    out->cols = matrix->desc.cols;
    out->logical_ld = matrix->desc.logical_ld;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->source_version = matrix->source_version;
    out->host_residues_current = matrix->host_residues_current ? 1u : 0u;
    out->device_residues_current = matrix->device_residues_current ? 1u : 0u;
    out->host_byte_limbs_current = matrix->host_byte_limbs_current ? 1u : 0u;
    out->device_byte_limbs_current = matrix->device_byte_limbs_current ? 1u : 0u;
    out->host_native_current = matrix->host_native_current ? 1u : 0u;
    out->device_native_current = matrix->device_native_current ? 1u : 0u;
    out->uses_residue_storage = (!matrix->residues.empty() || matrix->hip_residues != nullptr) ? 1u : 0u;
    out->uses_byte_limb_storage = (!matrix->byte_limbs.empty() || matrix->hip_byte_limbs != nullptr) ? 1u : 0u;
    out->uses_native_storage = matrix_has_native_storage(*matrix) ? 1u : 0u;
    out->hip_device_id = static_cast<int32_t>(matrix->hip_device_id);
    out->flags = 0;
    out->host_residue_bytes = static_cast<uint64_t>(matrix->residues.size() * sizeof(int8_t));
    out->device_residue_bytes = static_cast<uint64_t>(matrix->hip_residue_bytes);
    out->host_byte_limb_bytes = static_cast<uint64_t>(matrix->byte_limbs.size() * sizeof(uint8_t));
    out->device_byte_limb_bytes = static_cast<uint64_t>(matrix->hip_byte_limb_bytes);
    out->host_native_bytes = static_cast<uint64_t>(
        matrix->native_i64.size() * sizeof(int64_t) + matrix->native_u64.size() * sizeof(uint64_t));
    out->device_native_bytes =
        static_cast<uint64_t>(matrix->hip_native_i64_bytes + matrix->hip_native_u64_bytes);
    set_text(
        out->layout_version,
        sizeof(out->layout_version),
        native_vector_backend(matrix->backend) ? native_layout_version_for_semantics(matrix->desc.semantics)
                                               : persistent_layout_version_for_semantics(matrix->desc.semantics));
    set_text(out->storage_scope, sizeof(out->storage_scope), storage_scope_for_matrix(*matrix));
    set_text(out->detail, sizeof(out->detail), storage_detail_for_matrix(*matrix));
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_prepack_cache_key_info(
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache_key_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!prepack_operand_matrix_compatible(*plan, *matrix, operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    const std::string matrix_layout_version = persistent_layout_version_for_semantics(matrix->desc.semantics);
    const std::string operand_layout_version = prepack_operand_layout_version_for_plan(*plan, operand_role);
    const uint64_t plan_fingerprint = plan_workspace_fingerprint(*plan);
    const uint64_t cache_key_hash =
        prepack_cache_key_hash(*plan, *matrix, operand_role, matrix_layout_version, operand_layout_version);
    const std::string cache_key = build_prepack_cache_key(
        *plan,
        *matrix,
        operand_role,
        plan_fingerprint,
        cache_key_hash,
        matrix_layout_version,
        operand_layout_version);

    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->semantics = plan->desc.semantics;
    out->operand_role = operand_role;
    out->cache_key_valid = 1;
    out->reusable_prepack_cache_available =
        (operand_role == RNS8_OPERAND_B && rocwmma_b_prepack_cache_supported(*plan)) ? 1u : 0u;
    out->production_prepack_cache_available = 0;
    out->flags = 0;
    out->hip_device_id = matrix->hip_device_id;
    out->reserved0 = 0;
    out->matrix_rows = matrix->desc.rows;
    out->matrix_cols = matrix->desc.cols;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->source_version = matrix->source_version;
    out->plan_fingerprint = plan_fingerprint;
    out->cache_key_hash = cache_key_hash;
    set_text(out->matrix_layout_version, sizeof(out->matrix_layout_version), matrix_layout_version);
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version), operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope), "validated_key_no_production_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache_key);
    set_text(
        out->detail,
        sizeof(out->detail),
        "Plan and operand matrix are compatible for future prepack cache keying; no production cache is available.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_prepack_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!valid_prepack_operand_role(operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (operand_role != RNS8_OPERAND_B || !rocwmma_b_prepack_cache_supported(*plan)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!context_accepts_backend(*ctx, plan->backend) || !prepack_operand_matrix_compatible(*plan, *matrix, operand_role) ||
        matrix->hip_device_id != ctx->device_id || !matrix->device_residues_current) {
      return RNS8_INVALID_ARGUMENT;
    }

    std::size_t b_pack_bytes = 0;
    std::size_t total_cache_bytes = 0;
    if (!rocwmma_b_prepack_bytes_for_plan(*plan, b_pack_bytes, total_cache_bytes)) {
      return RNS8_RANGE_ERROR;
    }

    auto* cache = new (std::nothrow) rns8_prepack_cache();
    if (!cache) {
      return RNS8_INTERNAL_ERROR;
    }
    cache->backend = plan->backend;
    cache->semantics = plan->desc.semantics;
    cache->operand_role = operand_role;
    cache->rows = matrix->desc.rows;
    cache->cols = matrix->desc.cols;
    cache->k = plan->desc.k;
    cache->tile_m = plan->desc.tile_m;
    cache->tile_n = plan->desc.tile_n;
    cache->prefix = plan->prefix;
    cache->finite_modulus = matrix->finite_modulus;
    cache->prefix_schedule_fingerprint = prepack_prefix_schedule_fingerprint(*plan);
    cache->k_block_size = rocwmma_b_prepack_k_block_size(*plan);
    cache->k_block_cap = rocwmma_b_prepack_k_block_cap();
    cache->source_version = matrix->source_version;
    cache->plan_fingerprint = plan_workspace_fingerprint(*plan);
    cache->target_id = prepack_target_id_for_device(matrix->hip_device_id);
    cache->selected_kernel = plan->backend_selected_kernel;
    cache->prepack_kernel_variant = rocwmma_b_prepack_kernel_variant();
    cache->matrix_layout_version = persistent_layout_version_for_semantics(matrix->desc.semantics);
    cache->operand_layout_version = prepack_operand_layout_version_for_plan(*plan, operand_role);
    cache->cache_key_hash =
        prepack_cache_key_hash(*plan, *matrix, operand_role, cache->matrix_layout_version, cache->operand_layout_version);
    cache->cache_key = build_prepack_cache_key(
        *plan,
        *matrix,
        operand_role,
        cache->plan_fingerprint,
        cache->cache_key_hash,
        cache->matrix_layout_version,
        cache->operand_layout_version);
    cache->hip_device_id = ctx->device_id;
    cache->device_bytes = total_cache_bytes;
    cache->operand_pack_bytes = b_pack_bytes;

    rns8_status status = rns8::detail::hip_direct_allocate(ctx->device_id, total_cache_bytes, &cache->device_data);
    if (status != RNS8_SUCCESS) {
      delete cache;
      return status;
    }
    status = rns8::detail::rocwmma_prepack_b_rns_device(
        ctx->device_id,
        matrix->hip_residues,
        cache->device_data,
        cache->device_bytes,
        plan->desc.k,
        plan->desc.n,
        matrix->desc.cols,
        plan->prefix);
    if (status != RNS8_SUCCESS) {
      (void)rns8::detail::hip_direct_free(ctx->device_id, cache->device_data);
      cache->device_data = nullptr;
      delete cache;
      return status;
    }
    *out = cache;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_prepack_cache_info(const rns8_prepack_cache* cache, rns8_prepack_cache_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!cache || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = cache->backend;
    out->semantics = cache->semantics;
    out->operand_role = cache->operand_role;
    out->cache_key_valid = cache->cache_key_hash != 0 && !cache->cache_key.empty() ? 1u : 0u;
    out->reusable_prepack_cache_available = 1;
    out->production_prepack_cache_available = 0;
    out->flags = 0;
    out->hip_device_id = cache->hip_device_id;
    out->reserved0 = 0;
    out->matrix_rows = cache->rows;
    out->matrix_cols = cache->cols;
    out->k = cache->k;
    out->max_prefix = cache->prefix;
    out->finite_modulus = cache->finite_modulus;
    out->source_version = cache->source_version;
    out->plan_fingerprint = cache->plan_fingerprint;
    out->cache_key_hash = cache->cache_key_hash;
    out->device_bytes = static_cast<uint64_t>(cache->device_bytes);
    out->operand_pack_bytes = static_cast<uint64_t>(cache->operand_pack_bytes);
    set_text(out->matrix_layout_version, sizeof(out->matrix_layout_version), cache->matrix_layout_version);
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version), cache->operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope), "runtime_reusable_b_prepack_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache->cache_key);
    set_text(
        out->detail,
        sizeof(out->detail),
        "Created reusable accelerator prepack cache; no production cache availability is reported.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_prepack_cache(rns8_prepack_cache* cache) {
  if (!cache) {
    return RNS8_SUCCESS;
  }
  rns8_status status = RNS8_SUCCESS;
  if (cache->device_data) {
    status = rns8::detail::hip_direct_free(cache->hip_device_id, cache->device_data);
    cache->device_data = nullptr;
    cache->device_bytes = 0;
  }
  delete cache;
  return status;
}
