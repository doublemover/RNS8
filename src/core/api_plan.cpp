#include "core/api_internal.hpp"

namespace rns8::detail::api {

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
    if (plan.schedule_prefix_group_count == 1 && plan.schedule_min_required_prefix == plan.prefix &&
        plan.schedule_max_required_prefix == plan.prefix && plan.schedule_min_selected_prefix == plan.prefix &&
        plan.schedule_max_selected_prefix == plan.prefix && plan.schedule_adaptive_prefix_active == 0 &&
        plan.schedule_adaptive_skip_active == 0) {
      plan.tile_schedule.clear();
    }
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

const char* semantics_name_for_key(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
}

std::string selected_kernel_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_CPU_REFERENCE) {
    return uses_finite_storage(plan.desc.semantics) ? "cpu_reference_finite_u8_gemm_v1"
                                                    : "cpu_reference_scalar_rns_gemm_v1";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "cpu_wrap64_byte_limb_reference_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return "direct_hip_wrap64_byte_gemm36_tiled_2d_v3";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      if (plan.desc.finite_modulus == 256) {
        return "direct_hip_tiled_finite_u8_gemm_mod256_v1";
      }
      if (plan.desc.finite_modulus == 255) {
        return "direct_hip_tiled_finite_u8_gemm_mod255_v1";
      }
      if (plan.desc.finite_modulus == 251) {
        return "direct_hip_tiled_finite_u8_gemm_mod251_v1";
      }
      return "direct_hip_tiled_finite_u8_gemm_v1";
    }
    return "direct_hip_tiled_rns_gemm_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return plan.desc.semantics == RNS8_BOUNDED_I64 ? "hip_vector_alu_i64_exact_192b_v1"
                                                   : "hip_vector_alu_u64_exact_192b_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_int8_i32_scratch_reduce_baseline_v1";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (!plan.tile_schedule.empty()) {
      return "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1";
    }
    return "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    if (!plan.tile_schedule.empty()) {
      return "rocwmma_i8_i32_signed_tiled_hot_residue_v1";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1";
    }
    return "rocwmma_i8_i32_signed_hot_residue_v1";
  }
  return "not_implemented";
}

std::string epilogue_mode_for_plan(const rns8_plan& plan) {
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "low64_wrap_export";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "direct_int64_export";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "separate_i32_scratch_reduce_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "separate_i32_scratch_reduce_rns_output";
    }
    return "separate_i32_scratch_reduce_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "ck_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "ck_fused_i32_to_centered_residue_rns_output";
    }
    return "ck_fused_i32_to_centered_residue_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "rocwmma_fused_i32_to_centered_residue_rns_output";
    }
    return "rocwmma_fused_i32_to_centered_residue_then_crt_export";
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return "fused_centered_residue_then_canonical_u8_export";
  }
  if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    return "fused_centered_residue_rns_output";
  }
  return "fused_centered_residue_then_crt_export";
}

std::string workspace_mode_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    return plan.tile_schedule.empty() ? "resident_device_buffers"
                                      : "resident_device_buffers_with_tiled_schedule";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "native_device_i64_u64_buffers";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "resident_device_buffers_with_hipblaslt_scratch";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return "resident_device_buffers_with_ck_canonical_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    return "resident_device_buffers_with_rocwmma_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "host_byte_limb_reference_workspace";
  }
  return "host_reference_workspace";
}

rns8_output_domain output_domain_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB;
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return RNS8_OUTPUT_DOMAIN_FINITE_U8;
  }
  return RNS8_OUTPUT_DOMAIN_RNS_RESIDUE;
}

const char* output_domain_name(rns8_output_domain domain) {
  switch (domain) {
    case RNS8_OUTPUT_DOMAIN_RNS_RESIDUE:
      return "rns_residue_current";
    case RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64:
      return "native_i64_u64_current";
    case RNS8_OUTPUT_DOMAIN_FINITE_U8:
      return "finite_u8_current";
    case RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB:
      return "wrap64_byte_limb_current";
  }
  return "unknown";
}

bool plan_output_device_current(const rns8_plan& plan) {
  return hip_device_backend(plan.backend) ? true : false;
}

uint32_t next_op_flags_for_plan(const rns8_plan& plan) {
  uint32_t flags = RNS8_NEXT_OP_FINAL_EXPORT;
  const rns8_output_domain domain = output_domain_for_plan(plan);
  if (domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE) {
    flags |= RNS8_NEXT_OP_RNS_GEMM;
  }
  if (domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64) {
    flags |= RNS8_NEXT_OP_NATIVE_GEMM;
    if (plan.desc.semantics == RNS8_BOUNDED_I64 || plan.desc.semantics == RNS8_BOUNDED_U64) {
      flags |= RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE;
    }
  }
  if (rocwmma_b_prepack_cache_supported(plan)) {
    flags |= RNS8_NEXT_OP_REUSABLE_B_PREPACK;
  }
  return flags;
}

const char* next_op_hint_for_plan(const rns8_plan& plan) {
  const rns8_output_domain domain = output_domain_for_plan(plan);
  if (domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64) {
    return "produces native bounded device output; next native GEMM can consume it directly, and native-to-RNS conversion is available for mixed-storage AUTO chains";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE) {
    return "produces residue-current RNS output; next same-contract RNS GEMM can consume it without logical export";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_FINITE_U8) {
    return "produces finite-current centered residues; export converts to canonical uint8 output";
  }
  if (domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB) {
    return "produces wrap64 byte-limb-current output; export reconstructs low 64-bit words";
  }
  return "unknown output domain";
}

std::string isa_evidence_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine";
    }
    if (uses_finite_storage(plan.desc.semantics) &&
        (plan.desc.finite_modulus == 251 || plan.desc.finite_modulus == 255 || plan.desc.finite_modulus == 256)) {
      return "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide";
    }
    return "rns8_hip_direct_reciprocal_isa_gate";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "source_level_192bit_limb_accumulator_no_matrix_engine";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_library_int8_matmul_baseline";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide";
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    return "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide";
  }
  return "not_applicable_cpu";
}

uint64_t workspace_required_bytes_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    std::size_t scratch_bytes = 0;
    std::size_t workspace_bytes = 0;
    if (!rns8::detail::hipblaslt_baseline_workspace_requirements(
            plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, workspace_bytes)) {
      return 0;
    }
    if (scratch_bytes > std::numeric_limits<uint64_t>::max() - workspace_bytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(scratch_bytes) + static_cast<uint64_t>(workspace_bytes);
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    int64_t max_m = plan.desc.m;
    int64_t max_n = plan.desc.n;
    if (!plan.tile_schedule.empty()) {
      max_m = 0;
      max_n = 0;
      for (const auto& entry : plan.tile_schedule) {
        max_m = std::max(max_m, entry.row_extent);
        max_n = std::max(max_n, entry.col_extent);
      }
    }
    std::size_t a_pack_bytes = 0;
    std::size_t b_pack_bytes = 0;
    std::size_t temp_c_bytes = 0;
    std::size_t total_bytes = 0;
    if (!rns8::detail::ck_workspace_requirements(
            max_m, max_n, plan.desc.k, a_pack_bytes, b_pack_bytes, temp_c_bytes, total_bytes)) {
      return 0;
    }
    return static_cast<uint64_t>(total_bytes);
  }
  if (plan.backend == RNS8_BACKEND_ROCWMMA) {
    int64_t max_m = plan.desc.m;
    int64_t max_n = plan.desc.n;
    if (!plan.tile_schedule.empty()) {
      max_m = 0;
      max_n = 0;
      for (const auto& entry : plan.tile_schedule) {
        max_m = std::max(max_m, entry.row_extent);
        max_n = std::max(max_n, entry.col_extent);
      }
    }
    std::size_t a_pack_bytes = 0;
    std::size_t b_pack_bytes = 0;
    std::size_t total_bytes = 0;
    if (!rns8::detail::rocwmma_workspace_requirements(
            max_m, max_n, plan.desc.k, a_pack_bytes, b_pack_bytes, total_bytes)) {
      return 0;
    }
    return static_cast<uint64_t>(total_bytes);
  }
  if (plan.backend != RNS8_BACKEND_HIP_DIRECT || plan.tile_schedule.empty()) {
    return 0;
  }
  return static_cast<uint64_t>(plan.tile_schedule.size()) * sizeof(rns8_plan_tile_schedule_entry);
}

bool accelerator_workspace_shape_for_plan(const rns8_plan& plan, int64_t& max_m, int64_t& max_n) {
  max_m = plan.desc.m;
  max_n = plan.desc.n;
  if (!plan.tile_schedule.empty()) {
    max_m = 0;
    max_n = 0;
    for (const auto& entry : plan.tile_schedule) {
      max_m = std::max(max_m, entry.row_extent);
      max_n = std::max(max_n, entry.col_extent);
    }
  }
  return max_m > 0 && max_n > 0 && plan.desc.k > 0;
}

bool hipblaslt_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& library_workspace_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  accumulator_bytes = 0;
  library_workspace_bytes = 0;
  total_bytes = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
  if (!rns8::detail::hipblaslt_baseline_workspace_requirements(
          plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, workspace_bytes)) {
    return false;
  }
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  uint64_t padded_k = 0;
  const uint64_t max_k_block =
      static_cast<uint64_t>(plan.desc.k) < static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK)
          ? static_cast<uint64_t>(plan.desc.k)
          : static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK);
  if (!rns8::detail::hipblaslt_round_up_aligned(static_cast<uint64_t>(plan.desc.m), padded_m) ||
      !rns8::detail::hipblaslt_round_up_aligned(static_cast<uint64_t>(plan.desc.n), padded_n) ||
      !rns8::detail::hipblaslt_round_up_aligned(max_k_block, padded_k)) {
    return false;
  }
  if (!rns8::detail::checked_mul_u64(padded_m, padded_k, a_pack_bytes) ||
      !rns8::detail::checked_mul_u64(padded_n, padded_k, b_pack_bytes) ||
      workspace_bytes < a_pack_bytes || workspace_bytes - a_pack_bytes < b_pack_bytes) {
    return false;
  }
  accumulator_bytes = static_cast<uint64_t>(scratch_bytes);
  library_workspace_bytes = static_cast<uint64_t>(workspace_bytes) - a_pack_bytes - b_pack_bytes;
  if (accumulator_bytes > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(workspace_bytes)) {
    return false;
  }
  total_bytes = accumulator_bytes + static_cast<uint64_t>(workspace_bytes);
  return total_bytes == plan.backend_workspace_required_bytes;
}

bool ck_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  accumulator_bytes = 0;
  total_bytes = 0;
  int64_t max_m = 0;
  int64_t max_n = 0;
  if (!accelerator_workspace_shape_for_plan(plan, max_m, max_n)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t temp_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::ck_workspace_requirements(
          max_m, max_n, plan.desc.k, a_bytes, b_bytes, temp_bytes, total_workspace)) {
    return false;
  }
  a_pack_bytes = static_cast<uint64_t>(a_bytes);
  b_pack_bytes = static_cast<uint64_t>(b_bytes);
  accumulator_bytes = static_cast<uint64_t>(temp_bytes);
  total_bytes = static_cast<uint64_t>(total_workspace);
  return total_bytes == plan.backend_workspace_required_bytes;
}

bool rocwmma_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  total_bytes = 0;
  int64_t max_m = 0;
  int64_t max_n = 0;
  if (!accelerator_workspace_shape_for_plan(plan, max_m, max_n)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::rocwmma_workspace_requirements(max_m, max_n, plan.desc.k, a_bytes, b_bytes, total_workspace)) {
    return false;
  }
  a_pack_bytes = static_cast<uint64_t>(a_bytes);
  b_pack_bytes = static_cast<uint64_t>(b_bytes);
  total_bytes = static_cast<uint64_t>(total_workspace);
  return total_bytes == plan.backend_workspace_required_bytes;
}

const char* persistent_layout_version_for_semantics(rns8_semantics semantics) {
  if (semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "wrap64_byte_limb_v1";
  }
  if (uses_finite_storage(semantics)) {
    return "finite_u8_centered_residue_v1";
  }
  return "rns_centered_residue_planes_v1";
}

const char* native_layout_version_for_semantics(rns8_semantics semantics) {
  if (semantics == RNS8_BOUNDED_I64) {
    return "native_i64_rowmajor_v1";
  }
  if (semantics == RNS8_BOUNDED_U64) {
    return "native_u64_rowmajor_v1";
  }
  return "not_applicable";
}

const char* persistent_layout_version_for_plan(const rns8_plan& plan) {
  return native_vector_backend(plan.backend) ? native_layout_version_for_semantics(plan.desc.semantics)
                                             : persistent_layout_version_for_semantics(plan.desc.semantics);
}

const char* storage_scope_for_matrix(const rns8_matrix& matrix) {
  if (native_vector_backend(matrix.backend)) {
    return "native_device_storage";
  }
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return matrix.backend == RNS8_BACKEND_HIP_DIRECT ? "device_byte_limb_storage" : "host_byte_limb_storage";
  }
  return hip_resident_rns_backend(matrix.backend) ? "device_resident_storage" : "host_resident_storage";
}

const char* storage_detail_for_matrix(const rns8_matrix& matrix) {
  if (native_vector_backend(matrix.backend)) {
    return "Bounded vector-ALU matrix owns compact native integer storage on the selected HIP device.";
  }
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return matrix.backend == RNS8_BACKEND_HIP_DIRECT
               ? "Strict wrap64 matrix owns compact byte-limb storage on the selected HIP device."
               : "Strict wrap64 matrix owns compact host byte-limb storage.";
  }
  if (uses_finite_storage(matrix.desc.semantics)) {
    return hip_resident_rns_backend(matrix.backend)
               ? "Finite-u8 matrix owns one centered residue plane on the selected HIP device."
               : "Finite-u8 matrix owns one centered host residue plane.";
  }
  return hip_resident_rns_backend(matrix.backend)
             ? "RNS matrix owns centered residue planes on the selected HIP device."
             : "RNS matrix owns centered host residue planes.";
}

bool hipblaslt_scratch_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes) {
  bytes = 0;
  if (plan.backend != RNS8_BACKEND_HIPBLASLT) {
    return false;
  }
  std::size_t workspace_bytes = 0;
  return rns8::detail::hipblaslt_baseline_workspace_requirements(
      plan.desc.m, plan.desc.n, plan.desc.k, bytes, workspace_bytes);
}

bool hipblaslt_workspace_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes) {
  bytes = 0;
  if (plan.backend != RNS8_BACKEND_HIPBLASLT) {
    return false;
  }
  std::size_t scratch_bytes = 0;
  return rns8::detail::hipblaslt_baseline_workspace_requirements(
      plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, bytes);
}

std::string build_autotune_key(const rns8_plan& plan) {
  std::string key = "backend=";
  key += backend_name(plan.backend);
  key += ";semantics=";
  key += semantics_name_for_key(plan.desc.semantics);
  key += ";m=" + std::to_string(plan.desc.m);
  key += ";n=" + std::to_string(plan.desc.n);
  key += ";k=" + std::to_string(plan.desc.k);
  if (uses_finite_storage(plan.desc.semantics)) {
    key += ";finite_modulus=" + std::to_string(plan.desc.finite_modulus);
  }
  key += ";prefix=" + std::to_string(plan.prefix);
  key += ";tile_m=" + std::to_string(plan.desc.tile_m);
  key += ";tile_n=" + std::to_string(plan.desc.tile_n);
  key += ";groups=" + std::to_string(plan.schedule_prefix_group_count);
  key += ";adaptive_prefix=" + std::to_string(plan.schedule_adaptive_prefix_active);
  key += ";adaptive_skip=" + std::to_string(plan.schedule_adaptive_skip_active);
  key += ";kernel=" + plan.backend_selected_kernel;
  key += ";epilogue=" + plan.backend_epilogue_mode;
  return key;
}

bool backend_library_version_matches_plan(const rns8_plan& plan, const rns8_backend_capability_info& capability) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return plan.backend_library_version.rfind("hipBLASLt ", 0) == 0 ||
           plan.backend_library_version == capability.library_version;
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return plan.backend_library_version.rfind("HIP runtime ", 0) == 0 ||
           plan.backend_library_version == capability.library_version;
  }
  return plan.backend_library_version == capability.library_version;
}

rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity_for_plan(
    const rns8_context& ctx,
    const rns8_plan& plan) {
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  const std::string target = ctx.device_info.gcn_arch;
  if (!target.empty() && target != "none") {
    runtime.target_id = target;
  } else if (!ctx.device_info.hip_available) {
    runtime.target_id = "cpu";
  }
  runtime.hip_sdk_or_library_version = plan.backend_library_version;
  return runtime;
}

void configure_plan_backend_metadata(rns8_plan& plan, const rns8_context* ctx) {
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(plan.backend, capability);
  plan.backend_selected_kernel = selected_kernel_for_plan(plan);
  plan.backend_library = capability.library_name;
  plan.backend_library_version = capability.library_version;
  if (plan.backend == RNS8_BACKEND_HIPBLASLT && ctx && !ctx->hipblaslt_library_version.empty()) {
    plan.backend_library_version = ctx->hipblaslt_library_version;
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 && ctx && ctx->device_info.hip_runtime_version != 0) {
    plan.backend_library_version = "HIP runtime " + std::to_string(ctx->device_info.hip_runtime_version);
  }
  plan.backend_capability_status = capability.status;
  plan.backend_epilogue_mode = epilogue_mode_for_plan(plan);
  plan.backend_workspace_mode = workspace_mode_for_plan(plan);
  plan.backend_isa_evidence = isa_evidence_for_plan(plan);
  plan.backend_workspace_required_bytes = workspace_required_bytes_for_plan(plan);
  plan.backend_performance_validated = capability.performance_validated;
  plan.backend_autotune_key = build_autotune_key(plan);
}

bool prepare_auto_candidate_backend(rns8_context& ctx, rns8_backend_kind backend) {
  static_cast<void>(ctx);
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(backend, capability);
  if (!capability.is_available || !capability.compiled_kernel_available ||
      !capability.exact_differential_validated) {
    return false;
  }

  rns8_device_info probe_info{};
  probe_info.struct_size = sizeof(probe_info);
  probe_info.abi_version = RNS8_ABI_VERSION;
  switch (backend) {
    case RNS8_BACKEND_HIPBLASLT:
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (ctx.hipblaslt_handle) {
        return true;
      }
      return rns8::detail::hipblaslt_create_context(
                 ctx.device_id, probe_info, &ctx.hipblaslt_handle, ctx.hipblaslt_library_version) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_CK:
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      return rns8::detail::ck_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_ROCWMMA:
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      return rns8::detail::rocwmma_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      return rns8::detail::vector_alu_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    default:
      return false;
  }
}

void select_auto_backend_from_reviewed_cache(rns8_context& ctx, rns8_plan& plan) {
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !backend_supports_semantics(ctx.backend, plan.desc.semantics)) {
    return;
  }
  const rns8_backend_kind candidates[] = {
      RNS8_BACKEND_HIP_VECTOR_ALU_INT64,
      RNS8_BACKEND_HIPBLASLT,
      RNS8_BACKEND_CK,
      RNS8_BACKEND_ROCWMMA,
  };
  const auto snapshot = rns8::detail::read_autotune_cache();
  const rns8::detail::AutotuneCacheEntry* best_hit = nullptr;
  rns8_plan best_plan{};

  for (const rns8_backend_kind candidate_backend : candidates) {
    if (!backend_supports_semantics(candidate_backend, plan.desc.semantics)) {
      continue;
    }
    if (candidate_backend == RNS8_BACKEND_HIPBLASLT && is_per_tile_bound_kind(plan.desc.bound_kind)) {
      continue;
    }
    if (!prepare_auto_candidate_backend(ctx, candidate_backend)) {
      continue;
    }
    rns8_plan candidate = plan;
    candidate.backend = candidate_backend;
    configure_plan_backend_metadata(candidate, &ctx);
    const auto runtime = autotune_runtime_identity_for_plan(ctx, candidate);
    const auto* hit =
        rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, candidate.backend_autotune_key, runtime);
    if (!hit || hit->workspace_bytes != candidate.backend_workspace_required_bytes) {
      continue;
    }
    if (!best_hit || hit->measured_median_end_to_end_us < best_hit->measured_median_end_to_end_us) {
      best_hit = hit;
      best_plan = std::move(candidate);
    }
  }

  if (best_hit) {
    plan = std::move(best_plan);
    plan.backend_performance_validated = 1;
  }
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
  hash = workspace_fingerprint_mix(hash, plan.backend_workspace_required_bytes);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_selected_kernel);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library_version);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_capability_status);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_epilogue_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_workspace_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_isa_evidence);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_autotune_key);
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

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

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
        requested != RNS8_BACKEND_HIPBLASLT && requested != RNS8_BACKEND_CK &&
        requested != RNS8_BACKEND_ROCWMMA && requested != RNS8_BACKEND_WRAP64_BYTE_LIMB &&
        requested != RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (requested == RNS8_BACKEND_HIPBLASLT && is_per_tile_bound_kind(desc->bound_kind)) {
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
    configure_plan_backend_metadata(*plan, ctx);
    if (desc->requested_backend == RNS8_BACKEND_AUTO) {
      select_auto_backend_from_reviewed_cache(*ctx, *plan);
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
      if (is_per_tile_bound_kind(plan->desc.bound_kind) &&
          static_cast<uint64_t>(plan->tile_bounds.size()) == plan->schedule_tile_count) {
        const uint64_t bound = plan->tile_bounds[static_cast<std::size_t>(index)];
        const boost::multiprecision::cpp_int range = bounded_range_from_bound(plan->desc.semantics, bound);
        entries[index] = make_tile_schedule_entry(
            *plan,
            index,
            rns8::detail::required_prefix_for_range(range),
            plan->schedule_min_selected_prefix,
            0,
            rns8::detail::bit_length(range));
        continue;
      }
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

rns8_status rns8_get_plan_backend_info(const rns8_plan* plan, rns8_plan_backend_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_backend_capability_info capability{};
    capability.struct_size = sizeof(capability);
    capability.abi_version = RNS8_ABI_VERSION;
    fill_backend_capability_info(plan->backend, capability);

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->is_accelerator = capability.is_accelerator;
    out->is_correctness_backend = capability.is_correctness_backend;
    out->is_matrix_engine_backend = capability.is_matrix_engine_backend;
    out->compiled_kernel_available = capability.compiled_kernel_available;
    out->exact_differential_validated = capability.exact_differential_validated;
    out->performance_validated = plan->backend_performance_validated;
    out->flags = capability.flags;
    out->workspace_required_bytes = plan->backend_workspace_required_bytes;
    set_text(out->selected_kernel, sizeof(out->selected_kernel), plan->backend_selected_kernel);
    set_text(out->accelerator_library, sizeof(out->accelerator_library), plan->backend_library);
    set_text(out->accelerator_version, sizeof(out->accelerator_version), plan->backend_library_version);
    set_text(out->capability_status, sizeof(out->capability_status), plan->backend_capability_status);
    set_text(out->epilogue_mode, sizeof(out->epilogue_mode), plan->backend_epilogue_mode);
    set_text(out->workspace_mode, sizeof(out->workspace_mode), plan->backend_workspace_mode);
    set_text(out->isa_evidence, sizeof(out->isa_evidence), plan->backend_isa_evidence);
    set_text(out->autotune_key, sizeof(out->autotune_key), plan->backend_autotune_key);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_plan_packing_info(const rns8_plan* plan, rns8_plan_packing_info* out) {
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
    out->backend = plan->backend;
    out->semantics = plan->desc.semantics;
    out->uses_resident_matrix_inputs = 1;
    out->reusable_prepack_cache_available = 0;
    out->production_prepack_cache_available = 0;
    out->flags = 0;
    out->input_domain = output_domain_for_plan(*plan);
    out->output_domain = output_domain_for_plan(*plan);
    out->output_host_current = plan_output_device_current(*plan) ? 0u : 1u;
    out->output_device_current = plan_output_device_current(*plan) ? 1u : 0u;
    out->next_op_flags = next_op_flags_for_plan(*plan);
    out->reserved0 = 0;
    set_text(out->input_domain_name, sizeof(out->input_domain_name), output_domain_name(out->input_domain));
    set_text(out->output_domain_name, sizeof(out->output_domain_name), output_domain_name(out->output_domain));
    set_text(out->next_op_hint, sizeof(out->next_op_hint), next_op_hint_for_plan(*plan));

    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!hipblaslt_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->accumulator_workspace_bytes,
              out->library_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "hipblaslt_a_transposed_centered_i8_mk16_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "hipblaslt_b_transposed_centered_i8_nk16_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
      set_text(
          out->detail,
          sizeof(out->detail),
          "hipBLASLt uses aligned INT8 matrix-engine pack layouts, INT32 scratch, and workspace-local repeated-operand caches; no reusable production prepack cache.");
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_CK) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!ck_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->accumulator_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "ck_b_canonical_colmajor_i8_n64_kblock32768_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
      set_text(
          out->detail,
          sizeof(out->detail),
          "CK packs A/B into transient canonical INT8 workspaces and uses temporary centered output storage; no reusable production prepack cache.");
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_ROCWMMA) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!rocwmma_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "rocwmma_a_rowmajor_i8_m16_kblock65536_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "rns_i8_tile_swizzled_b_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      if (rocwmma_b_prepack_cache_supported(*plan)) {
        out->reusable_prepack_cache_available = 1;
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "reusable_b_prepack_cache");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA supports a reusable B prepack cache for this non-tiled RNS plan; A remains transient per dispatch and no production cache is reported.");
      } else {
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA packs A/B into transient INT8 matrix-engine workspaces; no reusable production prepack cache.");
      }
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      set_text(out->a_layout_version, sizeof(out->a_layout_version), native_layout_version_for_semantics(plan->desc.semantics));
      set_text(out->b_layout_version, sizeof(out->b_layout_version), native_layout_version_for_semantics(plan->desc.semantics));
      set_text(
          out->output_layout_version,
          sizeof(out->output_layout_version),
          native_layout_version_for_semantics(plan->desc.semantics));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "native_device_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Native vector-ALU backend consumes compact bounded i64/u64 device storage directly and exports native C without CRT reconstruction.");
      return RNS8_SUCCESS;
    }

    set_text(out->a_layout_version, sizeof(out->a_layout_version), persistent_layout_version_for_plan(*plan));
    set_text(out->b_layout_version, sizeof(out->b_layout_version), persistent_layout_version_for_plan(*plan));
    set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "device_resident_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Direct HIP consumes persistent device-resident matrix storage directly; no transient matrix-engine pack workspace or reusable prepack cache.");
    } else if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "host_byte_limb_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Strict wrap64 byte-limb reference consumes persistent byte-limb storage directly; no matrix-engine prepack cache.");
    } else {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "host_resident_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "CPU reference consumes persistent host-resident storage directly; no matrix-engine prepack cache.");
    }
    return RNS8_SUCCESS;
  });
}
