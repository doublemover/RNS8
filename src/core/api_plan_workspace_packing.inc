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
  uint64_t active_entry_count = 0;
  for (const auto& entry : plan.tile_schedule) {
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
      continue;
    }
    if (entry.selected_prefix > std::numeric_limits<uint64_t>::max() - active_entry_count) {
      return std::numeric_limits<uint64_t>::max();
    }
    active_entry_count += entry.selected_prefix;
  }
  const uint64_t schedule_entries =
      hip_direct_gemm_requires_device_tile_schedule(plan) ? static_cast<uint64_t>(plan.tile_schedule.size()) : 0;
  if (schedule_entries > std::numeric_limits<uint64_t>::max() - active_entry_count) {
    return std::numeric_limits<uint64_t>::max();
  }
  const uint64_t total_entries = schedule_entries + active_entry_count;
  if (total_entries > std::numeric_limits<uint64_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return total_entries * sizeof(rns8_plan_tile_schedule_entry);
}

bool hip_direct_gemm_requires_device_tile_schedule(const rns8_plan& plan) {
  if (plan.backend != RNS8_BACKEND_HIP_DIRECT || plan.tile_schedule.empty()) {
    return false;
  }
  bool saw_zero_output_tile = false;
  bool every_tile_zero_output = true;
  uint32_t uniform_zero_selected_prefix = 0;
  for (const auto& entry : plan.tile_schedule) {
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
      every_tile_zero_output = false;
      continue;
    }
    saw_zero_output_tile = true;
    if (uniform_zero_selected_prefix == 0) {
      uniform_zero_selected_prefix = entry.selected_prefix;
    } else if (uniform_zero_selected_prefix != entry.selected_prefix) {
      every_tile_zero_output = false;
    }
  }
  return saw_zero_output_tile && !(every_tile_zero_output && uniform_zero_selected_prefix != 0);
}

bool plan_all_zero_output_tiles(const rns8_plan& plan) {
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
  key += ";target_id=";
  key += target_id_for_plan_key(plan);
  key += ";semantics=";
  key += semantics_name_for_key(plan.desc.semantics);
  key += ";m=" + std::to_string(plan.desc.m);
  key += ";n=" + std::to_string(plan.desc.n);
  key += ";k=" + std::to_string(plan.desc.k);
  key += ";bound_kind=";
  key += bound_kind_name_for_key(plan.desc.bound_kind);
  if (plan.desc.bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K) {
    key += ";effective_bound=" + std::to_string(plan.desc.bound);
    key += ";lhs_bound=" + std::to_string(plan.desc.lhs_bound);
    key += ";rhs_bound=" + std::to_string(plan.desc.rhs_bound);
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    key += ";finite_modulus=" + std::to_string(plan.desc.finite_modulus);
  }
  key += ";prefix=" + std::to_string(plan.prefix);
  if (uses_rns_storage(plan.desc.semantics)) {
    key += ";requested_max_prefix=" + std::to_string(plan.desc.max_prefix);
    key += ";prefix_policy=";
    if ((plan.desc.flags & RNS8_PLAN_FORCE_FIXED_PREFIX) != 0) {
      key += "fixed_requested";
    } else if (is_per_tile_bound_kind(plan.desc.bound_kind)) {
      key += "per_tile_minimum";
    } else {
      key += "minimum_proven";
    }
  }
  key += ";tile_m=" + std::to_string(plan.desc.tile_m);
  key += ";tile_n=" + std::to_string(plan.desc.tile_n);
  key += ";groups=" + std::to_string(plan.schedule_prefix_group_count);
  key += ";adaptive_prefix=" + std::to_string(plan.schedule_adaptive_prefix_active);
  key += ";adaptive_skip=" + std::to_string(plan.schedule_adaptive_skip_active);
  key += ";schedule_flags=" + std::to_string(plan.schedule_flags);
  key += ";zero_a_rows=" + std::to_string(plan.zero_a_row_count);
  key += ";zero_b_cols=" + std::to_string(plan.zero_b_col_count);
  key += ";zero_row_col_products=" + std::to_string(plan.zero_row_col_product_count);
  uint64_t zero_output_tile_count = 0;
  for (const auto& entry : plan.tile_schedule) {
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
      ++zero_output_tile_count;
    }
  }
  key += ";zero_output_tiles=" + std::to_string(zero_output_tile_count);
  key += ";accumulator_type=" + plan.backend_accumulator_type;
  key += ";accumulator_signedness=" + plan.backend_accumulator_signedness;
  key += ";accumulator_modulus_policy=" + plan.backend_accumulator_modulus_policy;
  key += ";k_block_size=" + std::to_string(plan.backend_accumulator_k_block_size);
  key += ";k_block_cap=" + std::to_string(plan.backend_accumulator_k_block_cap);
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
  if (!plan.backend_target_id.empty()) {
    runtime.target_id = plan.backend_target_id;
  } else if (hip_device_backend(plan.backend) && concrete_device_target_id(ctx.device_info.gcn_arch)) {
    runtime.target_id = ctx.device_info.gcn_arch;
  } else if (!hip_device_backend(plan.backend) || !ctx.device_info.hip_available) {
    runtime.target_id = "cpu";
  } else {
    runtime.target_id = "unknown";
  }
  runtime.hip_sdk_or_library_version = plan.backend_library_version;
  return runtime;
}

