
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

