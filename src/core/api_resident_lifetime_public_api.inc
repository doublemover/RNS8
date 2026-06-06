rns8_output_domain resident_current_output_domain(const rns8_matrix& matrix) {
  if (matrix.host_byte_limbs_current || matrix.device_byte_limbs_current) {
    return RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB;
  }
  if (matrix.host_native_current || matrix.device_native_current) {
    return RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64;
  }
  if (uses_finite_storage(matrix.desc.semantics) &&
      (matrix.host_residues_current || matrix.device_residues_current)) {
    return RNS8_OUTPUT_DOMAIN_FINITE_U8;
  }
  return RNS8_OUTPUT_DOMAIN_RNS_RESIDUE;
}

const char* resident_current_domain_name(const rns8_matrix& matrix) {
  if (matrix.host_byte_limbs_current || matrix.device_byte_limbs_current) {
    return matrix.device_byte_limbs_current ? "wrap64_byte_limb_device_current"
                                            : "wrap64_byte_limb_host_current";
  }
  if (matrix.host_native_current || matrix.device_native_current) {
    return matrix.device_native_current ? "native_i64_u64_device_current"
                                        : "native_i64_u64_host_current";
  }
  if (uses_finite_storage(matrix.desc.semantics)) {
    if (matrix.device_residues_current) {
      return "finite_u8_device_current";
    }
    if (matrix.host_residues_current) {
      return "finite_u8_host_current";
    }
    return "finite_u8_not_current";
  }
  if (matrix.device_residues_current) {
    return "rns_residue_device_current";
  }
  if (matrix.host_residues_current) {
    return "rns_residue_host_current";
  }
  return "no_current_output_domain";
}

const char* resident_matrix_role_name(rns8_resident_matrix_role role) {
  switch (role) {
    case RNS8_RESIDENT_MATRIX_ROLE_UNKNOWN:
      return "unknown";
    case RNS8_RESIDENT_MATRIX_ROLE_A:
      return "A";
    case RNS8_RESIDENT_MATRIX_ROLE_B:
      return "B";
    case RNS8_RESIDENT_MATRIX_ROLE_C:
      return "C";
  }
  return "invalid";
}

bool valid_resident_matrix_role(rns8_resident_matrix_role role) {
  return role == RNS8_RESIDENT_MATRIX_ROLE_UNKNOWN ||
         role == RNS8_RESIDENT_MATRIX_ROLE_A ||
         role == RNS8_RESIDENT_MATRIX_ROLE_B ||
         role == RNS8_RESIDENT_MATRIX_ROLE_C;
}

bool resident_plan_role_shape(
    const rns8_plan& plan,
    rns8_resident_matrix_role role,
    int64_t& rows,
    int64_t& cols) {
  switch (role) {
    case RNS8_RESIDENT_MATRIX_ROLE_A:
      rows = plan.desc.m;
      cols = plan.desc.k;
      return true;
    case RNS8_RESIDENT_MATRIX_ROLE_B:
      rows = plan.desc.k;
      cols = plan.desc.n;
      return true;
    case RNS8_RESIDENT_MATRIX_ROLE_C:
      rows = plan.desc.m;
      cols = plan.desc.n;
      return true;
    case RNS8_RESIDENT_MATRIX_ROLE_UNKNOWN:
      rows = 0;
      cols = 0;
      return false;
  }
  rows = 0;
  cols = 0;
  return false;
}

bool resident_matrix_matches_plan_role(
    const rns8_matrix& matrix,
    const rns8_plan& plan,
    rns8_resident_matrix_role role) {
  int64_t rows = 0;
  int64_t cols = 0;
  if (!resident_plan_role_shape(plan, role, rows, cols)) {
    return false;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  const rns8_bound_kind storage_bound_kind = storage_bound_kind_for_plan(plan);
  if (!matrix_descriptor_matches(
          matrix,
          plan.desc.semantics,
          storage_bound_kind,
          rows,
          cols,
          storage_prefix,
          plan.desc.tile_m,
          plan.desc.tile_n)) {
    return false;
  }
  if (matrix.backend != plan.backend) {
    return false;
  }
  if (uses_finite_storage(plan.desc.semantics) &&
      matrix.finite_modulus != 0 &&
      matrix.finite_modulus != plan.desc.finite_modulus) {
    return false;
  }
  return true;
}

bool resident_matrix_has_host_current(const rns8_matrix& matrix) {
  return matrix.host_residues_current || matrix.host_byte_limbs_current || matrix.host_native_current;
}

bool resident_matrix_has_device_current(const rns8_matrix& matrix) {
  return matrix.device_residues_current || matrix.device_byte_limbs_current || matrix.device_native_current;
}

bool resident_matrix_has_device_storage(const rns8_matrix& matrix) {
  return matrix.hip_residues || matrix.hip_byte_limbs || matrix.hip_native_i64 ||
         matrix.hip_native_u64;
}

rns8_status rns8_get_resident_lifetime_info(
    const rns8_matrix* matrix,
    const rns8_plan* plan,
    const rns8_workspace* workspace,
    rns8_resident_matrix_role matrix_role,
    rns8_resident_lifetime_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!matrix || !out || !valid_resident_matrix_role(matrix_role) || (workspace && !plan)) {
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
    out->matrix_role = matrix_role;
    out->current_output_domain = resident_current_output_domain(*matrix);
    out->rows = matrix->desc.rows;
    out->cols = matrix->desc.cols;
    out->logical_ld = matrix->desc.logical_ld;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->source_version = matrix->source_version;
    out->hip_device_id = static_cast<int32_t>(matrix->hip_device_id);
    out->source_version_valid = matrix->source_version != 0 ? 1u : 0u;
    out->host_current = resident_matrix_has_host_current(*matrix) ? 1u : 0u;
    out->device_current = resident_matrix_has_device_current(*matrix) ? 1u : 0u;
    out->device_resident = resident_matrix_has_device_storage(*matrix) ? 1u : 0u;
    out->plan_binding_available = plan ? 1u : 0u;
    out->workspace_binding_available = workspace ? 1u : 0u;

    if (plan) {
      out->plan_fingerprint = plan_workspace_fingerprint(*plan);
      out->plan_binding_matches = resident_matrix_matches_plan_role(*matrix, *plan, matrix_role) ? 1u : 0u;
      out->device_id_matches = 1u;
    } else {
      out->device_id_matches = 1u;
    }

    if (plan && workspace) {
      const bool identity_matches =
          workspace->backend == plan->backend &&
          workspace->semantics == plan->desc.semantics &&
          workspace->bound_kind == plan->desc.bound_kind &&
          workspace->m == plan->desc.m &&
          workspace->n == plan->desc.n &&
          workspace->k == plan->desc.k &&
          workspace->prefix == plan->prefix;
      out->workspace_fingerprint = workspace->schedule_fingerprint;
      out->workspace_identity_matches = identity_matches ? 1u : 0u;
      out->workspace_schedule_matches = workspace_schedule_contract_matches_plan(*plan, *workspace) ? 1u : 0u;
      out->workspace_backend_matches = workspace_backend_metadata_matches_plan(*plan, *workspace) ? 1u : 0u;
      out->workspace_binding_matches =
          out->workspace_identity_matches &&
          out->workspace_schedule_matches &&
          out->workspace_backend_matches &&
          out->plan_binding_matches ? 1u : 0u;
      if (hip_device_backend(plan->backend)) {
        out->device_id_matches =
            matrix->hip_device_id == workspace->hip_device_id ? 1u : 0u;
        out->workspace_binding_matches =
            out->workspace_binding_matches && out->device_id_matches ? 1u : 0u;
      }
    }

    const bool plan_allows_reuse = !plan || out->plan_binding_matches != 0;
    out->can_feed_rns_gemm =
        plan_allows_reuse && uses_rns_storage(matrix->desc.semantics) &&
        (matrix->host_residues_current || matrix->device_residues_current) ? 1u : 0u;
    out->can_feed_native_gemm =
        plan_allows_reuse &&
        (matrix->desc.semantics == RNS8_BOUNDED_I64 || matrix->desc.semantics == RNS8_BOUNDED_U64) &&
        (matrix->host_native_current || matrix->device_native_current) ? 1u : 0u;
    out->can_feed_final_export =
        (out->host_current || out->device_current) && plan_allows_reuse ? 1u : 0u;

    set_text(out->current_domain, sizeof(out->current_domain), resident_current_domain_name(*matrix));
    set_text(out->matrix_role_name, sizeof(out->matrix_role_name), resident_matrix_role_name(matrix_role));
    set_text(
        out->source_version_policy,
        sizeof(out->source_version_policy),
        "monotonic_per_import_pack_or_gemm_output_hash");
    set_text(
        out->workspace_binding_policy,
        sizeof(out->workspace_binding_policy),
        plan ? "plan_fingerprint_plus_workspace_identity_schedule_backend"
             : "not_bound_to_plan_or_workspace");
    set_text(
        out->output_currentness_policy,
        sizeof(out->output_currentness_policy),
        "explicit_host_or_device_current_domain_no_implicit_semantic_inference");
    set_text(
        out->lifetime_policy,
        sizeof(out->lifetime_policy),
        "caller_owned_matrix_handle_until_destroy_versioned_by_source_and_plan");
    const bool mismatch =
        (plan && !out->plan_binding_matches) ||
        (workspace && !out->workspace_binding_matches) ||
        (plan && hip_device_backend(plan->backend) && !out->device_id_matches);
    set_text(
        out->mismatch_policy,
        sizeof(out->mismatch_policy),
        mismatch
            ? "reject_source_version_descriptor_semantic_prefix_target_workspace_mismatch"
            : "source_version_descriptor_semantic_prefix_target_workspace_match");
    set_text(
        out->detail,
        sizeof(out->detail),
        workspace
            ? "Resident matrix role is bound to the supplied plan/workspace contract."
            : plan ? "Resident matrix role is checked against the supplied plan; workspace is unchecked."
                   : "Resident matrix lifetime is reported without plan/workspace binding.");
    return RNS8_SUCCESS;
  });
}
