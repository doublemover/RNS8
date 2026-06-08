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
        requested != RNS8_BACKEND_HIP_VECTOR_ALU_INT64 && requested != RNS8_BACKEND_AMDGPU_BUILTINS) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if ((requested == RNS8_BACKEND_CK || requested == RNS8_BACKEND_ROCWMMA ||
         requested == RNS8_BACKEND_AMDGPU_BUILTINS) &&
        uses_finite_storage(desc->semantics) &&
        !rns8::detail::finite_u8::static_byte_modulus_supported(desc->finite_modulus)) {
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
    if (plan->desc.bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K &&
        !rns8::detail::input_range_output_bound(plan->desc, plan->desc.bound)) {
      delete plan;
      return RNS8_RANGE_ERROR;
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
    out->bound_kind = plan->desc.bound_kind;
    out->effective_bound = plan->desc.bound;
    out->lhs_bound = plan->desc.lhs_bound;
    out->rhs_bound = plan->desc.rhs_bound;
    out->zero_a_row_count = plan->zero_a_row_count;
    out->zero_b_col_count = plan->zero_b_col_count;
    out->zero_row_col_product_count = plan->zero_row_col_product_count;
    set_text(out->bound_contract, sizeof(out->bound_contract), bound_contract_name_for_plan(*plan));
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
            rns8::detail::bit_length(range),
            0);
        continue;
      }
      entries[index] = make_tile_schedule_entry(
          *plan,
          index,
          plan->schedule_min_required_prefix,
          plan->schedule_min_selected_prefix,
          0,
          plan->schedule_range_bit_length,
          0);
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
    out->accumulator_k_block_size = plan->backend_accumulator_k_block_size;
    out->accumulator_k_block_cap = plan->backend_accumulator_k_block_cap;
    out->accumulator_modulus = plan->backend_accumulator_modulus;
    out->accumulator_max_lhs_abs = plan->backend_accumulator_max_lhs_abs;
    out->accumulator_max_rhs_abs = plan->backend_accumulator_max_rhs_abs;
    out->accumulator_max_product = plan->backend_accumulator_max_product;
    out->accumulator_uses_int32_inner_product = plan->backend_accumulator_uses_int32_inner_product;
    out->accumulator_safe_for_k_block = plan->backend_accumulator_safe_for_k_block;
    set_text(out->selected_kernel, sizeof(out->selected_kernel), plan->backend_selected_kernel);
    set_text(out->accelerator_library, sizeof(out->accelerator_library), plan->backend_library);
    set_text(out->accelerator_version, sizeof(out->accelerator_version), plan->backend_library_version);
    set_text(out->capability_status, sizeof(out->capability_status), plan->backend_capability_status);
    set_text(out->epilogue_mode, sizeof(out->epilogue_mode), plan->backend_epilogue_mode);
    set_text(out->workspace_mode, sizeof(out->workspace_mode), plan->backend_workspace_mode);
    set_text(out->isa_evidence, sizeof(out->isa_evidence), plan->backend_isa_evidence);
    set_text(out->autotune_key, sizeof(out->autotune_key), plan->backend_autotune_key);
    set_text(
        out->accumulator_input_domain,
        sizeof(out->accumulator_input_domain),
        plan->backend_accumulator_input_domain);
    set_text(out->accumulator_signedness, sizeof(out->accumulator_signedness), plan->backend_accumulator_signedness);
    set_text(out->accumulator_type, sizeof(out->accumulator_type), plan->backend_accumulator_type);
    set_text(
        out->accumulator_modulus_policy,
        sizeof(out->accumulator_modulus_policy),
        plan->backend_accumulator_modulus_policy);
    set_text(
        out->accumulator_safety_status,
        sizeof(out->accumulator_safety_status),
        plan->backend_accumulator_safety_status);
    return RNS8_SUCCESS;
  });
}

bool grouped_dispatch_contract_semantics_supported(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
         semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
         semantics == RNS8_FINITE_FIELD_U8 || semantics == RNS8_FINITE_RING_U8;
}

const char* grouped_dispatch_contract_output_domain(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_BOUNDED_U64:
      return "native_i64_u64_host";
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_limb_host";
    case RNS8_FINITE_FIELD_U8:
    case RNS8_FINITE_RING_U8:
      return "finite_u8_host";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap64_u64_host";
  }
  return "unknown";
}

rns8_status rns8_get_grouped_dispatch_contract_info(
    const rns8_plan* plan,
    uint32_t task_count,
    rns8_grouped_dispatch_contract_info* out) {
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
    out->task_count = task_count;

    out->same_shape_required = 1;
    out->same_semantics_required = 1;
    out->compact_row_major_required = 1;
    out->unique_matrix_handles_required = 1;
    out->unique_workspace_handles_required = 1;
    out->same_plan_fingerprint_required = 1;
    out->same_device_required = 1;
    out->device_resident_inputs_required = 1;
    out->per_task_source_versions_required = 1;
    out->produces_device_current_output = 1;
    out->final_export_required_for_host_output = 1;
    out->per_task_status_required = 1;
    out->checksum_policy_required = 1;
    out->descriptor_reuse_validated = 1;
    out->flags = RNS8_GROUPED_DISPATCH_CONTRACT_SAME_SHAPE_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_COMPACT_ROW_MAJOR_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_UNIQUE_TASK_HANDLES_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_ONE_WORKSPACE_PER_TASK_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_SAME_DEVICE_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_PER_TASK_SOURCE_VERSION_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_DEVICE_CURRENT_OUTPUT_REQUIRED |
                 RNS8_GROUPED_DISPATCH_CONTRACT_PUBLIC_PACK_EXPLICIT |
                 RNS8_GROUPED_DISPATCH_CONTRACT_PUBLIC_EXPORT_EXPLICIT |
                 RNS8_GROUPED_DISPATCH_CONTRACT_GENERIC_BUCKETING_UNAVAILABLE |
                 RNS8_GROUPED_DISPATCH_CONTRACT_AUTO_ROUTING_DISABLED;
    set_text(out->descriptor_layout, sizeof(out->descriptor_layout), "same_shape_resident_task_triplets_v1");
    set_text(out->bucket_policy, sizeof(out->bucket_policy), "single_same_shape_bucket");
    set_text(
        out->source_version_policy,
        sizeof(out->source_version_policy),
        "per_task_monotonic_source_version_repack");
    set_text(out->workspace_policy, sizeof(out->workspace_policy), "one_workspace_per_task_shared_plan");
    set_text(
        out->matrix_ownership_policy,
        sizeof(out->matrix_ownership_policy),
        "caller_owns_unique_task_triplets_until_grouped_call_returns");
    set_text(
        out->descriptor_reuse_policy,
        sizeof(out->descriptor_reuse_policy),
        "reuse_after_shape_workspace_source_validation");
    set_text(
        out->stride_policy,
        sizeof(out->stride_policy),
        "matrix_ld_matches_logical_shape_host_output_ld_explicit");
    set_text(
        out->output_currentness_policy,
        sizeof(out->output_currentness_policy),
        "device_residue_current_after_grouped_gemm_host_output_after_export");
    set_text(
        out->lifetime_policy,
        sizeof(out->lifetime_policy),
        "task_matrices_and_workspaces_valid_until_grouped_call_returns");
    set_text(out->checksum_policy, sizeof(out->checksum_policy), "combined_per_task_checksum_u64");
    set_text(out->status_policy, sizeof(out->status_policy), "fail_fast_per_task_operation_status");
    set_text(
        out->device_descriptor_policy,
        sizeof(out->device_descriptor_policy),
        "device_pointer_tables_and_compact_slabs");
    set_text(
        out->output_domain,
        sizeof(out->output_domain),
        grouped_dispatch_contract_output_domain(plan->desc.semantics));

    if (plan->backend != RNS8_BACKEND_HIP_DIRECT) {
      set_text(out->unsupported_reason, sizeof(out->unsupported_reason), "backend_not_hip_direct");
    } else if (!direct_hip_compiled()) {
      set_text(out->unsupported_reason, sizeof(out->unsupported_reason), "direct_hip_not_compiled");
    } else if (task_count <= 1) {
      set_text(out->unsupported_reason, sizeof(out->unsupported_reason), "task_count_must_exceed_one");
    } else if (!grouped_dispatch_contract_semantics_supported(plan->desc.semantics)) {
      set_text(out->unsupported_reason, sizeof(out->unsupported_reason), "unsupported_grouped_semantics");
    } else if (plan->desc.m <= 0 || plan->desc.n <= 0 || plan->desc.k <= 0) {
      set_text(out->unsupported_reason, sizeof(out->unsupported_reason), "invalid_plan_shape");
    } else {
      out->descriptor_contract_supported = 1;
      out->public_execution_available = 1;
    }

    if (out->descriptor_contract_supported) {
      set_text(
          out->detail,
          sizeof(out->detail),
          "Direct-HIP same-shape grouped resident GEMM is public for this plan; callers must pack inputs before the grouped call and export outputs after it; generic bucketing and AUTO routing remain disabled.");
    } else {
      out->flags |= RNS8_GROUPED_DISPATCH_CONTRACT_BENCHMARK_ONLY_EXECUTION;
      set_text(
          out->detail,
          sizeof(out->detail),
          "Grouped dispatch remains unavailable for this plan; public pack/export batching, generic bucketing, and AUTO routing are not exposed.");
    }
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
#if RNS8_CK_USE_XDL
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "ck_a_centered_rowmajor_i8_m128_kblock32768_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "ck_b_centered_colmajor_i8_n128_kblock32768_v1");
#elif RNS8_CK_SHAPE_ALIGNMENT == 128
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "ck_a_canonical_rowmajor_i8_m128_kblock32768_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "ck_b_canonical_colmajor_i8_n128_kblock32768_v1");
#else
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "ck_b_canonical_colmajor_i8_n64_kblock32768_v1");
#endif
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
      set_text(
          out->detail,
          sizeof(out->detail),
#if RNS8_CK_USE_XDL
          "CK packs A/B into transient centered signed INT8 workspaces and uses temporary centered output storage; no reusable production prepack cache.");
#else
          "CK packs A/B into transient canonical INT8 workspaces and uses temporary centered output storage; no reusable production prepack cache.");
#endif
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
        out->production_prepack_cache_available = 1;
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "production_reusable_b_prepack_cache");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA supports the production B-side prepack cache for this non-tiled RNS plan; A remains transient per dispatch and cache keys reject stale source versions, target, shape, prefix, and layout mismatches.");
      } else {
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA packs A/B into transient INT8 matrix-engine workspaces; no reusable production prepack cache.");
      }
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      out->uses_transient_pack_workspace = 0;
      out->uses_matrix_engine_pack_layout = 1;
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "amdgpu_builtin_a_centered_i8_matrix_core_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "amdgpu_builtin_b_centered_i8_matrix_core_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "resident_dense_direct_or_explicit_sparse_a_handle");
      set_text(
          out->detail,
          sizeof(out->detail),
          "AMDGPU builtin dense kernels read resident RNS/finite byte planes directly; sparse-A uses a separate public sparse handle and is not inferred from dense inputs.");
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
