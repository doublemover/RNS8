void configure_plan_backend_metadata(rns8_plan& plan, const rns8_context* ctx) {
  plan.validated_contract.reset();
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(plan.backend, capability);
  plan.backend_target_id = target_id_for_plan_context(plan, ctx);
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
  plan.backend_accumulator_k_block_size = accumulator_k_block_size_for_plan(plan);
  plan.backend_accumulator_k_block_cap = accumulator_k_block_cap_for_plan(plan);
  plan.backend_accumulator_modulus = accumulator_modulus_for_plan(plan);
  plan.backend_accumulator_max_lhs_abs = accumulator_max_abs_input_for_plan(plan);
  plan.backend_accumulator_max_rhs_abs = accumulator_max_abs_input_for_plan(plan);
  plan.backend_accumulator_max_product = accumulator_max_product_for_plan(plan);
  plan.backend_accumulator_uses_int32_inner_product =
      accumulator_uses_int32_inner_product_for_plan(plan) ? 1u : 0u;
  plan.backend_accumulator_safe_for_k_block = accumulator_safe_for_k_block_for_plan(plan);
  plan.backend_accumulator_input_domain = accumulator_input_domain_for_plan(plan);
  plan.backend_accumulator_signedness = accumulator_signedness_for_plan(plan);
  plan.backend_accumulator_type = accumulator_type_for_plan(plan);
  plan.backend_accumulator_modulus_policy = accumulator_modulus_policy_for_plan(plan);
  plan.backend_accumulator_safety_status = accumulator_safety_status_for_plan(plan);
  plan.arithmetic_guard = make_plan_arithmetic_guard(plan);
  plan.backend_autotune_key = build_autotune_key(plan);
}

bool prepare_auto_candidate_backend(rns8_context& ctx, rns8_backend_kind backend) {
  static_cast<void>(ctx);
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(backend, capability);
  // Inventory carries no validation receipt. The exact-match reviewed cache
  // below supplies scoped evidence after the backend has been probed.
  if (!capability.is_available || !capability.compiled_kernel_available) {
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
    case RNS8_BACKEND_AMDGPU_BUILTINS:
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
      return rns8::detail::amdgpu_builtins_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
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

static const char* auto_reason_text(uint32_t reason) {
  switch(reason){case RNS8_AUTO_UNSUPPORTED_SEMANTICS:return "unsupported semantics";
    case RNS8_AUTO_UNSUPPORTED_MODULUS:return "unsupported modulus";case RNS8_AUTO_PER_TILE_UNSUPPORTED:return "per-tile unsupported";
    case RNS8_AUTO_NOT_COMPILED:return "backend not compiled";case RNS8_AUTO_PROBE_FAILED:return "probe failed";
    case RNS8_AUTO_NO_EXACT_ENTRY:return "no exact entry";case RNS8_AUTO_UNVALIDATED_ENTRY:return "unvalidated entry";
    case RNS8_AUTO_IDENTITY_MISMATCH:return "identity/runtime mismatch";case RNS8_AUTO_WORKSPACE_MISMATCH:return "workspace mismatch";
    case RNS8_AUTO_SLOWER:return "slower than selected";case RNS8_AUTO_SELECTED:return "reviewed exact entry selected";
    case RNS8_AUTO_TIED:return "tied with selected";case RNS8_AUTO_ARITHMETIC_REJECTED:return "arithmetic proof rejected";default:return "baseline policy";}
}
void select_auto_backend_from_reviewed_cache(rns8_context& ctx, rns8_plan& plan) {
  rns8_auto_trace_info trace{};trace.struct_size=sizeof(trace);trace.abi_version=RNS8_ABI_VERSION;
  std::vector<rns8_auto_decision> decisions;
  rns8_auto_decision baseline{};baseline.struct_size=sizeof(baseline);baseline.abi_version=RNS8_ABI_VERSION;
  baseline.backend=plan.backend;baseline.disposition=2;baseline.reason_code=RNS8_AUTO_BASELINE_POLICY;
  set_text(baseline.reason,sizeof(baseline.reason),auto_reason_text(baseline.reason_code));decisions.push_back(baseline);
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !backend_supports_semantics(ctx.backend, plan.desc.semantics)) {
    set_text(trace.decision,sizeof(trace.decision),"baseline context; accelerator candidates not evaluated");
    plan.auto_trace=trace;plan.auto_decisions=std::move(decisions);return;
  }
  trace.selection_performed=1;
  const rns8_backend_kind candidates[] = {RNS8_BACKEND_HIP_VECTOR_ALU_INT64,RNS8_BACKEND_HIPBLASLT,
      RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_AMDGPU_BUILTINS};
  const auto snapshot = rns8::detail::read_autotune_cache();
  trace.cache_exists=snapshot.exists;trace.cache_loaded=snapshot.loaded;trace.cache_entry_count=snapshot.entries.size();
  set_text(trace.cache_path,sizeof(trace.cache_path),snapshot.path.string());
  const rns8::detail::AutotuneCacheEntry* best_hit = nullptr;
  rns8_plan best_plan{};size_t best_index=0;
  for (const auto candidate_backend:candidates) {
    rns8_auto_decision decision{};decision.struct_size=sizeof(decision);decision.abi_version=RNS8_ABI_VERSION;decision.backend=candidate_backend;
    const auto evaluate=[&]() -> uint32_t {
      if(!backend_supports_semantics(candidate_backend,plan.desc.semantics))return RNS8_AUTO_UNSUPPORTED_SEMANTICS;
      if((candidate_backend==RNS8_BACKEND_CK || candidate_backend==RNS8_BACKEND_ROCWMMA || candidate_backend==RNS8_BACKEND_AMDGPU_BUILTINS) &&
          uses_finite_storage(plan.desc.semantics) && !rns8::detail::finite_u8::static_byte_modulus_supported(plan.desc.finite_modulus))return RNS8_AUTO_UNSUPPORTED_MODULUS;
      if(candidate_backend==RNS8_BACKEND_HIPBLASLT && is_per_tile_bound_kind(plan.desc.bound_kind))return RNS8_AUTO_PER_TILE_UNSUPPORTED;
      rns8_backend_capability_info capability{};capability.struct_size=sizeof(capability);capability.abi_version=RNS8_ABI_VERSION;
      fill_backend_capability_info(candidate_backend,capability);
      if(!capability.is_available || !capability.compiled_kernel_available)return RNS8_AUTO_NOT_COMPILED;
      if(!prepare_auto_candidate_backend(ctx,candidate_backend))return RNS8_AUTO_PROBE_FAILED;
      rns8_plan candidate=plan;candidate.backend=candidate_backend;configure_plan_backend_metadata(candidate,&ctx);
      if(!candidate.arithmetic_guard || !candidate.arithmetic_guard->checked.admitted())return RNS8_AUTO_ARITHMETIC_REJECTED;
      decision.required_workspace_bytes=candidate.backend_workspace_required_bytes;
      const auto hash=rns8::detail::sha256(candidate.backend_autotune_key);set_text(decision.key_sha256,sizeof(decision.key_sha256),hash);
      const auto* exact=rns8::detail::find_exact_autotune_entry(snapshot,candidate.backend_autotune_key);
      decision.exact_entry_present=exact!=nullptr;if(!exact)return RNS8_AUTO_NO_EXACT_ENTRY;
      if(!rns8::detail::find_validated_autotune_entry(snapshot,candidate.backend_autotune_key))return RNS8_AUTO_UNVALIDATED_ENTRY;
      const auto runtime=autotune_runtime_identity_for_plan(ctx,candidate);
      const auto* hit=rns8::detail::find_validated_autotune_entry_for_runtime(snapshot,candidate.backend_autotune_key,runtime);
      decision.validated_entry_present=hit!=nullptr;if(!hit)return RNS8_AUTO_IDENTITY_MISMATCH;
      if(hit->workspace_bytes!=candidate.backend_workspace_required_bytes)return RNS8_AUTO_WORKSPACE_MISMATCH;
      decision.reviewed_median_us=hit->measured_median_end_to_end_us;
      if(!best_hit || hit->measured_median_end_to_end_us<best_hit->measured_median_end_to_end_us){
        best_hit=hit;best_plan=std::move(candidate);best_index=decisions.size();}
      return RNS8_AUTO_SLOWER; // final winner is marked after all actual candidates
    };
    decision.reason_code=evaluate();set_text(decision.reason,sizeof(decision.reason),auto_reason_text(decision.reason_code));decisions.push_back(decision);
  }
  if(best_hit){plan=std::move(best_plan);plan.backend_performance_validated=1;
    for(size_t i=1;i<decisions.size();++i)if(i!=best_index && decisions[i].reason_code==RNS8_AUTO_SLOWER &&
        decisions[i].reviewed_median_us==best_hit->measured_median_end_to_end_us){decisions[i].reason_code=RNS8_AUTO_TIED;set_text(decisions[i].reason,sizeof(decisions[i].reason),auto_reason_text(RNS8_AUTO_TIED));}
    decisions[best_index].disposition=1;decisions[best_index].reason_code=RNS8_AUTO_SELECTED;
    set_text(decisions[best_index].reason,sizeof(decisions[best_index].reason),auto_reason_text(RNS8_AUTO_SELECTED));
    trace.selected_exact_entry=1;trace.selected_validated_entry=1;set_text(trace.decision,sizeof(trace.decision),"reviewed exact entry selected");
  }else set_text(trace.decision,sizeof(trace.decision),"baseline retained; no admissible reviewed exact candidate");
  plan.auto_trace=trace;plan.auto_decisions=std::move(decisions);
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

