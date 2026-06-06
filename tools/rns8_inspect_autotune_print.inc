void print_autotune_text(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    const rns8::detail::PlanLoweringDescription* plan_lowering,
    bool show_entries) {
  std::cout << "autotune_cache_path:   " << snapshot.path.string() << "\n";
  std::cout << "autotune_cache_loaded: " << (snapshot.loaded ? 1 : 0) << "\n";
  std::cout << "autotune_cache_exists: " << (snapshot.exists ? 1 : 0) << "\n";
  std::cout << "autotune_entry_count:  " << snapshot.entries.size() << "\n";
  if (!snapshot.error.empty()) {
    std::cout << "autotune_cache_error:  " << snapshot.error << "\n";
  }
  if (!autotune_key.empty()) {
    const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
    std::cout << "autotune_key:          " << autotune_key << "\n";
    std::cout << "autotune_exact_hit:    " << (hit ? 1 : 0) << "\n";
    std::cout << "autotune_runtime_target:  " << runtime.target_id << "\n";
    std::cout << "autotune_runtime_version: " << runtime.hip_sdk_or_library_version << "\n";
    std::cout << "selection_rationale:   "
              << rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend, runtime)
              << "\n";
    if (hit) {
      std::cout << "autotune_backend:      " << hit->selected_backend << "\n";
      std::cout << "autotune_kernel:       " << hit->selected_kernel << "\n";
      std::cout << "autotune_median_e2e:   " << hit->measured_median_end_to_end_us << "\n";
    }
    if (plan_packing) {
      print_plan_packing_text(*plan_packing);
    }
    if (plan_lowering) {
      print_plan_lowering_text(*plan_lowering);
    }
  }
  if (show_entries) {
    for (const auto& entry : snapshot.entries) {
      std::cout << "autotune_entry:        " << entry.selected_backend << " " << entry.selected_kernel
                << " " << entry.key << "\n";
    }
  }
}

void print_autotune_json(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    const rns8::detail::PlanLoweringDescription* plan_lowering,
    bool show_entries) {
  const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
  std::cout << "  \"autotune_cache\": {\n";
  std::cout << "    \"path\": \"" << json_escape(snapshot.path.string()) << "\",\n";
  std::cout << "    \"loaded\": " << (snapshot.loaded ? "true" : "false") << ",\n";
  std::cout << "    \"exists\": " << (snapshot.exists ? "true" : "false") << ",\n";
  std::cout << "    \"schema_version\": " << snapshot.schema_version << ",\n";
  std::cout << "    \"entry_count\": " << snapshot.entries.size() << ",\n";
  std::cout << "    \"error\": \"" << json_escape(snapshot.error) << "\",\n";
  std::cout << "    \"queried_key\": \"" << json_escape(autotune_key) << "\",\n";
  std::cout << "    \"exact_hit\": " << (hit ? "true" : "false") << ",\n";
  std::cout << "    \"runtime_target_id\": \"" << json_escape(runtime.target_id) << "\",\n";
  std::cout << "    \"runtime_version\": \"" << json_escape(runtime.hip_sdk_or_library_version) << "\",\n";
  std::cout << "    \"selection_rationale\": \""
            << json_escape(rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend, runtime))
            << "\",\n";
  std::cout << "    \"selector_report\": {\n";
  std::cout << "      \"source\": \"private_inspect_selector_report\",\n";
  std::cout << "      \"selected_backend\": \"" << json_escape(selected_backend) << "\",\n";
  std::cout << "      \"selected_key\": \"" << json_escape(autotune_key) << "\",\n";
  std::cout << "      \"cache_load_state\": \"" << (snapshot.loaded ? "loaded" : snapshot.exists ? "failed" : "missing")
            << "\",\n";
  std::cout << "      \"runtime_target_id\": \"" << json_escape(runtime.target_id) << "\",\n";
  std::cout << "      \"runtime_version\": \"" << json_escape(runtime.hip_sdk_or_library_version) << "\",\n";
  std::cout << "      \"fallback_reason\": \""
            << json_escape(rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend, runtime))
            << "\",\n";
  std::cout << "      \"rejection_reason_vocabulary\": ["
            << "\"unsupported semantics\", "
            << "\"per-tile unsupported\", "
            << "\"backend not compiled\", "
            << "\"probe failed\", "
            << "\"no exact entry\", "
            << "\"unvalidated entry\", "
            << "\"identity/runtime mismatch\", "
            << "\"workspace mismatch\", "
            << "\"slower than selected\"]\n";
  std::cout << "    }";
  if (plan_packing) {
    std::cout << ",\n";
    print_plan_packing_json(*plan_packing, plan_lowering != nullptr || hit != nullptr || show_entries);
  }
  if (plan_lowering) {
    if (!plan_packing) {
      std::cout << ",\n";
    }
    print_plan_lowering_json(*plan_lowering, hit != nullptr || show_entries);
  }
  if (hit) {
    if (!plan_packing && !plan_lowering) {
      std::cout << ",\n";
    }
    std::cout << "    \"entry\": {\n";
    std::cout << "      \"selected_backend\": \"" << json_escape(hit->selected_backend) << "\",\n";
    std::cout << "      \"selected_kernel\": \"" << json_escape(hit->selected_kernel) << "\",\n";
    std::cout << "      \"target_id\": \"" << json_escape(hit->target_id) << "\",\n";
    std::cout << "      \"validation_status\": \"" << json_escape(hit->validation_status) << "\",\n";
    std::cout << "      \"performance_validated\": " << (hit->performance_validated ? "true" : "false") << ",\n";
    std::cout << "      \"measured_median_end_to_end_us\": " << hit->measured_median_end_to_end_us << "\n";
    std::cout << "    }";
  }
  if (show_entries) {
    std::cout << ",\n";
    std::cout << "    \"entries\": [";
    for (std::size_t i = 0; i < snapshot.entries.size(); ++i) {
      const auto& entry = snapshot.entries[i];
      std::cout << (i == 0 ? "\n" : ",\n");
      std::cout << "      {\"key\": \"" << json_escape(entry.key) << "\", \"selected_backend\": \""
                << json_escape(entry.selected_backend) << "\", \"selected_kernel\": \""
                << json_escape(entry.selected_kernel) << "\"}";
    }
    if (!snapshot.entries.empty()) {
      std::cout << "\n    ";
    }
    std::cout << "]";
  }
  std::cout << "\n";
  std::cout << "  }\n";
}

const char* selector_family_for_backend(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_HIP_DIRECT:
      return "direct_hip_correctness_family";
    case RNS8_BACKEND_HIPBLASLT:
    case RNS8_BACKEND_CK:
    case RNS8_BACKEND_ROCWMMA:
      return "matrix_engine_accelerator_family";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "native_vector_alu_family";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64_byte_limb_family";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu_reference_family";
    case RNS8_BACKEND_AUTO:
      return "auto_exact_cache_family";
  }
  return "unknown_family";
}

std::string selector_shadow_recommendation(
    rns8_backend_kind backend,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot) {
  if (backend == RNS8_BACKEND_AUTO) {
    return "exact_cache_only";
  }
  if (!capability.compiled_kernel_available) {
    return "blocked_backend_not_compiled";
  }
  if (capability.candidate_evidence_only && (!snapshot || !snapshot->loaded)) {
    return "blocked_unvalidated_entry";
  }
  if (capability.is_accelerator) {
    return "recommend_only_with_reviewed_same_contract_margin";
  }
  return "conservative_supported_family";
}

