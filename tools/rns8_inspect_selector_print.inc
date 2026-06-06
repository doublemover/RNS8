void print_selector_shadow_text(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot) {
  std::cout << "selector_shadow_source: private_inspect_selector_shadow\n";
  std::cout << "selector_shadow_family: " << selector_family_for_backend(info.backend) << "\n";
  std::cout << "selector_shadow_boundary: exact_cache_selection_unchanged_family_advisory_only\n";
  std::cout << "selector_shadow_recommendation: "
            << selector_shadow_recommendation(info.backend, capability, snapshot) << "\n";
  std::cout << "selector_shadow_cache_loaded: " << (snapshot && snapshot->loaded ? 1 : 0) << "\n";
  std::cout << "selector_shadow_blockers: unsupported semantics; per-tile unsupported; backend not compiled; "
               "probe failed; no exact entry; unvalidated entry; identity/runtime mismatch; workspace mismatch; "
               "slower than selected\n";
}

void print_selector_shadow_json(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    bool trailing_comma) {
  std::cout << "  \"selector_shadow\": {\n";
  std::cout << "    \"source\": \"private_inspect_selector_shadow\",\n";
  std::cout << "    \"family_recommendation\": \""
            << json_escape(selector_shadow_recommendation(info.backend, capability, snapshot)) << "\",\n";
  std::cout << "    \"selected_family\": \"" << selector_family_for_backend(info.backend) << "\",\n";
  std::cout << "    \"conservative_family_boundary\": \"exact_cache_selection_unchanged_family_advisory_only\",\n";
  std::cout << "    \"cache_loaded\": " << (snapshot && snapshot->loaded ? "true" : "false") << ",\n";
  std::cout << "    \"blocker_vocabulary\": ["
            << "\"unsupported semantics\", "
            << "\"per-tile unsupported\", "
            << "\"backend not compiled\", "
            << "\"probe failed\", "
            << "\"no exact entry\", "
            << "\"unvalidated entry\", "
            << "\"identity/runtime mismatch\", "
            << "\"workspace mismatch\", "
            << "\"slower than selected\"]\n";
  std::cout << "  }" << (trailing_comma ? "," : "") << "\n";
}

void print_text(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    const rns8::detail::PlanLoweringDescription* plan_lowering,
    bool show_autotune_cache,
    bool selector_shadow) {
  std::cout << "RNS8 inspect\n";
  std::cout << "backend:       " << backend_name(info.backend) << "\n";
  std::cout << "device_id:     " << info.device_id << "\n";
  std::cout << "name:          " << info.name << "\n";
  std::cout << "gcn_arch:      " << info.gcn_arch << "\n";
  std::cout << "hip_available: " << info.hip_available << "\n";
  std::cout << "hip_runtime:   " << info.hip_runtime_version << "\n";
  std::cout << "hip_driver:    " << info.hip_driver_version << "\n";
  std::cout << "global_mem:    " << info.global_mem_bytes << "\n";
  std::cout << "detail:        " << info.detail << "\n";
  print_capability_text(capability);
  if (snapshot) {
    print_autotune_text(
        *snapshot,
        autotune_key,
        backend_name(info.backend),
        runtime,
        plan_packing,
        plan_lowering,
        show_autotune_cache);
  }
  if (selector_shadow) {
    print_selector_shadow_text(info, capability, snapshot);
  }
}

void print_json(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    const rns8::detail::PlanLoweringDescription* plan_lowering,
    bool show_autotune_cache,
    bool selector_shadow) {
  std::cout << "{\n";
  std::cout << "  \"backend\": \"" << backend_name(info.backend) << "\",\n";
  std::cout << "  \"device_id\": " << info.device_id << ",\n";
  std::cout << "  \"name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "  \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "  \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "  \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "  \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "  \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  std::cout << "  \"detail\": \"" << json_escape(info.detail) << "\",\n";
  print_capability_json(capability, snapshot != nullptr || selector_shadow);
  if (snapshot) {
    print_autotune_json(
        *snapshot,
        autotune_key,
        backend_name(info.backend),
        runtime,
        plan_packing,
        plan_lowering,
        show_autotune_cache);
    if (selector_shadow) {
      std::cout << ",\n";
    }
  }
  if (selector_shadow) {
    print_selector_shadow_json(info, capability, snapshot, false);
  }
  std::cout << "}\n";
}

}  // namespace
