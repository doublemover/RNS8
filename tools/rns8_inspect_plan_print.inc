void print_capability_text(const rns8_backend_capability_info& capability) {
  std::cout << "capability_status: " << capability.status << "\n";
  std::cout << "selected_kernel:   " << capability.selected_kernel << "\n";
  std::cout << "accelerator:       " << capability.is_accelerator << "\n";
  std::cout << "correctness:       " << capability.is_correctness_backend << "\n";
  std::cout << "matrix_engine:     " << capability.is_matrix_engine_backend << "\n";
  std::cout << "compiled_kernel:   " << capability.compiled_kernel_available << "\n";
  std::cout << "exact_validated:   " << capability.exact_differential_validated << "\n";
  std::cout << "perf_validated:    " << capability.performance_validated << "\n";
  std::cout << "library:           " << capability.library_name << "\n";
  std::cout << "enable_flag:       " << capability.enable_flag << "\n";
  std::cout << "epilogue_mode:     " << capability.epilogue_mode << "\n";
  std::cout << "workspace_mode:    " << capability.workspace_mode << "\n";
  std::cout << "isa_evidence:      " << capability.isa_evidence << "\n";
  std::cout << "capability_detail: " << capability.detail << "\n";
}

void print_capability_json(const rns8_backend_capability_info& capability, bool trailing_comma) {
  std::cout << "  \"capability\": {\n";
  std::cout << "    \"backend\": \"" << backend_name(capability.backend) << "\",\n";
  std::cout << "    \"backend_name\": \"" << json_escape(capability.backend_name) << "\",\n";
  std::cout << "    \"status\": \"" << json_escape(capability.status) << "\",\n";
  std::cout << "    \"detail\": \"" << json_escape(capability.detail) << "\",\n";
  std::cout << "    \"selected_kernel\": \"" << json_escape(capability.selected_kernel) << "\",\n";
  std::cout << "    \"accelerator_backend\": " << (capability.is_accelerator ? "true" : "false") << ",\n";
  std::cout << "    \"correctness_backend\": " << (capability.is_correctness_backend ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_engine_backend\": " << (capability.is_matrix_engine_backend ? "true" : "false") << ",\n";
  std::cout << "    \"requires_feature_detection\": "
            << (capability.requires_feature_detection ? "true" : "false") << ",\n";
  std::cout << "    \"enable_flag_fail_fast\": "
            << (capability.enable_flag_fail_fast ? "true" : "false") << ",\n";
  std::cout << "    \"candidate_evidence_only\": "
            << (capability.candidate_evidence_only ? "true" : "false") << ",\n";
  std::cout << "    \"compiled_kernel_available\": "
            << (capability.compiled_kernel_available ? "true" : "false") << ",\n";
  std::cout << "    \"exact_differential_validated\": "
            << (capability.exact_differential_validated ? "true" : "false") << ",\n";
  std::cout << "    \"performance_validated\": " << (capability.performance_validated ? "true" : "false") << ",\n";
  std::cout << "    \"library_name\": \"" << json_escape(capability.library_name) << "\",\n";
  std::cout << "    \"library_version\": \"" << json_escape(capability.library_version) << "\",\n";
  std::cout << "    \"enable_flag\": \"" << json_escape(capability.enable_flag) << "\",\n";
  std::cout << "    \"epilogue_mode\": \"" << json_escape(capability.epilogue_mode) << "\",\n";
  std::cout << "    \"workspace_mode\": \"" << json_escape(capability.workspace_mode) << "\",\n";
  std::cout << "    \"isa_evidence\": \"" << json_escape(capability.isa_evidence) << "\"\n";
  std::cout << "  }" << (trailing_comma ? "," : "") << "\n";
}

void print_plan_packing_text(const rns8_plan_packing_info& packing) {
  std::cout << "autotune_plan_backend:            " << backend_name(packing.backend) << "\n";
  std::cout << "autotune_plan_semantics:          " << semantics_name(packing.semantics) << "\n";
  std::cout << "autotune_plan_resident_inputs:    " << packing.uses_resident_matrix_inputs << "\n";
  std::cout << "autotune_plan_transient_pack:     " << packing.uses_transient_pack_workspace << "\n";
  std::cout << "autotune_plan_matrix_engine_pack: " << packing.uses_matrix_engine_pack_layout << "\n";
  std::cout << "autotune_plan_reusable_prepack:   " << packing.reusable_prepack_cache_available << "\n";
  std::cout << "autotune_plan_production_prepack: " << packing.production_prepack_cache_available << "\n";
  std::cout << "autotune_plan_input_domain:       " << output_domain_name(packing.input_domain) << "\n";
  std::cout << "autotune_plan_output_domain:      " << output_domain_name(packing.output_domain) << "\n";
  std::cout << "autotune_plan_source_versioned:   "
            << ((packing.flags & RNS8_PLAN_PACKING_SOURCE_VERSIONED_INPUTS) ? 1 : 0) << "\n";
  std::cout << "autotune_plan_pack_elision:       "
            << ((packing.flags & RNS8_PLAN_PACKING_SAME_SOURCE_VERSION_PACK_ELISION) ? 1 : 0) << "\n";
  std::cout << "autotune_plan_packing_flags:      " << packing.flags << "\n";
  std::cout << "autotune_plan_next_op_flags:      " << packing.next_op_flags << "\n";
  std::cout << "autotune_plan_total_pack_bytes:   " << packing.total_transient_workspace_bytes << "\n";
  std::cout << "autotune_plan_prepack_scope:      " << packing.prepack_cache_scope << "\n";
  std::cout << "autotune_plan_next_op_hint:       " << packing.next_op_hint << "\n";
}

void print_plan_packing_json(const rns8_plan_packing_info& packing, bool trailing_comma) {
  std::cout << "    \"plan_packing\": {\n";
  std::cout << "      \"backend\": \"" << backend_name(packing.backend) << "\",\n";
  std::cout << "      \"semantics\": \"" << semantics_name(packing.semantics) << "\",\n";
  std::cout << "      \"uses_resident_matrix_inputs\": "
            << (packing.uses_resident_matrix_inputs ? "true" : "false") << ",\n";
  std::cout << "      \"uses_transient_pack_workspace\": "
            << (packing.uses_transient_pack_workspace ? "true" : "false") << ",\n";
  std::cout << "      \"uses_matrix_engine_pack_layout\": "
            << (packing.uses_matrix_engine_pack_layout ? "true" : "false") << ",\n";
  std::cout << "      \"reusable_prepack_cache_available\": "
            << (packing.reusable_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "      \"production_prepack_cache_available\": "
            << (packing.production_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "      \"input_domain\": \"" << json_escape(output_domain_name(packing.input_domain)) << "\",\n";
  std::cout << "      \"output_domain\": \"" << json_escape(output_domain_name(packing.output_domain)) << "\",\n";
  std::cout << "      \"input_domain_name\": \"" << json_escape(packing.input_domain_name) << "\",\n";
  std::cout << "      \"output_domain_name\": \"" << json_escape(packing.output_domain_name) << "\",\n";
  std::cout << "      \"output_host_current\": " << (packing.output_host_current ? "true" : "false") << ",\n";
  std::cout << "      \"output_device_current\": " << (packing.output_device_current ? "true" : "false") << ",\n";
  std::cout << "      \"source_versioned_inputs\": "
            << ((packing.flags & RNS8_PLAN_PACKING_SOURCE_VERSIONED_INPUTS) ? "true" : "false") << ",\n";
  std::cout << "      \"same_source_version_pack_elision_available\": "
            << ((packing.flags & RNS8_PLAN_PACKING_SAME_SOURCE_VERSION_PACK_ELISION) ? "true" : "false") << ",\n";
  std::cout << "      \"flags\": " << packing.flags << ",\n";
  std::cout << "      \"next_op_flags\": " << packing.next_op_flags << ",\n";
  std::cout << "      \"next_op_hint\": \"" << json_escape(packing.next_op_hint) << "\",\n";
  std::cout << "      \"a_pack_workspace_bytes\": " << packing.a_pack_workspace_bytes << ",\n";
  std::cout << "      \"b_pack_workspace_bytes\": " << packing.b_pack_workspace_bytes << ",\n";
  std::cout << "      \"accumulator_workspace_bytes\": " << packing.accumulator_workspace_bytes << ",\n";
  std::cout << "      \"library_workspace_bytes\": " << packing.library_workspace_bytes << ",\n";
  std::cout << "      \"total_transient_workspace_bytes\": " << packing.total_transient_workspace_bytes << ",\n";
  std::cout << "      \"a_layout_version\": \"" << json_escape(packing.a_layout_version) << "\",\n";
  std::cout << "      \"b_layout_version\": \"" << json_escape(packing.b_layout_version) << "\",\n";
  std::cout << "      \"output_layout_version\": \"" << json_escape(packing.output_layout_version) << "\",\n";
  std::cout << "      \"prepack_cache_scope\": \"" << json_escape(packing.prepack_cache_scope) << "\",\n";
  std::cout << "      \"detail\": \"" << json_escape(packing.detail) << "\"\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_plan_lowering_text(const rns8::detail::PlanLoweringDescription& lowering) {
  std::cout << "autotune_plan_lowering_operation:      " << lowering.operation << "\n";
  std::cout << "autotune_plan_lowering_semantics:      " << lowering.semantic_contract << "\n";
  std::cout << "autotune_plan_lowering_backend_family: " << lowering.backend_family << "\n";
  std::cout << "autotune_plan_lowering_desired_output: " << lowering.desired_output << "\n";
  std::cout << "autotune_plan_lowering_schedule:       " << lowering.schedule_strategy << "\n";
  std::cout << "autotune_plan_lowering_packing:        " << lowering.packing_strategy << "\n";
  std::cout << "autotune_plan_lowering_reuse:          " << lowering.reuse_strategy << "\n";
  std::cout << "autotune_plan_lowering_conversion:     " << lowering.conversion_strategy << "\n";
  std::cout << "autotune_plan_lowering_path:           " << lowering.lowering_path << "\n";
}

void print_plan_lowering_json(const rns8::detail::PlanLoweringDescription& lowering, bool trailing_comma) {
  std::cout << "    \"plan_lowering\": {\n";
  std::cout << "      \"operation\": \"" << json_escape(lowering.operation.c_str()) << "\",\n";
  std::cout << "      \"semantic_contract\": \"" << json_escape(lowering.semantic_contract.c_str()) << "\",\n";
  std::cout << "      \"backend_family\": \"" << json_escape(lowering.backend_family.c_str()) << "\",\n";
  std::cout << "      \"input_domain\": \"" << json_escape(lowering.input_domain.c_str()) << "\",\n";
  std::cout << "      \"output_domain\": \"" << json_escape(lowering.output_domain.c_str()) << "\",\n";
  std::cout << "      \"desired_output\": \"" << json_escape(lowering.desired_output.c_str()) << "\",\n";
  std::cout << "      \"schedule_strategy\": \"" << json_escape(lowering.schedule_strategy.c_str()) << "\",\n";
  std::cout << "      \"packing_strategy\": \"" << json_escape(lowering.packing_strategy.c_str()) << "\",\n";
  std::cout << "      \"reuse_strategy\": \"" << json_escape(lowering.reuse_strategy.c_str()) << "\",\n";
  std::cout << "      \"conversion_strategy\": \"" << json_escape(lowering.conversion_strategy.c_str()) << "\",\n";
  std::cout << "      \"lowering_path\": \"" << json_escape(lowering.lowering_path.c_str()) << "\",\n";
  std::cout << "      \"final_export_available\": " << (lowering.final_export_available ? "true" : "false") << ",\n";
  std::cout << "      \"rns_continuation_available\": "
            << (lowering.rns_continuation_available ? "true" : "false") << ",\n";
  std::cout << "      \"native_continuation_available\": "
            << (lowering.native_continuation_available ? "true" : "false") << ",\n";
  std::cout << "      \"native_to_rns_available\": " << (lowering.native_to_rns_available ? "true" : "false")
            << ",\n";
  std::cout << "      \"reusable_b_prepack_available\": "
            << (lowering.reusable_b_prepack_available ? "true" : "false") << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability) {
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  const std::string target = info.gcn_arch;
  if (!target.empty() && target != "none") {
    runtime.target_id = target;
  } else if (!info.hip_available) {
    runtime.target_id = "cpu";
  }
  if (capability.library_version[0] != '\0' &&
      std::string(capability.library_version) != "runtime_queried_in_context") {
    runtime.hip_sdk_or_library_version = capability.library_version;
  }
  return runtime;
}

