    output << "  \"gpu_event_timings_us\": null,\n";
    output << "  \"gpu_event_timing_summary_us\": null,\n";
  }
  print_first_call_setup_inventory_json(output, args, result);
  output << "  \"plan_us\": " << result.plan_us << ",\n";
  if (global_bound_scan_available) {
    output << "  \"global_bound_scan_us\": " << result.global_bound_scan_us << ",\n";
    output << "  \"avg_global_bound_scan_us\": " << static_cast<double>(result.global_bound_scan_us) << ",\n";
  }
  output << "  \"avg_planning_us\": " << static_cast<double>(result.plan_us) << ",\n";
  output << "  \"schedule_query_us\": " << result.schedule_query_us << ",\n";
  output << "  \"avg_scheduling_us\": " << static_cast<double>(result.schedule_query_us) << ",\n";
  if (result.tile_bound_scan_available) {
    output << "  \"tile_bound_scan_us\": " << result.tile_bound_scan_us << ",\n";
    output << "  \"avg_tile_bound_scan_us\": " << static_cast<double>(result.tile_bound_scan_us) << ",\n";
  }
  output << "  \"matrix_alloc_us\": " << result.matrix_alloc_us << ",\n";
  output << "  \"avg_matrix_alloc_us\": " << static_cast<double>(result.matrix_alloc_us) << ",\n";
  output << "  \"avg_prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    output << static_cast<double>(result.prepack_setup_us);
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  if (pack_a_timing_available) {
    output << "  \"avg_pack_a_us\": " << avg_pack_a_us << ",\n";
  }
  if (pack_b_timing_available) {
    output << "  \"avg_pack_b_us\": " << avg_pack_b_us << ",\n";
  }
  output << "  \"avg_pack_per_task_us\": "
            << (avg_pack_us / task_count_denominator) << ",\n";
  output << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  output << "  \"avg_rns_gemm_per_task_us\": "
            << (avg_gemm_us / task_count_denominator) << ",\n";
  output << "  \"per_modulus_gemm_estimate_applicable\": "
            << (per_modulus_estimate_applicable ? "true" : "false") << ",\n";
  output << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  output << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  output << "  \"avg_crt_export_per_task_us\": "
            << (avg_export_us / task_count_denominator) << ",\n";
  output << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  output << "  \"avg_end_to_end_per_task_us\": "
            << (avg_end_to_end_us / task_count_denominator) << ",\n";
  output << "  \"raw_timings_us\": {\n";
  if (global_bound_scan_available) {
    output << "    \"global_bound_scan\": ";
    print_single_timing_array(output, result.global_bound_scan_us);
    output << ",\n";
  }
  output << "    \"planning\": ";
  print_single_timing_array(output, result.plan_us);
  output << ",\n";
  output << "    \"scheduling\": ";
  print_single_timing_array(output, result.schedule_query_us);
  output << ",\n";
  if (result.tile_bound_scan_available) {
    output << "    \"tile_bound_scan\": ";
    print_single_timing_array(output, result.tile_bound_scan_us);
    output << ",\n";
  }
  output << "    \"matrix_alloc\": ";
  print_single_timing_array(output, result.matrix_alloc_us);
  output << ",\n";
  output << "    \"pack\": ";
  print_double_array(output, result.samples.pack_us);
  output << ",\n";
  if (pack_a_timing_available) {
    output << "    \"pack_a\": ";
    print_double_array(output, result.samples.pack_a_us);
    output << ",\n";
  }
  if (pack_b_timing_available) {
    output << "    \"pack_b\": ";
    print_double_array(output, result.samples.pack_b_us);
    output << ",\n";
  }
  output << "    \"rns_gemm\": ";
  print_double_array(output, result.samples.gemm_us);
  output << ",\n";
  output << "    \"crt_export\": ";
  print_double_array(output, result.samples.export_us);
  output << ",\n";
  output << "    \"end_to_end\": ";
  print_double_array(output, result.samples.end_to_end_us);
  output << "\n";
  output << "  },\n";
  output << "  \"timing_summary_us\": {\n";
  if (global_bound_scan_available) {
    print_single_timing_summary(output, "global_bound_scan", result.global_bound_scan_us, true);
  }
  print_single_timing_summary(output, "planning", result.plan_us, true);
  print_single_timing_summary(output, "scheduling", result.schedule_query_us, true);
  if (result.tile_bound_scan_available) {
    print_single_timing_summary(output, "tile_bound_scan", result.tile_bound_scan_us, true);
  }
  print_single_timing_summary(output, "matrix_alloc", result.matrix_alloc_us, true);
  print_timing_summary(output, "pack", result.samples.pack_us, true);
  if (pack_a_timing_available) {
    print_timing_summary(output, "pack_a", result.samples.pack_a_us, true);
  }
  if (pack_b_timing_available) {
    print_timing_summary(output, "pack_b", result.samples.pack_b_us, true);
  }
  print_timing_summary(output, "rns_gemm", result.samples.gemm_us, true);
  print_timing_summary(output, "crt_export", result.samples.export_us, true);
  print_timing_summary(output, "end_to_end", result.samples.end_to_end_us, false);
  output << "  },\n";
  output << "  \"checksum_u64\": " << result.checksum << "\n";
  output << "}\n";
