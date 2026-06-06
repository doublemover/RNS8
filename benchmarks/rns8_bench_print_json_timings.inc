    std::cout << "  \"gpu_event_timings_us\": null,\n";
    std::cout << "  \"gpu_event_timing_summary_us\": null,\n";
  }
  std::cout << "  \"plan_us\": " << result.plan_us << ",\n";
  if (global_bound_scan_available) {
    std::cout << "  \"global_bound_scan_us\": " << result.global_bound_scan_us << ",\n";
    std::cout << "  \"avg_global_bound_scan_us\": " << static_cast<double>(result.global_bound_scan_us) << ",\n";
  }
  std::cout << "  \"avg_planning_us\": " << static_cast<double>(result.plan_us) << ",\n";
  std::cout << "  \"schedule_query_us\": " << result.schedule_query_us << ",\n";
  std::cout << "  \"avg_scheduling_us\": " << static_cast<double>(result.schedule_query_us) << ",\n";
  if (result.tile_bound_scan_available) {
    std::cout << "  \"tile_bound_scan_us\": " << result.tile_bound_scan_us << ",\n";
    std::cout << "  \"avg_tile_bound_scan_us\": " << static_cast<double>(result.tile_bound_scan_us) << ",\n";
  }
  std::cout << "  \"matrix_alloc_us\": " << result.matrix_alloc_us << ",\n";
  std::cout << "  \"avg_matrix_alloc_us\": " << static_cast<double>(result.matrix_alloc_us) << ",\n";
  std::cout << "  \"avg_prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    std::cout << static_cast<double>(result.prepack_setup_us);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  std::cout << "  \"avg_pack_per_task_us\": "
            << (avg_pack_us / task_count_denominator) << ",\n";
  std::cout << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"avg_rns_gemm_per_task_us\": "
            << (avg_gemm_us / task_count_denominator) << ",\n";
  std::cout << "  \"per_modulus_gemm_estimate_applicable\": "
            << (per_modulus_estimate_applicable ? "true" : "false") << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_crt_export_per_task_us\": "
            << (avg_export_us / task_count_denominator) << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  std::cout << "  \"avg_end_to_end_per_task_us\": "
            << (avg_end_to_end_us / task_count_denominator) << ",\n";
  std::cout << "  \"raw_timings_us\": {\n";
  if (global_bound_scan_available) {
    std::cout << "    \"global_bound_scan\": ";
    print_single_u64_array(result.global_bound_scan_us);
    std::cout << ",\n";
  }
  std::cout << "    \"planning\": ";
  print_single_u64_array(result.plan_us);
  std::cout << ",\n";
  std::cout << "    \"scheduling\": ";
  print_single_u64_array(result.schedule_query_us);
  std::cout << ",\n";
  if (result.tile_bound_scan_available) {
    std::cout << "    \"tile_bound_scan\": ";
    print_single_u64_array(result.tile_bound_scan_us);
    std::cout << ",\n";
  }
  std::cout << "    \"matrix_alloc\": ";
  print_single_u64_array(result.matrix_alloc_us);
  std::cout << ",\n";
  std::cout << "    \"pack\": ";
  print_u64_array(result.samples.pack_us);
  std::cout << ",\n";
  std::cout << "    \"rns_gemm\": ";
  print_u64_array(result.samples.gemm_us);
  std::cout << ",\n";
  std::cout << "    \"crt_export\": ";
  print_u64_array(result.samples.export_us);
  std::cout << ",\n";
  std::cout << "    \"end_to_end\": ";
  print_u64_array(result.samples.end_to_end_us);
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"timing_summary_us\": {\n";
  if (global_bound_scan_available) {
    print_single_timing_summary("global_bound_scan", result.global_bound_scan_us, true);
  }
  print_single_timing_summary("planning", result.plan_us, true);
  print_single_timing_summary("scheduling", result.schedule_query_us, true);
  if (result.tile_bound_scan_available) {
    print_single_timing_summary("tile_bound_scan", result.tile_bound_scan_us, true);
  }
  print_single_timing_summary("matrix_alloc", result.matrix_alloc_us, true);
  print_timing_summary("pack", result.samples.pack_us, true);
  print_timing_summary("rns_gemm", result.samples.gemm_us, true);
  print_timing_summary("crt_export", result.samples.export_us, true);
  print_timing_summary("end_to_end", result.samples.end_to_end_us, false);
  std::cout << "  },\n";
  std::cout << "  \"checksum_u64\": " << result.checksum << "\n";
  std::cout << "}\n";
