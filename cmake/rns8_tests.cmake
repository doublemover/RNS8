if(BUILD_TESTING AND RNS8_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(RNS8_FAIL_FAST_ACCELERATOR_FLAGS RNS8_ENABLE_AMDGPU_BUILTINS)
  foreach(RNS8_ACCELERATOR_ENABLE_FLAG ${RNS8_FAIL_FAST_ACCELERATOR_FLAGS})
    string(TOLOWER "${RNS8_ACCELERATOR_ENABLE_FLAG}" RNS8_ACCELERATOR_ENABLE_NAME)
    add_test(
      NAME "accelerator_fail_fast_${RNS8_ACCELERATOR_ENABLE_NAME}"
      COMMAND
        "${CMAKE_COMMAND}"
        "-DRNS8_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DRNS8_ACCEL_FAIL_FAST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/accelerator-fail-fast/${RNS8_ACCELERATOR_ENABLE_FLAG}"
        "-DRNS8_ACCELERATOR_ENABLE_FLAG=${RNS8_ACCELERATOR_ENABLE_FLAG}"
        "-DRNS8_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/accelerator_enable_fail_fast.cmake"
    )
    set_tests_properties("accelerator_fail_fast_${RNS8_ACCELERATOR_ENABLE_NAME}" PROPERTIES LABELS "accelerator;fail-fast")
  endforeach()

  add_test(
    NAME benchmark_schema_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_execution_modes_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_execution_modes.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_execution_modes_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_contract_metadata_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_contract_metadata.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_contract_metadata_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_helper_metadata_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_helper_metadata.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_helper_metadata_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_backend_metadata_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_backend_metadata.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_backend_metadata_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_semantic_contracts_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_semantic_contracts.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_semantic_contracts_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_reuse_timing_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_reuse_timing.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_reuse_timing_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_gpu_events_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_gpu_events.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_gpu_events_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME benchmark_schema_output_policies_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_schema_output_policies.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_output_policies_self_test PROPERTIES LABELS "benchmark;schema")

  add_test(
    NAME metadata_registry_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_metadata_registry.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(metadata_registry_self_test PROPERTIES LABELS "metadata;schema")

  add_test(
    NAME claim_validation_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_claim_validation.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(claim_validation_self_test PROPERTIES LABELS "docs;schema")

  add_test(
    NAME benchmark_sweep_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_sweep.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_sweep_self_test PROPERTIES LABELS "benchmark;sweep;autotune")

  add_test(
    NAME benchmark_sweep_failure_summary_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_sweep_failure_summary.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_sweep_failure_summary_self_test PROPERTIES LABELS "benchmark;sweep;evidence")

  add_test(
    NAME evidence_database_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_evidence_database.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(evidence_database_self_test PROPERTIES LABELS "benchmark;evidence;roofline")

  add_test(
    NAME gpu_counter_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_gpu_counter_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(gpu_counter_report_self_test PROPERTIES LABELS "benchmark;evidence;gpu-counters")

  add_test(
    NAME gpu_isa_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_gpu_isa_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(gpu_isa_report_self_test PROPERTIES LABELS "benchmark;evidence;isa")

  add_test(
    NAME amd_matrix_instruction_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_amd_matrix_instruction_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(amd_matrix_instruction_report_self_test PROPERTIES LABELS "benchmark;evidence;isa")

  add_test(
    NAME target_validation_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_target_validation_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(target_validation_report_self_test PROPERTIES LABELS "benchmark;evidence;targets")

  add_test(
    NAME cdna_env_summary_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cdna_env_summary.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(cdna_env_summary_self_test PROPERTIES LABELS "benchmark;evidence;targets")

  add_test(
    NAME report_capture_inputs_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_report_capture_inputs.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(report_capture_inputs_self_test PROPERTIES LABELS "benchmark;evidence;schema")

  add_test(
    NAME bounded_i64_1024_review_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_bounded_i64_1024_review.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(bounded_i64_1024_review_self_test PROPERTIES LABELS "benchmark;evidence;autotune")

  add_test(
    NAME tile_shape_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_tile_shape_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(tile_shape_report_self_test PROPERTIES LABELS "benchmark;evidence;perf")

  add_test(
    NAME starfoundry_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_starfoundry_reports.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(starfoundry_report_self_test PROPERTIES LABELS "benchmark;evidence;schema")

  add_test(
    NAME adaptive_grouped_scheduler_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_adaptive_grouped_scheduler_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(adaptive_grouped_scheduler_report_self_test PROPERTIES LABELS "benchmark;evidence;scheduler")

  add_test(
    NAME streaming_overlap_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_streaming_overlap_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(streaming_overlap_report_self_test PROPERTIES LABELS "benchmark;evidence;scheduler")

  add_test(
    NAME perf_variance_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_perf_variance_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(perf_variance_report_self_test PROPERTIES LABELS "benchmark;evidence;perf")

  add_test(
    NAME error_detection_policy_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_error_detection_policy_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(error_detection_policy_report_self_test PROPERTIES LABELS "benchmark;evidence;schema")

  add_test(
    NAME cache_promotion_closeout_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cache_promotion_closeout.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(cache_promotion_closeout_self_test PROPERTIES LABELS "tools;autotune;evidence")

  add_test(
    NAME fhe_workload_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_fhe_workload_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(fhe_workload_report_self_test PROPERTIES LABELS "benchmark;evidence;schema")

  add_test(
    NAME cpu_small_shape_selector_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cpu_small_shape_selector_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(cpu_small_shape_selector_report_self_test PROPERTIES LABELS "benchmark;evidence;selector")

  add_test(
    NAME incremental_result_cache_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_incremental_result_cache_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(incremental_result_cache_report_self_test PROPERTIES LABELS "benchmark;evidence;schema")

  add_test(
    NAME shape_family_shadow_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_shape_family_shadow_report.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(shape_family_shadow_report_self_test PROPERTIES LABELS "benchmark;evidence;autotune")

  add_test(
    NAME autotune_cache_install_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_autotune_cache_install.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(autotune_cache_install_self_test PROPERTIES LABELS "tools;autotune")

  if(TARGET rns8-bench)
    add_test(
      NAME benchmark_raw_autotune_cache_write_refused
      COMMAND
        "$<TARGET_FILE:rns8-bench>"
        --backend cpu
        --semantics bounded-i64
        --m 1
        --n 1
        --k 1
        --write-autotune-cache
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(
      benchmark_raw_autotune_cache_write_refused
      PROPERTIES
        LABELS "benchmark;autotune"
        WILL_FAIL TRUE
    )

    add_test(
      NAME benchmark_reuse_packed_inputs_smoke
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_reuse_packed_inputs.py"
        "$<TARGET_FILE:rns8-bench>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-reuse-packed-inputs-smoke"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(benchmark_reuse_packed_inputs_smoke PROPERTIES LABELS "benchmark;schema")

    add_test(
      NAME benchmark_exact_wide_smoke
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_exact_wide.py"
        "$<TARGET_FILE:rns8-bench>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-exact-wide-smoke"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(benchmark_exact_wide_smoke PROPERTIES LABELS "benchmark;schema")
  endif()

  add_test(
    NAME dependency_checker_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_check_dependencies.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(dependency_checker_self_test PROPERTIES LABELS "tools;dependencies;accelerator")

  add_test(
    NAME release_tree_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_check_release_tree.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(release_tree_self_test PROPERTIES LABELS "tools;release")

  add_test(
    NAME bootstrap_rocm_accelerators_dry_run_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_bootstrap_rocm_accelerators.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(bootstrap_rocm_accelerators_dry_run_self_test PROPERTIES LABELS "tools;accelerator")

  add_test(
    NAME release_tree_current_check
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_release_tree.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(release_tree_current_check PROPERTIES LABELS "tools;release")

  if(RNS8_ENABLE_GMP OR RNS8_ENABLE_FLINT)
    add_executable(rns8_optional_exact_libs_smoke tests/unit/test_optional_exact_libs.cpp)
    if(RNS8_ENABLE_GMP)
      target_compile_definitions(rns8_optional_exact_libs_smoke PRIVATE RNS8_ENABLE_GMP=1)
      target_link_libraries(rns8_optional_exact_libs_smoke PRIVATE RNS8::GMP)
    endif()
    if(RNS8_ENABLE_FLINT)
      target_compile_definitions(rns8_optional_exact_libs_smoke PRIVATE RNS8_ENABLE_FLINT=1)
      target_link_libraries(rns8_optional_exact_libs_smoke PRIVATE RNS8::FLINT)
      if(TARGET RNS8::GMP)
        target_link_libraries(rns8_optional_exact_libs_smoke PRIVATE RNS8::GMP)
      endif()
    endif()
    add_test(NAME optional_exact_libs_link_smoke COMMAND rns8_optional_exact_libs_smoke)
    set_tests_properties(optional_exact_libs_link_smoke PROPERTIES LABELS "tools;dependencies;optional-exact")
  endif()

  if(NOT RNS8_ENABLE_WINDOWS_CLANG_ASAN)
    add_test(
      NAME install_downstream_cmake_smoke
      COMMAND
        "${CMAKE_COMMAND}"
        "-DRNS8_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DRNS8_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DRNS8_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
        "-DRNS8_DOWNSTREAM_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DRNS8_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/temp/install-rns8/${CMAKE_BUILD_TYPE}"
        "-DRNS8_DOWNSTREAM_BINARY_DIR=${CMAKE_CURRENT_SOURCE_DIR}/temp/downstream-rns8/${CMAKE_BUILD_TYPE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/install_downstream_smoke.cmake"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(install_downstream_cmake_smoke PROPERTIES LABELS "install;package;examples;cpu")
  endif()

  add_test(
    NAME benchmark_schema_current_fixtures
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_wrap64_hip.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_adaptive_hip.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_i64_adaptive_hip.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_i64_hipblaslt.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_i64_ck.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_adaptive_ck.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_i64_rocwmma.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_adaptive_rocwmma.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_i64_vector_alu.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_vector_alu.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_finite_ring_u8_ck.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_finite_field_u8_rocwmma.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_current_fixtures PROPERTIES LABELS "benchmark;schema")

  if(TARGET rns8-bench AND RNS8_ENABLE_HIP)
    add_test(
      NAME benchmark_auto_backend_smoke
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_backend.py"
        "$<TARGET_FILE:rns8-bench>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-smoke"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(benchmark_auto_backend_smoke PROPERTIES LABELS "benchmark;hip;autotune")

    if(RNS8_ENABLE_HIPBLASLT)
      add_test(
        NAME benchmark_auto_default_cache_hit_hipblaslt
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          hipblaslt
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_hit_hipblaslt
        PROPERTIES LABELS "benchmark;hip;autotune;hipblaslt"
      )
      add_test(
        NAME benchmark_auto_default_cache_exact_wide_hit_hipblaslt
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          hipblaslt
          exact-wide-signed
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_exact_wide_hit_hipblaslt
        PROPERTIES LABELS "benchmark;hip;autotune;hipblaslt"
      )
      add_test(
        NAME benchmark_auto_default_cache_finite_field_hit_hipblaslt
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          hipblaslt
          finite-u8-field
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_finite_field_hit_hipblaslt
        PROPERTIES LABELS "benchmark;hip;autotune;hipblaslt;finite-u8"
      )
    endif()

    if(RNS8_ENABLE_CK)
      add_test(
        NAME benchmark_auto_default_cache_hit_ck
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          ck
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_hit_ck
        PROPERTIES LABELS "benchmark;hip;autotune;ck"
      )
      add_test(
        NAME benchmark_auto_default_cache_exact_wide_hit_ck
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          ck
          exact-wide-signed
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_exact_wide_hit_ck
        PROPERTIES LABELS "benchmark;hip;autotune;ck"
      )
      add_test(
        NAME benchmark_auto_default_cache_finite_ring_hit_ck
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          ck
          finite-u8-ring
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_finite_ring_hit_ck
        PROPERTIES LABELS "benchmark;hip;autotune;ck;finite-u8"
      )
    endif()

    if(RNS8_ENABLE_ROCWMMA)
      if(RNS8_ENABLE_ROCWMMA_WRAP64_CANDIDATE_TESTS)
        add_test(
          NAME benchmark_rocwmma_wrap64_candidate_smoke
          COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_rocwmma_wrap64_candidate.py"
            "$<TARGET_FILE:rns8-bench>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-rocwmma-wrap64-candidate-smoke"
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        set_tests_properties(
          benchmark_rocwmma_wrap64_candidate_smoke
          PROPERTIES LABELS "benchmark;hip;rocwmma;wrap64"
        )
      endif()

      add_test(
        NAME benchmark_auto_default_cache_hit_rocwmma
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          rocwmma
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_hit_rocwmma
        PROPERTIES LABELS "benchmark;hip;autotune;rocwmma"
      )
      add_test(
        NAME benchmark_auto_default_cache_exact_wide_hit_rocwmma
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          rocwmma
          exact-wide-signed
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_exact_wide_hit_rocwmma
        PROPERTIES LABELS "benchmark;hip;autotune;rocwmma"
      )
      add_test(
        NAME benchmark_auto_default_cache_finite_field_hit_rocwmma
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_benchmark_auto_default_cache.py"
          "$<TARGET_FILE:rns8-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/temp/benchmark-auto-default-cache-smoke"
          rocwmma
          finite-u8-field
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      set_tests_properties(
        benchmark_auto_default_cache_finite_field_hit_rocwmma
        PROPERTIES LABELS "benchmark;hip;autotune;rocwmma;finite-u8"
      )
    endif()

    add_test(
      NAME benchmark_vector_alu_baseline_smoke
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_vector_alu_baseline.py"
        "$<TARGET_FILE:rns8-bench>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_schema.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/temp/vector-alu-baseline-smoke"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(benchmark_vector_alu_baseline_smoke PROPERTIES LABELS "benchmark;hip;baseline")
  endif()

  add_test(
    NAME benchmark_result_compare_same_contract
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/result_compare.py"
      --json
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_adaptive_hip.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/benchmark_schema/v4_bounded_u64_adaptive_hip.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_result_compare_same_contract PROPERTIES LABELS "benchmark;schema;compare")

  add_test(
    NAME benchmark_result_compare_self_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_result_compare.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_result_compare_self_test PROPERTIES LABELS "benchmark;schema;compare")

  if(TARGET rns8-inspect)
    add_test(
      NAME inspect_cli_hard_cut_diagnostics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_inspect_cli.py" "$<TARGET_FILE:rns8-inspect>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(inspect_cli_hard_cut_diagnostics PROPERTIES LABELS "tools;cli;hard-cut")
  endif()

  find_package(Catch2 3 CONFIG REQUIRED)
  rns8_assert_no_linux_windows_vcpkg_target(Catch2::Catch2)
  rns8_assert_no_linux_windows_vcpkg_target(Catch2::Catch2WithMain)
  set(
    RNS8_TEST_SOURCES
    tests/unit/test_moduli.cpp
    tests/unit/test_residues.cpp
    tests/unit/test_ring_gemm.cpp
    tests/unit/test_crt.cpp
    tests/unit/test_api.cpp
    tests/unit/test_autotune_cache.cpp
    tests/unit/test_bounded_gemm.cpp
    tests/unit/test_bounded_reference_sweeps.cpp
    tests/unit/test_exact_wide.cpp
    tests/unit/test_semantics.cpp
    tests/unit/test_wrap64.cpp
  )
  if(RNS8_ENABLE_HIP)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_hip_direct.cpp)
  endif()
  if(RNS8_ENABLE_HIPBLASLT)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_hipblaslt.cpp)
  endif()
  if(RNS8_ENABLE_CK)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_ck.cpp)
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_rocwmma.cpp)
  endif()
  add_executable(rns8_tests ${RNS8_TEST_SOURCES})
  target_include_directories(
    rns8_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_HIPBLASLT}>:RNS8_ENABLE_HIPBLASLT=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_CK}>:RNS8_ENABLE_CK=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_ROCWMMA}>:RNS8_ENABLE_ROCWMMA=1>)
  target_compile_definitions(rns8_tests PRIVATE RNS8_CK_USE_XDL=${RNS8_CK_USE_XDL})
  target_compile_definitions(rns8_tests PRIVATE RNS8_CK_SHAPE_ALIGNMENT=${RNS8_CK_SHAPE_ALIGNMENT})
  target_compile_definitions(
    rns8_tests
    PRIVATE $<$<BOOL:${RNS8_ENABLE_ROCWMMA_WRAP64_CANDIDATE_TESTS}>:RNS8_ENABLE_ROCWMMA_WRAP64_CANDIDATE_TESTS=1>
  )
  if(RNS8_ENABLE_HIP)
    target_include_directories(rns8_tests PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_link_libraries(rns8_tests PRIVATE ${RNS8_HIP_LIBRARIES})
    target_compile_definitions(rns8_tests PRIVATE __HIP_PLATFORM_AMD__=1)
  endif()
  target_link_libraries(rns8_tests PRIVATE rns8_static Catch2::Catch2WithMain)
  rns8_copy_windows_clang_asan_runtime(rns8_tests)
  include(Catch)
  catch_discover_tests(rns8_tests)
  if(RNS8_ENABLE_HIP)
    list(GET RNS8_AMDGPU_TARGETS 0 RNS8_HIP_ISA_CHECK_TARGET)
    add_test(
      NAME hip_direct_kernel_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_hip_kernel_isa.py"
        --object "${RNS8_HIP_DIRECT_KERNEL_OBJECT}"
        --target "${RNS8_HIP_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(hip_direct_kernel_isa_check PROPERTIES LABELS "hip;isa;hard-cut")
    add_test(
      NAME generated_reducer_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_generated_reducer_isa.py"
        --object "${RNS8_HIP_DIRECT_KERNEL_OBJECT}"
        --target "${RNS8_HIP_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(generated_reducer_isa_check PROPERTIES LABELS "hip;isa;generated-reducer")
    add_test(
      NAME wrap64_kernel_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_wrap64_kernel_isa.py"
        --object "${RNS8_WRAP64_HIP_KERNEL_OBJECT}"
        --target "${RNS8_HIP_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(wrap64_kernel_isa_check PROPERTIES LABELS "hip;isa;wrap64")
  endif()
  if(RNS8_ENABLE_CK)
    list(GET RNS8_AMDGPU_TARGETS 0 RNS8_CK_ISA_CHECK_TARGET)
    add_test(
      NAME ck_kernel_matrix_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_ck_kernel_isa.py"
        --object "${RNS8_CK_KERNEL_OBJECT}"
        --target "${RNS8_CK_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(ck_kernel_matrix_isa_check PROPERTIES LABELS "ck;isa;accelerator")
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    list(GET RNS8_AMDGPU_TARGETS 0 RNS8_ROCWMMA_ISA_CHECK_TARGET)
    add_test(
      NAME rocwmma_kernel_matrix_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_rocwmma_kernel_isa.py"
        --object "${RNS8_ROCWMMA_KERNEL_OBJECT}"
        --target "${RNS8_ROCWMMA_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(rocwmma_kernel_matrix_isa_check PROPERTIES LABELS "rocwmma;isa;accelerator")
  endif()
endif()

