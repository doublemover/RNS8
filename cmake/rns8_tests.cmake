if(BUILD_TESTING AND RNS8_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME residue_dot_collector_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_residue_dot.py")
  if(TARGET rns8-residue-dot-calibrate AND RNS8_ENABLE_HIP AND (RNS8_EXPERIMENTAL_RESIDUE_DOT4 OR RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL))
    foreach(_residue_case IN ITEMS independent-p20-sparse grouped-p20-sparse scheduled-p20-sparse region-p20-sparse grouped-fullwidth-k65 grouped-finite247)
      add_test(NAME residue_dot_${_residue_case}_smoke COMMAND rns8-residue-dot-calibrate "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/residue-dot/${_residue_case}.json" ctest-${_residue_case})
      set_tests_properties(residue_dot_${_residue_case}_smoke PROPERTIES LABELS "hip;residue-dot;benchmark-smoke" TIMEOUT 120)
    endforeach()
  endif()
  add_test(NAME queue_calibration_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_queue_calibration.py")
  add_test(NAME hipblaslt_metadata_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_hipblaslt_metadata.py")
  add_test(NAME streaming_measurement_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_streaming_measurement.py")
  add_test(NAME public_measurement_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_public_measurement_contracts.py")
  add_test(NAME wide_magnitude_receipts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_wide_magnitude.py")
  add_test(NAME ci_benchmark_smoke COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_ci_benchmark_smoke.py")
  add_test(NAME observation_cohorts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_observation_cohorts.py")
  add_test(NAME measurement_evidence_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_measurement_evidence_contracts.py")
  add_test(NAME tooling_contract_regressions COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_tooling_contracts.py")
  add_test(NAME dependency_policy_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_dependency_policy.py")
  add_test(NAME documentation_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_documentation.py")
  add_test(NAME documentation_links COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_documentation.py")
  add_test(NAME dependency_resolution_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_dependency_resolution.py")
  add_test(NAME locality_schedule_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_locality_schedule.py")
  add_test(NAME calibration_manifest_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_calibration_manifest.py")
  add_test(NAME narrow_public_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_narrow_public_calibration.py")
  add_test(NAME int4_public_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_int4_calibration.py")
  add_test(NAME public_epilogue_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_epilogue_calibration.py")
  add_test(NAME native_calibration_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_native_calibration.py")
  add_test(NAME finite_verify_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_finite_verify.py")
  add_test(NAME block_sparse_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_block_sparse.py")
  add_test(NAME wrap64_calibration_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_wrap64.py")
  if(TARGET rns8-wrap64-calibrate)
    set_tests_properties(wrap64_calibration_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_WRAP64_CALIBRATION_EXE=$<TARGET_FILE:rns8-wrap64-calibrate>"
      LABELS "tools;wrap64;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_wrap_candidate IN ITEMS signed_radix native_u8 direct_v4 rocwmma_corrected)
        if(_wrap_candidate STREQUAL "rocwmma_corrected" AND NOT RNS8_ENABLE_ROCWMMA)
          continue()
        endif()
        foreach(_wrap_scope IN ITEMS resident roundtrip cold)
          add_test(NAME wrap64_${_wrap_candidate}_${_wrap_scope}_smoke
            COMMAND rns8-wrap64-calibrate 0 ${_wrap_candidate} ${_wrap_scope}
              2 3 17 carry 1 2 1 ctest-${_wrap_candidate}-${_wrap_scope})
          set_tests_properties(wrap64_${_wrap_candidate}_${_wrap_scope}_smoke PROPERTIES
            LABELS "hip;wrap64;benchmark-smoke" TIMEOUT 60 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
    endif()
  endif()
  if(TARGET rns8-block-sparse-calibrate)
    set_tests_properties(block_sparse_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_BLOCK_SPARSE_EXE=$<TARGET_FILE:rns8-block-sparse-calibrate>"
      LABELS "tools;sparse;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_sparse_mode IN ITEMS dense sparse_scalar sparse_wmma sparse_hybrid sparse_scalar_cold sparse_wmma_cold sparse_hybrid_cold)
        if(_sparse_mode MATCHES "hybrid")
          set(_sparse_threshold 1)
        else()
          set(_sparse_threshold 0)
        endif()
        add_test(NAME block_sparse_${_sparse_mode}_smoke
          COMMAND rns8-block-sparse-calibrate 0 ${_sparse_mode} 17 19 129 mixed
            1 ${_sparse_threshold} 2 1 ctest-${_sparse_mode})
        set_tests_properties(block_sparse_${_sparse_mode}_smoke PROPERTIES
          LABELS "hip;sparse;benchmark-smoke" TIMEOUT 60 SKIP_RETURN_CODE 77)
      endforeach()
    endif()
  endif()
  if(TARGET rns8-finite-verify-calibrate)
    set_tests_properties(finite_verify_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_FINITE_VERIFY_EXE=$<TARGET_FILE:rns8-finite-verify-calibrate>"
      LABELS "tools;finite;verification;cli")
    foreach(_finite_mode IN ITEMS verify_only compute_verify full_reference)
      if(_finite_mode STREQUAL "full_reference")
        set(_finite_trials 0)
        set(_finite_seed 0)
      else()
        set(_finite_trials 4)
        set(_finite_seed 3135)
      endif()
      add_test(NAME finite_verify_cpu_${_finite_mode}_smoke
        COMMAND rns8-finite-verify-calibrate cpu-reference -1 ${_finite_mode}
          2 3 65 250 ${_finite_trials} 2 1 ${_finite_seed} ctest-cpu-${_finite_mode})
      set_tests_properties(finite_verify_cpu_${_finite_mode}_smoke PROPERTIES
        LABELS "cpu;finite;verification;benchmark-smoke" TIMEOUT 60)
      if(RNS8_ENABLE_HIP)
        add_test(NAME finite_verify_hip_${_finite_mode}_smoke
          COMMAND rns8-finite-verify-calibrate hip-direct 0 ${_finite_mode}
            2 3 65 256 ${_finite_trials} 2 1 ${_finite_seed} ctest-hip-${_finite_mode})
        set_tests_properties(finite_verify_hip_${_finite_mode}_smoke PROPERTIES
          LABELS "hip;finite;verification;benchmark-smoke" TIMEOUT 60 SKIP_RETURN_CODE 77)
      endif()
    endforeach()
  endif()
  add_test(NAME arithmetic_composition_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_arithmetic_composition.py")
  add_test(NAME modulus_generator_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_generate_modulus_kernels.py")
  add_test(NAME native_wmma_logical_checker_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_native_wmma_logic.py")
  add_test(NAME native_wmma_generator_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_generate_native_wmma.py")
  add_test(NAME block_gemm_research_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_block_gemm_research.py")
  add_test(NAME native_integrity_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_native_integrity.py")
  add_test(NAME native_interop_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_native_interop.py")
  add_test(NAME lift_recipe_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_lift_recipe.py")
  add_test(NAME bilinear_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_bilinear_calibration.py")
  add_test(NAME rank23_october55_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_rank23_october55.py")
  add_test(NAME rank23_slp_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_rank23_slp.py")
  add_test(NAME residue_integrity_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_residue_integrity.py")
  add_test(NAME analytical_planner_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_analytical_planner.py")
  add_test(NAME planning_recipe_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_planning_recipe.py")
  add_test(NAME modulus_scheme_optimizer_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_modulus_scheme_optimizer.py")
  add_test(NAME encoding_catalogue_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_encoding_catalogue.py")
  add_test(NAME instruction_roof_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_instruction_roof.py")
  add_test(NAME split_k_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_split_k.py")
  add_test(NAME ragged_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_ragged.py")
  add_test(NAME stream_panel_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_stream_panels.py")
  if(TARGET rns8-stream-panel-calibrate)
    set_tests_properties(stream_panel_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_STREAM_PANEL_EXE=$<TARGET_FILE:rns8-stream-panel-calibrate>"
      LABELS "tools;stream-panels;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_stream_domain IN ITEMS signed unsigned)
        foreach(_stream_mode IN ITEMS incore stream)
          if(_stream_mode STREQUAL "stream")
            set(_stream_panel_arguments 2 2 5 2048 192)
          else()
            set(_stream_panel_arguments 0 0 0 0 0)
          endif()
          add_test(NAME stream_panel_${_stream_domain}_${_stream_mode}_smoke
            COMMAND rns8-stream-panel-calibrate 0 ${_stream_mode} 2 3 17 ${_stream_domain}
              3 9 ${_stream_panel_arguments} 2 2 1 ctest-panels-${_stream_domain}-${_stream_mode})
          set_tests_properties(stream_panel_${_stream_domain}_${_stream_mode}_smoke PROPERTIES
            LABELS "hip;stream-panels;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
    endif()
  endif()
  if(TARGET rns8-ragged-calibrate)
    set_tests_properties(ragged_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_RAGGED_EXE=$<TARGET_FILE:rns8-ragged-calibrate>"
      LABELS "tools;ragged;cli")
    foreach(_ragged_backend IN ITEMS cpu-reference hip-direct)
      if(_ragged_backend STREQUAL "hip-direct")
        if(NOT RNS8_ENABLE_HIP)
          continue()
        endif()
        set(_ragged_device 0)
        set(_ragged_label hip)
      else()
        set(_ragged_device -1)
        set(_ragged_label cpu)
      endif()
      foreach(_ragged_domain IN ITEMS signed unsigned)
        foreach(_ragged_mode IN ITEMS uniform_export ragged_export ragged_resident)
          add_test(NAME ragged_${_ragged_label}_${_ragged_domain}_${_ragged_mode}_smoke
            COMMAND rns8-ragged-calibrate ${_ragged_backend} ${_ragged_device} ${_ragged_mode}
              17 19 mixed ${_ragged_domain} 3 5 2 2 1 ctest-ragged-${_ragged_label}-${_ragged_domain}-${_ragged_mode})
          set_tests_properties(ragged_${_ragged_label}_${_ragged_domain}_${_ragged_mode}_smoke PROPERTIES
            LABELS "${_ragged_label};ragged;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
      add_test(NAME ragged_${_ragged_label}_wide140_smoke
        COMMAND rns8-ragged-calibrate ${_ragged_backend} ${_ragged_device} ragged_export
          4 4 wide140 signed 1 1 1 2 1 ctest-ragged-${_ragged_label}-wide140)
      set_tests_properties(ragged_${_ragged_label}_wide140_smoke PROPERTIES
        LABELS "${_ragged_label};ragged;wide;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
  endif()
  if(TARGET rns8-split-k-calibrate)
    set_tests_properties(split_k_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_SPLIT_K_EXE=$<TARGET_FILE:rns8-split-k-calibrate>"
      LABELS "tools;split-k;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_split_domain IN ITEMS signed unsigned)
        foreach(_split_mode IN ITEMS unsplit full window)
          if(_split_mode STREQUAL "unsplit")
            set(_split_fragments 1)
            set(_split_window 0)
          elseif(_split_mode STREQUAL "full")
            set(_split_fragments 3)
            set(_split_window 3)
          else()
            set(_split_fragments 3)
            set(_split_window 2)
          endif()
          add_test(NAME split_k_${_split_domain}_${_split_mode}_smoke
            COMMAND rns8-split-k-calibrate 0 ${_split_mode} 2 3 17 ${_split_domain}
              ${_split_fragments} ${_split_window} 2 2 1 ctest-split-${_split_domain}-${_split_mode})
          set_tests_properties(split_k_${_split_domain}_${_split_mode}_smoke PROPERTIES
            LABELS "hip;split-k;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
    endif()
  endif()
  if(TARGET rns8-instruction-roof-calibrate)
    set_tests_properties(instruction_roof_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_INSTRUCTION_ROOF_EXE=$<TARGET_FILE:rns8-instruction-roof-calibrate>"
      LABELS "tools;instruction-chain;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_roof_mode IN ITEMS raw reduce251)
        foreach(_roof_chains IN ITEMS 1 2 4 8)
          add_test(NAME instruction_roof_${_roof_mode}_${_roof_chains}_smoke
            COMMAND rns8-instruction-roof-calibrate 0 ${_roof_mode} ${_roof_chains} 17 5 2 1
              ctest-roof-${_roof_mode}-${_roof_chains})
          set_tests_properties(instruction_roof_${_roof_mode}_${_roof_chains}_smoke PROPERTIES
            LABELS "hip;instruction-chain;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
    endif()
  endif()
  add_test(NAME word_mod_calibration_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_word_mod_calibration.py")
  if(TARGET rns8-word-mod-calibrate)
    set_tests_properties(word_mod_calibration_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_WORD_MOD_CALIBRATION_EXE=$<TARGET_FILE:rns8-word-mod-calibrate>"
      LABELS "tools;word-modular;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_word_method IN ITEMS baseline packed_dot4 byte_wmma)
        foreach(_word_scope IN ITEMS resident cold)
          foreach(_word_domain IN ITEMS ring prime_field)
            if(_word_domain STREQUAL "ring")
              set(_word_modulus 256)
            else()
              set(_word_modulus 251)
            endif()
            add_test(NAME word_mod_${_word_method}_${_word_scope}_${_word_domain}_smoke
              COMMAND rns8-word-mod-calibrate 0 ${_word_method} ${_word_scope} 2 3 17
                ${_word_domain} ${_word_modulus} 2 1 ctest-${_word_method}-${_word_scope}-${_word_domain})
            set_tests_properties(word_mod_${_word_method}_${_word_scope}_${_word_domain}_smoke PROPERTIES
              LABELS "hip;word-modular;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
          endforeach()
        endforeach()
      endforeach()
      foreach(_word_scope IN ITEMS resident cold)
        foreach(_word_domain IN ITEMS ring prime_field)
          if(_word_domain STREQUAL "ring")
            set(_word_modulus 256)
          else()
            set(_word_modulus 251)
          endif()
          foreach(_word_left IN ITEMS 1 2)
            if(_word_left EQUAL 1)
              set(_word_right 2)
            else()
              set(_word_right 1)
            endif()
            add_test(NAME word_mod_binary64_${_word_left}_${_word_right}_${_word_scope}_${_word_domain}_smoke
              COMMAND rns8-word-mod-calibrate 0 asymmetric_binary64 ${_word_scope} 2 3 17
                ${_word_domain} ${_word_modulus} 2 1 ${_word_left} ${_word_right} 17
                ctest-binary64-${_word_left}-${_word_right}-${_word_scope}-${_word_domain})
            set_tests_properties(word_mod_binary64_${_word_left}_${_word_right}_${_word_scope}_${_word_domain}_smoke PROPERTIES
              LABELS "hip;word-modular;binary64;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
          endforeach()
        endforeach()
      endforeach()
    endif()
  endif()
  add_test(NAME crt_calibration_collector_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_collect_crt_calibration.py")
  if(TARGET rns8-crt-calibrate)
    set_tests_properties(crt_calibration_collector_self_test PROPERTIES
      ENVIRONMENT "RNS8_CRT_CALIBRATION_EXE=$<TARGET_FILE:rns8-crt-calibrate>"
      LABELS "tools;conversion;cli")
    if(RNS8_ENABLE_HIP)
      foreach(_conversion_candidate IN ITEMS garner direct balanced fixed_dot4 tile_balanced tile_convert_balanced tile_convert_garner import_scalar_signed import_scalar_unsigned import_dot4_signed import_dot4_unsigned fixed_wmma import_wmma_signed import_wmma_unsigned)
        foreach(_conversion_scope IN ITEMS resident cold)
          add_test(NAME conversion_${_conversion_candidate}_${_conversion_scope}_smoke
            COMMAND rns8-crt-calibrate ${_conversion_candidate} 17 3 2 1
              ctest-${_conversion_candidate}-${_conversion_scope} ${_conversion_scope})
          set_tests_properties(conversion_${_conversion_candidate}_${_conversion_scope}_smoke PROPERTIES
            LABELS "hip;conversion;benchmark-smoke" TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
      endforeach()
    endif()
  endif()
  add_test(NAME crt_research_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_crt_research.py")
  add_test(NAME floating_rewrite_oracle_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_floating_rewrite_oracle.py")
  add_test(NAME arithmetic_certificate_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_arithmetic_certificate.py")
  add_test(NAME build_manifest_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_build_manifest.py")

  add_test(
    NAME performance_dashboard_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_performance_dashboard.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(performance_dashboard_self_test PROPERTIES LABELS "tools;schema")
  add_test(
    NAME gpu_qualification_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_gpu_qualification.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(gpu_qualification_self_test PROPERTIES LABELS "tools;qualification")
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
  add_test(NAME cdna_scripts_self_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cdna_scripts.py")
  set_tests_properties(cdna_scripts_self_test PROPERTIES LABELS "tools;qualification")

  add_test(
    NAME report_capture_inputs_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_report_capture_inputs.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(report_capture_inputs_self_test PROPERTIES LABELS "benchmark;evidence;schema")


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
    string(TOUPPER "${CMAKE_BUILD_TYPE}" RNS8_SMOKE_BUILD_TYPE)
    add_test(
      NAME install_downstream_cmake_smoke
      COMMAND
        "${CMAKE_COMMAND}"
        "-DRNS8_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DRNS8_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
        "-DRNS8_DOWNSTREAM_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DRNS8_DOWNSTREAM_TEST_SHARED=${RNS8_BUILD_SHARED}"
        "-DRNS8_DOWNSTREAM_EXAMPLE_RELATIVE_DIR=${CMAKE_INSTALL_DATADIR}/RNS8/examples/downstream-cmake"
        "-DRNS8_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install-smoke/prefix"
        "-DRNS8_DOWNSTREAM_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-smoke/packaged-consumer"
        "-DRNS8_DOWNSTREAM_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DRNS8_DOWNSTREAM_CXX_FLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${RNS8_SMOKE_BUILD_TYPE}}"
        "-DRNS8_DOWNSTREAM_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${RNS8_SMOKE_BUILD_TYPE}}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/install_downstream_smoke.cmake"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(install_downstream_cmake_smoke PROPERTIES LABELS "install;package;examples;cpu" TIMEOUT 120)
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

  # Isolate replacement global allocators from normal tests and shipping libraries.
  add_executable(rns8_exact_cpu_allocation_faults
    tests/unit/test_exact_cpu_allocation_faults.cpp
    tests/support/allocation_fault.cpp)
  target_include_directories(rns8_exact_cpu_allocation_faults PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}")
  target_link_libraries(rns8_exact_cpu_allocation_faults PRIVATE rns8_static)
  if(MSVC)
    target_compile_options(rns8_exact_cpu_allocation_faults PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(rns8_exact_cpu_allocation_faults PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  rns8_copy_windows_clang_asan_runtime(rns8_exact_cpu_allocation_faults)
  add_test(NAME exact_cpu_allocation_faults COMMAND rns8_exact_cpu_allocation_faults)
  set_tests_properties(exact_cpu_allocation_faults PROPERTIES LABELS "cpu;exact;allocation;transaction" TIMEOUT 60 SKIP_RETURN_CODE 77)

  find_package(Catch2 3 CONFIG REQUIRED)
  rns8_assert_no_linux_windows_vcpkg_target(Catch2::Catch2)
  set(
    RNS8_TEST_SOURCES
    tests/unit/test_public_abi.cpp
    tests/unit/test_moduli.cpp
    tests/unit/test_residues.cpp
    tests/unit/test_ring_gemm.cpp
    tests/unit/test_residue_dot.cpp
    tests/unit/test_residue_epilogue.cpp
    tests/unit/test_exact_lift_workspace.cpp
    tests/unit/test_residue_layout.cpp
    tests/unit/test_crt.cpp
    tests/unit/test_direct_crt_constants.cpp
    tests/unit/test_rank23_slp.cpp
    tests/unit/test_integrity_identity.cpp
    tests/unit/test_distributed_plan.cpp
    tests/unit/test_int4_encoding.cpp
    tests/unit/test_native_tile_bound.cpp
    tests/unit/test_native_wmma_queue.cpp
    tests/unit/test_native64_reducer.cpp
    tests/unit/test_complex_residue_plan.cpp
    tests/unit/test_ragged_residues.cpp
    tests/unit/test_block_sparse.cpp
    tests/unit/test_generated_reducers.cpp
    tests/unit/test_generated_encodings.cpp
    tests/unit/test_api.cpp
    tests/unit/test_autotune_cache.cpp
    tests/unit/test_bounded_gemm.cpp
    tests/unit/test_bounded_reference_sweeps.cpp
    tests/unit/test_hipblaslt_metadata.cpp
    tests/unit/test_streaming_measurement.cpp
    tests/unit/test_public_sustained.cpp
    tests/unit/test_wide_magnitude_fixture.cpp
    tests/unit/test_exact_wide.cpp
    tests/unit/test_semantics.cpp
    tests/unit/test_wrap64.cpp
  )
  # Host-only execution of scalar HIP arithmetic; this is not a GPU test.
  if(NOT MSVC)
    list(APPEND RNS8_TEST_SOURCES tests/unit/test_hip_arithmetic_host.cpp)
  endif()
  if(RNS8_ENABLE_HIP)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_hip_direct.cpp)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_hip_direct_persistent_kernels.cpp)
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
  if(RNS8_ENABLE_AMDGPU_BUILTINS)
    list(APPEND RNS8_TEST_SOURCES tests/differential/test_amdgpu_builtins.cpp)
  endif()
  set(_rns8_reducer_dir "${CMAKE_CURRENT_BINARY_DIR}/generated-reducer-tests")
  set(_rns8_reducer_header "${_rns8_reducer_dir}/rns8_generated_centered_reducers.cuh")
  add_custom_command(OUTPUT "${_rns8_reducer_header}"
    BYPRODUCTS "${_rns8_reducer_dir}/rns8_generated_centered_reducers.json"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_modulus_kernels.py"
            --out-dir "${_rns8_reducer_dir}"
    DEPENDS tools/generate_modulus_kernels.py tools/arithmetic_certificate.py tools/modulus_set_search.py
    VERBATIM)
  set(_rns8_encoding_dir "${CMAKE_CURRENT_BINARY_DIR}/generated-encoding-tests")
  set(_rns8_encoding_header "${_rns8_encoding_dir}/encodings.hpp")
  add_custom_command(OUTPUT "${_rns8_encoding_header}"
    BYPRODUCTS "${_rns8_encoding_dir}/awe6_251.hpp" "${_rns8_encoding_dir}/awe6_256.hpp"
               "${_rns8_encoding_dir}/base13_251.hpp" "${_rns8_encoding_dir}/base13_256.hpp"
               "${_rns8_encoding_dir}/exact_divide.hpp" "${_rns8_encoding_dir}/encodings.json"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_encoding_test_fixtures.py"
            --out-dir "${_rns8_encoding_dir}"
    DEPENDS tools/generate_encoding_test_fixtures.py tools/encoding_catalogue.py tools/arithmetic_certificate.py
    VERBATIM)
  add_executable(rns8_tests tests/support/test_main.cpp ${RNS8_TEST_SOURCES} "${_rns8_reducer_header}" "${_rns8_encoding_header}")
  target_include_directories(
    rns8_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${_rns8_reducer_dir}"
      "${_rns8_encoding_dir}"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>)
  if(RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA)
    target_sources(rns8_tests PRIVATE "${RNS8_GENERATED_NATIVE_DIR}/native_variants.hpp")
    target_include_directories(rns8_tests PRIVATE "${RNS8_GENERATED_NATIVE_DIR}")
    target_compile_definitions(rns8_tests PRIVATE RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA=1)
    add_test(NAME generated_native_wmma_isa_check
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_generated_native_wmma.py"
              --object "${RNS8_GENERATED_NATIVE_OBJECT}" --generated-dir "${RNS8_GENERATED_NATIVE_DIR}"
              --target "${RNS8_AMDGPU_TARGETS}" --hipcc "${RNS8_HIP_HIPCC}"
              --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
              --output "${CMAKE_CURRENT_BINARY_DIR}/generated-native-wmma/isa.json"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(generated_native_wmma_isa_check PROPERTIES LABELS "hip;isa;generated-native")
  endif()
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_HIPBLASLT}>:RNS8_ENABLE_HIPBLASLT=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_CK}>:RNS8_ENABLE_CK=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_ROCWMMA}>:RNS8_ENABLE_ROCWMMA=1>)
  target_compile_definitions(rns8_tests PRIVATE $<$<BOOL:${RNS8_ENABLE_AMDGPU_BUILTINS}>:RNS8_ENABLE_AMDGPU_BUILTINS=1>)
  target_compile_definitions(
    rns8_tests
    PRIVATE $<$<BOOL:${RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE}>:RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE=1>
  )
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
  target_link_libraries(rns8_tests PRIVATE rns8_static Catch2::Catch2)
  rns8_copy_windows_clang_asan_runtime(rns8_tests)
  include(Catch)
  # Older distro Catch2 discovery scripts omit this property even though the
  # runtime returns 4 when every selected test is explicitly skipped.
  catch_discover_tests(rns8_tests PROPERTIES SKIP_RETURN_CODE 4)
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

