include_guard(GLOBAL)

# One native CTest denominator for the ordinary build and the compiler-free
# dependency CI project. These controls use inert files/mocks, never fake GPU
# execution as qualification. Real package/compiler/device lanes stay separate.
function(rns8_add_dependency_contract_tests source python)
  foreach(_row IN ITEMS
      "matrix_target_contracts|test_matrix_targets.py|tools,admission|60"
      "dependency_policy_self_test|test_dependency_policy.py|tools,dependencies|180"
      "dependency_resolution_self_test|test_dependency_resolution.py|tools,dependencies|180"
      "rocm_sdk_init_self_test|test_rocm_sdk.py|tools,dependencies,sdk|180"
      "native_build_tools_self_test|test_init_build_tools.py|tools,dependencies,bootstrap|180"
      "host_llvm_bootstrap_self_test|test_init_host_llvm.py|tools,dependencies,bootstrap|180"
      "sdk_package_policy_self_test|test_sdk_package.py|tools,dependencies,packaging|180"
      "ck_input_contract_self_test|test_ck_input_contract.py|tools,dependencies,accelerator|180"
      "dependency_checker_self_test|test_check_dependencies.py|tools,dependencies,accelerator|180"
      "reference_probe_self_test|test_reference_probes.py|tools,dependencies,reference|180"
      "ck_target_probe_self_test|test_ck_target_probes.py|tools,dependencies,accelerator|180"
      "accelerator_target_probe_self_test|test_accelerator_target_probes.py|tools,dependencies,accelerator|180"
      "bootstrap_native_contract_self_test|test_bootstrap_native_contract.py|tools,dependencies,accelerator|180"
      "bootstrap_rocm_accelerators_dry_run_self_test|test_bootstrap_rocm_accelerators.py|tools,dependencies,accelerator|180")
    string(REPLACE "|" ";" _fields "${_row}")
    list(GET _fields 0 _name)
    list(GET _fields 1 _script)
    list(GET _fields 2 _labels)
    list(GET _fields 3 _timeout)
    string(REPLACE "," ";" _labels "${_labels}")
    add_test(NAME "${_name}" COMMAND "${python}" "${source}/tools/${_script}")
    set_tests_properties("${_name}" PROPERTIES
      WORKING_DIRECTORY "${source}" LABELS "${_labels}" TIMEOUT "${_timeout}")
  endforeach()
endfunction()
