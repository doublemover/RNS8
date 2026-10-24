cmake_minimum_required(VERSION 3.22)

# Also configure this installed source directory as a standalone public consumer.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(RNS8Workloads LANGUAGES CXX)
  include(CTest)
  find_package(RNS8 CONFIG REQUIRED)
  find_path(RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR NAMES boost/multiprecision/cpp_int.hpp REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
endif()

option(RNS8_WORKLOAD_USE_SHARED "Link workloads against the exported shared RNS8 target" OFF)
option(RNS8_WORKLOAD_GPU_TESTS "Run explicit direct-HIP workload smoke cases" ${RNS8_ENABLE_HIP})
if(RNS8_WORKLOAD_USE_SHARED)
  if(NOT TARGET rns8::rns8)
    message(FATAL_ERROR "The selected package has no public shared RNS8 target")
  endif()
  set(_workload_library rns8::rns8)
else()
  set(_workload_library rns8::rns8_static)
endif()

set(RNS8_WORKLOAD_TARGETS
  integer-cross-product
  gf251-transform
  resident-chain
  complex-product
  exactness-walkthrough
)

# Bind example code/data separately: the library manifest does not hash examples/.
file(GLOB _workload_inputs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/recipes.json" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_workload_inputs})
set(_workload_identity "{}")
foreach(_input IN LISTS _workload_inputs)
  file(SHA256 "${_input}" _hash)
  get_filename_component(_name "${_input}" NAME)
  string(JSON _workload_identity SET "${_workload_identity}" "${_name}" "\"${_hash}\"")
endforeach()
set(_identity_header "#pragma once\ninline constexpr char rns8_workload_source_identity_json[] = R\"rns8(${_workload_identity})rns8\";\n#define RNS8_WORKLOAD_SOURCE_IDENTITY_JSON rns8_workload_source_identity_json\n")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rns8_library_manifest.json")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/rns8_library_manifest.json" _library_manifest)
  string(APPEND _identity_header "inline constexpr char rns8_workload_library_manifest_json[] = R\"rns8(${_library_manifest})rns8\";\n#define RNS8_WORKLOAD_LIBRARY_MANIFEST_JSON rns8_workload_library_manifest_json\n")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/workload_source_identity.hpp" "${_identity_header}")

foreach(_name IN LISTS RNS8_WORKLOAD_TARGETS)
  string(REPLACE "-" "_" _source "${_name}")
  add_executable("rns8-workload-${_name}" "${_source}.cpp")
  target_link_libraries("rns8-workload-${_name}" PRIVATE ${_workload_library} nlohmann_json::nlohmann_json)
  target_include_directories("rns8-workload-${_name}" PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_include_directories("rns8-workload-${_name}" SYSTEM PRIVATE "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}")
  target_compile_features("rns8-workload-${_name}" PRIVATE cxx_std_17)
  if(COMMAND rns8_bind_manifest)
    rns8_bind_manifest("rns8-workload-${_name}")
  endif()
  if(COMMAND rns8_copy_windows_clang_asan_runtime)
    rns8_copy_windows_clang_asan_runtime("rns8-workload-${_name}")
  endif()
  if(MSVC)
    target_compile_options("rns8-workload-${_name}" PRIVATE /W4 /permissive-)
  else()
    target_compile_options("rns8-workload-${_name}" PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(WIN32 AND RNS8_WORKLOAD_USE_SHARED)
    add_custom_command(TARGET "rns8-workload-${_name}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:${_workload_library}>" "$<TARGET_FILE_DIR:rns8-workload-${_name}>")
  endif()
  if(BUILD_TESTING)
    add_test(NAME "workload_${_source}_cpu" COMMAND "rns8-workload-${_name}" --reference-only --size tiny --repeats 1 --warmups 0 --json)
    set_tests_properties("workload_${_source}_cpu" PROPERTIES LABELS "examples;workloads;cpu" TIMEOUT 30)
    if(RNS8_WORKLOAD_GPU_TESTS)
      add_test(NAME "workload_${_source}_hip" COMMAND "rns8-workload-${_name}" --backend hip-direct --device 0 --size tiny --repeats 1 --warmups 0 --json)
      set_tests_properties("workload_${_source}_hip" PROPERTIES LABELS "examples;workloads;hip" TIMEOUT 60 SKIP_RETURN_CODE 77)
    endif()
  endif()
endforeach()

if(BUILD_TESTING)
  foreach(_gram_export IN ITEMS each once)
    add_test(NAME workload_gram_reuse_${_gram_export}_cpu COMMAND rns8-workload-integer-cross-product
      --reference-only --size tiny --gram --gram-reuse 3 --gram-export ${_gram_export}
      --repeats 2 --warmups 1 --sustained-ms 1 --json)
    set_tests_properties(workload_gram_reuse_${_gram_export}_cpu PROPERTIES LABELS "examples;workloads;cpu;gram" TIMEOUT 60)
    if(RNS8_WORKLOAD_GPU_TESTS)
      add_test(NAME workload_gram_reuse_${_gram_export}_hip COMMAND rns8-workload-integer-cross-product
        --backend hip-direct --device 0 --size tiny --gram --gram-reuse 3 --gram-export ${_gram_export}
        --repeats 2 --warmups 1 --sustained-ms 1 --json)
      set_tests_properties(workload_gram_reuse_${_gram_export}_hip PROPERTIES LABELS "examples;workloads;hip;gram" TIMEOUT 90 SKIP_RETURN_CODE 77)
    endif()
  endforeach()
  add_test(NAME workload_gram_prepared_expanded_proof_cpu COMMAND rns8-workload-integer-cross-product
    --reference-only --size tiny --gram --gram-reuse 3 --gram-export once --pattern zero
    --repeats 2 --warmups 1 --sustained-ms 1 --json)
  set_tests_properties(workload_gram_prepared_expanded_proof_cpu PROPERTIES LABELS "examples;workloads;cpu;gram" TIMEOUT 60)
  if(RNS8_WORKLOAD_GPU_TESTS)
    add_test(NAME workload_gram_prepared_expanded_proof_hip COMMAND rns8-workload-integer-cross-product
      --backend hip-direct --device 0 --size tiny --gram --gram-reuse 3 --gram-export once --pattern zero
      --repeats 2 --warmups 1 --sustained-ms 1 --json)
    set_tests_properties(workload_gram_prepared_expanded_proof_hip PROPERTIES LABELS "examples;workloads;hip;gram" TIMEOUT 90 SKIP_RETURN_CODE 77)
  endif()
  add_executable(rns8-workload-self-test workload_self_test.cpp)
  target_link_libraries(rns8-workload-self-test PRIVATE ${_workload_library} nlohmann_json::nlohmann_json)
  target_include_directories(rns8-workload-self-test SYSTEM PRIVATE "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}")
  target_compile_features(rns8-workload-self-test PRIVATE cxx_std_17)
  if(COMMAND rns8_copy_windows_clang_asan_runtime)
    rns8_copy_windows_clang_asan_runtime(rns8-workload-self-test)
  endif()
  add_test(NAME workload_independent_baselines COMMAND rns8-workload-self-test)
  set_tests_properties(workload_independent_baselines PROPERTIES LABELS "examples;workloads;cpu" TIMEOUT 20)
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/RNS8/examples/workloads"
    COMPONENT Development FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cpp" PATTERN "*.json" PATTERN "CMakeLists.txt" PATTERN "README.md")
  string(REGEX REPLACE "\\.hpp$" ".json" _library_manifest_json "${RNS8_MANIFEST_HEADER}")
  install(FILES "${_library_manifest_json}" DESTINATION "${CMAKE_INSTALL_DATADIR}/RNS8/examples/workloads"
    RENAME rns8_library_manifest.json COMPONENT Development)
  if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME workload_report_tooling COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/test_workload_examples.py"
      --bin-dir "$<TARGET_FILE_DIR:rns8-workload-integer-cross-product>")
    set_tests_properties(workload_report_tooling PROPERTIES LABELS "examples;workloads;cpu;schema" TIMEOUT 90)
    add_test(NAME workload_installed_public_consumer COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/test_workload_install.py"
      --build-dir "${PROJECT_BINARY_DIR}" --config "$<CONFIG>" --backend cpu-reference)
    set_tests_properties(workload_installed_public_consumer PROPERTIES LABELS "examples;workloads;install;cpu" TIMEOUT 600 RUN_SERIAL TRUE)
    if(RNS8_WORKLOAD_GPU_TESTS)
      add_test(NAME workload_installed_public_gpu_consumer COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/test_workload_install.py"
        --build-dir "${PROJECT_BINARY_DIR}" --config "$<CONFIG>" --backend hip-direct)
      set_tests_properties(workload_installed_public_gpu_consumer PROPERTIES LABELS "examples;workloads;install;hip" TIMEOUT 600 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
    endif()
  endif()
endif()
