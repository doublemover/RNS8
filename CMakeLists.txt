cmake_minimum_required(VERSION 3.28)

project(
  RNS8
  VERSION 0.1.0
  DESCRIPTION "Exact integer GEMM with AMD GPU matrix engines, RNS, and CRT reconstruction"
  LANGUAGES CXX
)

include(CTest)

option(RNS8_ENABLE_HIP "Build the direct HIP backend with explicit hipcc integration" OFF)
option(RNS8_BUILD_TESTS "Build RNS8 correctness tests" ON)
option(RNS8_BUILD_TOOLS "Build RNS8 command-line tools" ON)
option(RNS8_BUILD_BENCHMARKS "Build RNS8 benchmark shell" ON)
option(RNS8_ENABLE_GMP "Detect GMP for future optional differential references" OFF)
option(RNS8_ENABLE_FLINT "Detect FLINT for future optional differential references" OFF)
option(RNS8_PROBE_ACCELERATORS "Run CMake evidence-only discovery for optional accelerator components" OFF)
option(RNS8_ENABLE_HIPBLASLT "Enable hipBLASLt accelerator backend after validated implementation" OFF)
option(RNS8_ENABLE_CK "Enable CK accelerator backend after validated implementation" OFF)
option(RNS8_ENABLE_ROCWMMA "Enable rocWMMA accelerator backend after validated implementation" OFF)
option(RNS8_ENABLE_AMDGPU_BUILTINS "Enable target-specific AMDGPU builtin accelerator kernels after validated implementation" OFF)

set(RNS8_AMDGPU_TARGETS "gfx1100" CACHE STRING "Semicolon-separated AMDGPU offload targets for direct HIP sources")
set(RNS8_HIP_ROOT "" CACHE PATH "AMD HIP SDK or ROCm root used by explicit HIP integration")
set(RNS8_ROCM_COVERAGE_TARGETS "" CACHE STRING "Source-level ROCm target family coverage metadata; does not add offload architectures")
set(RNS8_ROCM_DEPS_BUILD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/out/third_party/rocm/build/windows-gfx1100" CACHE PATH "Ignored repo-local build/generated-header root for optional ROCm accelerator dependencies")
set(RNS8_ROCM_DEPS_INSTALL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/out/third_party/rocm/install/windows-gfx1100" CACHE PATH "Ignored repo-local staged install root for optional ROCm accelerator dependencies")
set(RNS8_CK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rocm/composable_kernel" CACHE PATH "Repo-local Composable Kernel source or install root")
set(RNS8_ROCWMMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rocm/rocWMMA" CACHE PATH "Repo-local rocWMMA source or install root")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

if(RNS8_ENABLE_HIPBLASLT AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_HIPBLASLT requires RNS8_ENABLE_HIP because hipBLASLt uses resident HIP buffers")
endif()

if(RNS8_ENABLE_CK AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_CK requires RNS8_ENABLE_HIP because CK uses resident HIP buffers")
endif()

if(RNS8_ENABLE_ROCWMMA AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_ROCWMMA requires RNS8_ENABLE_HIP because rocWMMA uses resident HIP buffers")
endif()

if(RNS8_ENABLE_AMDGPU_BUILTINS)
  message(
    FATAL_ERROR
      "AMDGPU builtin accelerator backend is not implemented. "
      "Use RNS8_PROBE_ACCELERATORS=ON or tools/check_dependencies.py --accelerator-probes for evidence-only probes."
  )
endif()

if(RNS8_PROBE_ACCELERATORS)
  find_package(RNS8HIPBLASLT QUIET)
  find_package(RNS8CK QUIET)
  find_package(RNS8ROCWMMA QUIET)
  message(STATUS "RNS8 hipBLASLt candidate evidence: ${RNS8_HIPBLASLT_EVIDENCE}")
  message(STATUS "RNS8 CK candidate evidence: ${RNS8_CK_EVIDENCE}")
  message(STATUS "RNS8 CK compile probe: ${RNS8_CK_COMPILE_PROBE_STATUS}")
  message(STATUS "RNS8 CK primitive probe: ${RNS8_CK_PRIMITIVE_PROBE_STATUS}")
  message(STATUS "RNS8 rocWMMA candidate evidence: ${RNS8_ROCWMMA_EVIDENCE}")
  message(STATUS "RNS8 rocWMMA compile probe: ${RNS8_ROCWMMA_COMPILE_PROBE_STATUS}")
  message(STATUS "RNS8 rocWMMA primitive probe: ${RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS}")
  message(STATUS "RNS8 AMDGPU builtin candidate evidence: disabled until target-specific exact kernels exist")
  message(STATUS "RNS8 accelerator backend enablement: disabled")
endif()

find_path(
  RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR
  NAMES boost/multiprecision/cpp_int.hpp
  REQUIRED
)
find_package(nlohmann_json CONFIG REQUIRED)

set(RNS8_PUBLIC_HEADERS
  include/rns8/status.h
  include/rns8/semantics.h
  include/rns8/bounds.h
  include/rns8/moduli.h
  include/rns8/rns8.h
  include/rns8/rns8.hpp
)

set(RNS8_SOURCES
  src/core/accelerator_backend.cpp
  src/core/autotune_cache.cpp
  src/core/api.cpp
  src/core/moduli.cpp
  src/core/status.cpp
  src/cpu/cpu_reference.cpp
  src/reconstruct/crt.cpp
  src/backend_hip_direct/hip_backend.cpp
  src/backend_ck/ck_backend.cpp
  src/backend_wmma/wmma_backend.cpp
  src/backend_wrap64/wrap64_reference.cpp
  src/backend_wrap64/wrap64_hip.cpp
)

if(RNS8_ENABLE_HIP)
  find_package(RNS8HIP REQUIRED)
  rns8_compile_hip_source(
    RNS8_HIP_DIRECT_KERNEL_OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hip_direct/hip_direct_kernels.hip"
  )
  rns8_compile_hip_source(
    RNS8_WRAP64_HIP_KERNEL_OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_wrap64/wrap64_hip_kernels.hip"
  )
  list(APPEND RNS8_SOURCES "${RNS8_HIP_DIRECT_KERNEL_OBJECT}" "${RNS8_WRAP64_HIP_KERNEL_OBJECT}")
  if(RNS8_ENABLE_HIPBLASLT)
    find_package(RNS8HIPBLASLT REQUIRED)
    rns8_compile_hip_source(
      RNS8_HIPBLASLT_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hipblaslt/hipblaslt_kernels.hip"
    )
    list(APPEND RNS8_SOURCES src/backend_hipblaslt/hipblaslt_backend.cpp "${RNS8_HIPBLASLT_KERNEL_OBJECT}")
  endif()
  if(RNS8_ENABLE_CK)
    find_package(RNS8CK REQUIRED)
    set(RNS8_HIP_SOURCE_INCLUDE_DIRS "${RNS8_CK_GENERATED_INCLUDE_DIR}" "${RNS8_CK_INCLUDE_DIR}")
    rns8_compile_hip_source(
      RNS8_CK_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_ck/ck_backend_kernels.hip"
    )
    unset(RNS8_HIP_SOURCE_INCLUDE_DIRS)
    list(APPEND RNS8_SOURCES "${RNS8_CK_KERNEL_OBJECT}")
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    find_package(RNS8ROCWMMA REQUIRED)
    set(RNS8_HIP_SOURCE_INCLUDE_DIRS "${RNS8_ROCWMMA_INCLUDE_DIR}")
    rns8_compile_hip_source(
      RNS8_WMMA_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_wmma/wmma_backend_kernels.hip"
    )
    unset(RNS8_HIP_SOURCE_INCLUDE_DIRS)
    list(APPEND RNS8_SOURCES "${RNS8_WMMA_KERNEL_OBJECT}")
  endif()
endif()

function(rns8_configure_library target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_17)
  target_include_directories(
    ${target_name}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE
      $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>
      $<$<BOOL:${RNS8_ENABLE_HIPBLASLT}>:RNS8_ENABLE_HIPBLASLT=1>
      $<$<BOOL:${RNS8_ENABLE_CK}>:RNS8_ENABLE_CK=1>
      $<$<BOOL:${RNS8_ENABLE_ROCWMMA}>:RNS8_ENABLE_ROCWMMA=1>
  )
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(RNS8_ENABLE_HIP)
    target_include_directories(${target_name} PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${RNS8_HIP_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE __HIP_PLATFORM_AMD__=1)
  endif()
  if(RNS8_ENABLE_HIPBLASLT)
    if(RNS8_HIPBLASLT_INCLUDE_DIR)
      target_include_directories(${target_name} PRIVATE "${RNS8_HIPBLASLT_INCLUDE_DIR}")
    endif()
    if(RNS8_HIPBLASLT_TARGET)
      target_link_libraries(${target_name} PRIVATE "${RNS8_HIPBLASLT_TARGET}")
    else()
      target_link_libraries(${target_name} PRIVATE "${RNS8_HIPBLASLT_LIBRARY}")
    endif()
  endif()
  if(RNS8_ENABLE_CK)
    target_include_directories(${target_name} PRIVATE "${RNS8_CK_GENERATED_INCLUDE_DIR}" "${RNS8_CK_INCLUDE_DIR}")
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    target_include_directories(${target_name} PRIVATE "${RNS8_ROCWMMA_INCLUDE_DIR}")
  endif()
  target_link_libraries(${target_name} PRIVATE nlohmann_json::nlohmann_json)
endfunction()

add_library(rns8 SHARED ${RNS8_PUBLIC_HEADERS} ${RNS8_SOURCES})
add_library(rns8::rns8 ALIAS rns8)
rns8_configure_library(rns8)
target_compile_definitions(rns8 PRIVATE RNS8_BUILDING_LIBRARY)
set_target_properties(
  rns8
  PROPERTIES
    OUTPUT_NAME rns8
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_library(rns8_static STATIC ${RNS8_PUBLIC_HEADERS} ${RNS8_SOURCES})
add_library(rns8::rns8_static ALIAS rns8_static)
rns8_configure_library(rns8_static)
target_compile_definitions(rns8_static PUBLIC RNS8_STATIC)
set_target_properties(rns8_static PROPERTIES OUTPUT_NAME rns8_static)

if(RNS8_BUILD_TOOLS)
  add_executable(rns8-inspect tools/rns8_inspect.cpp)
  target_include_directories(rns8-inspect PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(rns8-inspect PRIVATE rns8_static)

  add_executable(rns8-verify tools/rns8_verify.cpp)
  target_include_directories(
    rns8-verify
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8-verify PRIVATE rns8_static)
endif()

if(RNS8_BUILD_BENCHMARKS)
  set(RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS}")
  string(REPLACE ";" "," RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS_TEXT}")
  set(RNS8_HIPCC_PATH_TEXT "")
  set(RNS8_HIPCC_VERSION_TEXT "")
  set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "")
  set(RNS8_HIP_ROOT_TEXT "${RNS8_HIP_ROOT}")
  string(REPLACE "\\" "/" RNS8_HIP_ROOT_TEXT "${RNS8_HIP_ROOT_TEXT}")
  if(RNS8_HIP_ROOT)
    get_filename_component(RNS8_HIP_ROOT_BASENAME "${RNS8_HIP_ROOT}" NAME)
    if(RNS8_HIP_ROOT_BASENAME MATCHES "^[0-9]+(\\.[0-9]+)*$")
      set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "${RNS8_HIP_ROOT_BASENAME}")
    endif()
  endif()
  if(RNS8_ENABLE_HIP AND RNS8_HIP_HIPCC)
    set(RNS8_HIPCC_PATH_TEXT "${RNS8_HIP_HIPCC}")
    string(REPLACE "\\" "/" RNS8_HIPCC_PATH_TEXT "${RNS8_HIPCC_PATH_TEXT}")
    execute_process(
      COMMAND "${RNS8_HIP_HIPCC}" --version
      RESULT_VARIABLE RNS8_HIPCC_VERSION_RESULT
      OUTPUT_VARIABLE RNS8_HIPCC_VERSION_OUTPUT
      ERROR_VARIABLE RNS8_HIPCC_VERSION_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(RNS8_HIPCC_VERSION_RESULT EQUAL 0)
      set(RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_OUTPUT}")
      if(RNS8_HIPCC_VERSION_ERROR)
        string(APPEND RNS8_HIPCC_VERSION_TEXT " ${RNS8_HIPCC_VERSION_ERROR}")
      endif()
      string(REPLACE "\n" " | " RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\r" " " RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\\" "/" RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\"" "'" RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE ";" "," RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
    endif()
  endif()
  set(RNS8_GIT_COMMIT "unknown")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE RNS8_GIT_RESULT
      OUTPUT_VARIABLE RNS8_GIT_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(RNS8_GIT_RESULT EQUAL 0 AND NOT RNS8_GIT_OUTPUT STREQUAL "")
      set(RNS8_GIT_COMMIT "${RNS8_GIT_OUTPUT}")
    endif()
  endif()
  add_executable(rns8-bench benchmarks/rns8_bench.cpp)
  target_compile_definitions(
    rns8-bench
    PRIVATE
      RNS8_CONFIGURED_AMDGPU_TARGETS="${RNS8_AMDGPU_TARGETS_TEXT}"
      RNS8_CONFIGURED_HIP_ENABLED=$<IF:$<BOOL:${RNS8_ENABLE_HIP}>,1,0>
      RNS8_CONFIGURED_HIP_ROOT="${RNS8_HIP_ROOT_TEXT}"
      RNS8_CONFIGURED_HIPCC_PATH="${RNS8_HIPCC_PATH_TEXT}"
      RNS8_CONFIGURED_HIPCC_VERSION="${RNS8_HIPCC_VERSION_TEXT}"
      RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION="${RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT}"
      RNS8_GIT_COMMIT="${RNS8_GIT_COMMIT}"
      RNS8_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_include_directories(rns8-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(rns8-bench PRIVATE rns8_static)
endif()

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
    NAME dependency_checker_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_check_dependencies.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(dependency_checker_self_test PROPERTIES LABELS "tools;dependencies;accelerator")

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
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(benchmark_schema_current_fixtures PROPERTIES LABELS "benchmark;schema")

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

  if(TARGET rns8-inspect)
    add_test(
      NAME inspect_cli_hard_cut_diagnostics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_inspect_cli.py" "$<TARGET_FILE:rns8-inspect>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(inspect_cli_hard_cut_diagnostics PROPERTIES LABELS "tools;cli;hard-cut")
  endif()

  find_package(Catch2 3 CONFIG REQUIRED)
  add_executable(
    rns8_tests
    tests/unit/test_moduli.cpp
    tests/unit/test_residues.cpp
    tests/unit/test_ring_gemm.cpp
    tests/unit/test_crt.cpp
    tests/unit/test_api.cpp
    tests/unit/test_bounded_gemm.cpp
    tests/unit/test_bounded_reference_sweeps.cpp
    tests/unit/test_exact_wide.cpp
    tests/unit/test_semantics.cpp
    tests/unit/test_wrap64.cpp
    tests/differential/test_hip_direct.cpp
    tests/differential/test_hipblaslt.cpp
    tests/differential/test_ck.cpp
    tests/differential/test_wmma.cpp
  )
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
  target_link_libraries(rns8_tests PRIVATE rns8_static Catch2::Catch2WithMain)
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
    list(GET RNS8_AMDGPU_TARGETS 0 RNS8_WMMA_ISA_CHECK_TARGET)
    add_test(
      NAME wmma_kernel_matrix_isa_check
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_wmma_kernel_isa.py"
        --object "${RNS8_WMMA_KERNEL_OBJECT}"
        --target "${RNS8_WMMA_ISA_CHECK_TARGET}"
        --hipcc "${RNS8_HIP_HIPCC}"
        --scratch-root "${CMAKE_CURRENT_SOURCE_DIR}/temp"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(wmma_kernel_matrix_isa_check PROPERTIES LABELS "rocwmma;isa;accelerator")
  endif()
endif()
