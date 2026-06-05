cmake_minimum_required(VERSION 3.28)

project(
  RNS8
  VERSION 0.1.0
  DESCRIPTION "Exact integer GEMM with AMD GPU matrix engines, RNS, and CRT reconstruction"
  LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(RNS8_ENABLE_HIP "Build the direct HIP backend with explicit hipcc integration" OFF)
option(RNS8_BUILD_TESTS "Build RNS8 correctness tests" ON)
option(RNS8_BUILD_TOOLS "Build RNS8 command-line tools" ON)
option(RNS8_BUILD_EXAMPLES "Build CPU-only public API examples" ON)
option(RNS8_BUILD_BENCHMARKS "Build RNS8 benchmark shell" ON)
option(RNS8_ENABLE_GMP "Require GMP and build the optional exact-reference link smoke" OFF)
option(RNS8_ENABLE_FLINT "Require FLINT and build the optional exact-reference link smoke" OFF)
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
get_target_property(RNS8_NLOHMANN_JSON_INCLUDE_DIRS nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
if(NOT RNS8_NLOHMANN_JSON_INCLUDE_DIRS)
  set(RNS8_NLOHMANN_JSON_INCLUDE_DIRS "")
endif()

if(RNS8_ENABLE_GMP OR RNS8_ENABLE_FLINT)
  find_package(RNS8ThirdParty REQUIRED)
endif()

if(RNS8_ENABLE_GMP AND NOT RNS8_GMP_FOUND)
  message(
    FATAL_ERROR
      "RNS8_ENABLE_GMP=ON requires GMP headers and library. "
      "Install vcpkg feature optional-exact-libs or point VCPKG_ROOT/RNS8_HIP_ROOT at a tree containing gmp.h and gmp/libgmp."
  )
endif()

if(RNS8_ENABLE_FLINT AND NOT RNS8_FLINT_FOUND)
  message(
    FATAL_ERROR
      "RNS8_ENABLE_FLINT=ON requires FLINT headers and library. "
      "Install vcpkg feature optional-exact-libs or point VCPKG_ROOT/RNS8_HIP_ROOT at a tree containing flint/flint.h and flint/libflint."
  )
endif()

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
  src/core/api_backend_info.cpp
  src/core/api_context.cpp
  src/core/api_export.cpp
  src/core/api_gemm.cpp
  src/core/api_matrix_workspace.cpp
  src/core/api_oneshot.cpp
  src/core/api_pack.cpp
  src/core/api_plan.cpp
  src/core/backend_common.cpp
  src/core/moduli.cpp
  src/core/plan_lowering.cpp
  src/core/status.cpp
  src/cpu/cpu_reference.cpp
  src/reconstruct/crt.cpp
  src/backend_hip_direct/hip_backend.cpp
  src/backend_hip_direct/hip_timing.cpp
  src/backend_vector_alu/vector_alu_backend.cpp
  src/backend_ck/ck_backend.cpp
  src/backend_rocwmma/rocwmma_backend.cpp
  src/backend_wrap64/wrap64_reference.cpp
  src/backend_wrap64/wrap64_hip.cpp
)

function(rns8_apply_ck_wmma_no_divide_patch out_include_dir out_header)
  set(_ck_gridwise_source "${RNS8_CK_INCLUDE_DIR}/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma.hpp")
  set(_ck_patch_include_dir "${CMAKE_CURRENT_BINARY_DIR}/rns8_ck_wmma_no_divide_include")
  set(_ck_gridwise "${_ck_patch_include_dir}/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma.hpp")
  if(NOT EXISTS "${_ck_gridwise_source}")
    message(FATAL_ERROR "CK WMMA gridwise header not found: ${_ck_gridwise_source}")
  endif()

  set(_ck_old [=[    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }]=])
  set(_ck_new [=[    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t M01, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n, M01);
    }]=])

  file(READ "${_ck_gridwise_source}" _ck_gridwise_text)
  string(FIND "${_ck_gridwise_text}" "${_ck_new}" _ck_new_offset)
  if(_ck_new_offset EQUAL -1)
    string(FIND "${_ck_gridwise_text}" "${_ck_old}" _ck_old_offset)
    if(_ck_old_offset EQUAL -1)
      message(
        FATAL_ERROR
        "CK WMMA no-divide integration patch does not match ${_ck_gridwise_source}; "
        "inspect CK's MakeDefaultBlock2CTileMap before enabling RNS8_ENABLE_CK"
      )
    endif()
    string(REPLACE "${_ck_old}" "${_ck_new}" _ck_gridwise_text "${_ck_gridwise_text}")
    message(STATUS "RNS8 generated CK WMMA no-divide block-map include overlay: ${_ck_gridwise}")
  else()
    message(STATUS "RNS8 CK WMMA no-divide block-map patch already present in source: ${_ck_gridwise_source}")
  endif()
  get_filename_component(_ck_gridwise_dir "${_ck_gridwise}" DIRECTORY)
  file(MAKE_DIRECTORY "${_ck_gridwise_dir}")
  file(WRITE "${_ck_gridwise}" "${_ck_gridwise_text}")
  set(${out_include_dir} "${_ck_patch_include_dir}" PARENT_SCOPE)
  set(${out_header} "${_ck_gridwise}" PARENT_SCOPE)
endfunction()

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
  rns8_compile_hip_source(
    RNS8_VECTOR_ALU_KERNEL_OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_vector_alu/vector_alu_kernels.hip"
  )
  list(
    APPEND
    RNS8_SOURCES
    "${RNS8_HIP_DIRECT_KERNEL_OBJECT}"
    "${RNS8_WRAP64_HIP_KERNEL_OBJECT}"
    "${RNS8_VECTOR_ALU_KERNEL_OBJECT}"
  )
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
    rns8_apply_ck_wmma_no_divide_patch(RNS8_CK_WMMA_PATCH_INCLUDE_DIR RNS8_CK_WMMA_PATCHED_HEADER)
    set(RNS8_HIP_SOURCE_INCLUDE_DIRS "${RNS8_CK_WMMA_PATCH_INCLUDE_DIR}" "${RNS8_CK_GENERATED_INCLUDE_DIR}" "${RNS8_CK_INCLUDE_DIR}")
    set(RNS8_HIP_SOURCE_DEPENDS "${RNS8_CK_WMMA_PATCHED_HEADER}")
    if(MSVC)
      set(RNS8_HIP_SOURCE_COMPILE_OPTIONS "-D_CRT_SECURE_NO_WARNINGS")
    endif()
    rns8_compile_hip_source(
      RNS8_CK_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_ck/ck_backend_kernels.hip"
    )
    unset(RNS8_HIP_SOURCE_INCLUDE_DIRS)
    unset(RNS8_HIP_SOURCE_DEPENDS)
    unset(RNS8_HIP_SOURCE_COMPILE_OPTIONS)
    list(APPEND RNS8_SOURCES "${RNS8_CK_KERNEL_OBJECT}")
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    find_package(RNS8ROCWMMA REQUIRED)
    set(RNS8_HIP_SOURCE_INCLUDE_DIRS "${RNS8_ROCWMMA_INCLUDE_DIR}")
    rns8_compile_hip_source(
      RNS8_ROCWMMA_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_rocwmma/rocwmma_backend_kernels.hip"
    )
    unset(RNS8_HIP_SOURCE_INCLUDE_DIRS)
    list(APPEND RNS8_SOURCES "${RNS8_ROCWMMA_KERNEL_OBJECT}")
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
      ${RNS8_NLOHMANN_JSON_INCLUDE_DIRS}
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE
      $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>
      $<$<BOOL:${RNS8_ENABLE_HIPBLASLT}>:RNS8_ENABLE_HIPBLASLT=1>
      $<$<BOOL:${RNS8_ENABLE_CK}>:RNS8_ENABLE_CK=1>
      $<$<BOOL:${RNS8_ENABLE_ROCWMMA}>:RNS8_ENABLE_ROCWMMA=1>
      $<$<BOOL:${RNS8_ENABLE_GMP}>:RNS8_ENABLE_GMP=1>
      $<$<BOOL:${RNS8_ENABLE_FLINT}>:RNS8_ENABLE_FLINT=1>
  )
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    if(RNS8_ENABLE_HIP)
      # AMD HIP SDK vector headers intentionally use nameless unions that trip MSVC /W4.
      target_compile_options(${target_name} PRIVATE /wd4201)
    endif()
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
  set(
    RNS8_BENCHMARK_SOURCES
    benchmarks/rns8_bench.cpp
    benchmarks/rns8_bench_modes.cpp
    benchmarks/rns8_bench_support.cpp
  )
  if(RNS8_ENABLE_HIP)
    rns8_compile_hip_source(
      RNS8_BENCH_VECTOR_ALU_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/hip_vector_alu_baseline_kernels.hip"
    )
    list(APPEND RNS8_BENCHMARK_SOURCES "${RNS8_BENCH_VECTOR_ALU_KERNEL_OBJECT}")
  endif()
  add_executable(rns8-bench ${RNS8_BENCHMARK_SOURCES})
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
  target_include_directories(
    rns8-bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8-bench PRIVATE rns8_static)
  if(RNS8_ENABLE_HIP)
    target_include_directories(rns8-bench PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_compile_definitions(rns8-bench PRIVATE __HIP_PLATFORM_AMD__=1 RNS8_ENABLE_HIP=1)
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    target_compile_definitions(rns8-bench PRIVATE RNS8_ENABLE_ROCWMMA=1)
  endif()
endif()

if(RNS8_BUILD_EXAMPLES)
  add_subdirectory(examples)
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
    NAME starfoundry_report_self_test
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_starfoundry_reports.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(starfoundry_report_self_test PROPERTIES LABELS "benchmark;evidence;schema")

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
  add_executable(
    rns8_tests
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
    tests/differential/test_hip_direct.cpp
    tests/differential/test_hipblaslt.cpp
    tests/differential/test_ck.cpp
    tests/differential/test_rocwmma.cpp
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
  if(RNS8_ENABLE_HIP)
    target_include_directories(rns8_tests PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_link_libraries(rns8_tests PRIVATE ${RNS8_HIP_LIBRARIES})
    target_compile_definitions(rns8_tests PRIVATE __HIP_PLATFORM_AMD__=1)
  endif()
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

set(RNS8_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/RNS8")

install(
  TARGETS rns8 rns8_static
  EXPORT RNS8Targets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Runtime
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Development
)

install(
  DIRECTORY include/rns8
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Development
)

install(
  EXPORT RNS8Targets
  FILE RNS8Targets.cmake
  NAMESPACE rns8::
  DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
  COMPONENT Development
)

configure_package_config_file(
  cmake/RNS8Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/RNS8Config.cmake"
  INSTALL_DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RNS8ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/RNS8Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/RNS8ConfigVersion.cmake"
  DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
  COMPONENT Development
)
