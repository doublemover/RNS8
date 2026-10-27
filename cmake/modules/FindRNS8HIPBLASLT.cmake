set(_RNS8_HIPBLASLT_HINTS)
if(RNS8_HIPBLASLT_ROOT)
  list(APPEND _RNS8_HIPBLASLT_HINTS "${RNS8_HIPBLASLT_ROOT}")
endif()

function(_rns8_hipblaslt_append_hint path)
  if(path)
    file(TO_CMAKE_PATH "${path}" _rns8_hipblaslt_hint)
    list(APPEND _RNS8_HIPBLASLT_HINTS "${_rns8_hipblaslt_hint}")
    set(_RNS8_HIPBLASLT_HINTS "${_RNS8_HIPBLASLT_HINTS}" PARENT_SCOPE)
  endif()
endfunction()

if(RNS8_HIP_ROOT)
  _rns8_hipblaslt_append_hint("${RNS8_HIP_ROOT}")
endif()
if(DEFINED ENV{ROCM_PATH})
  _rns8_hipblaslt_append_hint("$ENV{ROCM_PATH}")
endif()
if(DEFINED ENV{HIP_PATH})
  _rns8_hipblaslt_append_hint("$ENV{HIP_PATH}")
endif()
if(WIN32)
  rns8_append_windows_vcpkg_hints(_RNS8_HIPBLASLT_HINTS)
  list(APPEND _RNS8_HIPBLASLT_HINTS "C:/Program Files/AMD/ROCm/7.1" "C:/Program Files/AMD/ROCm/6.4")
endif()
list(APPEND _RNS8_HIPBLASLT_HINTS "/opt/rocm")

if(RNS8_DISCOVERY_IDENTITY_SHA256 AND RNS8_HIPBLASLT_ROOT)
  # NO_DEFAULT_PATH does not exclude other HINTS. A missing package in the
  # selected root must not import targets from another SDK or vcpkg prefix.
  set(_RNS8_HIPBLASLT_HINTS "${RNS8_HIPBLASLT_ROOT}")
  set(_RNS8_HIPBLASLT_PACKAGE_POLICY NO_DEFAULT_PATH)
  if(hipblaslt_DIR AND NOT hipblaslt_DIR MATCHES "-NOTFOUND$")
    file(REAL_PATH "${hipblaslt_DIR}" _rns8_package_dir)
    file(REAL_PATH "${RNS8_HIPBLASLT_ROOT}" _rns8_package_root)
    string(FIND "${_rns8_package_dir}/" "${_rns8_package_root}/" _rns8_package_prefix)
    if(NOT _rns8_package_prefix EQUAL 0)
      message(FATAL_ERROR "hipblaslt_DIR is outside the shared selected RNS8_HIPBLASLT_ROOT; remove the stale cache or select one coherent installation")
    endif()
  endif()
endif()
find_package(
  hipblaslt
  CONFIG
  QUIET
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES lib/cmake/hipblaslt
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

find_path(
  RNS8_HIPBLASLT_INCLUDE_DIR
  NAMES hipblaslt/hipblaslt.h hipblaslt.h
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES include
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

find_library(
  RNS8_HIPBLASLT_LIBRARY
  NAMES hipblaslt
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES lib lib64 bin
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

find_file(
  RNS8_HIPBLASLT_DLL_A
  NAMES libhipblaslt.dll.a
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES lib lib64
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

find_file(
  RNS8_HIPBLASLT_DLL
  NAMES libhipblaslt.dll hipblaslt.dll
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES bin
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

if(NOT RNS8_HIPBLASLT_LIBRARY AND RNS8_HIPBLASLT_DLL_A)
  set(RNS8_HIPBLASLT_LIBRARY "${RNS8_HIPBLASLT_DLL_A}")
endif()

set(RNS8_HIPBLASLT_TARGET "")
set(RNS8_HIPBLASLT_IMPORTED_IMPLIB "")
set(RNS8_HIPBLASLT_IMPORTED_LOCATION "")
if(TARGET roc::hipblaslt)
  set(RNS8_HIPBLASLT_TARGET "roc::hipblaslt")
  get_target_property(RNS8_HIPBLASLT_IMPORTED_IMPLIB roc::hipblaslt IMPORTED_IMPLIB_RELEASE)
  get_target_property(RNS8_HIPBLASLT_IMPORTED_LOCATION roc::hipblaslt IMPORTED_LOCATION_RELEASE)
  if(NOT RNS8_HIPBLASLT_IMPORTED_IMPLIB OR RNS8_HIPBLASLT_IMPORTED_IMPLIB STREQUAL "RNS8_HIPBLASLT_IMPORTED_IMPLIB-NOTFOUND")
    set(RNS8_HIPBLASLT_IMPORTED_IMPLIB "")
  endif()
  if(NOT RNS8_HIPBLASLT_IMPORTED_LOCATION OR RNS8_HIPBLASLT_IMPORTED_LOCATION STREQUAL "RNS8_HIPBLASLT_IMPORTED_LOCATION-NOTFOUND")
    set(RNS8_HIPBLASLT_IMPORTED_LOCATION "")
  endif()
  if(NOT RNS8_HIPBLASLT_INCLUDE_DIR)
    get_target_property(_RNS8_HIPBLASLT_TARGET_INCLUDES roc::hipblaslt INTERFACE_INCLUDE_DIRECTORIES)
    if(_RNS8_HIPBLASLT_TARGET_INCLUDES)
      list(GET _RNS8_HIPBLASLT_TARGET_INCLUDES 0 RNS8_HIPBLASLT_INCLUDE_DIR)
    endif()
  endif()
  if(NOT RNS8_HIPBLASLT_LIBRARY AND RNS8_HIPBLASLT_IMPORTED_IMPLIB)
    set(RNS8_HIPBLASLT_LIBRARY "${RNS8_HIPBLASLT_IMPORTED_IMPLIB}")
  endif()
  if(NOT RNS8_HIPBLASLT_DLL AND RNS8_HIPBLASLT_IMPORTED_LOCATION)
    set(RNS8_HIPBLASLT_DLL "${RNS8_HIPBLASLT_IMPORTED_LOCATION}")
  endif()
endif()

find_program(
  RNS8_HIPBLASLT_BENCH
  NAMES hipblaslt-bench hipblaslt-bench.exe
  HINTS ${_RNS8_HIPBLASLT_HINTS}
  PATH_SUFFIXES bin
  ${_RNS8_HIPBLASLT_PACKAGE_POLICY}
)

set(RNS8_HIPBLASLT_CANDIDATE FALSE)
set(RNS8_HIPBLASLT_EVIDENCE "not discovered")
if(RNS8_HIPBLASLT_TARGET OR RNS8_HIPBLASLT_INCLUDE_DIR OR RNS8_HIPBLASLT_LIBRARY OR RNS8_HIPBLASLT_DLL OR RNS8_HIPBLASLT_BENCH)
  rns8_assert_no_linux_windows_vcpkg_paths("hipBLASLt include directory" "${RNS8_HIPBLASLT_INCLUDE_DIR}")
  rns8_assert_no_linux_windows_vcpkg_paths("hipBLASLt library" "${RNS8_HIPBLASLT_LIBRARY}")
  rns8_assert_no_linux_windows_vcpkg_paths("hipBLASLt DLL import library" "${RNS8_HIPBLASLT_DLL_A}")
  rns8_assert_no_linux_windows_vcpkg_paths("hipBLASLt DLL" "${RNS8_HIPBLASLT_DLL}")
  rns8_assert_no_linux_windows_vcpkg_paths("hipBLASLt bench tool" "${RNS8_HIPBLASLT_BENCH}")
  if(RNS8_HIPBLASLT_TARGET)
    rns8_assert_no_linux_windows_vcpkg_target("${RNS8_HIPBLASLT_TARGET}")
  endif()
  set(RNS8_HIPBLASLT_CANDIDATE TRUE)
  set(RNS8_HIPBLASLT_EVIDENCE)
  if(RNS8_HIPBLASLT_TARGET)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "target=${RNS8_HIPBLASLT_TARGET}")
  endif()
  if(RNS8_HIPBLASLT_INCLUDE_DIR)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "header=${RNS8_HIPBLASLT_INCLUDE_DIR}")
  endif()
  if(RNS8_HIPBLASLT_LIBRARY)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "library=${RNS8_HIPBLASLT_LIBRARY}")
  endif()
  if(RNS8_HIPBLASLT_IMPORTED_IMPLIB)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "imported_implib=${RNS8_HIPBLASLT_IMPORTED_IMPLIB}")
  endif()
  if(RNS8_HIPBLASLT_DLL)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "dll=${RNS8_HIPBLASLT_DLL}")
  endif()
  if(RNS8_HIPBLASLT_BENCH)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "tool=${RNS8_HIPBLASLT_BENCH}")
  endif()
endif()

set(RNS8_HIPBLASLT_MSVC_LINK_PROBE "not_run")
set(RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT "")
if(RNS8_PROBE_ACCELERATORS AND MSVC AND TARGET roc::hipblaslt)
  set(_RNS8_HIPBLASLT_PROBE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-msvc-link-probe.cpp")
  file(
    WRITE
    "${_RNS8_HIPBLASLT_PROBE_SOURCE}"
    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hipblaslt/hipblaslt_headers.hpp\"\nint main() { hipblasLtHandle_t h = nullptr; return hipblasLtCreate(&h) == HIPBLAS_STATUS_SUCCESS && h ? (hipblasLtDestroy(h), 0) : 0; }\n"
  )
  try_compile(
    RNS8_HIPBLASLT_MSVC_LINK_PROBE_PASSED
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-msvc-link-probe"
    SOURCES "${_RNS8_HIPBLASLT_PROBE_SOURCE}"
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "-D__HIP_PLATFORM_AMD__=1"
    LINK_LIBRARIES roc::hipblaslt
    OUTPUT_VARIABLE RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT
  )
  if(RNS8_HIPBLASLT_MSVC_LINK_PROBE_PASSED)
    set(RNS8_HIPBLASLT_MSVC_LINK_PROBE "passed")
  elseif(RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT MATCHES "Cannot open include file: '(memory|cstdint)'")
    set(RNS8_HIPBLASLT_MSVC_LINK_PROBE "not_run_missing_msvc_environment")
  else()
    set(RNS8_HIPBLASLT_MSVC_LINK_PROBE "failed")
  endif()
  string(REPLACE "\n" " | " RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT "${RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT}")
  string(REPLACE "\r" " " RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT "${RNS8_HIPBLASLT_MSVC_LINK_PROBE_OUTPUT}")
  if(RNS8_HIPBLASLT_EVIDENCE)
    list(APPEND RNS8_HIPBLASLT_EVIDENCE "msvc_link_probe=${RNS8_HIPBLASLT_MSVC_LINK_PROBE}")
  endif()
endif()

set(RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE FALSE)
# This selects an optional compiled API in an explicitly enabled backend.
# It is independent of the evidence-only component-discovery probes above.
if(RNS8_ENABLE_HIPBLASLT AND RNS8_HIP_HIPCC AND RNS8_HIPBLASLT_LIBRARY)
  set(_rns8_algo_index_source "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-algorithm-index-probe.hip")
  file(WRITE "${_rns8_algo_index_source}"
    "#include <hip/hip_runtime.h>\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hipblaslt/hipblaslt_algorithm_index_bridge.hpp\"\n")
  set(_rns8_algo_index_args -std=c++17)
  if(NOT WIN32)
    list(APPEND _rns8_algo_index_args -fPIC)
  endif()
  foreach(_target IN LISTS RNS8_AMDGPU_TARGETS)
    list(APPEND _rns8_algo_index_args "--offload-arch=${_target}")
  endforeach()
  if(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND _rns8_algo_index_args -fms-runtime-lib=dll_dbg -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=2)
    else()
      list(APPEND _rns8_algo_index_args -fms-runtime-lib=dll -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0)
    endif()
  endif()
  foreach(_include IN ITEMS "${RNS8_HIP_STL_OVERLAY}" "${RNS8_HIPBLASLT_INCLUDE_DIR}")
    if(_include)
      if(WIN32)
        # hipcc forwards attached include flags as command text on Windows.
        list(APPEND _rns8_algo_index_args "-I\"${_include}\"")
      else()
        list(APPEND _rns8_algo_index_args "-I${_include}")
      endif()
    endif()
  endforeach()
  set(_rns8_algo_index_object "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-algorithm-index-probe${CMAKE_CXX_OUTPUT_EXTENSION}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${RNS8_HIP_COMPILER_ENVIRONMENT}
      "${RNS8_HIP_HIPCC}" ${_rns8_algo_index_args} -c "${_rns8_algo_index_source}" -o "${_rns8_algo_index_object}"
    RESULT_VARIABLE _rns8_algo_index_result OUTPUT_VARIABLE _rns8_algo_index_stdout
    ERROR_VARIABLE _rns8_algo_index_stderr TIMEOUT 60)
  set(RNS8_HIPBLASLT_ALGO_INDEX_PROBE_OUTPUT "${_rns8_algo_index_stdout}\n${_rns8_algo_index_stderr}")
  if(_rns8_algo_index_result STREQUAL "0" AND TARGET roc::hipblaslt)
    set(_rns8_algo_index_main "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-algorithm-index-main.cpp")
    file(WRITE "${_rns8_algo_index_main}"
      "#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hipblaslt/hipblaslt_headers.hpp\"\nextern \"C\" int rns8_hipblaslt_algorithm_index(const hipblasLtMatmulAlgo_t*);\nint main() { hipblasLtMatmulAlgo_t algo{}; return rns8_hipblaslt_algorithm_index(&algo); }\n")
    if(DEFINED CMAKE_TRY_COMPILE_CONFIGURATION)
      set(_rns8_had_try_configuration TRUE)
      set(_rns8_saved_try_configuration "${CMAKE_TRY_COMPILE_CONFIGURATION}")
    else()
      set(_rns8_had_try_configuration FALSE)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_TRY_COMPILE_CONFIGURATION Debug)
    else()
      set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
    endif()
    try_compile(_rns8_algo_index_linked
      "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-algorithm-index-link-probe"
      SOURCES "${_rns8_algo_index_main}" CXX_STANDARD 17
      COMPILE_DEFINITIONS "-D__HIP_PLATFORM_AMD__=1"
      LINK_LIBRARIES "${_rns8_algo_index_object}" roc::hipblaslt ${RNS8_HIP_LIBRARIES}
      OUTPUT_VARIABLE _rns8_algo_index_link_output)
    if(_rns8_had_try_configuration)
      set(CMAKE_TRY_COMPILE_CONFIGURATION "${_rns8_saved_try_configuration}")
    else()
      unset(CMAKE_TRY_COMPILE_CONFIGURATION)
    endif()
    string(APPEND RNS8_HIPBLASLT_ALGO_INDEX_PROBE_OUTPUT "\n${_rns8_algo_index_link_output}")
    if(_rns8_algo_index_linked)
      set(RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE TRUE)
    endif()
  endif()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-algorithm-index-probe.log"
    "compiler=${RNS8_HIP_HIPCC}\ncompile_result=${_rns8_algo_index_result}\navailable=${RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE}\n${RNS8_HIPBLASLT_ALGO_INDEX_PROBE_OUTPUT}")
endif()
if(RNS8_HIPBLASLT_EVIDENCE)
  list(APPEND RNS8_HIPBLASLT_EVIDENCE "algorithm_index_query=${RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE}")
endif()

set(RNS8_HIPBLASLT_BACKEND_READY FALSE)
set(RNS8_HIPBLASLT_BACKEND_ENABLEMENT "disabled")
set(RNS8HIPBLASLT_FOUND "${RNS8_HIPBLASLT_CANDIDATE}")
