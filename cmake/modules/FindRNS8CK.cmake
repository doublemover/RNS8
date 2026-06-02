set(_RNS8_CK_HINTS)
foreach(_RNS8_CK_HINT IN ITEMS
    "${RNS8_CK_ROOT}"
    "${RNS8_ROCM_DEPS_INSTALL_ROOT}"
    "${RNS8_ROCM_DEPS_BUILD_ROOT}/composable_kernel"
    "${RNS8_HIP_ROOT}"
    "$ENV{ROCM_PATH}"
    "$ENV{HIP_PATH}"
    "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}"
    "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}"
    "/opt/rocm")
  if(_RNS8_CK_HINT)
    list(APPEND _RNS8_CK_HINTS "${_RNS8_CK_HINT}")
  endif()
endforeach()

find_path(
  RNS8_CK_INCLUDE_DIR
  NAMES ck/ck.hpp
  HINTS ${_RNS8_CK_HINTS}
  PATH_SUFFIXES include
)

set(RNS8_CK_CANDIDATE FALSE)
set(RNS8_CK_EVIDENCE "not discovered")
set(RNS8_CK_COMPILE_PROBE_STATUS "not_run_missing_header")
set(RNS8_CK_COMPILE_PROBE_OUTPUT "")
set(RNS8_CK_GENERATED_INCLUDE_DIR "")

if(RNS8_CK_INCLUDE_DIR)
  set(RNS8_CK_CANDIDATE TRUE)
  set(RNS8_CK_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/accelerator-dependency-probes/ck-generated/include")
  file(MAKE_DIRECTORY "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck")

  set(_RNS8_CK_COMMIT_ID "unknown")
  find_package(Git QUIET)
  if(Git_FOUND AND RNS8_CK_ROOT AND EXISTS "${RNS8_CK_ROOT}/.git")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${RNS8_CK_ROOT}" rev-parse HEAD
      RESULT_VARIABLE _RNS8_CK_GIT_RESULT
      OUTPUT_VARIABLE _RNS8_CK_GIT_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_RNS8_CK_GIT_RESULT EQUAL 0 AND NOT _RNS8_CK_GIT_OUTPUT STREQUAL "")
      set(_RNS8_CK_COMMIT_ID "${_RNS8_CK_GIT_OUTPUT}")
    endif()
  endif()

  file(
    WRITE "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck/config.h"
    "#ifndef CK_CONFIG_H_IN\n"
    "#define CK_CONFIG_H_IN\n"
    "#define CK_ENABLE_INT8 ON\n"
    "#define CK_ENABLE_FP8 ON\n"
    "#define CK_ENABLE_BF8 ON\n"
    "#define CK_ENABLE_FP16 ON\n"
    "#define CK_ENABLE_BF16 ON\n"
    "#define CK_ENABLE_FP32 ON\n"
    "#define CK_ENABLE_FP64 ON\n"
    "#define CK_ENABLE_DL_KERNELS ON\n"
    "#define CK_ENABLE_DPP_KERNELS ON\n"
    "#define CK_USE_WMMA ON\n"
    "#endif\n"
  )
  file(
    WRITE "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck/version.h"
    "#ifndef CK_VERSION_H_\n"
    "#define CK_VERSION_H_\n"
    "#define CK_VERSION 1.1.0\n"
    "#define CK_VERSION_MAJOR 1\n"
    "#define CK_VERSION_MINOR 1\n"
    "#define CK_VERSION_PATCH 0\n"
    "#define CK_COMMIT_ID ${_RNS8_CK_COMMIT_ID}\n"
    "#endif\n"
  )

  set(_RNS8_CK_TARGET "gfx1100")
  if(RNS8_AMDGPU_TARGETS)
    set(_RNS8_CK_TARGETS ${RNS8_AMDGPU_TARGETS})
    list(GET _RNS8_CK_TARGETS 0 _RNS8_CK_TARGET)
  endif()

  find_program(
    RNS8_CK_HIPCC
    NAMES hipcc hipcc.exe hipcc.bat
    HINTS "${RNS8_HIP_ROOT}/bin" "$ENV{HIP_PATH}/bin" "$ENV{ROCM_PATH}/bin"
  )

  if(RNS8_CK_HIPCC)
    set(_RNS8_CK_PROBE_DIR "${CMAKE_CURRENT_BINARY_DIR}/accelerator-dependency-probes")
    file(MAKE_DIRECTORY "${_RNS8_CK_PROBE_DIR}")
    set(_RNS8_CK_PROBE_SOURCE "${_RNS8_CK_PROBE_DIR}/ck_probe.cpp")
    set(_RNS8_CK_PROBE_BINARY "${_RNS8_CK_PROBE_DIR}/ck_probe${CMAKE_EXECUTABLE_SUFFIX}")
    file(
      WRITE "${_RNS8_CK_PROBE_SOURCE}"
      "#include <ck/ck.hpp>\n"
      "#include <ck_tile/core.hpp>\n"
      "__global__ void rns8_ck_dependency_probe_kernel() {}\n"
      "int main() { return 0; }\n"
    )
    execute_process(
      COMMAND
        "${RNS8_CK_HIPCC}"
        "-std=c++17"
        "--offload-arch=${_RNS8_CK_TARGET}"
        "-I${RNS8_CK_GENERATED_INCLUDE_DIR}"
        "-I${RNS8_CK_INCLUDE_DIR}"
        "${_RNS8_CK_PROBE_SOURCE}"
        "-o"
        "${_RNS8_CK_PROBE_BINARY}"
      RESULT_VARIABLE _RNS8_CK_PROBE_RESULT
      OUTPUT_VARIABLE _RNS8_CK_PROBE_STDOUT
      ERROR_VARIABLE _RNS8_CK_PROBE_STDERR
      TIMEOUT 90
    )
    set(RNS8_CK_COMPILE_PROBE_OUTPUT "${_RNS8_CK_PROBE_STDOUT}${_RNS8_CK_PROBE_STDERR}")
    if(_RNS8_CK_PROBE_RESULT EQUAL 0)
      set(RNS8_CK_COMPILE_PROBE_STATUS "compile_probe_pass")
    else()
      set(RNS8_CK_COMPILE_PROBE_STATUS "compile_probe_fail")
    endif()
  else()
    set(RNS8_CK_COMPILE_PROBE_STATUS "not_run_missing_hipcc")
  endif()

  set(
    RNS8_CK_EVIDENCE
    "header=${RNS8_CK_INCLUDE_DIR};generated_include=${RNS8_CK_GENERATED_INCLUDE_DIR};compile_probe=${RNS8_CK_COMPILE_PROBE_STATUS}"
  )
endif()

set(RNS8_CK_BACKEND_READY FALSE)
set(RNS8_CK_BACKEND_ENABLEMENT "disabled")
set(RNS8CK_FOUND FALSE)
