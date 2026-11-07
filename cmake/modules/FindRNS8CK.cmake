set(_RNS8_CK_HINTS)
foreach(_RNS8_CK_HINT IN ITEMS
    "${RNS8_CK_ROOT}"
    "${RNS8_ROCM_DEPS_INSTALL_ROOT}"
    "${RNS8_ROCM_DEPS_BUILD_ROOT}/composable_kernel"
    "${RNS8_HIP_ROOT}"
    "$ENV{ROCM_PATH}"
    "$ENV{HIP_PATH}"
    "/opt/rocm")
  if(_RNS8_CK_HINT)
    list(APPEND _RNS8_CK_HINTS "${_RNS8_CK_HINT}")
  endif()
endforeach()
rns8_append_windows_vcpkg_hints(_RNS8_CK_HINTS)

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
set(RNS8_CK_PRIMITIVE_PROBE_STATUS "not_run_missing_header")
set(RNS8_CK_PRIMITIVE_PROBE_OUTPUT "")
set(RNS8_CK_GENERATED_INCLUDE_DIR "")

if(RNS8_CK_INCLUDE_DIR)
  include("${CMAKE_CURRENT_LIST_DIR}/../MatrixTargetContract.cmake")
  set(_RNS8_CK_TARGETS ${RNS8_AMDGPU_TARGETS})
  if(NOT _RNS8_CK_TARGETS)
    set(_RNS8_CK_TARGETS gfx1100)
  endif()
  list(REMOVE_DUPLICATES _RNS8_CK_TARGETS)
  rns8_matrix_target_contract(_rns8_ck_wave CK ${_RNS8_CK_TARGETS})
  if(_rns8_ck_wave EQUAL 64)
    set(RNS8_CK_USE_XDL 1)
    set(RNS8_CK_SHAPE_ALIGNMENT 128)
  else()
    set(RNS8_CK_USE_XDL 0)
    set(RNS8_CK_SHAPE_ALIGNMENT 64)
  endif()
  include("${CMAKE_CURRENT_LIST_DIR}/RNS8CKInputContract.cmake")
  if(RNS8_CK_USE_XDL)
    set(RNS8_CK_WMMA_SIGNED_INPUTS 1)
  else()
    set(_rns8_ck_wmma_contract "${RNS8_CK_INCLUDE_DIR}/ck/tensor_operation/gpu/warp/wmma_gemm.hpp")
    rns8_ck_wmma_input_contract("${_rns8_ck_wmma_contract}" RNS8_CK_WMMA_SIGNED_INPUTS)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rns8_ck_wmma_contract}")
  endif()
  rns8_assert_no_linux_windows_vcpkg_paths("CK include directory" "${RNS8_CK_INCLUDE_DIR}")
  # Identify the selected dependency by content, including SDK installs without
  # Git metadata. The full dependency-tree manifest remains the wider receipt.
  if(RNS8_CK_USE_XDL)
    set(_rns8_ck_selected_header "${RNS8_CK_INCLUDE_DIR}/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp")
  else()
    set(_rns8_ck_selected_header "${RNS8_CK_INCLUDE_DIR}/ck/tensor_operation/gpu/warp/wmma_gemm.hpp")
  endif()
  file(SHA256 "${_rns8_ck_selected_header}" _rns8_ck_implementation_sha)
  set(_rns8_ck_identity "${_rns8_ck_implementation_sha}")
  foreach(_rns8_ck_identity_header IN ITEMS ck/ck.hpp ck/version.h)
    if(EXISTS "${RNS8_CK_INCLUDE_DIR}/${_rns8_ck_identity_header}")
      file(SHA256 "${RNS8_CK_INCLUDE_DIR}/${_rns8_ck_identity_header}" _rns8_ck_identity_sha)
      string(APPEND _rns8_ck_identity ":${_rns8_ck_identity_sha}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RNS8_CK_INCLUDE_DIR}/${_rns8_ck_identity_header}")
    endif()
  endforeach()
  string(SHA256 _rns8_ck_dependency_sha "${_rns8_ck_identity}")
  string(SUBSTRING "${_rns8_ck_dependency_sha}" 0 48 _rns8_ck_public_sha)
  set(RNS8_CK_DEPENDENCY_ID "CK-sha256-192:${_rns8_ck_public_sha}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rns8_ck_selected_header}")
  set(RNS8_CK_CANDIDATE TRUE)
  set(RNS8_CK_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/accelerator-dependency-probes/ck-generated/include")
  file(MAKE_DIRECTORY "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck")

  set(_RNS8_CK_COMMIT_ID "unknown")
  find_package(Git QUIET)
  if(Git_FOUND AND RNS8_CK_ROOT AND
      (EXISTS "${RNS8_CK_ROOT}/.git" OR EXISTS "${RNS8_CK_ROOT}/../../.git"))
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
    "#define CK_USE_XDL ON\n"
    "#define CK_USE_WMMA ON\n"
    "#endif\n"
  )
  if(EXISTS "${RNS8_CK_INCLUDE_DIR}/ck/version.h")
    configure_file("${RNS8_CK_INCLUDE_DIR}/ck/version.h"
      "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck/version.h" COPYONLY)
  else()
    # Source distributions generate this header. Read their declared version
    # instead of labelling every supported CK checkout as the original 1.1.0.
    if(NOT EXISTS "${RNS8_CK_ROOT}/CMakeLists.txt")
      message(FATAL_ERROR "CK requires an installed version.h or source CMakeLists.txt")
    endif()
    file(STRINGS "${RNS8_CK_ROOT}/CMakeLists.txt" _ck_version_line
      REGEX "^set\\(version [0-9]+\\.[0-9]+\\.[0-9]+\\)")
    if(NOT _ck_version_line MATCHES "^set\\(version ([0-9]+)\\.([0-9]+)\\.([0-9]+)\\)$")
      message(FATAL_ERROR "Cannot determine the selected CK source version")
    endif()
    set(_ck_major "${CMAKE_MATCH_1}")
    set(_ck_minor "${CMAKE_MATCH_2}")
    set(_ck_patch "${CMAKE_MATCH_3}")
    file(WRITE "${RNS8_CK_GENERATED_INCLUDE_DIR}/ck/version.h"
    "#ifndef CK_VERSION_H_\n"
    "#define CK_VERSION_H_\n"
    "#define CK_VERSION ${_ck_major}.${_ck_minor}.${_ck_patch}\n"
    "#define CK_VERSION_MAJOR ${_ck_major}\n"
    "#define CK_VERSION_MINOR ${_ck_minor}\n"
    "#define CK_VERSION_PATCH ${_ck_patch}\n"
    "#define CK_COMMIT_ID ${_RNS8_CK_COMMIT_ID}\n"
    "#endif\n"
  )
  endif()

  if(RNS8_DISCOVERY_IDENTITY_SHA256)
    set(RNS8_CK_HIPCC "${RNS8_HIP_HIPCC}" CACHE FILEPATH "Shared selected HIP compiler" FORCE)
  endif()
  find_program(
    RNS8_CK_HIPCC
    NAMES hipcc hipcc.exe hipcc.bat
    HINTS "${RNS8_HIP_ROOT}/bin" "$ENV{HIP_PATH}/bin" "$ENV{ROCM_PATH}/bin"
  )

  if(RNS8_CK_HIPCC)
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _probe_run)
    set(RNS8_CK_COMPILE_PROBE_STATUS "compile_probe_pass")
    set(RNS8_CK_PRIMITIVE_PROBE_STATUS "primitive_object_probe_pass")
    set(RNS8_CK_TARGET_PROBE_RESULTS)
    foreach(_RNS8_CK_TARGET IN LISTS _RNS8_CK_TARGETS)
      string(SHA256 _target_id "${_RNS8_CK_TARGET}")
      set(_RNS8_CK_PROBE_DIR "${CMAKE_CURRENT_BINARY_DIR}/accelerator-dependency-probes/ck-${_probe_run}/${_target_id}")
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
      file(WRITE "${_RNS8_CK_PROBE_DIR}/compile.log" "${_RNS8_CK_PROBE_STDOUT}${_RNS8_CK_PROBE_STDERR}")
      string(APPEND RNS8_CK_COMPILE_PROBE_OUTPUT "\nTarget ${_RNS8_CK_TARGET}\n${_RNS8_CK_PROBE_STDOUT}${_RNS8_CK_PROBE_STDERR}")
      if(_RNS8_CK_PROBE_RESULT EQUAL 0)
        set(_target_compile_status "compile_probe_pass")
      else()
        set(_target_compile_status "compile_probe_fail")
        set(RNS8_CK_COMPILE_PROBE_STATUS "compile_probe_fail")
      endif()

      if(WIN32)
        set(_RNS8_CK_PRIMITIVE_OBJECT "${_RNS8_CK_PROBE_DIR}/ck_primitive_probe.obj")
      else()
        set(_RNS8_CK_PRIMITIVE_OBJECT "${_RNS8_CK_PROBE_DIR}/ck_primitive_probe.o")
      endif()
      set(_RNS8_CK_PRIMITIVE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../tools/probes/ck_primitive.hip")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_RNS8_CK_PRIMITIVE_SOURCE}")
      execute_process(
        COMMAND
          "${RNS8_CK_HIPCC}"
          "-std=c++17"
          "-O2"
          "-DRNS8_CK_PRIMITIVE_USE_XDL=${RNS8_CK_USE_XDL}"
          "--offload-arch=${_RNS8_CK_TARGET}"
          "-c"
          "${_RNS8_CK_PRIMITIVE_SOURCE}"
          "-I${RNS8_CK_GENERATED_INCLUDE_DIR}"
          "-I${CMAKE_CURRENT_LIST_DIR}/../../src/backend_common"
          "-I${RNS8_CK_INCLUDE_DIR}"
          "-o"
          "${_RNS8_CK_PRIMITIVE_OBJECT}"
        RESULT_VARIABLE _RNS8_CK_PRIMITIVE_RESULT
        OUTPUT_VARIABLE _RNS8_CK_PRIMITIVE_STDOUT
        ERROR_VARIABLE _RNS8_CK_PRIMITIVE_STDERR
        TIMEOUT 180
      )
      file(WRITE "${_RNS8_CK_PROBE_DIR}/primitive.log" "${_RNS8_CK_PRIMITIVE_STDOUT}${_RNS8_CK_PRIMITIVE_STDERR}")
      string(APPEND RNS8_CK_PRIMITIVE_PROBE_OUTPUT "\nTarget ${_RNS8_CK_TARGET}\n${_RNS8_CK_PRIMITIVE_STDOUT}${_RNS8_CK_PRIMITIVE_STDERR}")
      if(_RNS8_CK_PRIMITIVE_RESULT EQUAL 0)
        set(_target_primitive_status "primitive_object_probe_pass")
      else()
        set(_target_primitive_status "primitive_object_probe_fail")
        set(RNS8_CK_PRIMITIVE_PROBE_STATUS "primitive_object_probe_fail")
      endif()
      list(APPEND RNS8_CK_TARGET_PROBE_RESULTS "${_RNS8_CK_TARGET}|${_target_compile_status}|${_target_primitive_status}|${_RNS8_CK_PROBE_DIR}")
    endforeach()
  else()
    set(RNS8_CK_COMPILE_PROBE_STATUS "not_run_missing_hipcc")
    set(RNS8_CK_PRIMITIVE_PROBE_STATUS "not_run_missing_hipcc")
  endif()

  set(
    RNS8_CK_EVIDENCE
    "header=${RNS8_CK_INCLUDE_DIR};generated_include=${RNS8_CK_GENERATED_INCLUDE_DIR};compile_probe=${RNS8_CK_COMPILE_PROBE_STATUS};primitive_probe=${RNS8_CK_PRIMITIVE_PROBE_STATUS};targets=${RNS8_CK_TARGET_PROBE_RESULTS}"
  )
endif()

set(RNS8_CK_BACKEND_READY FALSE)
set(RNS8_CK_BACKEND_ENABLEMENT "disabled")
set(RNS8CK_FOUND FALSE)
if(RNS8_CK_CANDIDATE AND RNS8_CK_COMPILE_PROBE_STATUS STREQUAL "compile_probe_pass" AND RNS8_CK_PRIMITIVE_PROBE_STATUS STREQUAL "primitive_object_probe_pass")
  set(RNS8_CK_BACKEND_READY TRUE)
  set(RNS8_CK_BACKEND_ENABLEMENT "dependency_ready_for_opt_in_backend_build")
  set(RNS8CK_FOUND TRUE)
endif()
