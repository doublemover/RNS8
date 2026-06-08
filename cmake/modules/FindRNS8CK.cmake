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
  rns8_assert_no_linux_windows_vcpkg_paths("CK include directory" "${RNS8_CK_INCLUDE_DIR}")
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
    "#define CK_USE_XDL ON\n"
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

    set(_RNS8_CK_PRIMITIVE_SOURCE "${_RNS8_CK_PROBE_DIR}/ck_primitive_probe.cpp")
    if(WIN32)
      set(_RNS8_CK_PRIMITIVE_OBJECT "${_RNS8_CK_PROBE_DIR}/ck_primitive_probe.obj")
    else()
      set(_RNS8_CK_PRIMITIVE_OBJECT "${_RNS8_CK_PROBE_DIR}/ck_primitive_probe.o")
    endif()
    if(_RNS8_CK_TARGET MATCHES "^gfx9")
      file(WRITE "${_RNS8_CK_PRIMITIVE_SOURCE}" [=[
#include <cstdint>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1, 256, 128, 128, 64, 16, 16, 32, 32, 4, 2,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, true,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
    1, 1, S<1, 32, 1, 4>, 16>;

extern "C" float rns8_ck_i8_xdl_primitive_probe(const int8_t* a, const int8_t* b, int8_t* c) {
  auto arg = DeviceGemmInstance::MakeArgument(
      a, b, c, 64, 128, 64, 64, 64, 128, PassThrough{}, PassThrough{}, PassThrough{});
  if (!DeviceGemmInstance::IsSupportedArgument(arg) || !DeviceGemmInstance::IsValidCompilationParameter()) {
    return -1.0f;
  }
  auto invoker = DeviceGemmInstance::MakeInvoker();
  return invoker.Run(arg);
}
]=])
    else()
      file(WRITE "${_RNS8_CK_PRIMITIVE_SOURCE}" [=[
#include <cstdint>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1, 128, 64, 128, 64, 2, 16, 16, 2, 4,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true,
    1, 1, S<1, 32, 1, 4>, 8>;

extern "C" float rns8_ck_i8_wmma_primitive_probe(const int8_t* a, const int8_t* b, int8_t* c) {
  auto arg = DeviceGemmInstance::MakeArgument(
      a, b, c, 64, 128, 64, 64, 128, 128, PassThrough{}, PassThrough{}, PassThrough{});
  if (!DeviceGemmInstance::IsSupportedArgument(arg) || !DeviceGemmInstance::IsValidCompilationParameter()) {
    return -1.0f;
  }
  auto invoker = DeviceGemmInstance::MakeInvoker();
  return invoker.Run(arg);
}
]=])
    endif()
    execute_process(
      COMMAND
        "${RNS8_CK_HIPCC}"
        "-std=c++17"
        "-O2"
        "--offload-arch=${_RNS8_CK_TARGET}"
        "-c"
        "${_RNS8_CK_PRIMITIVE_SOURCE}"
        "-I${RNS8_CK_GENERATED_INCLUDE_DIR}"
        "-I${RNS8_CK_INCLUDE_DIR}"
        "-o"
        "${_RNS8_CK_PRIMITIVE_OBJECT}"
      RESULT_VARIABLE _RNS8_CK_PRIMITIVE_RESULT
      OUTPUT_VARIABLE _RNS8_CK_PRIMITIVE_STDOUT
      ERROR_VARIABLE _RNS8_CK_PRIMITIVE_STDERR
      TIMEOUT 180
    )
    set(RNS8_CK_PRIMITIVE_PROBE_OUTPUT "${_RNS8_CK_PRIMITIVE_STDOUT}${_RNS8_CK_PRIMITIVE_STDERR}")
    if(_RNS8_CK_PRIMITIVE_RESULT EQUAL 0)
      set(RNS8_CK_PRIMITIVE_PROBE_STATUS "primitive_object_probe_pass")
    else()
      set(RNS8_CK_PRIMITIVE_PROBE_STATUS "primitive_object_probe_fail")
    endif()
  else()
    set(RNS8_CK_COMPILE_PROBE_STATUS "not_run_missing_hipcc")
    set(RNS8_CK_PRIMITIVE_PROBE_STATUS "not_run_missing_hipcc")
  endif()

  set(
    RNS8_CK_EVIDENCE
    "header=${RNS8_CK_INCLUDE_DIR};generated_include=${RNS8_CK_GENERATED_INCLUDE_DIR};compile_probe=${RNS8_CK_COMPILE_PROBE_STATUS};primitive_probe=${RNS8_CK_PRIMITIVE_PROBE_STATUS}"
  )
endif()

set(RNS8_CK_BACKEND_READY FALSE)
set(RNS8_CK_BACKEND_ENABLEMENT "disabled")
set(RNS8CK_FOUND FALSE)
if(RNS8_CK_CANDIDATE AND RNS8_CK_PRIMITIVE_PROBE_STATUS STREQUAL "primitive_object_probe_pass")
  set(RNS8_CK_BACKEND_READY TRUE)
  set(RNS8_CK_BACKEND_ENABLEMENT "dependency_ready_for_opt_in_backend_build")
  set(RNS8CK_FOUND TRUE)
endif()
