set(_RNS8_ROCWMMA_HINTS)
foreach(_RNS8_ROCWMMA_HINT IN ITEMS
    "${RNS8_ROCWMMA_ROOT}"
    "${RNS8_ROCM_DEPS_INSTALL_ROOT}"
    "${RNS8_HIP_ROOT}"
    "$ENV{ROCM_PATH}"
    "$ENV{HIP_PATH}"
    "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}"
    "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}"
    "/opt/rocm")
  if(_RNS8_ROCWMMA_HINT)
    list(APPEND _RNS8_ROCWMMA_HINTS "${_RNS8_ROCWMMA_HINT}")
  endif()
endforeach()

find_path(
  RNS8_ROCWMMA_INCLUDE_DIR
  NAMES rocwmma/rocwmma.hpp
  HINTS ${_RNS8_ROCWMMA_HINTS}
  PATH_SUFFIXES include library/include
)

set(RNS8_ROCWMMA_CANDIDATE FALSE)
set(RNS8_ROCWMMA_EVIDENCE "not discovered")
set(RNS8_ROCWMMA_COMPILE_PROBE_STATUS "not_run_missing_header")
set(RNS8_ROCWMMA_COMPILE_PROBE_OUTPUT "")
set(RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS "not_run_missing_header")
set(RNS8_ROCWMMA_PRIMITIVE_PROBE_OUTPUT "")

if(RNS8_ROCWMMA_INCLUDE_DIR)
  set(RNS8_ROCWMMA_CANDIDATE TRUE)
  set(_RNS8_ROCWMMA_TARGET "gfx1100")
  if(RNS8_AMDGPU_TARGETS)
    set(_RNS8_ROCWMMA_TARGETS ${RNS8_AMDGPU_TARGETS})
    list(GET _RNS8_ROCWMMA_TARGETS 0 _RNS8_ROCWMMA_TARGET)
  endif()

  find_program(
    RNS8_ROCWMMA_HIPCC
    NAMES hipcc hipcc.exe hipcc.bat
    HINTS "${RNS8_HIP_ROOT}/bin" "$ENV{HIP_PATH}/bin" "$ENV{ROCM_PATH}/bin"
  )

  if(RNS8_ROCWMMA_HIPCC)
    set(_RNS8_ROCWMMA_PROBE_DIR "${CMAKE_CURRENT_BINARY_DIR}/accelerator-dependency-probes")
    file(MAKE_DIRECTORY "${_RNS8_ROCWMMA_PROBE_DIR}")
    set(_RNS8_ROCWMMA_PROBE_SOURCE "${_RNS8_ROCWMMA_PROBE_DIR}/rocwmma_probe.cpp")
    set(_RNS8_ROCWMMA_PROBE_BINARY "${_RNS8_ROCWMMA_PROBE_DIR}/rocwmma_probe${CMAKE_EXECUTABLE_SUFFIX}")
    file(
      WRITE "${_RNS8_ROCWMMA_PROBE_SOURCE}"
      "#include <rocwmma/rocwmma.hpp>\n"
      "__global__ void rns8_rocwmma_dependency_probe_kernel() {}\n"
      "int main() { return 0; }\n"
    )
    execute_process(
      COMMAND
        "${RNS8_ROCWMMA_HIPCC}"
        "-std=c++17"
        "--offload-arch=${_RNS8_ROCWMMA_TARGET}"
        "-I${RNS8_ROCWMMA_INCLUDE_DIR}"
        "${_RNS8_ROCWMMA_PROBE_SOURCE}"
        "-o"
        "${_RNS8_ROCWMMA_PROBE_BINARY}"
      RESULT_VARIABLE _RNS8_ROCWMMA_PROBE_RESULT
      OUTPUT_VARIABLE _RNS8_ROCWMMA_PROBE_STDOUT
      ERROR_VARIABLE _RNS8_ROCWMMA_PROBE_STDERR
      TIMEOUT 90
    )
    set(RNS8_ROCWMMA_COMPILE_PROBE_OUTPUT "${_RNS8_ROCWMMA_PROBE_STDOUT}${_RNS8_ROCWMMA_PROBE_STDERR}")
    if(_RNS8_ROCWMMA_PROBE_RESULT EQUAL 0)
      set(RNS8_ROCWMMA_COMPILE_PROBE_STATUS "compile_probe_pass")
    else()
      set(RNS8_ROCWMMA_COMPILE_PROBE_STATUS "compile_probe_fail")
    endif()

    set(_RNS8_ROCWMMA_PRIMITIVE_SOURCE "${_RNS8_ROCWMMA_PROBE_DIR}/rocwmma_primitive_probe.cpp")
    if(WIN32)
      set(_RNS8_ROCWMMA_PRIMITIVE_OBJECT "${_RNS8_ROCWMMA_PROBE_DIR}/rocwmma_primitive_probe.obj")
    else()
      set(_RNS8_ROCWMMA_PRIMITIVE_OBJECT "${_RNS8_ROCWMMA_PROBE_DIR}/rocwmma_primitive_probe.o")
    endif()
    file(WRITE "${_RNS8_ROCWMMA_PRIMITIVE_SOURCE}" [=[
#include <cstdint>
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void rns8_rocwmma_i8_mma_primitive_probe(
    const int8_t* a, const int8_t* b, int32_t* c) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, 16, 16, 32, int8_t, row_major>;
  using FragB = fragment<matrix_b, 16, 16, 32, int8_t, col_major>;
  using FragAcc = fragment<accumulator, 16, 16, 32, int32_t>;
  FragA frag_a;
  FragB frag_b;
  FragAcc acc;
  fill_fragment(acc, 0);
  load_matrix_sync(frag_a, a, 32);
  load_matrix_sync(frag_b, b, 32);
  mma_sync(acc, frag_a, frag_b, acc);
  store_matrix_sync(c, acc, 16, mem_row_major);
}
]=])
    execute_process(
      COMMAND
        "${RNS8_ROCWMMA_HIPCC}"
        "-std=c++17"
        "-O2"
        "--offload-arch=${_RNS8_ROCWMMA_TARGET}"
        "-c"
        "${_RNS8_ROCWMMA_PRIMITIVE_SOURCE}"
        "-I${RNS8_ROCWMMA_INCLUDE_DIR}"
        "-o"
        "${_RNS8_ROCWMMA_PRIMITIVE_OBJECT}"
      RESULT_VARIABLE _RNS8_ROCWMMA_PRIMITIVE_RESULT
      OUTPUT_VARIABLE _RNS8_ROCWMMA_PRIMITIVE_STDOUT
      ERROR_VARIABLE _RNS8_ROCWMMA_PRIMITIVE_STDERR
      TIMEOUT 120
    )
    set(RNS8_ROCWMMA_PRIMITIVE_PROBE_OUTPUT "${_RNS8_ROCWMMA_PRIMITIVE_STDOUT}${_RNS8_ROCWMMA_PRIMITIVE_STDERR}")
    if(_RNS8_ROCWMMA_PRIMITIVE_RESULT EQUAL 0)
      set(RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS "primitive_object_probe_pass")
    else()
      set(RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS "primitive_object_probe_fail")
    endif()
  else()
    set(RNS8_ROCWMMA_COMPILE_PROBE_STATUS "not_run_missing_hipcc")
    set(RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS "not_run_missing_hipcc")
  endif()

  set(
    RNS8_ROCWMMA_EVIDENCE
    "header=${RNS8_ROCWMMA_INCLUDE_DIR};compile_probe=${RNS8_ROCWMMA_COMPILE_PROBE_STATUS};primitive_probe=${RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS}"
  )
endif()

set(RNS8_ROCWMMA_BACKEND_READY FALSE)
set(RNS8_ROCWMMA_BACKEND_ENABLEMENT "disabled")
set(RNS8ROCWMMA_FOUND FALSE)
