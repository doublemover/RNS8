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

