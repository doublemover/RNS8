cmake_minimum_required(VERSION 3.22)

include(cmake/rns8_discovery.cmake)

project(
  RNS8
  VERSION 0.1.0
  DESCRIPTION "Exact integer GEMM with AMD GPU matrix engines, RNS, and CRT reconstruction"
  LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(cmake/rns8_platform_guards.cmake)
rns8_assert_linux_native_discovery_context()
include(cmake/rns8_options.cmake)
include(cmake/rns8_provenance.cmake)
include(cmake/rns8_sources.cmake)
include(cmake/RNS8Runtime.cmake)
set(RNS8_WINDOWS_RUNTIME_DLLS "")
set(RNS8_WINDOWS_RUNTIME_NAMES "")
if(WIN32 AND RNS8_ENABLE_HIP)
  list(APPEND RNS8_WINDOWS_RUNTIME_DLLS "${RNS8_HIP_RUNTIME}")
  get_filename_component(_rns8_hip_runtime_dir "${RNS8_HIP_RUNTIME}" DIRECTORY)
  # ROCm Core's HIP runtime imports this SDK-owned loader. System DLLs and
  # GPU-driver components remain OS-managed; no system directories are changed.
  if(EXISTS "${_rns8_hip_runtime_dir}/rocm_kpack.dll")
    list(APPEND RNS8_WINDOWS_RUNTIME_DLLS "${_rns8_hip_runtime_dir}/rocm_kpack.dll")
  endif()
  foreach(_runtime IN LISTS RNS8_WINDOWS_RUNTIME_DLLS)
    get_filename_component(_name "${_runtime}" NAME)
    list(APPEND RNS8_WINDOWS_RUNTIME_NAMES "${_name}")
  endforeach()
endif()
include(cmake/rns8_targets.cmake)
include(cmake/rns8_tests.cmake)
include(cmake/rns8_install.cmake)
get_property(_rns8_local_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _rns8_local_targets)
  get_target_property(_type ${_target} TYPE)
  if(_type STREQUAL "EXECUTABLE" OR _type STREQUAL "SHARED_LIBRARY")
    rns8_copy_runtime_dependencies(${_target})
  endif()
endforeach()
