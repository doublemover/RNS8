include(FindPackageHandleStandardArgs)

set(_RNS8_HIP_HINTS)
if(RNS8_HIP_ROOT)
  list(APPEND _RNS8_HIP_HINTS "${RNS8_HIP_ROOT}")
endif()
if(DEFINED ENV{HIP_PATH})
  list(APPEND _RNS8_HIP_HINTS "$ENV{HIP_PATH}")
endif()
if(DEFINED ENV{ROCM_PATH})
  list(APPEND _RNS8_HIP_HINTS "$ENV{ROCM_PATH}")
endif()
if(WIN32)
  list(APPEND _RNS8_HIP_HINTS "C:/Program Files/AMD/ROCm/7.1" "C:/Program Files/AMD/ROCm/6.4")
else()
  list(APPEND _RNS8_HIP_HINTS "/opt/rocm")
endif()

find_program(
  RNS8_HIP_HIPCC
  NAMES hipcc hipcc.bat hipcc.exe
  HINTS ${_RNS8_HIP_HINTS}
  PATH_SUFFIXES bin
)

find_path(
  RNS8_HIP_INCLUDE_DIR
  NAMES hip/hip_runtime_api.h
  HINTS ${_RNS8_HIP_HINTS}
  PATH_SUFFIXES include
)

find_library(
  RNS8_HIP_LIBRARY
  NAMES amdhip64
  HINTS ${_RNS8_HIP_HINTS}
  PATH_SUFFIXES lib lib64 bin
)

find_package_handle_standard_args(
  RNS8HIP
  REQUIRED_VARS RNS8_HIP_HIPCC RNS8_HIP_INCLUDE_DIR RNS8_HIP_LIBRARY
)

if(RNS8HIP_FOUND)
  rns8_assert_no_linux_windows_vcpkg_paths("HIP include directory" "${RNS8_HIP_INCLUDE_DIR}")
  rns8_assert_no_linux_windows_vcpkg_paths("HIP library" "${RNS8_HIP_LIBRARY}")
  rns8_assert_no_linux_windows_vcpkg_paths("hipcc executable" "${RNS8_HIP_HIPCC}")
  set(RNS8_HIP_INCLUDE_DIRS "${RNS8_HIP_INCLUDE_DIR}")
  set(RNS8_HIP_LIBRARIES "${RNS8_HIP_LIBRARY}")
endif()

function(rns8_compile_hip_source out_var source)
  if(NOT RNS8HIP_FOUND)
    message(FATAL_ERROR "rns8_compile_hip_source requires RNS8HIP")
  endif()

  get_filename_component(_source_abs "${source}" ABSOLUTE)
  get_filename_component(_source_name "${source}" NAME_WE)
  set(_object "${CMAKE_CURRENT_BINARY_DIR}/hip/${_source_name}${CMAKE_CXX_OUTPUT_EXTENSION}")

  set(_arch_args)
  foreach(_target IN LISTS RNS8_AMDGPU_TARGETS)
    if(_target)
      list(APPEND _arch_args "--offload-arch=${_target}")
    endif()
  endforeach()

  set(_host_runtime_args)
  if(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND _host_runtime_args -fms-runtime-lib=dll_dbg -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=2)
    else()
      list(APPEND _host_runtime_args -fms-runtime-lib=dll -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0)
    endif()
  endif()

  set(_pic_args)
  if(NOT WIN32)
    list(APPEND _pic_args -fPIC)
  endif()

  set(_include_args)
  foreach(_include_dir IN LISTS RNS8_HIP_SOURCE_INCLUDE_DIRS)
    if(_include_dir)
      list(APPEND _include_args "-I${_include_dir}")
    endif()
  endforeach()

  set(_source_compile_options)
  foreach(_option IN LISTS RNS8_HIP_SOURCE_COMPILE_OPTIONS)
    if(_option)
      list(APPEND _source_compile_options "${_option}")
    endif()
  endforeach()

  add_custom_command(
    OUTPUT "${_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/hip"
    COMMAND
      "${RNS8_HIP_HIPCC}"
      ${_arch_args}
      ${_host_runtime_args}
      ${_pic_args}
      -std=c++17
      -O2
      "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
      ${_include_args}
      ${_source_compile_options}
      -c "${_source_abs}"
      -o "${_object}"
    DEPENDS "${_source_abs}" ${RNS8_HIP_SOURCE_DEPENDS}
    VERBATIM
    COMMENT "Compiling HIP source ${_source_name} with explicit hipcc integration"
  )

  set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  set(${out_var} "${_object}" PARENT_SCOPE)
endfunction()
