include(FindPackageHandleStandardArgs)

set(RNS8_HIP_CMAKE_MODULE "${CMAKE_CURRENT_LIST_FILE}")

if(NOT RNS8_DISCOVERY_IDENTITY_SHA256)
  message(FATAL_ERROR "FindRNS8HIP requires the shared pre-project cmake/rns8_discovery.cmake selection; it cannot independently mix PATH headers/libraries/toolchains")
endif()
# Compiler/include/link/runtime/device-library cache bindings come from one
# shared SDK selection. No independent find_* fallback is permitted here.

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

# LLVM llvm-project#201563 fixes MSVC 14.51 constexpr math declarations colliding
# with HIP's late device declarations. Backport only that include-order change
# into the build tree for older SDK wrappers; never modify the installed SDK.
set(RNS8_HIP_STL_OVERLAY "")
if(RNS8HIP_FOUND AND WIN32 AND MSVC_VERSION GREATER_EQUAL 1951)
  get_filename_component(_hip_bin "${RNS8_HIP_HIPCC}" DIRECTORY)
  execute_process(COMMAND "${RNS8_HIP_CLANG}" -print-resource-dir
    OUTPUT_VARIABLE _hip_resource OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _hip_resource_status)
  if(_hip_resource_status EQUAL 0)
    set(_wrapper "${_hip_resource}/include/__clang_hip_runtime_wrapper.h")
    if(EXISTS "${_wrapper}")
      file(READ "${_wrapper}" _wrapper_text)
      string(FIND "${_wrapper_text}" "#include <cmath>" _cmath_position)
      string(FIND "${_wrapper_text}" "#include <__clang_cuda_math_forward_declares.h>" _forward_position)
      if(_cmath_position GREATER_EQUAL 0 AND _forward_position GREATER _cmath_position)
        string(REPLACE "#include <__clang_cuda_math_forward_declares.h>" "" _wrapper_text "${_wrapper_text}")
        string(REPLACE "#include <cmath>" "#include <__clang_cuda_math_forward_declares.h>\n#include <cmath>" _wrapper_text "${_wrapper_text}")
        set(RNS8_HIP_STL_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/hip-stl-overlay")
        file(MAKE_DIRECTORY "${RNS8_HIP_STL_OVERLAY}")
        file(CONFIGURE OUTPUT "${RNS8_HIP_STL_OVERLAY}/__clang_hip_runtime_wrapper.h" CONTENT "${_wrapper_text}" @ONLY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_wrapper}")
        message(STATUS "HIP/MSVC 14.51: applying LLVM #201563 include-order fix in build-local wrapper")
      endif()
    endif()
  endif()
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
  set(_compat_dependencies)
  if(RNS8_HIP_STL_OVERLAY)
    list(APPEND _include_args "-I${RNS8_HIP_STL_OVERLAY}")
    list(APPEND _compat_dependencies "${RNS8_HIP_STL_OVERLAY}/__clang_hip_runtime_wrapper.h")
  endif()
  foreach(_include_dir IN LISTS RNS8_HIP_SOURCE_INCLUDE_DIRS)
    if(_include_dir)
      list(APPEND _include_args "-I${_include_dir}")
    endif()
  endforeach()

  set(_source_compile_options)
  if(RNS8_EXPERIMENTAL_GARNER_TABLE)
    list(APPEND _source_compile_options "-DRNS8_EXPERIMENTAL_GARNER_TABLE=1")
  endif()
  foreach(_option IN LISTS RNS8_HIP_SOURCE_COMPILE_OPTIONS)
    if(_option)
      list(APPEND _source_compile_options "${_option}")
    endif()
  endforeach()

  add_custom_command(
    OUTPUT "${_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/hip"
    COMMAND
      "${CMAKE_COMMAND}" -E env ${RNS8_HIP_COMPILER_ENVIRONMENT}
      "${RNS8_HIP_HIPCC}"
      ${_arch_args}
      ${_host_runtime_args}
      ${_pic_args}
      -include "${RNS8_MANIFEST_HEADER}"
      -std=c++17
      -O2
      "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
      ${_include_args}
      ${_source_compile_options}
      -c "${_source_abs}"
      -o "${_object}"
    DEPENDS rns8_build_manifest "${RNS8_MANIFEST_HEADER}" "${_source_abs}" "${RNS8_HIP_CMAKE_MODULE}" ${RNS8_HIP_SOURCE_DEPENDS} ${_compat_dependencies}
    VERBATIM
    COMMENT "Compiling HIP source ${_source_name} with explicit hipcc integration"
  )

  set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  set(${out_var} "${_object}" PARENT_SCOPE)
endfunction()
