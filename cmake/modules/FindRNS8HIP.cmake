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
  if(NOT TARGET RNS8::hip_runtime)
    add_library(RNS8::hip_runtime INTERFACE IMPORTED)
    set_target_properties(RNS8::hip_runtime PROPERTIES INTERFACE_LINK_LIBRARIES "${RNS8_HIP_LIBRARY}")
  else()
    get_target_property(_rns8_runtime_library RNS8::hip_runtime INTERFACE_LINK_LIBRARIES)
    if(NOT _rns8_runtime_library STREQUAL RNS8_HIP_LIBRARY)
      message(FATAL_ERROR "Existing RNS8::hip_runtime differs from the shared selected runtime; configure a fresh coherent build")
    endif()
  endif()
  # Export a dependency name, never the producer machine's runtime library path.
  set(RNS8_HIP_LIBRARIES RNS8::hip_runtime)
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

  set(_source_compile_options ${RNS8_DEVICE_ASAN_COMPILE_FLAGS})
  if(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS)
    list(APPEND _source_compile_options "-DRNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS=1")
  endif()
  if(RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL)
    list(APPEND _source_compile_options "-DRNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL=1")
  endif()
  if(RNS8_EXPERIMENTAL_RESIDUE_DOT4)
    list(APPEND _source_compile_options "-DRNS8_EXPERIMENTAL_RESIDUE_DOT4=1")
  endif()
  if(RNS8_EXPERIMENTAL_GARNER_TABLE)
    list(APPEND _source_compile_options "-DRNS8_EXPERIMENTAL_GARNER_TABLE=1")
  endif()
  foreach(_option IN LISTS RNS8_HIP_SOURCE_COMPILE_OPTIONS)
    if(_option)
      list(APPEND _source_compile_options "${_option}")
    endif()
  endforeach()

  # Build execution and analyzer inventory consume this same argv list.
  set(_compiler_args
      "${RNS8_HIP_HIPCC}"
      ${_arch_args}
      ${_host_runtime_args}
      ${_pic_args}
      -include "${RNS8_MANIFEST_HEADER}"
      -std=c++17
      -O2
      "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
      ${_include_args}
      ${_source_compile_options}
      -c "${_source_abs}"
      -o "${_object}")
  rns8_record_hip_compile("${_source_abs}" "${_object}" "${_compiler_args}"
    "${RNS8_HIP_COMPILER_ENVIRONMENT}" "${RNS8_MANIFEST_HEADER};${_compat_dependencies}")
  set(_execute_args "${_compiler_args}")
  set(_receipt_dependencies)
  set(_receipt_byproducts)
  if(RNS8_CAPTURE_HIP_COMPILE_COST)
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
      message(FATAL_ERROR "HIP compile receipts require a named configuration, for example CMAKE_BUILD_TYPE=Release")
    endif()
    set(_receipt_context "{}")
    foreach(_field IN ITEMS schema source_root build_root directory file output configuration)
      if(_field STREQUAL "schema")
        set(_value "rns8_native_compile_context_v1")
      elseif(_field STREQUAL "source_root")
        set(_value "${PROJECT_SOURCE_DIR}")
      elseif(_field STREQUAL "build_root")
        set(_value "${CMAKE_BINARY_DIR}")
      elseif(_field STREQUAL "directory")
        set(_value "${CMAKE_CURRENT_BINARY_DIR}")
      elseif(_field STREQUAL "file")
        set(_value "${_source_abs}")
      elseif(_field STREQUAL "output")
        set(_value "${_object}")
      else()
        set(_value "$<CONFIG>")
      endif()
      rns8_json_string(_quoted "${_value}")
      string(JSON _receipt_context SET "${_receipt_context}" "${_field}" "${_quoted}")
    endforeach()
    set(_receipt_inputs "${_source_abs}" "${RNS8_MANIFEST_HEADER}" "${RNS8_HIP_CMAKE_MODULE}" ${RNS8_HIP_SOURCE_DEPENDS} ${_compat_dependencies})
    foreach(_compiler IN ITEMS "${RNS8_HIP_CLANG}" "${CMAKE_CXX_COMPILER}")
      if(_compiler)
        list(APPEND _receipt_inputs "${_compiler}")
      endif()
    endforeach()
    set(_absolute_inputs)
    foreach(_input IN LISTS _receipt_inputs)
      get_filename_component(_absolute_input "${_input}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
      list(APPEND _absolute_inputs "${_absolute_input}")
    endforeach()
    list(REMOVE_DUPLICATES _absolute_inputs)
    foreach(_field IN ITEMS arguments inputs targets)
      if(_field STREQUAL "arguments")
        set(_values "${_compiler_args}")
      elseif(_field STREQUAL "inputs")
        set(_values "${_absolute_inputs}")
      else()
        set(_values "${RNS8_AMDGPU_TARGETS}")
      endif()
      rns8_json_array(_array _values)
      string(JSON _receipt_context SET "${_receipt_context}" "${_field}" "${_array}")
    endforeach()
    set(_receipt_context_file "${_object}.compile-context-$<CONFIG>.json")
    file(GENERATE OUTPUT "${_receipt_context_file}" CONTENT "${_receipt_context}\n")
    set(_execute_args "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/compile_receipt.py"
      --execute --context "${_receipt_context_file}")
    set(_receipt_dependencies "${_receipt_context_file}" "${PROJECT_SOURCE_DIR}/tools/compile_receipt.py"
      "${PROJECT_SOURCE_DIR}/tools/compiler_context.py" "${PROJECT_SOURCE_DIR}/tools/process_progress.py"
      "${PROJECT_SOURCE_DIR}/tools/benchmark_request.py" "${PROJECT_SOURCE_DIR}/tools/compiled_binary_identity.py"
      "${PROJECT_SOURCE_DIR}/tools/json_evidence.py")
    set(_receipt_byproducts "${_object}.compile.json")
  endif()
  add_custom_command(
    OUTPUT "${_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/hip"
    COMMAND
      "${CMAKE_COMMAND}" -E env ${RNS8_HIP_COMPILER_ENVIRONMENT}
      ${_execute_args}
    DEPENDS rns8_build_manifest "${RNS8_MANIFEST_HEADER}" "${_source_abs}" "${RNS8_HIP_CMAKE_MODULE}" ${RNS8_HIP_SOURCE_DEPENDS} ${_compat_dependencies} ${_receipt_dependencies}
    BYPRODUCTS ${_receipt_byproducts}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM
    COMMENT "Compiling HIP source ${_source_name} with explicit hipcc integration"
  )

  set_source_files_properties("${_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  set(${out_var} "${_object}" PARENT_SCOPE)
endfunction()
