cmake_minimum_required(VERSION 3.22)
project(RNS8ExactReferenceProbe LANGUAGES CXX)
get_filename_component(_rns8_source "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT RNS8_REFERENCE_COMPONENT MATCHES "^(gmp|flint)$")
  message(FATAL_ERROR "Choose exactly one RNS8_REFERENCE_COMPONENT: gmp or flint")
endif()
set(RNS8_ENABLE_GMP OFF)
set(RNS8_ENABLE_FLINT OFF)
if(RNS8_REFERENCE_COMPONENT STREQUAL "gmp")
  set(RNS8_ENABLE_GMP ON)
else()
  set(RNS8_ENABLE_FLINT ON)
endif()
# This probe always links an executable; a toolchain's static-only try_compile
# default cannot turn unresolved external symbols into a compatibility pass.
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
include("${_rns8_source}/cmake/rns8_platform_guards.cmake")
rns8_assert_linux_native_discovery_context()
include("${_rns8_source}/cmake/modules/FindRNS8ThirdParty.cmake")
set(_result "{}")
foreach(_pair IN ITEMS
    "schema|rns8_exact_reference_link_probe_v1"
    "component|${RNS8_REFERENCE_COMPONENT}"
    "compile_link|${RNS8_EXACT_REFERENCE_LINK_PROBE}"
    "compiler|${CMAKE_CXX_COMPILER}"
    "compiler_id|${CMAKE_CXX_COMPILER_ID}"
    "compiler_version|${CMAKE_CXX_COMPILER_VERSION}"
    "system|${CMAKE_SYSTEM_NAME}"
    "processor|${CMAKE_SYSTEM_PROCESSOR}"
    "runtime_validation|not_run")
  string(FIND "${_pair}" "|" _separator)
  string(SUBSTRING "${_pair}" 0 ${_separator} _key)
  math(EXPR _start "${_separator}+1")
  string(SUBSTRING "${_pair}" ${_start} -1 _value)
  string(REPLACE "\\" "\\\\" _value "${_value}")
  string(REPLACE "\"" "\\\"" _value "${_value}")
  string(REPLACE "\n" "\\n" _value "${_value}")
  string(REPLACE "\r" "\\r" _value "${_value}")
  string(REPLACE "\t" "\\t" _value "${_value}")
  string(JSON _result SET "${_result}" "${_key}" "\"${_value}\"")
endforeach()
string(JSON _result SET "${_result}" pointer_bytes ${CMAKE_SIZEOF_VOID_P})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reference-probe.json" "${_result}\n")
