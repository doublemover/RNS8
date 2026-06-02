cmake_minimum_required(VERSION 3.28)

project(
  RNS8
  VERSION 0.1.0
  DESCRIPTION "Exact integer GEMM with AMD GPU matrix engines, RNS, and CRT reconstruction"
  LANGUAGES CXX
)

include(CTest)

option(RNS8_ENABLE_HIP "Build the direct HIP backend with explicit hipcc integration" OFF)
option(RNS8_BUILD_TESTS "Build RNS8 correctness tests" ON)
option(RNS8_BUILD_TOOLS "Build RNS8 command-line tools" ON)
option(RNS8_BUILD_BENCHMARKS "Build RNS8 benchmark shell" ON)
option(RNS8_ENABLE_GMP "Detect GMP for future optional differential references" OFF)
option(RNS8_ENABLE_FLINT "Detect FLINT for future optional differential references" OFF)

set(RNS8_AMDGPU_TARGETS "gfx1100" CACHE STRING "Semicolon-separated AMDGPU offload targets for direct HIP sources")
set(RNS8_HIP_ROOT "" CACHE PATH "AMD HIP SDK or ROCm root used by explicit HIP integration")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_path(
  RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR
  NAMES boost/multiprecision/cpp_int.hpp
  REQUIRED
)

set(RNS8_PUBLIC_HEADERS
  include/rns8/status.h
  include/rns8/semantics.h
  include/rns8/bounds.h
  include/rns8/moduli.h
  include/rns8/rns8.h
  include/rns8/rns8.hpp
)

set(RNS8_SOURCES
  src/core/api.cpp
  src/core/moduli.cpp
  src/core/status.cpp
  src/cpu/cpu_reference.cpp
  src/reconstruct/crt.cpp
  src/backend_hip_direct/hip_backend.cpp
  src/backend_wrap64/wrap64_reference.cpp
  src/backend_wrap64/wrap64_hip.cpp
)

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
  list(APPEND RNS8_SOURCES "${RNS8_HIP_DIRECT_KERNEL_OBJECT}" "${RNS8_WRAP64_HIP_KERNEL_OBJECT}")
endif()

function(rns8_configure_library target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_17)
  target_include_directories(
    ${target_name}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_compile_definitions(${target_name} PRIVATE $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(RNS8_ENABLE_HIP)
    target_include_directories(${target_name} PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${RNS8_HIP_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE __HIP_PLATFORM_AMD__=1)
  endif()
endfunction()

add_library(rns8 SHARED ${RNS8_PUBLIC_HEADERS} ${RNS8_SOURCES})
add_library(rns8::rns8 ALIAS rns8)
rns8_configure_library(rns8)
target_compile_definitions(rns8 PRIVATE RNS8_BUILDING_LIBRARY)
set_target_properties(
  rns8
  PROPERTIES
    OUTPUT_NAME rns8
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_library(rns8_static STATIC ${RNS8_PUBLIC_HEADERS} ${RNS8_SOURCES})
add_library(rns8::rns8_static ALIAS rns8_static)
rns8_configure_library(rns8_static)
target_compile_definitions(rns8_static PUBLIC RNS8_STATIC)
set_target_properties(rns8_static PROPERTIES OUTPUT_NAME rns8_static)

if(RNS8_BUILD_TOOLS)
  add_executable(rns8-inspect tools/rns8_inspect.cpp)
  target_link_libraries(rns8-inspect PRIVATE rns8_static)

  add_executable(rns8-verify tools/rns8_verify.cpp)
  target_include_directories(
    rns8-verify
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8-verify PRIVATE rns8_static)
endif()

if(RNS8_BUILD_BENCHMARKS)
  set(RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS}")
  string(REPLACE ";" "," RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS_TEXT}")
  set(RNS8_GIT_COMMIT "unknown")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE RNS8_GIT_RESULT
      OUTPUT_VARIABLE RNS8_GIT_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(RNS8_GIT_RESULT EQUAL 0 AND NOT RNS8_GIT_OUTPUT STREQUAL "")
      set(RNS8_GIT_COMMIT "${RNS8_GIT_OUTPUT}")
    endif()
  endif()
  add_executable(rns8-bench benchmarks/rns8_bench.cpp)
  target_compile_definitions(
    rns8-bench
    PRIVATE
      RNS8_CONFIGURED_AMDGPU_TARGETS="${RNS8_AMDGPU_TARGETS_TEXT}"
      RNS8_GIT_COMMIT="${RNS8_GIT_COMMIT}"
      RNS8_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_include_directories(rns8-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(rns8-bench PRIVATE rns8_static)
endif()

if(BUILD_TESTING AND RNS8_BUILD_TESTS)
  find_package(Catch2 3 CONFIG REQUIRED)
  add_executable(
    rns8_tests
    tests/unit/test_moduli.cpp
    tests/unit/test_residues.cpp
    tests/unit/test_ring_gemm.cpp
    tests/unit/test_crt.cpp
    tests/unit/test_bounded_gemm.cpp
    tests/unit/test_bounded_reference_sweeps.cpp
    tests/unit/test_exact_wide.cpp
    tests/unit/test_semantics.cpp
    tests/unit/test_wrap64.cpp
    tests/differential/test_hip_direct.cpp
  )
  target_include_directories(
    rns8_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8_tests PRIVATE rns8_static Catch2::Catch2WithMain)
  include(Catch)
  catch_discover_tests(rns8_tests)
endif()
