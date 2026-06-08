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
      ${RNS8_NLOHMANN_JSON_INCLUDE_DIRS}
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE
      $<$<BOOL:${RNS8_ENABLE_HIP}>:RNS8_ENABLE_HIP=1>
      $<$<BOOL:${RNS8_ENABLE_HIPBLASLT}>:RNS8_ENABLE_HIPBLASLT=1>
      $<$<BOOL:${RNS8_ENABLE_CK}>:RNS8_ENABLE_CK=1>
      $<$<BOOL:${RNS8_ENABLE_ROCWMMA}>:RNS8_ENABLE_ROCWMMA=1>
      $<$<BOOL:${RNS8_ENABLE_GMP}>:RNS8_ENABLE_GMP=1>
      $<$<BOOL:${RNS8_ENABLE_FLINT}>:RNS8_ENABLE_FLINT=1>
      $<$<BOOL:${RNS8_CPU_PARALLEL_OPENMP}>:RNS8_CPU_PARALLEL_OPENMP=1>
      RNS8_CK_SHAPE_ALIGNMENT=${RNS8_CK_SHAPE_ALIGNMENT}
  )
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    if(RNS8_ENABLE_HIP)
      # AMD HIP SDK vector headers intentionally use nameless unions that trip MSVC /W4.
      target_compile_options(${target_name} PRIVATE /wd4201)
    endif()
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(RNS8_ENABLE_HIP)
    target_include_directories(${target_name} PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${RNS8_HIP_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE __HIP_PLATFORM_AMD__=1)
  endif()
  if(RNS8_CPU_PARALLEL_OPENMP)
    target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(RNS8_ENABLE_HIPBLASLT)
    if(RNS8_HIPBLASLT_INCLUDE_DIR)
      target_include_directories(${target_name} PRIVATE "${RNS8_HIPBLASLT_INCLUDE_DIR}")
    endif()
    if(RNS8_HIPBLASLT_TARGET)
      target_link_libraries(${target_name} PRIVATE "${RNS8_HIPBLASLT_TARGET}")
    else()
      target_link_libraries(${target_name} PRIVATE "${RNS8_HIPBLASLT_LIBRARY}")
    endif()
  endif()
  if(RNS8_ENABLE_CK)
    target_include_directories(${target_name} PRIVATE "${RNS8_CK_GENERATED_INCLUDE_DIR}" "${RNS8_CK_INCLUDE_DIR}")
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    target_include_directories(${target_name} PRIVATE "${RNS8_ROCWMMA_INCLUDE_DIR}")
  endif()
endfunction()

if(RNS8_BUILD_SHARED)
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
endif()

add_library(rns8_static STATIC ${RNS8_PUBLIC_HEADERS} ${RNS8_SOURCES})
add_library(rns8::rns8_static ALIAS rns8_static)
rns8_configure_library(rns8_static)
target_compile_definitions(rns8_static PUBLIC RNS8_STATIC)
set_target_properties(rns8_static PROPERTIES OUTPUT_NAME rns8_static)

if(RNS8_BUILD_TOOLS)
  add_executable(rns8-inspect tools/rns8_inspect.cpp)
  target_include_directories(rns8-inspect PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(rns8-inspect PRIVATE rns8_static)
  rns8_copy_windows_clang_asan_runtime(rns8-inspect)

  add_executable(rns8-verify tools/rns8_verify.cpp)
  target_include_directories(
    rns8-verify
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8-verify PRIVATE rns8_static)
  rns8_copy_windows_clang_asan_runtime(rns8-verify)
endif()

if(RNS8_BUILD_BENCHMARKS)
  set(RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS}")
  string(REPLACE ";" "," RNS8_AMDGPU_TARGETS_TEXT "${RNS8_AMDGPU_TARGETS_TEXT}")
  set(RNS8_HIPCC_PATH_TEXT "")
  set(RNS8_HIPCC_VERSION_TEXT "")
  set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "")
  set(RNS8_HIP_ROOT_TEXT "${RNS8_HIP_ROOT}")
  string(REPLACE "\\" "/" RNS8_HIP_ROOT_TEXT "${RNS8_HIP_ROOT_TEXT}")
  if(RNS8_HIP_ROOT)
    get_filename_component(RNS8_HIP_ROOT_BASENAME "${RNS8_HIP_ROOT}" NAME)
    if(RNS8_HIP_ROOT_BASENAME MATCHES "^[0-9]+(\\.[0-9]+)*$")
      set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "${RNS8_HIP_ROOT_BASENAME}")
    endif()
  endif()
  if(RNS8_ENABLE_HIP AND RNS8_HIP_HIPCC)
    set(RNS8_HIPCC_PATH_TEXT "${RNS8_HIP_HIPCC}")
    string(REPLACE "\\" "/" RNS8_HIPCC_PATH_TEXT "${RNS8_HIPCC_PATH_TEXT}")
    execute_process(
      COMMAND "${RNS8_HIP_HIPCC}" --version
      RESULT_VARIABLE RNS8_HIPCC_VERSION_RESULT
      OUTPUT_VARIABLE RNS8_HIPCC_VERSION_OUTPUT
      ERROR_VARIABLE RNS8_HIPCC_VERSION_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(RNS8_HIPCC_VERSION_RESULT EQUAL 0)
      set(RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_OUTPUT}")
      if(RNS8_HIPCC_VERSION_ERROR)
        string(APPEND RNS8_HIPCC_VERSION_TEXT " ${RNS8_HIPCC_VERSION_ERROR}")
      endif()
      string(REPLACE "\n" " | " RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\r" " " RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\\" "/" RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE "\"" "'" RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      string(REPLACE ";" "," RNS8_HIPCC_VERSION_TEXT "${RNS8_HIPCC_VERSION_TEXT}")
      if(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT STREQUAL "" AND RNS8_HIPCC_VERSION_TEXT MATCHES "HIP version[^0-9]*([0-9][^ |,;]*)")
        set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "${CMAKE_MATCH_1}")
      elseif(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT STREQUAL "" AND RNS8_HIPCC_VERSION_TEXT MATCHES "HIP_VERSION[^0-9]*([0-9][^ |,;]*)")
        set(RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
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
  set(
    RNS8_BENCHMARK_SOURCES
    benchmarks/rns8_bench.cpp
    benchmarks/rns8_bench_args.cpp
    benchmarks/rns8_bench_modes.cpp
    benchmarks/rns8_bench_support.cpp
  )
  if(RNS8_ENABLE_HIP)
    rns8_compile_hip_source(
      RNS8_BENCH_VECTOR_ALU_KERNEL_OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/hip_vector_alu_baseline_kernels.hip"
    )
    list(APPEND RNS8_BENCHMARK_SOURCES "${RNS8_BENCH_VECTOR_ALU_KERNEL_OBJECT}")
  endif()
  add_executable(rns8-bench ${RNS8_BENCHMARK_SOURCES})
  target_compile_definitions(
    rns8-bench
    PRIVATE
      RNS8_CONFIGURED_AMDGPU_TARGETS="${RNS8_AMDGPU_TARGETS_TEXT}"
      RNS8_CONFIGURED_HIP_ENABLED=$<IF:$<BOOL:${RNS8_ENABLE_HIP}>,1,0>
      RNS8_CONFIGURED_HIP_ROOT="${RNS8_HIP_ROOT_TEXT}"
      RNS8_CONFIGURED_HIPCC_PATH="${RNS8_HIPCC_PATH_TEXT}"
      RNS8_CONFIGURED_HIPCC_VERSION="${RNS8_HIPCC_VERSION_TEXT}"
      RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION="${RNS8_HIP_SDK_OR_ROCM_VERSION_TEXT}"
      RNS8_GIT_COMMIT="${RNS8_GIT_COMMIT}"
      RNS8_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_include_directories(
    rns8-bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
  )
  target_link_libraries(rns8-bench PRIVATE rns8_static)
  rns8_copy_windows_clang_asan_runtime(rns8-bench)
  if(RNS8_ENABLE_HIP)
    target_include_directories(rns8-bench PRIVATE ${RNS8_HIP_INCLUDE_DIRS})
    target_compile_definitions(rns8-bench PRIVATE __HIP_PLATFORM_AMD__=1 RNS8_ENABLE_HIP=1)
  endif()
  if(RNS8_ENABLE_ROCWMMA)
    target_compile_definitions(rns8-bench PRIVATE RNS8_ENABLE_ROCWMMA=1)
  endif()
endif()

if(RNS8_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(RNS8_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "RNS8_BUILD_FUZZERS requires an LLVM Clang or clang-cl compiler")
  endif()

  function(rns8_add_cpu_fuzzer target_name source_file)
    add_executable(${target_name} "${source_file}")
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_include_directories(
      ${target_name}
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(${target_name} PRIVATE rns8_static)
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /fsanitize=address -fsanitize=fuzzer /W4 /permissive-)
      if(RNS8_ENABLE_WINDOWS_CLANG_ASAN)
        target_link_libraries(${target_name} PRIVATE "${RNS8_CLANG_FUZZER_LIB}")
      else()
        target_link_options(${target_name} PRIVATE -fsanitize=fuzzer,address)
      endif()
    else()
      target_compile_options(${target_name} PRIVATE -fsanitize=fuzzer,address)
      target_link_options(${target_name} PRIVATE -fsanitize=fuzzer,address)
      target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    rns8_copy_windows_clang_asan_runtime(${target_name})
  endfunction()

  rns8_add_cpu_fuzzer(rns8_fuzz_plan_descriptor tests/fuzz/rns8_fuzz_plan_descriptor.cpp)
  rns8_add_cpu_fuzzer(rns8_fuzz_export_contract tests/fuzz/rns8_fuzz_export_contract.cpp)
  rns8_add_cpu_fuzzer(rns8_fuzz_metadata_json tests/fuzz/rns8_fuzz_metadata_json.cpp)
  target_link_libraries(rns8_fuzz_metadata_json PRIVATE nlohmann_json::nlohmann_json)
endif()

