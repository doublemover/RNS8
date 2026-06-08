option(RNS8_ENABLE_HIP "Build the direct HIP backend with explicit hipcc integration" OFF)
option(RNS8_BUILD_TESTS "Build RNS8 correctness tests" ON)
option(RNS8_BUILD_TOOLS "Build RNS8 command-line tools" ON)
option(RNS8_BUILD_EXAMPLES "Build CPU-only public API examples" ON)
option(RNS8_BUILD_BENCHMARKS "Build RNS8 benchmark shell" ON)
option(RNS8_BUILD_SHARED "Build the public RNS8 shared library" ON)
option(RNS8_BUILD_FUZZERS "Build CPU-only LLVM libFuzzer hardening harnesses" OFF)
option(RNS8_ENABLE_CPU_PARALLEL "Enable OpenMP-backed CPU reference/validation parallelism when OpenMP is available" ON)
option(RNS8_ENABLE_WINDOWS_CLANG_ASAN "Link Windows clang-cl CPU targets against LLVM AddressSanitizer" OFF)
option(RNS8_ENABLE_GMP "Require GMP and build the optional exact-reference link smoke" OFF)
option(RNS8_ENABLE_FLINT "Require FLINT and build the optional exact-reference link smoke" OFF)
option(RNS8_PROBE_ACCELERATORS "Run CMake evidence-only discovery for optional accelerator components" OFF)
option(RNS8_ENABLE_HIPBLASLT "Enable hipBLASLt accelerator backend after validated implementation" OFF)
option(RNS8_ENABLE_CK "Enable CK accelerator backend after validated implementation" OFF)
option(RNS8_ENABLE_ROCWMMA "Enable rocWMMA accelerator backend after validated implementation" OFF)
option(
  RNS8_ENABLE_ROCWMMA_WRAP64_CANDIDATE_TESTS
  "Run internal rocWMMA wrap64 byte-GEMM36 candidate tests and benchmark smoke"
  ON
)
option(RNS8_ENABLE_AMDGPU_BUILTINS "Enable target-specific AMDGPU builtin accelerator kernels after validated implementation" OFF)

set(RNS8_AMDGPU_TARGETS "gfx1100" CACHE STRING "Semicolon-separated AMDGPU offload targets for direct HIP sources")
set(RNS8_HIP_ROOT "" CACHE PATH "AMD HIP SDK or ROCm root used by explicit HIP integration")
set(RNS8_ROCM_COVERAGE_TARGETS "" CACHE STRING "Source-level ROCm target family coverage metadata; does not add offload architectures")
set(_RNS8_ROCM_DEPS_BUILD_ROOT_DEFAULT "")
set(_RNS8_ROCM_DEPS_INSTALL_ROOT_DEFAULT "")
if(WIN32)
  set(_RNS8_ROCM_DEPS_BUILD_ROOT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/out/third_party/rocm/build/windows-gfx1100")
  set(_RNS8_ROCM_DEPS_INSTALL_ROOT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/out/third_party/rocm/install/windows-gfx1100")
endif()
set(RNS8_ROCM_DEPS_BUILD_ROOT "${_RNS8_ROCM_DEPS_BUILD_ROOT_DEFAULT}" CACHE PATH "Ignored repo-local build/generated-header root for optional ROCm accelerator dependencies")
set(RNS8_ROCM_DEPS_INSTALL_ROOT "${_RNS8_ROCM_DEPS_INSTALL_ROOT_DEFAULT}" CACHE PATH "Ignored repo-local staged install root for optional ROCm accelerator dependencies")
set(RNS8_CK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rocm/composable_kernel" CACHE PATH "Repo-local Composable Kernel source or install root")
set(RNS8_ROCWMMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rocm/rocWMMA" CACHE PATH "Repo-local rocWMMA source or install root")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
rns8_assert_linux_native_discovery_context()

set(RNS8_CPU_PARALLEL_OPENMP OFF)
if(RNS8_ENABLE_CPU_PARALLEL)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    set(RNS8_CPU_PARALLEL_OPENMP ON)
  else()
    message(STATUS "RNS8 CPU parallelism requested but OpenMP CXX was not found; CPU validation will use the serial fallback")
  endif()
endif()

if(RNS8_ENABLE_HIPBLASLT AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_HIPBLASLT requires RNS8_ENABLE_HIP because hipBLASLt uses resident HIP buffers")
endif()

if(RNS8_ENABLE_CK AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_CK requires RNS8_ENABLE_HIP because CK uses resident HIP buffers")
endif()

if(RNS8_ENABLE_ROCWMMA AND NOT RNS8_ENABLE_HIP)
  message(FATAL_ERROR "RNS8_ENABLE_ROCWMMA requires RNS8_ENABLE_HIP because rocWMMA uses resident HIP buffers")
endif()

if(RNS8_ENABLE_AMDGPU_BUILTINS)
  message(
    FATAL_ERROR
      "AMDGPU builtin accelerator backend is not implemented. "
      "Use RNS8_PROBE_ACCELERATORS=ON or tools/check_dependencies.py --accelerator-probes for evidence-only probes."
  )
endif()

if(RNS8_PROBE_ACCELERATORS)
  find_package(RNS8HIPBLASLT QUIET)
  find_package(RNS8CK QUIET)
  find_package(RNS8ROCWMMA QUIET)
  message(STATUS "RNS8 hipBLASLt candidate evidence: ${RNS8_HIPBLASLT_EVIDENCE}")
  message(STATUS "RNS8 CK candidate evidence: ${RNS8_CK_EVIDENCE}")
  message(STATUS "RNS8 CK compile probe: ${RNS8_CK_COMPILE_PROBE_STATUS}")
  message(STATUS "RNS8 CK primitive probe: ${RNS8_CK_PRIMITIVE_PROBE_STATUS}")
  message(STATUS "RNS8 rocWMMA candidate evidence: ${RNS8_ROCWMMA_EVIDENCE}")
  message(STATUS "RNS8 rocWMMA compile probe: ${RNS8_ROCWMMA_COMPILE_PROBE_STATUS}")
  message(STATUS "RNS8 rocWMMA primitive probe: ${RNS8_ROCWMMA_PRIMITIVE_PROBE_STATUS}")
  message(STATUS "RNS8 AMDGPU builtin candidate evidence: disabled until target-specific exact kernels exist")
  message(
    STATUS
      "RNS8 accelerator backend enablement flags: "
      "hipBLASLt=${RNS8_ENABLE_HIPBLASLT}; "
      "CK=${RNS8_ENABLE_CK}; "
      "rocWMMA=${RNS8_ENABLE_ROCWMMA}; "
      "AMDGPU_BUILTINS=${RNS8_ENABLE_AMDGPU_BUILTINS}"
  )
endif()

find_path(
  RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR
  NAMES boost/multiprecision/cpp_int.hpp
  REQUIRED
)
rns8_assert_no_linux_windows_vcpkg_paths(
  "Boost.Multiprecision include directory"
  "${RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR}"
)
find_package(nlohmann_json CONFIG REQUIRED)
get_target_property(RNS8_NLOHMANN_JSON_INCLUDE_DIRS nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
if(NOT RNS8_NLOHMANN_JSON_INCLUDE_DIRS)
  set(RNS8_NLOHMANN_JSON_INCLUDE_DIRS "")
endif()
rns8_assert_no_linux_windows_vcpkg_paths(
  "nlohmann_json::nlohmann_json include directories"
  ${RNS8_NLOHMANN_JSON_INCLUDE_DIRS}
)

if(RNS8_ENABLE_GMP OR RNS8_ENABLE_FLINT)
  find_package(RNS8ThirdParty REQUIRED)
endif()

if(RNS8_ENABLE_WINDOWS_CLANG_ASAN)
  if(NOT WIN32 OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "RNS8_ENABLE_WINDOWS_CLANG_ASAN requires clang-cl on Windows")
  endif()
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
    OUTPUT_VARIABLE RNS8_CLANG_RESOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(RNS8_CLANG_RUNTIME_DIR "${RNS8_CLANG_RESOURCE_DIR}/lib/windows")
  set(RNS8_CLANG_ASAN_DLL "${RNS8_CLANG_RUNTIME_DIR}/clang_rt.asan_dynamic-x86_64.dll")
  set(RNS8_CLANG_ASAN_LIB "${RNS8_CLANG_RUNTIME_DIR}/clang_rt.asan_dynamic-x86_64.lib")
  set(RNS8_CLANG_ASAN_THUNK_LIB "${RNS8_CLANG_RUNTIME_DIR}/clang_rt.asan_dynamic_runtime_thunk-x86_64.lib")
  set(RNS8_CLANG_FUZZER_LIB "${RNS8_CLANG_RUNTIME_DIR}/clang_rt.fuzzer-x86_64.lib")
  foreach(RNS8_REQUIRED_CLANG_RUNTIME
      "${RNS8_CLANG_ASAN_DLL}"
      "${RNS8_CLANG_ASAN_LIB}"
      "${RNS8_CLANG_ASAN_THUNK_LIB}")
    if(NOT EXISTS "${RNS8_REQUIRED_CLANG_RUNTIME}")
      message(FATAL_ERROR "Missing required LLVM sanitizer runtime: ${RNS8_REQUIRED_CLANG_RUNTIME}")
    endif()
  endforeach()
  if(RNS8_BUILD_FUZZERS AND NOT EXISTS "${RNS8_CLANG_FUZZER_LIB}")
    message(FATAL_ERROR "Missing required LLVM libFuzzer runtime: ${RNS8_CLANG_FUZZER_LIB}")
  endif()
  link_libraries("${RNS8_CLANG_ASAN_LIB}" "${RNS8_CLANG_ASAN_THUNK_LIB}")
  add_compile_definitions(_DISABLE_STRING_ANNOTATION _DISABLE_VECTOR_ANNOTATION)
  configure_file("${RNS8_CLANG_ASAN_DLL}" "${CMAKE_CURRENT_BINARY_DIR}/clang_rt.asan_dynamic-x86_64.dll" COPYONLY)
endif()

function(rns8_copy_windows_clang_asan_runtime target_name)
  if(RNS8_ENABLE_WINDOWS_CLANG_ASAN)
    add_custom_command(
      TARGET "${target_name}"
      POST_BUILD
      COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
          "${RNS8_CLANG_ASAN_DLL}"
          "$<TARGET_FILE_DIR:${target_name}>"
    )
  endif()
endfunction()

if(RNS8_ENABLE_GMP AND NOT RNS8_GMP_FOUND)
  set(_RNS8_GMP_HINT "Install vcpkg feature optional-exact-libs or point VCPKG_ROOT/RNS8_HIP_ROOT at a tree containing gmp.h and gmp/libgmp.")
  if(UNIX AND NOT WIN32)
    set(_RNS8_GMP_HINT "Install native Linux GMP development packages or point CMAKE_PREFIX_PATH at a native Linux GMP install.")
  endif()
  message(
    FATAL_ERROR
      "RNS8_ENABLE_GMP=ON requires GMP headers and library. "
      "${_RNS8_GMP_HINT}"
  )
endif()

if(RNS8_ENABLE_FLINT AND NOT RNS8_FLINT_FOUND)
  set(_RNS8_FLINT_HINT "Install vcpkg feature optional-exact-libs or point VCPKG_ROOT/RNS8_HIP_ROOT at a tree containing flint/flint.h and flint/libflint.")
  if(UNIX AND NOT WIN32)
    set(_RNS8_FLINT_HINT "Install native Linux FLINT development packages or point CMAKE_PREFIX_PATH at a native Linux FLINT install.")
  endif()
  message(
    FATAL_ERROR
      "RNS8_ENABLE_FLINT=ON requires FLINT headers and library. "
      "${_RNS8_FLINT_HINT}"
  )
endif()

