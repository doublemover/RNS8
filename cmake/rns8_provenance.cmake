find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(RNS8_MANIFEST_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/rns8_build_manifest.hpp")
function(rns8_generate_manifest_configuration)
  set(_overlay_hash "none")
  if(RNS8_HIP_STL_OVERLAY)
    file(SHA256 "${RNS8_HIP_STL_OVERLAY}/__clang_hip_runtime_wrapper.h" _overlay_hash)
  endif()
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/configuration.txt"
    CONTENT "generated_native_wmma=${RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA}\ngarner_table=${RNS8_EXPERIMENTAL_GARNER_TABLE}\nconfig=$<CONFIG>\ncompiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\ncompiler_path=${CMAKE_CXX_COMPILER}\nflags=${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}>$<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>$<$<CONFIG:RelWithDebInfo>:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}>$<$<CONFIG:MinSizeRel>:${CMAKE_CXX_FLAGS_MINSIZEREL}>\nhip=${RNS8_ENABLE_HIP}\nhip_compiler=${RNS8_HIP_HIPCC}\nhip_version=${RNS8_HIPCC_VERSION_TEXT}\nhip_stl_overlay_sha256=${_overlay_hash}\ntargets=${RNS8_AMDGPU_TARGETS}\nhipblaslt=${RNS8_ENABLE_HIPBLASLT}\nck=${RNS8_ENABLE_CK}\nrocwmma=${RNS8_ENABLE_ROCWMMA}\nbuiltins=${RNS8_ENABLE_AMDGPU_BUILTINS}\nopenmp=${RNS8_CPU_PARALLEL_OPENMP}\nvcpkg_installed=${VCPKG_INSTALLED_DIR}\n")
endfunction()
# Capture resolved dependency/backend settings, after all find_package calls.
cmake_language(DEFER CALL rns8_generate_manifest_configuration)
add_custom_target(rns8_build_manifest
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_manifest.py"
    --source "${CMAKE_CURRENT_SOURCE_DIR}"
    --config "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/configuration.txt"
    --header "${RNS8_MANIFEST_HEADER}"
  BYPRODUCTS "${RNS8_MANIFEST_HEADER}"
  VERBATIM)
function(rns8_bind_manifest target)
  add_dependencies(${target} rns8_build_manifest)
  if(MSVC)
    target_compile_options(${target} PRIVATE "/FI${RNS8_MANIFEST_HEADER}")
  else()
    target_compile_options(${target} PRIVATE -include "${RNS8_MANIFEST_HEADER}")
  endif()
endfunction()
