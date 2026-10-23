find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(RNS8_MANIFEST_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/rns8_build_manifest.hpp")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/configuration.txt"
  CONTENT "config=$<CONFIG>\ncompiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\nflags=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}\nhip=${RNS8_ENABLE_HIP}\ntargets=${RNS8_AMDGPU_TARGETS}\nhipblaslt=${RNS8_ENABLE_HIPBLASLT}\nck=${RNS8_ENABLE_CK}\nrocwmma=${RNS8_ENABLE_ROCWMMA}\nbuiltins=${RNS8_ENABLE_AMDGPU_BUILTINS}\nopenmp=${RNS8_CPU_PARALLEL_OPENMP}\n")
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
