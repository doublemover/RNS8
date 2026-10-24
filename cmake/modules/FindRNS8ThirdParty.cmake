set(_RNS8_THIRD_PARTY_HINTS)
rns8_append_windows_vcpkg_hints(_RNS8_THIRD_PARTY_HINTS)
if(RNS8_HIP_ROOT)
  list(APPEND _RNS8_THIRD_PARTY_HINTS "${RNS8_HIP_ROOT}")
endif()
if(DEFINED ENV{ROCM_PATH})
  list(APPEND _RNS8_THIRD_PARTY_HINTS "$ENV{ROCM_PATH}")
endif()

find_path(
  RNS8_GMP_INCLUDE_DIR
  NAMES gmp.h
  HINTS ${_RNS8_THIRD_PARTY_HINTS}
  PATH_SUFFIXES include
)
find_library(
  RNS8_GMP_LIBRARY
  NAMES gmp libgmp
  HINTS ${_RNS8_THIRD_PARTY_HINTS}
  PATH_SUFFIXES lib lib64
)
set(RNS8_GMP_FOUND FALSE)
if(RNS8_GMP_INCLUDE_DIR AND RNS8_GMP_LIBRARY)
  rns8_assert_no_linux_windows_vcpkg_paths("GMP include directory" "${RNS8_GMP_INCLUDE_DIR}")
  rns8_assert_no_linux_windows_vcpkg_paths("GMP library" "${RNS8_GMP_LIBRARY}")
  set(RNS8_GMP_FOUND TRUE)
  if(NOT TARGET RNS8::GMP)
    add_library(RNS8::GMP UNKNOWN IMPORTED)
    set_target_properties(
      RNS8::GMP
      PROPERTIES
        IMPORTED_LOCATION "${RNS8_GMP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RNS8_GMP_INCLUDE_DIR}"
    )
  endif()
endif()

find_path(
  RNS8_FLINT_INCLUDE_DIR
  NAMES flint/flint.h
  HINTS ${_RNS8_THIRD_PARTY_HINTS}
  PATH_SUFFIXES include
)
find_library(
  RNS8_FLINT_LIBRARY
  NAMES flint libflint
  HINTS ${_RNS8_THIRD_PARTY_HINTS}
  PATH_SUFFIXES lib lib64
)
set(RNS8_FLINT_FOUND FALSE)
if(RNS8_FLINT_INCLUDE_DIR AND RNS8_FLINT_LIBRARY)
  rns8_assert_no_linux_windows_vcpkg_paths("FLINT include directory" "${RNS8_FLINT_INCLUDE_DIR}")
  rns8_assert_no_linux_windows_vcpkg_paths("FLINT library" "${RNS8_FLINT_LIBRARY}")
  set(RNS8_FLINT_FOUND TRUE)
  if(NOT TARGET RNS8::FLINT)
    add_library(RNS8::FLINT UNKNOWN IMPORTED)
    set_target_properties(
      RNS8::FLINT
      PROPERTIES
        IMPORTED_LOCATION "${RNS8_FLINT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RNS8_FLINT_INCLUDE_DIR}"
    )
  endif()
endif()

set(RNS8ThirdParty_FOUND TRUE)

# Opt-in reference libraries must compile/link the APIs actually used by the
# smoke program, even when the main build disables tests. Never execute here.
set(_RNS8_EXACT_PROBE_LIBRARIES)
set(_RNS8_EXACT_PROBE_DEFINITIONS)
if(RNS8_ENABLE_GMP AND RNS8_GMP_FOUND)
  list(APPEND _RNS8_EXACT_PROBE_LIBRARIES RNS8::GMP)
  list(APPEND _RNS8_EXACT_PROBE_DEFINITIONS -DRNS8_ENABLE_GMP=1)
endif()
if(RNS8_ENABLE_FLINT AND RNS8_FLINT_FOUND)
  list(PREPEND _RNS8_EXACT_PROBE_LIBRARIES RNS8::FLINT)
  if(TARGET RNS8::GMP)
    list(APPEND _RNS8_EXACT_PROBE_LIBRARIES RNS8::GMP)
  endif()
  list(APPEND _RNS8_EXACT_PROBE_DEFINITIONS -DRNS8_ENABLE_FLINT=1)
endif()
set(RNS8_EXACT_REFERENCE_LINK_PROBE "not_run")
if(_RNS8_EXACT_PROBE_LIBRARIES)
  list(REMOVE_DUPLICATES _RNS8_EXACT_PROBE_LIBRARIES)
  unset(_RNS8_EXACT_REFERENCE_LINK_PASSED CACHE)
  try_compile(_RNS8_EXACT_REFERENCE_LINK_PASSED
    "${CMAKE_CURRENT_BINARY_DIR}/exact-reference-link-probe"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_optional_exact_libs.cpp"
    CXX_STANDARD 17
    COMPILE_DEFINITIONS ${_RNS8_EXACT_PROBE_DEFINITIONS}
    LINK_LIBRARIES ${_RNS8_EXACT_PROBE_LIBRARIES}
    OUTPUT_VARIABLE RNS8_EXACT_REFERENCE_LINK_PROBE_OUTPUT)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/exact-reference-link-probe.log"
       "${RNS8_EXACT_REFERENCE_LINK_PROBE_OUTPUT}")
  if(_RNS8_EXACT_REFERENCE_LINK_PASSED)
    set(RNS8_EXACT_REFERENCE_LINK_PROBE "passed")
  else()
    set(RNS8_EXACT_REFERENCE_LINK_PROBE "failed")
    message(FATAL_ERROR "Selected optional exact-reference headers/libraries failed the compiler/link API probe. See ${CMAKE_CURRENT_BINARY_DIR}/exact-reference-link-probe.log; select a coherent compiler/ABI/dependency prefix.")
  endif()
endif()
