set(RNS8_FORBID_WINDOWS_VCPKG_PATHS FALSE)
set(
  RNS8_FORCE_LINUX_NATIVE_DISCOVERY_GUARD
  OFF
  CACHE BOOL
  "Force Linux native dependency-discovery guards for configure smoke tests"
)
mark_as_advanced(RNS8_FORCE_LINUX_NATIVE_DISCOVERY_GUARD)
if(RNS8_FORCE_LINUX_NATIVE_DISCOVERY_GUARD OR (UNIX AND NOT WIN32))
  set(RNS8_FORBID_WINDOWS_VCPKG_PATHS TRUE)
endif()

function(rns8_path_is_forbidden_linux_windows_vcpkg out_var path)
  set(_forbidden FALSE)
  if(path)
    file(TO_CMAKE_PATH "${path}" _rns8_guard_path)
    string(TOLOWER "${_rns8_guard_path}" _rns8_guard_path_lower)
    if(_rns8_guard_path_lower MATCHES "/mnt/c/vcpkg($|[/;:])" OR
       _rns8_guard_path_lower MATCHES "(^|[;:])c:/vcpkg($|[/;:])" OR
       _rns8_guard_path_lower MATCHES "(^|[/;:])x64-windows(-static)?($|[/;:])")
      set(_forbidden TRUE)
    endif()
  endif()
  set(${out_var} "${_forbidden}" PARENT_SCOPE)
endfunction()

function(rns8_assert_no_linux_windows_vcpkg_paths label)
  if(NOT RNS8_FORBID_WINDOWS_VCPKG_PATHS)
    return()
  endif()
  foreach(_rns8_guard_path IN LISTS ARGN)
    rns8_path_is_forbidden_linux_windows_vcpkg(
      _rns8_guard_forbidden
      "${_rns8_guard_path}"
    )
    if(_rns8_guard_forbidden)
      message(
        FATAL_ERROR
          "Linux/UNIX builds must use native system packages and native CMake discovery. "
          "${label} contains a Windows vcpkg path: ${_rns8_guard_path}"
      )
    endif()
  endforeach()
endfunction()

function(rns8_assert_no_linux_windows_vcpkg_target target_name)
  if(NOT RNS8_FORBID_WINDOWS_VCPKG_PATHS OR NOT TARGET "${target_name}")
    return()
  endif()
  get_target_property(_rns8_guard_includes "${target_name}" INTERFACE_INCLUDE_DIRECTORIES)
  if(_rns8_guard_includes AND NOT _rns8_guard_includes MATCHES "-NOTFOUND$")
    rns8_assert_no_linux_windows_vcpkg_paths(
      "${target_name} INTERFACE_INCLUDE_DIRECTORIES"
      ${_rns8_guard_includes}
    )
  endif()
  get_target_property(_rns8_guard_system_includes "${target_name}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if(_rns8_guard_system_includes AND NOT _rns8_guard_system_includes MATCHES "-NOTFOUND$")
    rns8_assert_no_linux_windows_vcpkg_paths(
      "${target_name} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES"
      ${_rns8_guard_system_includes}
    )
  endif()
endfunction()

function(rns8_append_windows_vcpkg_hints list_var)
  if(NOT WIN32)
    return()
  endif()
  set(_rns8_guard_hints "${${list_var}}")
  if(DEFINED ENV{VCPKG_ROOT} AND VCPKG_TARGET_TRIPLET)
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}" _rns8_env_vcpkg_hint)
    list(APPEND _rns8_guard_hints "${_rns8_env_vcpkg_hint}")
  endif()
  if(VCPKG_TARGET_TRIPLET)
    list(APPEND _rns8_guard_hints "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
  endif()
  set(${list_var} "${_rns8_guard_hints}" PARENT_SCOPE)
endfunction()

function(rns8_assert_linux_native_discovery_context)
  if(NOT RNS8_FORBID_WINDOWS_VCPKG_PATHS)
    return()
  endif()

  if(CMAKE_TOOLCHAIN_FILE)
    file(TO_CMAKE_PATH "${CMAKE_TOOLCHAIN_FILE}" _rns8_guard_toolchain)
    string(TOLOWER "${_rns8_guard_toolchain}" _rns8_guard_toolchain_lower)
    if(_rns8_guard_toolchain_lower MATCHES "vcpkg")
      message(
        FATAL_ERROR
          "Linux/UNIX builds must not use the vcpkg toolchain. "
          "Install native packages such as Catch2 and nlohmann-json, then use native CMake discovery."
      )
    endif()
  endif()

  if(VCPKG_TARGET_TRIPLET AND VCPKG_TARGET_TRIPLET MATCHES "x64-windows")
    message(
      FATAL_ERROR
        "Linux/UNIX builds must not use Windows vcpkg triplet ${VCPKG_TARGET_TRIPLET}."
    )
  endif()

  rns8_assert_no_linux_windows_vcpkg_paths("CMAKE_PREFIX_PATH" ${CMAKE_PREFIX_PATH})
  rns8_assert_no_linux_windows_vcpkg_paths("CMAKE_INCLUDE_PATH" ${CMAKE_INCLUDE_PATH})
  rns8_assert_no_linux_windows_vcpkg_paths("CMAKE_SYSTEM_PREFIX_PATH" ${CMAKE_SYSTEM_PREFIX_PATH})
  if(DEFINED ENV{CMAKE_PREFIX_PATH})
    rns8_assert_no_linux_windows_vcpkg_paths("environment CMAKE_PREFIX_PATH" "$ENV{CMAKE_PREFIX_PATH}")
  endif()
  if(DEFINED ENV{CMAKE_INCLUDE_PATH})
    rns8_assert_no_linux_windows_vcpkg_paths("environment CMAKE_INCLUDE_PATH" "$ENV{CMAKE_INCLUDE_PATH}")
  endif()
endfunction()
