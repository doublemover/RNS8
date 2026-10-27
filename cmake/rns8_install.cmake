set(RNS8_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/RNS8")
foreach(_rns8_license_component IN ITEMS Runtime Development)
  install(FILES third_party/rank48/Licence_CeCILL-B_V1-en.txt NOTICE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/RNS8/licenses/rank48" COMPONENT ${_rns8_license_component})
endforeach()
if(RNS8_WINDOWS_RUNTIME_DLLS)
  install(FILES ${RNS8_WINDOWS_RUNTIME_DLLS} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime)
endif()
set(RNS8_INSTALL_TARGETS rns8_static)
if(RNS8_BUILD_SHARED)
  list(PREPEND RNS8_INSTALL_TARGETS rns8)
endif()

install(
  TARGETS ${RNS8_INSTALL_TARGETS}
  EXPORT RNS8Targets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Runtime
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Development
)

install(
  DIRECTORY include/rns8
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Development
)

install(
  EXPORT RNS8Targets
  FILE RNS8Targets.cmake
  NAMESPACE rns8::
  DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
  COMPONENT Development
)

configure_package_config_file(
  cmake/RNS8Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/RNS8Config.cmake"
  INSTALL_DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RNS8ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/RNS8Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/RNS8ConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RNS8Runtime.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RNS8StageRuntime.cmake"
  DESTINATION "${RNS8_INSTALL_CONFIG_DIR}"
  COMPONENT Development
)

# Ship a self-contained public consumer alongside the development package.
install(
  FILES
    examples/downstream-cmake/CMakeLists.txt
    examples/downstream-cmake/README.md
    examples/downstream-cmake/main.cpp
    examples/downstream-cmake/exact_wide_cpu_continuation.cpp
    examples/downstream-cmake/materialized_product_recipe.cpp
    examples/downstream-cmake/checked_crt_export.cpp
    examples/downstream-cmake/sparse_finite_ring.cpp
    examples/downstream-cmake/stream_bounded_i64.cpp
    examples/downstream-cmake/sparse_bounded_i64.cpp
    examples/downstream-cmake/explicit_sparse_common.hpp
    examples/downstream-cmake/noninteractive_errors.hpp
  DESTINATION "${CMAKE_INSTALL_DATADIR}/RNS8/examples/downstream-cmake"
  COMPONENT Development
)

install(FILES examples/device-native/CMakeLists.txt examples/device-native/README.md
    examples/device-native/main.cpp examples/device-native/producer_consumer.hip
    examples/device-native/WindowsTestSDK.cmake.in
  DESTINATION "${CMAKE_INSTALL_DATADIR}/RNS8/examples/device-native"
  COMPONENT Development)
