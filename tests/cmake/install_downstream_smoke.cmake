if(NOT DEFINED RNS8_BINARY_DIR)
  message(FATAL_ERROR "RNS8_BINARY_DIR is required")
endif()
if(NOT DEFINED RNS8_CMAKE_GENERATOR)
  message(FATAL_ERROR "RNS8_CMAKE_GENERATOR is required")
endif()
if(NOT DEFINED RNS8_DOWNSTREAM_BUILD_TYPE)
  message(FATAL_ERROR "RNS8_DOWNSTREAM_BUILD_TYPE is required")
endif()
if(NOT DEFINED RNS8_INSTALL_PREFIX)
  message(FATAL_ERROR "RNS8_INSTALL_PREFIX is required")
endif()
if(NOT DEFINED RNS8_DOWNSTREAM_BINARY_DIR)
  message(FATAL_ERROR "RNS8_DOWNSTREAM_BINARY_DIR is required")
endif()

# The consumer needs only the staged development package, including its example
# sources. Configuration/build has no source-tree includes or vcpkg toolchain.
if(NOT DEFINED RNS8_DOWNSTREAM_EXAMPLE_RELATIVE_DIR)
  message(FATAL_ERROR "RNS8_DOWNSTREAM_EXAMPLE_RELATIVE_DIR is required")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${RNS8_BINARY_DIR}" --prefix "${RNS8_INSTALL_PREFIX}"
  TIMEOUT 30
  RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "RNS8 install failed with ${install_result}")
endif()

set(consumer_source "${RNS8_DOWNSTREAM_BINARY_DIR}/source")
file(MAKE_DIRECTORY "${consumer_source}")
set(package_example "${RNS8_INSTALL_PREFIX}/${RNS8_DOWNSTREAM_EXAMPLE_RELATIVE_DIR}")
foreach(source IN ITEMS CMakeLists.txt README.md main.cpp exact_wide_cpu_continuation.cpp noninteractive_errors.hpp)
  file(COPY "${package_example}/${source}" DESTINATION "${consumer_source}")
endforeach()

set(linkages static)
if(RNS8_DOWNSTREAM_TEST_SHARED)
  list(APPEND linkages shared)
endif()
foreach(linkage IN LISTS linkages)
  set(use_shared OFF)
  if(linkage STREQUAL "shared")
    set(use_shared ON)
  endif()
  set(consumer_binary "${RNS8_DOWNSTREAM_BINARY_DIR}/${linkage}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      -S "${consumer_source}"
      -B "${consumer_binary}"
      -G "${RNS8_CMAKE_GENERATOR}"
      "-DCMAKE_PREFIX_PATH=${RNS8_INSTALL_PREFIX}"
      -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
      -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
      "-DCMAKE_BUILD_TYPE=${RNS8_DOWNSTREAM_BUILD_TYPE}"
      "-DCMAKE_CXX_COMPILER=${RNS8_DOWNSTREAM_COMPILER}"
      "-DCMAKE_CXX_FLAGS=${RNS8_DOWNSTREAM_CXX_FLAGS}"
      "-DCMAKE_EXE_LINKER_FLAGS=${RNS8_DOWNSTREAM_LINKER_FLAGS}"
      "-DRNS8_DOWNSTREAM_USE_SHARED=${use_shared}"
    TIMEOUT 30
    RESULT_VARIABLE configure_result
  )
  if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "downstream ${linkage} configure failed with ${configure_result}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_binary}" --parallel 2 --verbose
    TIMEOUT 30
    RESULT_VARIABLE build_result
  )
  if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "downstream ${linkage} build failed with ${build_result}")
  endif()

  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_binary}" --verbose --timeout 10
    TIMEOUT 30
    RESULT_VARIABLE test_result
  )
  if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "downstream ${linkage} tests failed with ${test_result}")
  endif()
endforeach()
