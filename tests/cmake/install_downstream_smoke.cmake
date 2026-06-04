if(NOT DEFINED RNS8_SOURCE_DIR)
  message(FATAL_ERROR "RNS8_SOURCE_DIR is required")
endif()
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

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${RNS8_BINARY_DIR}" --prefix "${RNS8_INSTALL_PREFIX}"
  RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "RNS8 install failed with ${install_result}")
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    -S "${RNS8_SOURCE_DIR}/examples/downstream-cmake"
    -B "${RNS8_DOWNSTREAM_BINARY_DIR}"
    -G "${RNS8_CMAKE_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${RNS8_INSTALL_PREFIX}"
    "-DCMAKE_BUILD_TYPE=${RNS8_DOWNSTREAM_BUILD_TYPE}"
  RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "downstream configure failed with ${configure_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${RNS8_DOWNSTREAM_BINARY_DIR}"
  RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "downstream build failed with ${build_result}")
endif()

execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${RNS8_DOWNSTREAM_BINARY_DIR}" --output-on-failure
  RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR "downstream test failed with ${test_result}")
endif()
