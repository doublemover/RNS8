if(NOT DEFINED RNS8_SOURCE_DIR)
  message(FATAL_ERROR "RNS8_SOURCE_DIR is required")
endif()
if(NOT DEFINED RNS8_ACCEL_FAIL_FAST_BINARY_DIR)
  message(FATAL_ERROR "RNS8_ACCEL_FAIL_FAST_BINARY_DIR is required")
endif()
if(NOT DEFINED RNS8_ACCELERATOR_ENABLE_FLAG)
  message(FATAL_ERROR "RNS8_ACCELERATOR_ENABLE_FLAG is required")
endif()

set(configure_args
  -S "${RNS8_SOURCE_DIR}"
  -B "${RNS8_ACCEL_FAIL_FAST_BINARY_DIR}"
  "-D${RNS8_ACCELERATOR_ENABLE_FLAG}=ON"
  -DRNS8_BUILD_TESTS=OFF
  -DRNS8_BUILD_TOOLS=OFF
  -DRNS8_BUILD_BENCHMARKS=OFF
)
if(DEFINED RNS8_CMAKE_GENERATOR AND NOT RNS8_CMAKE_GENERATOR STREQUAL "")
  list(APPEND configure_args -G "${RNS8_CMAKE_GENERATOR}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" ${configure_args}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_stdout
  ERROR_VARIABLE configure_stderr
)

set(configure_output "${configure_stdout}\n${configure_stderr}")
if(configure_result EQUAL 0)
  message(FATAL_ERROR "${RNS8_ACCELERATOR_ENABLE_FLAG} configured successfully; accelerator enable flags must fail fast until a real correctness backend exists")
endif()
if(NOT configure_output MATCHES "accelerator backends are not[ \r\n]+implemented")
  message(FATAL_ERROR "${RNS8_ACCELERATOR_ENABLE_FLAG} did not report the hard-cut unimplemented-backend message")
endif()
if(NOT configure_output MATCHES "evidence-only probes")
  message(FATAL_ERROR "${RNS8_ACCELERATOR_ENABLE_FLAG} did not direct users to evidence-only probes")
endif()
