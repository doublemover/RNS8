cmake_minimum_required(VERSION 3.22)
project(RNS8IntegerSplitConsumer LANGUAGES C CXX)
find_package(RNS8 CONFIG REQUIRED)
option(RNS8_SPLIT_USE_SHARED "Use the installed shared target" ON)
if(RNS8_SPLIT_USE_SHARED)
  set(_rns8 rns8::rns8)
else()
  set(_rns8 rns8::rns8_static)
endif()
add_executable(rns8-integer-split-consumer main.c)
set_target_properties(rns8-integer-split-consumer PROPERTIES
  C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_link_libraries(rns8-integer-split-consumer PRIVATE ${_rns8})
rns8_copy_runtime_dependencies(rns8-integer-split-consumer)
if(WIN32 AND RNS8_SPLIT_USE_SHARED)
  # The import library need not change when the implementation DLL changes.
  add_custom_target(stage-integer-split-runtime ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:${_rns8}>"
      "$<TARGET_FILE_DIR:rns8-integer-split-consumer>"
    DEPENDS rns8-integer-split-consumer VERBATIM)
endif()
enable_testing()
add_test(NAME integer-split-public-c17 COMMAND rns8-integer-split-consumer)
set_tests_properties(integer-split-public-c17 PROPERTIES TIMEOUT 30 LABELS "cpu;package;abi")
