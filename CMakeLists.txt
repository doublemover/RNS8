cmake_minimum_required(VERSION 3.28)

project(
  RNS8
  VERSION 0.1.0
  DESCRIPTION "Exact integer GEMM with AMD GPU matrix engines, RNS, and CRT reconstruction"
  LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


include(cmake/rns8_options.cmake)
include(cmake/rns8_sources.cmake)
include(cmake/rns8_targets.cmake)
include(cmake/rns8_tests.cmake)
include(cmake/rns8_install.cmake)
