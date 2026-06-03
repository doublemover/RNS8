cmake_minimum_required(VERSION 3.28)

project(RNS8DownstreamSmoke LANGUAGES CXX)

find_package(RNS8 CONFIG REQUIRED)

add_executable(rns8-downstream-smoke main.cpp)
target_link_libraries(rns8-downstream-smoke PRIVATE rns8::rns8_static)
target_compile_features(rns8-downstream-smoke PRIVATE cxx_std_17)

enable_testing()
add_test(NAME rns8_downstream_smoke COMMAND rns8-downstream-smoke)
