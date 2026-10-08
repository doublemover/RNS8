set(RNS8_EXAMPLE_TARGETS
  bounded-i64-oneshot
  finite-ring-u8-oneshot
  wrap64-u64-oneshot
  exact-wide-limb-export
  exact-wide-cpu-continuation
)

foreach(example_name IN LISTS RNS8_EXAMPLE_TARGETS)
  string(REPLACE "-" "_" source_stem "${example_name}")
  if(example_name STREQUAL "exact-wide-cpu-continuation")
    set(example_source "downstream-cmake/${source_stem}.cpp")
  else()
    set(example_source "${source_stem}.cpp")
  endif()
  add_executable("rns8-example-${example_name}" "${example_source}")
  target_link_libraries("rns8-example-${example_name}" PRIVATE rns8::rns8_static)
  target_compile_features("rns8-example-${example_name}" PRIVATE cxx_std_17)
  rns8_copy_windows_clang_asan_runtime("rns8-example-${example_name}")
  if(BUILD_TESTING)
    add_test(NAME "example_${source_stem}" COMMAND "rns8-example-${example_name}")
    set_tests_properties("example_${source_stem}" PROPERTIES LABELS "examples;cpu" TIMEOUT 10)
  endif()
endforeach()
