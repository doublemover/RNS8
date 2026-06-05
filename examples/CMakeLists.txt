set(RNS8_EXAMPLE_TARGETS
  bounded-i64-oneshot
  finite-ring-u8-oneshot
  wrap64-u64-oneshot
  exact-wide-limb-export
)

foreach(example_name IN LISTS RNS8_EXAMPLE_TARGETS)
  string(REPLACE "-" "_" source_stem "${example_name}")
  add_executable("rns8-example-${example_name}" "${source_stem}.cpp")
  target_link_libraries("rns8-example-${example_name}" PRIVATE rns8::rns8_static)
  target_compile_features("rns8-example-${example_name}" PRIVATE cxx_std_17)
  rns8_copy_windows_clang_asan_runtime("rns8-example-${example_name}")
  if(BUILD_TESTING)
    add_test(NAME "example_${source_stem}" COMMAND "rns8-example-${example_name}")
    set_tests_properties("example_${source_stem}" PROPERTIES LABELS "examples;cpu")
  endif()
endforeach()
