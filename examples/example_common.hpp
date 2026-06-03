#ifndef RNS8_EXAMPLE_COMMON_HPP
#define RNS8_EXAMPLE_COMMON_HPP

#include <cstdlib>
#include <iostream>

#include "rns8/rns8.h"

inline int fail_status(const char* operation, rns8_status status) {
  std::cerr << operation << " failed: " << rns8_status_string(status) << '\n';
  return EXIT_FAILURE;
}

inline int fail_check(const char* message) {
  std::cerr << message << '\n';
  return EXIT_FAILURE;
}

inline rns8_context_options cpu_context_options() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  return options;
}

inline rns8_gemm_desc base_gemm_desc(rns8_semantics semantics, rns8_bound_kind bound_kind, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

inline rns8_matrix_desc row_major_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t max_prefix) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.max_prefix = max_prefix;
  return desc;
}

#endif
