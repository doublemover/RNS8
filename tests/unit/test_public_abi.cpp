#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>
#include "rns8/rns8.h"

#define CHECK_ENUM_ABI(type) \
  static_assert(sizeof(type) == sizeof(uint32_t)); \
  static_assert(std::is_same_v<std::underlying_type_t<type>, uint32_t>)
CHECK_ENUM_ABI(rns8_status);
CHECK_ENUM_ABI(rns8_semantics);
CHECK_ENUM_ABI(rns8_layout);
CHECK_ENUM_ABI(rns8_backend_kind);
CHECK_ENUM_ABI(rns8_bound_kind);
CHECK_ENUM_ABI(rns8_sparse_contract);
CHECK_ENUM_ABI(rns8_sparse_operand);
CHECK_ENUM_ABI(rns8_sparse_index_layout);
CHECK_ENUM_ABI(rns8_sparse_value_signedness);
CHECK_ENUM_ABI(rns8_output_domain);
CHECK_ENUM_ABI(rns8_next_op_flags);
CHECK_ENUM_ABI(rns8_resident_matrix_role);
CHECK_ENUM_ABI(rns8_operand_role);
#undef CHECK_ENUM_ABI

TEST_CASE("reserved research APIs fail closed without changing outputs", "[qualification][cpu]") {
  int8_t byte = 42;
  int8_t* component = &byte;
  int64_t component_ld = 77;
  int split_count = 88;
  CHECK(rns8_ozaki_decompose_i64(nullptr, nullptr, 1, 1, 1, &component, &component_ld, &split_count) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(component == &byte);
  CHECK(component_ld == 77);
  CHECK(split_count == 88);
  double overhead = -1.0;
  CHECK(rns8_strassen_gemm_research(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, &overhead) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(overhead == -1.0);
  double failure_probability = -1.0;
  CHECK(rns8_freivalds_verify(nullptr, nullptr, nullptr, nullptr, 1, 2, &failure_probability) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(failure_probability == -1.0);
}
