#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include "rns8/rns8.h"
#include "rns8/pipeline_recipe.h"
static_assert(sizeof(rns8_allocation_domain_observation)==56);
static_assert(sizeof(rns8_allocation_observation)==136);
static_assert(std::is_standard_layout_v<rns8_allocation_observation>);

#define CHECK_ENUM_ABI(type) \
  static_assert(sizeof(type) == sizeof(uint32_t)); \
  static_assert(std::is_same_v<std::underlying_type_t<type>, uint32_t>)
CHECK_ENUM_ABI(rns8_adaptive_bound_method);
CHECK_ENUM_ABI(rns8_async_failure_stage);
CHECK_ENUM_ABI(rns8_auto_reason);
CHECK_ENUM_ABI(rns8_backend_kind);
CHECK_ENUM_ABI(rns8_bilinear_method);
CHECK_ENUM_ABI(rns8_block_sparse_method);
CHECK_ENUM_ABI(rns8_block_sparse_role);
CHECK_ENUM_ABI(rns8_bound_kind);
CHECK_ENUM_ABI(rns8_complex_product_method);
CHECK_ENUM_ABI(rns8_crt_algorithm);
CHECK_ENUM_ABI(rns8_execution_mode_flags);
CHECK_ENUM_ABI(rns8_gram_input_policy);
CHECK_ENUM_ABI(rns8_gram_method);
CHECK_ENUM_ABI(rns8_gram_orientation);
CHECK_ENUM_ABI(rns8_integer_split_format);
CHECK_ENUM_ABI(rns8_layout);
CHECK_ENUM_ABI(rns8_locality_order);
CHECK_ENUM_ABI(rns8_narrow_product_method);
CHECK_ENUM_ABI(rns8_native_device_access);
CHECK_ENUM_ABI(rns8_native_device_policy);
CHECK_ENUM_ABI(rns8_native_device_type);
CHECK_ENUM_ABI(rns8_native_dispatch_policy);
CHECK_ENUM_ABI(rns8_next_op_flags);
CHECK_ENUM_ABI(rns8_operand_role);
CHECK_ENUM_ABI(rns8_operation_kind);
CHECK_ENUM_ABI(rns8_output_domain);
CHECK_ENUM_ABI(rns8_pipeline_reconstruction_kind);
CHECK_ENUM_ABI(rns8_pipeline_route_kind);
CHECK_ENUM_ABI(rns8_product_prefix_policy);
CHECK_ENUM_ABI(rns8_product_execution_policy);
CHECK_ENUM_ABI(rns8_proof_verdict);
CHECK_ENUM_ABI(rns8_resident_matrix_role);
CHECK_ENUM_ABI(rns8_resident_node_kind);
CHECK_ENUM_ABI(rns8_semantics);
CHECK_ENUM_ABI(rns8_sparse_contract);
CHECK_ENUM_ABI(rns8_sparse_index_layout);
CHECK_ENUM_ABI(rns8_sparse_operand);
CHECK_ENUM_ABI(rns8_sparse_value_signedness);
CHECK_ENUM_ABI(rns8_status);
CHECK_ENUM_ABI(rns8_stream_gemm_schedule);
CHECK_ENUM_ABI(rns8_support_reason);
#undef CHECK_ENUM_ABI
static_assert(sizeof(rns8_integer_split_i64_desc) == 56);
static_assert(sizeof(rns8_integer_split_i64_info) == 72);
static_assert(std::is_standard_layout_v<rns8_integer_split_i64_desc>);
static_assert(std::is_standard_layout_v<rns8_integer_split_i64_info>);

TEST_CASE("research APIs reject unsupported or malformed calls without changing outputs", "[qualification][cpu]") {
  double overhead = -1.0;
  CHECK(rns8_strassen_gemm_research(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, &overhead) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(overhead == -1.0);
  CHECK(rns8_gemm_bilinear_exact(nullptr,nullptr,nullptr,nullptr,nullptr,
      RNS8_BILINEAR_RANK23_SLP,UINT64_MAX,nullptr,&overhead)==RNS8_INVALID_ARGUMENT);
  CHECK(overhead == -1.0);
  double failure_probability = -1.0;
  CHECK(rns8_freivalds_verify(nullptr, nullptr, nullptr, nullptr, 1, 2, &failure_probability) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(failure_probability == -1.0);
}

TEST_CASE("compiled ABI inspection preserves rejected outputs and borrows a stable sorted table", "[abi][cpu]") {
  CHECK(rns8_get_public_abi_info(nullptr) == RNS8_INVALID_ARGUMENT);
  rns8_public_abi_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_public_abi_info(&info) == RNS8_SUCCESS);
  REQUIRE(info.metric_schema_version == RNS8_PUBLIC_ABI_METRICS_VERSION);
  REQUIRE(info.metrics != nullptr);
  REQUIRE(info.metric_count > 0);
  REQUIRE(info.declaration_sha256 != nullptr);
  CHECK(std::strlen(info.declaration_sha256) == 64);
  REQUIRE(info.compiler_id != nullptr);
  REQUIRE(info.runtime_model != nullptr);
  CHECK(std::strlen(info.compiler_id) > 0);
  CHECK(std::strlen(info.runtime_model) > 0);
  bool found_size = false, found_offset = false, found_success = false;
  std::string_view previous;
  for (uint64_t index = 0; index < info.metric_count; ++index) {
    REQUIRE(info.metrics[index].name != nullptr);
    const std::string_view name(info.metrics[index].name);
    CHECK(name > previous);
    previous = name;
    if (name == "sizeof:rns8_public_abi_info") {
      CHECK(info.metrics[index].value == sizeof(info));
      found_size = true;
    }
    if (name == "offsetof:rns8_public_abi_info.metrics") {
      CHECK(info.metrics[index].value == offsetof(rns8_public_abi_info, metrics));
      found_offset = true;
    }
    if (name == "constant:RNS8_SUCCESS") {
      CHECK(info.metrics[index].value == uint64_t(RNS8_SUCCESS));
      found_success = true;
    }
  }
  CHECK(found_size);
  CHECK(found_offset);
  CHECK(found_success);
  auto second = info;
  REQUIRE(rns8_get_public_abi_info(&second) == RNS8_SUCCESS);
  CHECK(second.metrics == info.metrics);
  CHECK(second.declaration_sha256 == info.declaration_sha256);
  for (unsigned mode = 0; mode != 3; ++mode) {
    auto rejected = info;
    if (mode == 0) rejected.struct_size = sizeof(rejected) - 1;
    else if (mode == 1) rejected.struct_size = sizeof(rejected) + 8;
    else rejected.abi_version = UINT32_MAX;
    unsigned char before[sizeof(rejected)];
    std::memcpy(before, &rejected, sizeof(rejected));
    CHECK(rns8_get_public_abi_info(&rejected) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(before, &rejected, sizeof(rejected)) == 0);
  }
  // This is a real, aligned short allocation, not a full struct with a fake size.
  struct alignas(rns8_public_abi_info) Header { uint64_t struct_size; };
  Header header{sizeof(Header)};
  CHECK(rns8_get_public_abi_info(reinterpret_cast<rns8_public_abi_info*>(&header)) == RNS8_INVALID_ARGUMENT);
  CHECK(header.struct_size == sizeof(Header));
}
