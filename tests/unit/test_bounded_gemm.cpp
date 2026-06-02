#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "rns8/rns8.h"

namespace {

rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc i64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc u64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

}  // namespace

TEST_CASE("bounded i64 oneshot handles signs and cancellation") {
  rns8_context* ctx = create_cpu();
  const int64_t A[] = {7, -3, 5, -11, 13, 17};
  const int64_t B[] = {2, -5, 19, 23, -29, 31};
  int64_t C[4] = {};
  auto desc = i64_desc(2, 2, 3, 100000);
  CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 3, B, 2, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == -188);
  CHECK(C[1] == 51);
  CHECK(C[2] == -268);
  CHECK(C[3] == 881);

  const int64_t A_cancel[] = {123456789, 123456789};
  const int64_t B_cancel[] = {1, -1};
  int64_t C_cancel[] = {99};
  auto cancel_desc = i64_desc(1, 1, 2, 1);
  CHECK(rns8_gemm_i64_oneshot(ctx, &cancel_desc, A_cancel, 2, B_cancel, 1, C_cancel, 1) == RNS8_SUCCESS);
  CHECK(C_cancel[0] == 0);

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded i64 and u64 oneshot handle full boundary outputs") {
  rns8_context* ctx = create_cpu();
  {
    const int64_t A[] = {std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<int64_t>::max());
  }
  {
    const int64_t A[] = {-std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == -std::numeric_limits<int64_t>::max());
  }
  {
    const int64_t A[] = {std::numeric_limits<int64_t>::min()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = i64_desc(1, 1, 1, 1ull << 63u);
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<int64_t>::min());
  }
  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t C[] = {0};
    auto desc = u64_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    CHECK(rns8_gemm_u64_oneshot(ctx, &desc, A, 1, B, 1, C, 1) == RNS8_SUCCESS);
    CHECK(C[0] == std::numeric_limits<uint64_t>::max());
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded plan creation rejects insufficient prefix range") {
  rns8_context* ctx = create_cpu();
  auto desc = i64_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  desc.max_prefix = 8;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_RANGE_ERROR);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded plan schedule exposes tile grid and fixed prefix groups") {
  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(130, 129, 7, 1000);
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_m == 64);
  CHECK(info.tile_n == 64);
  CHECK(info.tile_rows == 3);
  CHECK(info.tile_cols == 3);
  CHECK(info.tile_count == 9);
  CHECK(info.min_required_prefix == 2);
  CHECK(info.max_required_prefix == 2);
  CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.prefix_group_count == 1);
  CHECK(info.adaptive_prefix_active == 0);
  CHECK(info.adaptive_skip_active == 0);
  CHECK(info.range_bit_length == 10);

  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(plan, nullptr, 0, &written) == RNS8_SUCCESS);
  CHECK(written == 9);
  std::vector<rns8_plan_tile_schedule_entry> too_small(8);
  CHECK(rns8_get_plan_tile_schedule(plan, too_small.data(), too_small.size(), &written) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(written == 9);

  std::vector<rns8_plan_tile_schedule_entry> entries(9);
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries.front().tile_row == 0);
  CHECK(entries.front().tile_col == 0);
  CHECK(entries.front().row_offset == 0);
  CHECK(entries.front().col_offset == 0);
  CHECK(entries.front().row_extent == 64);
  CHECK(entries.front().col_extent == 64);
  CHECK(entries.front().required_prefix == 2);
  CHECK(entries.front().selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(entries.front().group_index == 0);
  CHECK(entries.front().range_bit_length == 10);
  CHECK(entries.back().tile_row == 2);
  CHECK(entries.back().tile_col == 2);
  CHECK(entries.back().row_offset == 128);
  CHECK(entries.back().col_offset == 128);
  CHECK(entries.back().row_extent == 2);
  CHECK(entries.back().col_extent == 1);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("wrap64 plan schedule reports non-RNS byte-limb scheduling") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = 65;
  desc.n = 64;
  desc.k = 3;
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_rows == 2);
  CHECK(info.tile_cols == 1);
  CHECK(info.tile_count == 2);
  CHECK(info.min_required_prefix == 0);
  CHECK(info.max_required_prefix == 0);
  CHECK(info.min_selected_prefix == 0);
  CHECK(info.max_selected_prefix == 0);
  CHECK(info.prefix_group_count == 0);
  CHECK(info.range_bit_length == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded i64 contract rejects magnitudes beyond int64 min") {
  rns8_context* ctx = create_cpu();
  auto desc = i64_desc(1, 1, 1, (1ull << 63u) + 1u);
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}
