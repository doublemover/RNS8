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

TEST_CASE("bounded CPU plan schedule uses copied per-tile unsigned bounds") {
  rns8_context* ctx = create_cpu();
  std::vector<uint64_t> bounds = {1, 1000, 1000000, uint64_t{1} << 40u, 5,
                                  2000, 70000,  uint64_t{1} << 20u, uint64_t{1} << 32u};
  auto desc = u64_desc(130, 129, 7, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  bounds[3] = 1;

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_m == 64);
  CHECK(info.tile_n == 64);
  CHECK(info.tile_rows == 3);
  CHECK(info.tile_cols == 3);
  CHECK(info.tile_count == 9);
  CHECK(info.min_required_prefix == 1);
  CHECK(info.max_required_prefix == 6);
  CHECK(info.min_selected_prefix == 1);
  CHECK(info.max_selected_prefix == 6);
  CHECK(info.prefix_group_count == 5);
  CHECK(info.adaptive_prefix_active == 1);
  CHECK(info.adaptive_skip_active == 1);
  CHECK(info.range_bit_length == 41);

  uint64_t written = 0;
  std::vector<rns8_plan_tile_schedule_entry> entries(9);
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries[0].required_prefix == 1);
  CHECK(entries[0].selected_prefix == 1);
  CHECK(entries[0].group_index == 0);
  CHECK(entries[1].required_prefix == 2);
  CHECK(entries[1].selected_prefix == 2);
  CHECK(entries[1].group_index == 1);
  CHECK(entries[2].required_prefix == 3);
  CHECK(entries[2].selected_prefix == 3);
  CHECK(entries[2].group_index == 2);
  CHECK(entries[3].required_prefix == 6);
  CHECK(entries[3].selected_prefix == 6);
  CHECK(entries[3].group_index == 4);
  CHECK(entries[8].required_prefix == 5);
  CHECK(entries[8].selected_prefix == 5);
  CHECK(entries[8].group_index == 3);
  CHECK(entries[8].row_offset == 128);
  CHECK(entries[8].col_offset == 128);
  CHECK(entries[8].row_extent == 2);
  CHECK(entries[8].col_extent == 1);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU oneshot executes and exports per-tile unsigned prefixes") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  std::vector<uint64_t> A(m * k);
  std::vector<uint64_t> B(k * n);
  std::vector<uint64_t> C(m * n, 0);
  for (int64_t row = 0; row < m; ++row) {
    A[row] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[col] = col < 64 ? 7 : 1000;
  }

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto desc = u64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[row * n + col] == A[row] * B[col]);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU oneshot executes and exports per-tile signed prefixes") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  std::vector<int64_t> A(m * k);
  std::vector<int64_t> B(k * n);
  std::vector<int64_t> C(m * n, 0);
  for (int64_t row = 0; row < m; ++row) {
    A[row] = row < 64 ? -2 : -1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[col] = col < 64 ? 3 : -1000;
  }

  const std::vector<uint64_t> bounds = {6, 2000, 3000000, 1000000000};
  auto desc = i64_desc(m, n, k, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[row * n + col] == A[row] * B[col]);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded per-tile bound contracts reject missing or inconsistent bounds") {
  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(65, 65, 1, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> too_few = {7, 1000, 7000000};
  desc.tile_bounds = too_few.data();
  desc.tile_bounds_count = too_few.size();
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  desc.bound = 1;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto signed_desc = i64_desc(1, 1, 1, 0);
  signed_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  const std::vector<uint64_t> too_large = {std::numeric_limits<uint64_t>::max()};
  signed_desc.tile_bounds = too_large.data();
  signed_desc.tile_bounds_count = too_large.size();
  CHECK(rns8_create_plan(ctx, &signed_desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

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
