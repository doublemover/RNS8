#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

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
