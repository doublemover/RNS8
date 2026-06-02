#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "rns8/rns8.h"

namespace {

rns8_context* create_cpu_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc exact_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 0;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc exact_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

}  // namespace

TEST_CASE("public exact-wide export ABI rejects invalid limb layout") {
  rns8_context* ctx = create_cpu_context();
  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;
  constexpr uint64_t sentinel = 0x5e5e5e5e5e5e5e5eull;
  std::vector<uint64_t> limbs(128, sentinel);
  const int64_t overflowing_ld = std::numeric_limits<int64_t>::max() / 32 + 1;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs.data(), n - 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs.data(), overflowing_ld, 32) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, nullptr, n, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs.data(), n - 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs.data(), overflowing_ld, 32) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, nullptr, n, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  for (const uint64_t limb : limbs) {
    CHECK(limb == sentinel);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("public exact-wide export rejects bounded and wrap shortcuts") {
  rns8_context* ctx = create_cpu_context();
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  uint64_t limbs[2] = {0x1111111111111111ull, 0x2222222222222222ull};

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_i64(ctx, plan, C, reinterpret_cast<int64_t*>(limbs), n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_u64(ctx, plan, C, limbs, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_wrap_u64(ctx, plan, C, limbs, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs, n, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_i64(ctx, plan, C, reinterpret_cast<int64_t*>(limbs), n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_u64(ctx, plan, C, limbs, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_wrap_u64(ctx, plan, C, limbs, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs, n, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  CHECK(limbs[0] == 0x1111111111111111ull);
  CHECK(limbs[1] == 0x2222222222222222ull);
  rns8_destroy_context(ctx);
}

TEST_CASE("public accelerator backend context kinds fail fast") {
  const rns8_backend_kind backends[] = {RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK, RNS8_BACKEND_WMMA};
  for (const rns8_backend_kind backend : backends) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);
  }
}
