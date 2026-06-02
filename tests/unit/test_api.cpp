#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <string>
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

rns8_context* create_wrap_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
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

rns8_matrix_desc bounded_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc wrap_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = 0;
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

TEST_CASE("public exact-wide descriptors require bound-none and no stale metadata") {
  rns8_context* ctx = create_cpu_context();
  const uint64_t tile_bound = 1;

  struct Case {
    rns8_semantics semantics;
    rns8_bound_kind stale_bound_kind;
  };
  const Case cases[] = {
      {RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_GLOBAL_MAX_ABS},
      {RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_GLOBAL_MAX_UNSIGNED},
  };

  for (const Case& item : cases) {
    {
      auto desc = exact_desc(item.semantics, 1, 1, 1);
      desc.bound = 1;
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);
    }

    {
      auto desc = exact_desc(item.semantics, 1, 1, 1);
      desc.bound_kind = item.stale_bound_kind;
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);
    }

    {
      auto desc = exact_desc(item.semantics, 1, 1, 1);
      desc.tile_bounds = &tile_bound;
      desc.tile_bounds_count = 1;
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);
    }

    {
      auto matrix = exact_matrix_desc(1, 1, item.semantics);
      matrix.bound_kind = item.stale_bound_kind;
      rns8_matrix* out = nullptr;
      CHECK(rns8_create_matrix(ctx, &matrix, &out) == RNS8_INVALID_ARGUMENT);
      CHECK(out == nullptr);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public exact-wide export rejects stale-prefix matrix handles") {
  rns8_context* ctx = create_cpu_context();
  constexpr uint64_t sentinel0 = 0x8a8a8a8a8a8a8a8aull;
  constexpr uint64_t sentinel1 = 0x8b8b8b8b8b8b8b8bull;
  uint64_t limbs[2] = {sentinel0, sentinel1};

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_i64(ctx, plan, C, reinterpret_cast<int64_t*>(limbs), 1) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_u64(ctx, plan, C, limbs, 1) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public exact-wide export rejects bounded and wrap matrix handles") {
  rns8_context* cpu = create_cpu_context();
  rns8_context* wrap = create_wrap_context();
  constexpr uint64_t sentinel0 = 0x3c3c3c3c3c3c3c3cull;
  constexpr uint64_t sentinel1 = 0x4d4d4d4d4d4d4d4dull;
  uint64_t limbs[2] = {sentinel0, sentinel1};

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* bounded_c = nullptr;
    rns8_matrix* wrap_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
    auto bounded_matrix = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
    REQUIRE(rns8_create_matrix(cpu, &bounded_matrix, &bounded_c) == RNS8_SUCCESS);
    auto wrap_matrix = wrap_matrix_desc(1, 1);
    REQUIRE(rns8_create_matrix(wrap, &wrap_matrix, &wrap_c) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(cpu, plan, bounded_c, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);
    CHECK(rns8_export_exact_wide_signed_limbs(cpu, plan, wrap_c, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);

    rns8_destroy_matrix(wrap_c);
    rns8_destroy_matrix(bounded_c);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* bounded_c = nullptr;
    rns8_matrix* wrap_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
    auto bounded_matrix = bounded_matrix_desc(1, 1, RNS8_BOUNDED_U64);
    REQUIRE(rns8_create_matrix(cpu, &bounded_matrix, &bounded_c) == RNS8_SUCCESS);
    auto wrap_matrix = wrap_matrix_desc(1, 1);
    REQUIRE(rns8_create_matrix(wrap, &wrap_matrix, &wrap_c) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, plan, bounded_c, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);
    CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, plan, wrap_c, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == sentinel0);
    CHECK(limbs[1] == sentinel1);

    rns8_destroy_matrix(wrap_c);
    rns8_destroy_matrix(bounded_c);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(wrap);
  rns8_destroy_context(cpu);
}

TEST_CASE("public exact-wide exports reject null handles without touching output") {
  rns8_context* ctx = create_cpu_context();
  constexpr uint64_t sentinel0 = 0x9999999999999999ull;
  constexpr uint64_t sentinel1 = 0xaaaaaaaaaaaaaaaaull;
  uint64_t limbs[2] = {sentinel0, sentinel1};

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(nullptr, plan, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, nullptr, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, nullptr, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, nullptr, 1, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &matrix, &C) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_unsigned_limbs(nullptr, plan, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, nullptr, C, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, nullptr, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, nullptr, 1, 2) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(C);
    rns8_destroy_plan(plan);
  }

  CHECK(limbs[0] == sentinel0);
  CHECK(limbs[1] == sentinel1);
  rns8_destroy_context(ctx);
}

TEST_CASE("public exact-wide fixed-width exports preserve stride and range contracts") {
  rns8_context* ctx = create_cpu_context();

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* A = nullptr;
    rns8_matrix* B = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &A) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &B) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &C) == RNS8_SUCCESS);
    const int64_t a_value = -1;
    const int64_t b_value = 1;
    REQUIRE(rns8_pack_i64(ctx, A, &a_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, B, &b_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, A, B, C, workspace) == RNS8_SUCCESS);

    uint64_t padded[2] = {0, 0x4545454545454545ull};
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, C, padded, 2, 1) == RNS8_SUCCESS);
    CHECK(padded[0] == std::numeric_limits<uint64_t>::max());
    CHECK(padded[1] == 0x4545454545454545ull);

    std::vector<uint64_t> wide(32, 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, C, wide.data(), 1, 32) == RNS8_SUCCESS);
    for (const uint64_t limb : wide) {
      CHECK(limb == std::numeric_limits<uint64_t>::max());
    }

    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* A = nullptr;
    rns8_matrix* B = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &A) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &B) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &C) == RNS8_SUCCESS);
    const int64_t a_value = std::numeric_limits<int64_t>::min();
    const int64_t b_value = 1;
    REQUIRE(rns8_pack_i64(ctx, A, &a_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, B, &b_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, A, B, C, workspace) == RNS8_SUCCESS);

    uint64_t one_limb[1] = {};
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, C, one_limb, 1, 1) == RNS8_SUCCESS);
    CHECK(one_limb[0] == 0x8000000000000000ull);

    uint64_t two_limbs[2] = {};
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, C, two_limbs, 1, 2) == RNS8_SUCCESS);
    CHECK(two_limbs[0] == 0x8000000000000000ull);
    CHECK(two_limbs[1] == std::numeric_limits<uint64_t>::max());

    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* A = nullptr;
    rns8_matrix* B = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &A) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &B) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &C) == RNS8_SUCCESS);
    const int64_t a_value = std::numeric_limits<int64_t>::max();
    const int64_t b_value = 2;
    REQUIRE(rns8_pack_i64(ctx, A, &a_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, B, &b_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, A, B, C, workspace) == RNS8_SUCCESS);

    uint64_t too_small[1] = {0x3434343434343434ull};
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, C, too_small, 1, 1) == RNS8_RANGE_ERROR);
    CHECK(too_small[0] == 0x3434343434343434ull);

    uint64_t two_limbs[2] = {};
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, C, two_limbs, 1, 2) == RNS8_SUCCESS);
    CHECK(two_limbs[0] == 0xfffffffffffffffeull);
    CHECK(two_limbs[1] == 0);

    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* A = nullptr;
    rns8_matrix* B = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    auto b_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    auto c_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &A) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &B) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &C) == RNS8_SUCCESS);
    const uint64_t a_value = std::numeric_limits<uint64_t>::max();
    const uint64_t b_value = std::numeric_limits<uint64_t>::max();
    REQUIRE(rns8_pack_u64(ctx, A, &a_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, B, &b_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, A, B, C, workspace) == RNS8_SUCCESS);

    uint64_t too_small[1] = {0x6969696969696969ull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, too_small, 1, 1) == RNS8_RANGE_ERROR);
    CHECK(too_small[0] == 0x6969696969696969ull);

    uint64_t two_limbs[2] = {};
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, two_limbs, 1, 2) == RNS8_SUCCESS);
    CHECK(two_limbs[0] == 1);
    CHECK(two_limbs[1] == 0xfffffffffffffffeull);

    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* A = nullptr;
    rns8_matrix* B = nullptr;
    rns8_matrix* C = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    auto b_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    auto c_desc = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &A) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &B) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &C) == RNS8_SUCCESS);
    const uint64_t a_value = std::numeric_limits<uint64_t>::max();
    const uint64_t b_value = 2;
    REQUIRE(rns8_pack_u64(ctx, A, &a_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, B, &b_value, 1, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, A, B, C, workspace) == RNS8_SUCCESS);

    uint64_t one_limb[1] = {0x5656565656565656ull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, one_limb, 1, 1) == RNS8_RANGE_ERROR);
    CHECK(one_limb[0] == 0x5656565656565656ull);

    uint64_t padded[4] = {0, 0, 0x6767676767676767ull, 0x7878787878787878ull};
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, C, padded, 2, 2) == RNS8_SUCCESS);
    CHECK(padded[0] == 0xfffffffffffffffeull);
    CHECK(padded[1] == 1);
    CHECK(padded[2] == 0x6767676767676767ull);
    CHECK(padded[3] == 0x7878787878787878ull);

    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
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

TEST_CASE("public status strings cover every ABI status") {
  struct Case {
    rns8_status status;
    const char* text;
  };
  const Case cases[] = {
      {RNS8_SUCCESS, "success"},
      {RNS8_INVALID_ARGUMENT, "invalid argument"},
      {RNS8_UNSUPPORTED_OS, "unsupported operating system"},
      {RNS8_UNSUPPORTED_ARCH, "unsupported architecture"},
      {RNS8_UNSUPPORTED_BACKEND, "unsupported backend"},
      {RNS8_RANGE_ERROR, "range error"},
      {RNS8_ACCUMULATION_OVERFLOW_RISK, "accumulation overflow risk"},
      {RNS8_WORKSPACE_TOO_SMALL, "workspace too small"},
      {RNS8_BACKEND_FAILURE, "backend failure"},
      {RNS8_VERIFICATION_FAILED, "verification failed"},
      {RNS8_INTERNAL_ERROR, "internal error"},
  };

  for (const Case& item : cases) {
    CHECK(std::string(rns8_status_string(item.status)) == item.text);
  }
  CHECK(std::string(rns8_status_string(static_cast<rns8_status>(0x7fff))) == "unknown status");
}
