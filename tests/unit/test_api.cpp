#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"
#include "rns8/rns8.hpp"

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

rns8::detail::PlanLoweringDescription lowering_for_plan(rns8_plan* plan) {
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);

  rns8_plan_schedule_info schedule{};
  schedule.struct_size = sizeof(schedule);
  schedule.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &schedule) == RNS8_SUCCESS);

  return rns8::detail::describe_plan_lowering(backend, packing, schedule);
}

void set_autotune_cache_path_for_test(const std::filesystem::path& path) {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str());
#else
  setenv("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str(), 1);
#endif
}

void clear_autotune_cache_path_for_test() {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", "");
#else
  unsetenv("RNS8_AUTOTUNE_CACHE_PATH");
#endif
}

struct ScopedAutotuneCachePath {
  explicit ScopedAutotuneCachePath(const std::filesystem::path& path) { set_autotune_cache_path_for_test(path); }
  ~ScopedAutotuneCachePath() { clear_autotune_cache_path_for_test(); }
};

std::filesystem::path unique_cache_fixture_path(const char* stem) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (std::string(stem) + "-" + std::to_string(tick) + ".json");
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

rns8_matrix_desc finite_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = 0;
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

rns8_matrix_storage_info matrix_storage_info(const rns8_matrix* matrix) {
  rns8_matrix_storage_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_storage_info(matrix, &info) == RNS8_SUCCESS);
  return info;
}

rns8_prepack_cache_key_info prepack_cache_key_info(
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role) {
  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_prepack_cache_key_info(plan, matrix, operand_role, &info) == RNS8_SUCCESS);
  return info;
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
  std::vector<rns8_backend_kind> backends;
#if !defined(RNS8_ENABLE_ROCWMMA) || !RNS8_ENABLE_ROCWMMA
  backends.push_back(RNS8_BACKEND_WMMA);
#endif
#if !defined(RNS8_ENABLE_CK) || !RNS8_ENABLE_CK
  backends.push_back(RNS8_BACKEND_CK);
#endif
#if !defined(RNS8_ENABLE_HIPBLASLT) || !RNS8_ENABLE_HIPBLASLT
  backends.push_back(RNS8_BACKEND_HIPBLASLT);
#endif
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

TEST_CASE("malformed accelerator plan descriptors fail before unsupported routing") {
  rns8_context* ctx = create_cpu_context();
  const rns8_backend_kind backends[] = {RNS8_BACKEND_CK, RNS8_BACKEND_WMMA};

  for (const rns8_backend_kind backend : backends) {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = backend;
    desc.m = 1;
    desc.n = 1;
    desc.k = 1;
    desc.bound = 1;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    auto malformed = desc;
    malformed.flags = 1;
    CHECK(rns8_create_plan(ctx, &malformed, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    malformed = desc;
    malformed.m = 0;
    CHECK(rns8_create_plan(ctx, &malformed, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public backend capability info separates correctness and accelerator candidates") {
  rns8_backend_capability_info cpu{};
  cpu.struct_size = sizeof(cpu);
  cpu.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_backend_capability_info(RNS8_BACKEND_CPU_REFERENCE, &cpu) == RNS8_SUCCESS);
  CHECK(cpu.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(cpu.is_accelerator == 0);
  CHECK(cpu.is_correctness_backend == 1);
  CHECK(cpu.compiled_kernel_available == 1);
  CHECK(cpu.exact_differential_validated == 1);
  CHECK(cpu.performance_validated == 0);
  CHECK(std::string(cpu.status) == "implemented_correctness_backend");
  CHECK(std::string(cpu.selected_kernel) == "cpu_reference_scalar_rns_gemm_v1");

  rns8_backend_capability_info wrap{};
  wrap.struct_size = sizeof(wrap);
  wrap.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_backend_capability_info(RNS8_BACKEND_WRAP64_BYTE_LIMB, &wrap) == RNS8_SUCCESS);
  CHECK(wrap.supports_wrap64 == 1);
  CHECK(std::string(wrap.selected_kernel) == "cpu_wrap64_byte_limb_reference_v1");

  const rns8_backend_kind accelerators[] = {RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK, RNS8_BACKEND_WMMA};
  for (const rns8_backend_kind backend : accelerators) {
    rns8_backend_capability_info capability{};
    capability.struct_size = sizeof(capability);
    capability.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_backend_capability_info(backend, &capability) == RNS8_SUCCESS);
    CHECK(capability.is_accelerator == 1);
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    if (backend == RNS8_BACKEND_HIPBLASLT) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_baseline_backend");
      CHECK(std::string(capability.selected_kernel) == "hipblaslt_int8_i32_scratch_reduce_baseline_v1");
      CHECK(std::string(capability.epilogue_mode) == "separate_i32_scratch_residue_reduce");
      CHECK(std::string(capability.isa_evidence) == "hipblaslt_library_int8_matmul_baseline");
      continue;
    }
#endif
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
    if (backend == RNS8_BACKEND_CK) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_opt_in_ck_backend");
      CHECK(std::string(capability.selected_kernel) == "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1");
      CHECK(std::string(capability.epilogue_mode) == "ck_fused_i32_to_centered_residue");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_ck_canonical_pack_workspace");
      CHECK(std::string(capability.isa_evidence) ==
            "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
      continue;
    }
#endif
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
    if (backend == RNS8_BACKEND_WMMA) {
      CHECK(capability.is_correctness_backend == 1);
      CHECK(capability.requires_feature_detection == 1);
      CHECK(capability.enable_flag_fail_fast == 0);
      CHECK(capability.candidate_evidence_only == 0);
      CHECK(capability.compiled_kernel_available == 1);
      CHECK(capability.exact_differential_validated == 1);
      CHECK(capability.performance_validated == 0);
      CHECK(std::string(capability.status) == "implemented_opt_in_rocwmma_backend");
      CHECK(std::string(capability.selected_kernel) == "rocwmma_i8_i32_signed_hot_residue_v1");
      CHECK(std::string(capability.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_rocwmma_pack_workspace");
      CHECK(std::string(capability.isa_evidence) == "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
      continue;
    }
#endif
    CHECK(capability.is_correctness_backend == 0);
    CHECK(capability.requires_feature_detection == 1);
    CHECK(capability.enable_flag_fail_fast == 1);
    CHECK(capability.candidate_evidence_only == 1);
    CHECK(capability.compiled_kernel_available == 0);
    CHECK(capability.exact_differential_validated == 0);
    CHECK(capability.performance_validated == 0);
    CHECK(std::string(capability.isa_evidence) == "not_validated");
    if (backend == RNS8_BACKEND_CK) {
      CHECK(std::string(capability.status) == "not_enabled_in_this_build");
      CHECK(std::string(capability.selected_kernel) == "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1_disabled");
      CHECK(std::string(capability.epilogue_mode) == "ck_fused_i32_to_centered_residue_disabled");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_ck_pack_workspace_disabled");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
    } else if (backend == RNS8_BACKEND_WMMA) {
      CHECK(std::string(capability.status) == "not_enabled_or_builtin_not_implemented");
      CHECK(std::string(capability.selected_kernel) == "rocwmma_i8_i32_signed_hot_residue_v1_disabled");
      CHECK(std::string(capability.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue_disabled");
      CHECK(std::string(capability.workspace_mode) == "resident_device_buffers_with_rocwmma_pack_workspace_disabled");
      CHECK(capability.supports_bounded_rns == 1);
      CHECK(capability.supports_exact_wide_rns == 1);
      CHECK(capability.supports_finite_u8 == 1);
      CHECK(capability.supports_wrap64 == 0);
    } else {
      CHECK(std::string(capability.status) == "not_implemented_evidence_only");
      CHECK(std::string(capability.epilogue_mode) == "not_implemented");
    }
  }

  rns8_backend_capability_info bad_abi{};
  bad_abi.struct_size = sizeof(bad_abi) - 1;
  bad_abi.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_backend_capability_info(RNS8_BACKEND_CPU_REFERENCE, &bad_abi) == RNS8_INVALID_ARGUMENT);

  rns8_backend_capability_info unknown{};
  unknown.struct_size = sizeof(unknown);
  unknown.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_backend_capability_info(static_cast<rns8_backend_kind>(999), &unknown) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("public plan backend info exposes selected kernel and autotune contract") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_U64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 2;
    desc.n = 3;
    desc.k = 4;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.is_accelerator == 0);
    CHECK(info.is_correctness_backend == 1);
    CHECK(info.is_matrix_engine_backend == 0);
    CHECK(info.compiled_kernel_available == 1);
    CHECK(info.exact_differential_validated == 1);
    CHECK(info.performance_validated == 0);
    CHECK(info.workspace_required_bytes == 0);
    CHECK(std::string(info.selected_kernel) == "cpu_reference_scalar_rns_gemm_v1");
    CHECK(std::string(info.capability_status) == "implemented_correctness_backend");
    CHECK(std::string(info.epilogue_mode) == "fused_centered_residue_then_crt_export");
    CHECK(std::string(info.workspace_mode) == "host_reference_workspace");
    CHECK(std::string(info.isa_evidence) == "not_applicable_cpu");
    CHECK(std::string(info.autotune_key).find("backend=cpu-reference;semantics=bounded_u64") == 0);
    CHECK(std::string(info.autotune_key).find(";kernel=cpu_reference_scalar_rns_gemm_v1;") != std::string::npos);

    rns8_plan_backend_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_plan_backend_info(plan, &bad_abi) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap_ctx = create_wrap_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_WRAP_U64_MOD_2_64;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    desc.m = 2;
    desc.n = 2;
    desc.k = 2;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(wrap_ctx, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(std::string(info.selected_kernel) == "cpu_wrap64_byte_limb_reference_v1");
    CHECK(std::string(info.epilogue_mode) == "low64_wrap_export");
    CHECK(std::string(info.workspace_mode) == "host_byte_limb_reference_workspace");
    CHECK(std::string(info.autotune_key).find("semantics=wrap_u64_mod_2_64") != std::string::npos);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_backend_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_plan_backend_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("public matrix storage info exposes source-versioned resident layouts") {
  rns8_context* cpu = create_cpu_context();
  {
    auto desc = bounded_matrix_desc(2, 3, RNS8_BOUNDED_I64);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.semantics == RNS8_BOUNDED_I64);
    CHECK(info.logical_layout == RNS8_LAYOUT_ROW_MAJOR);
    CHECK(info.bound_kind == RNS8_BOUND_GLOBAL_MAX_ABS);
    CHECK(info.rows == 2);
    CHECK(info.cols == 3);
    CHECK(info.logical_ld == 3);
    CHECK(info.max_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.source_version == 0);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_byte_limbs_current == 0);
    CHECK(info.device_byte_limbs_current == 0);
    CHECK(info.uses_residue_storage == 1);
    CHECK(info.uses_byte_limb_storage == 0);
    CHECK(info.hip_device_id == -1);
    CHECK(info.host_residue_bytes == 2 * 3 * RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(info.device_residue_bytes == 0);
    CHECK(info.host_byte_limb_bytes == 0);
    CHECK(info.device_byte_limb_bytes == 0);
    CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.storage_scope) == "host_resident_storage");

    const int64_t values[] = {1, -2, 3, 4, -5, 6};
    REQUIRE(rns8_pack_i64(cpu, matrix, values, 3, 42) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.source_version == 42);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);
    CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");

    rns8_matrix_storage_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_matrix_storage_info(matrix, &bad_abi) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(matrix);
  }
  {
    auto desc = finite_matrix_desc(2, 2, RNS8_FINITE_RING_U8);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.semantics == RNS8_FINITE_RING_U8);
    CHECK(info.max_prefix == 0);
    CHECK(info.finite_modulus == 0);
    CHECK(info.host_residues_current == 0);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_residue_bytes == 4);
    CHECK(std::string(info.layout_version) == "finite_u8_centered_residue_v1");
    CHECK(std::string(info.storage_scope) == "host_resident_storage");

    const uint8_t values[] = {1, 2, 3, 4};
    REQUIRE(rns8_pack_finite_u8(cpu, matrix, 251, values, 2, 7) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.finite_modulus == 251);
    CHECK(info.source_version == 7);
    CHECK(info.host_residues_current == 1);
    CHECK(info.device_residues_current == 0);

    rns8_destroy_matrix(matrix);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap = create_wrap_context();
  {
    auto desc = wrap_matrix_desc(2, 3);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(wrap, &desc, &matrix) == RNS8_SUCCESS);

    auto info = matrix_storage_info(matrix);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(info.semantics == RNS8_WRAP_U64_MOD_2_64);
    CHECK(info.max_prefix == 0);
    CHECK(info.uses_residue_storage == 0);
    CHECK(info.uses_byte_limb_storage == 1);
    CHECK(info.host_residues_current == 0);
    CHECK(info.device_residues_current == 0);
    CHECK(info.host_byte_limbs_current == 1);
    CHECK(info.device_byte_limbs_current == 0);
    CHECK(info.host_residue_bytes == 0);
    CHECK(info.host_byte_limb_bytes == 2 * 3 * 8);
    CHECK(std::string(info.layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.storage_scope) == "host_byte_limb_storage");

    const uint64_t values[] = {0, 1, 2, 3, 4, 5};
    REQUIRE(rns8_pack_u64(wrap, matrix, values, 3, 99) == RNS8_SUCCESS);
    info = matrix_storage_info(matrix);
    CHECK(info.source_version == 99);
    CHECK(info.host_byte_limbs_current == 1);
    CHECK(info.device_byte_limbs_current == 0);

    rns8_destroy_matrix(matrix);
  }
  rns8_destroy_context(wrap);

  rns8_matrix_storage_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_matrix_storage_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("C++ matrix wrapper exposes storage info") {
  rns8::Context context(-1, RNS8_BACKEND_CPU_REFERENCE);
  rns8::Matrix matrix(context, bounded_matrix_desc(1, 2, RNS8_BOUNDED_U64));

  const auto info = matrix.storage_info();
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(info.semantics == RNS8_BOUNDED_U64);
  CHECK(info.rows == 1);
  CHECK(info.cols == 2);
  CHECK(std::string(info.layout_version) == "rns_centered_residue_planes_v1");
}

TEST_CASE("public prepack cache key info validates operand role and source version") {
  rns8_context* cpu = create_cpu_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 2;
  desc.n = 3;
  desc.k = 4;
  desc.bound = 64;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  auto a_desc = bounded_matrix_desc(desc.m, desc.k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(desc.k, desc.n, RNS8_BOUNDED_I64);
  REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &b) == RNS8_SUCCESS);

  const int64_t a_values[] = {1, -2, 3, 4, -5, 6, 7, -8};
  const int64_t b_values[] = {1, 2, -3, 4, -5, 6, 7, 8, -9, 10, -11, 12};
  REQUIRE(rns8_pack_i64(cpu, a, a_values, desc.k, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, b, b_values, desc.n, 22) == RNS8_SUCCESS);

  auto a_key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(a_key.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(a_key.semantics == RNS8_BOUNDED_I64);
  CHECK(a_key.operand_role == RNS8_OPERAND_A);
  CHECK(a_key.cache_key_valid == 1);
  CHECK(a_key.reusable_prepack_cache_available == 0);
  CHECK(a_key.production_prepack_cache_available == 0);
  CHECK(a_key.hip_device_id == -1);
  CHECK(a_key.reserved0 == 0);
  CHECK(a_key.matrix_rows == desc.m);
  CHECK(a_key.matrix_cols == desc.k);
  CHECK(a_key.max_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(a_key.source_version == 11);
  CHECK(a_key.plan_fingerprint != 0);
  CHECK(a_key.cache_key_hash != 0);
  CHECK(std::string(a_key.matrix_layout_version) == "rns_centered_residue_planes_v1");
  CHECK(std::string(a_key.operand_layout_version) == "rns_centered_residue_planes_v1");
  CHECK(std::string(a_key.cache_scope) == "validated_key_no_production_cache");
  CHECK(std::string(a_key.cache_key).find("operand=A") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("target_id=cpu") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("kernel=cpu_reference_scalar_rns_gemm_v1") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("prefix_schedule_hash=") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("k_block_size=0") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("source_version=11") != std::string::npos);
  CHECK(std::string(a_key.cache_key).find("hip_device_id=-1") != std::string::npos);

  const auto b_key = prepack_cache_key_info(plan, b, RNS8_OPERAND_B);
  CHECK(b_key.operand_role == RNS8_OPERAND_B);
  CHECK(b_key.matrix_rows == desc.k);
  CHECK(b_key.matrix_cols == desc.n);
  CHECK(b_key.source_version == 22);
  CHECK(b_key.plan_fingerprint == a_key.plan_fingerprint);
  CHECK(b_key.cache_key_hash != a_key.cache_key_hash);
  CHECK(std::string(b_key.cache_key).find("operand=B") != std::string::npos);

  const uint64_t old_a_hash = a_key.cache_key_hash;
  REQUIRE(rns8_pack_i64(cpu, a, a_values, desc.k, 12) == RNS8_SUCCESS);
  a_key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(a_key.source_version == 12);
  CHECK(a_key.cache_key_hash != old_a_hash);
  CHECK(std::string(a_key.cache_key).find("source_version=12") != std::string::npos);

  rns8_prepack_cache_key_info bad_abi{};
  bad_abi.struct_size = sizeof(bad_abi) - 1;
  bad_abi.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &bad_abi) == RNS8_INVALID_ARGUMENT);

  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(nullptr, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, nullptr, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, static_cast<rns8_operand_role>(99), &info) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_B, &info) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(b);
  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(cpu);
}

TEST_CASE("public prepack cache key info rejects finite modulus mismatch") {
  rns8_context* cpu = create_cpu_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_RING_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.finite_modulus = 251;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  auto a_desc = finite_matrix_desc(desc.m, desc.k, RNS8_FINITE_RING_U8);
  REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &a) == RNS8_SUCCESS);

  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);

  const uint8_t values[] = {1, 2, 3, 4};
  REQUIRE(rns8_pack_finite_u8(cpu, a, 251, values, desc.k, 33) == RNS8_SUCCESS);
  const auto key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.finite_modulus == 251);
  CHECK(key.source_version == 33);
  CHECK(std::string(key.matrix_layout_version) == "finite_u8_centered_residue_v1");

  REQUIRE(rns8_pack_finite_u8(cpu, a, 255, values, desc.k, 34) == RNS8_SUCCESS);
  CHECK(rns8_get_prepack_cache_key_info(plan, a, RNS8_OPERAND_A, &info) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(cpu);
}

TEST_CASE("public prepack cache key info covers wrap64 byte-limb operands") {
  rns8_context* wrap = create_wrap_context();
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = 2;
  desc.n = 2;
  desc.k = 3;

  rns8_plan* plan = nullptr;
  rns8_matrix* a = nullptr;
  auto a_desc = wrap_matrix_desc(desc.m, desc.k);
  REQUIRE(rns8_create_plan(wrap, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap, &a_desc, &a) == RNS8_SUCCESS);

  const uint64_t values[] = {0, 1, 2, 3, 4, 5};
  REQUIRE(rns8_pack_u64(wrap, a, values, desc.k, 55) == RNS8_SUCCESS);
  const auto key = prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
  CHECK(key.semantics == RNS8_WRAP_U64_MOD_2_64);
  CHECK(key.operand_role == RNS8_OPERAND_A);
  CHECK(key.matrix_rows == desc.m);
  CHECK(key.matrix_cols == desc.k);
  CHECK(key.max_prefix == 0);
  CHECK(key.source_version == 55);
  CHECK(key.hip_device_id == -1);
  CHECK(std::string(key.matrix_layout_version) == "wrap64_byte_limb_v1");
  CHECK(std::string(key.operand_layout_version) == "wrap64_byte_limb_v1");
  CHECK(std::string(key.cache_key).find("semantics=wrap_u64_mod_2_64") != std::string::npos);

  rns8_destroy_matrix(a);
  rns8_destroy_plan(plan);
  rns8_destroy_context(wrap);
}

TEST_CASE("C++ prepack cache key helper exposes validated key material") {
  rns8::Context context(-1, RNS8_BACKEND_CPU_REFERENCE);
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 2;
  desc.bound = 8;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8::Plan plan(context, desc);
  rns8::Matrix a(context, bounded_matrix_desc(desc.m, desc.k, RNS8_BOUNDED_U64));
  const uint64_t values[] = {1, 2};
  REQUIRE(rns8_pack_u64(context.get(), a.get(), values, desc.k, 77) == RNS8_SUCCESS);

  const auto key = rns8::prepack_cache_key_info(plan, a, RNS8_OPERAND_A);
  CHECK(key.operand_role == RNS8_OPERAND_A);
  CHECK(key.source_version == 77);
  CHECK(key.hip_device_id == -1);
  CHECK(key.cache_key_hash != 0);
}

TEST_CASE("public plan packing info exposes resident and transient layout contracts") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 2;
    desc.n = 3;
    desc.k = 4;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_packing_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
    CHECK(info.semantics == RNS8_BOUNDED_I64);
    CHECK(info.uses_resident_matrix_inputs == 1);
    CHECK(info.uses_transient_pack_workspace == 0);
    CHECK(info.uses_matrix_engine_pack_layout == 0);
    CHECK(info.reusable_prepack_cache_available == 0);
    CHECK(info.production_prepack_cache_available == 0);
    CHECK(info.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
    CHECK(info.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
    CHECK(info.output_host_current == 1);
    CHECK(info.output_device_current == 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_FINAL_EXPORT) != 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
    CHECK((info.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) == 0);
    CHECK(std::string(info.input_domain_name) == "rns_residue_current");
    CHECK(std::string(info.output_domain_name) == "rns_residue_current");
    CHECK(std::string(info.next_op_hint).find("without logical export") != std::string::npos);
    CHECK(info.a_pack_workspace_bytes == 0);
    CHECK(info.b_pack_workspace_bytes == 0);
    CHECK(info.accumulator_workspace_bytes == 0);
    CHECK(info.library_workspace_bytes == 0);
    CHECK(info.total_transient_workspace_bytes == 0);
    CHECK(std::string(info.a_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.b_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.output_layout_version) == "rns_centered_residue_planes_v1");
    CHECK(std::string(info.prepack_cache_scope) == "host_resident_no_prepack_cache");

    rns8_plan_packing_info bad_abi{};
    bad_abi.struct_size = sizeof(bad_abi) - 1;
    bad_abi.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_plan_packing_info(plan, &bad_abi) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap_ctx = create_wrap_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_WRAP_U64_MOD_2_64;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    desc.m = 2;
    desc.n = 2;
    desc.k = 2;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(wrap_ctx, &desc, &plan) == RNS8_SUCCESS);

    rns8_plan_packing_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
    CHECK(info.semantics == RNS8_WRAP_U64_MOD_2_64);
    CHECK(info.uses_resident_matrix_inputs == 1);
    CHECK(info.uses_transient_pack_workspace == 0);
    CHECK(info.uses_matrix_engine_pack_layout == 0);
    CHECK(info.input_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB);
    CHECK(info.output_domain == RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB);
    CHECK(info.output_host_current == 1);
    CHECK(info.output_device_current == 0);
    CHECK(info.next_op_flags == RNS8_NEXT_OP_FINAL_EXPORT);
    CHECK(std::string(info.output_domain_name) == "wrap64_byte_limb_current");
    CHECK(std::string(info.a_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.b_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.output_layout_version) == "wrap64_byte_limb_v1");
    CHECK(std::string(info.prepack_cache_scope) == "host_byte_limb_no_prepack_cache");

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_packing_info null_info{};
  null_info.struct_size = sizeof(null_info);
  null_info.abi_version = RNS8_ABI_VERSION;
  CHECK(rns8_get_plan_packing_info(nullptr, &null_info) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("internal plan lowering description classifies domains and continuation choices") {
  rns8_context* cpu = create_cpu_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_BOUNDED_I64;
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = 4;
    desc.n = 5;
    desc.k = 6;
    desc.bound = 16;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(cpu, &desc, &plan) == RNS8_SUCCESS);
    const auto lowering = lowering_for_plan(plan);
    CHECK(lowering.operation == "MatMul");
    CHECK(lowering.semantic_contract == "bounded_i64");
    CHECK(lowering.backend_family == "cpu_reference");
    CHECK(lowering.input_domain == "rns_residue_current");
    CHECK(lowering.output_domain == "rns_residue_current");
    CHECK(lowering.desired_output == "final_export_or_rns_chain");
    CHECK(lowering.schedule_strategy == "fixed_prefix_9");
    CHECK(lowering.packing_strategy == "resident_matrix_inputs");
    CHECK(lowering.reuse_strategy == "resident_inputs_no_prepack");
    CHECK(lowering.conversion_strategy == "no_conversion_needed_for_rns_chain");
    CHECK(lowering.lowering_path.find("RnsResidueCurrent") != std::string::npos);
    CHECK(lowering.final_export_available);
    CHECK(lowering.rns_continuation_available);
    CHECK_FALSE(lowering.native_continuation_available);
    CHECK_FALSE(lowering.native_to_rns_available);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(cpu);

  rns8_context* wrap_ctx = create_wrap_context();
  {
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_WRAP_U64_MOD_2_64;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    desc.m = 2;
    desc.n = 2;
    desc.k = 2;

    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(wrap_ctx, &desc, &plan) == RNS8_SUCCESS);
    const auto lowering = lowering_for_plan(plan);
    CHECK(lowering.semantic_contract == "wrap_u64_mod_2_64");
    CHECK(lowering.backend_family == "wrap64_reference");
    CHECK(lowering.input_domain == "wrap64_byte_limb_current");
    CHECK(lowering.output_domain == "wrap64_byte_limb_current");
    CHECK(lowering.desired_output == "final_export");
    CHECK(lowering.schedule_strategy == "semantic_specific_no_rns_prefix_schedule");
    CHECK(lowering.conversion_strategy == "wrap64_byte_limb_final_export_or_same_semantic_reuse");
    CHECK(lowering.lowering_path.find("Low64Export") != std::string::npos);
    CHECK(lowering.final_export_available);
    CHECK_FALSE(lowering.rns_continuation_available);
    CHECK_FALSE(lowering.native_continuation_available);
    CHECK_FALSE(lowering.native_to_rns_available);

    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(wrap_ctx);

  rns8_plan_backend_info vector_backend{};
  vector_backend.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  rns8_plan_packing_info vector_packing{};
  vector_packing.semantics = RNS8_BOUNDED_I64;
  vector_packing.output_domain = RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64;
  vector_packing.next_op_flags =
      RNS8_NEXT_OP_FINAL_EXPORT | RNS8_NEXT_OP_NATIVE_GEMM | RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE;
  std::snprintf(vector_packing.input_domain_name, sizeof(vector_packing.input_domain_name), "%s", "native_i64_u64_current");
  std::snprintf(
      vector_packing.output_domain_name,
      sizeof(vector_packing.output_domain_name),
      "%s",
      "native_i64_u64_current");
  rns8_plan_schedule_info vector_schedule{};
  vector_schedule.max_selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  const auto vector_lowering =
      rns8::detail::describe_plan_lowering(vector_backend, vector_packing, vector_schedule);
  CHECK(vector_lowering.backend_family == "native_vector_alu");
  CHECK(vector_lowering.desired_output == "final_export_or_native_chain");
  CHECK(vector_lowering.conversion_strategy == "native_to_rns_available_for_mixed_storage_auto");
  CHECK(vector_lowering.lowering_path.find("NativeToRns") != std::string::npos);
  CHECK(vector_lowering.native_continuation_available);
  CHECK(vector_lowering.native_to_rns_available);
  CHECK_FALSE(vector_lowering.rns_continuation_available);
}

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
TEST_CASE("hipBLASLt plan packing info reports transient pack and scratch bytes") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIPBLASLT;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_HIPBLASLT;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_FINAL_EXPORT) != 0);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == static_cast<uint64_t>(m * n * sizeof(int32_t)));
  CHECK(packing.library_workspace_bytes > 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "hipblaslt_a_transposed_centered_i8_mk16_v1");
  CHECK(std::string(packing.prepack_cache_scope) == "transient_per_dispatch_workspace");
  CHECK(packing.production_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
TEST_CASE("CK plan packing info reports canonical transient pack workspace") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CK;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CK;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == static_cast<uint64_t>(m * n));
  CHECK(packing.library_workspace_bytes == 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1");
  CHECK(std::string(packing.b_layout_version) == "ck_b_canonical_colmajor_i8_n64_kblock32768_v1");
  CHECK(packing.reusable_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
TEST_CASE("rocWMMA plan packing info reports transient workspace and reusable B cache") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WMMA;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_WMMA;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);
  CHECK(packing.uses_transient_pack_workspace == 1);
  CHECK(packing.uses_matrix_engine_pack_layout == 1);
  CHECK(packing.input_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_domain == RNS8_OUTPUT_DOMAIN_RNS_RESIDUE);
  CHECK(packing.output_host_current == 0);
  CHECK(packing.output_device_current == 1);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_RNS_GEMM) != 0);
  CHECK((packing.next_op_flags & RNS8_NEXT_OP_REUSABLE_B_PREPACK) != 0);
  CHECK(packing.a_pack_workspace_bytes == static_cast<uint64_t>(m * k));
  CHECK(packing.b_pack_workspace_bytes == static_cast<uint64_t>(n * k));
  CHECK(packing.accumulator_workspace_bytes == 0);
  CHECK(packing.library_workspace_bytes == 0);
  CHECK(packing.total_transient_workspace_bytes == backend.workspace_required_bytes);
  CHECK(std::string(packing.a_layout_version) == "rocwmma_a_rowmajor_i8_m16_kblock65536_v1");
  CHECK(std::string(packing.b_layout_version) == "rns_i8_tile_swizzled_b_v1");
  CHECK(packing.reusable_prepack_cache_available == 1);
  CHECK(std::string(packing.prepack_cache_scope) == "reusable_b_prepack_cache");
  CHECK(packing.production_prepack_cache_available == 0);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
#endif

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
TEST_CASE("AUTO plan consumes reviewed CK cache entry with HIP resident matrices") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options ck_options{};
  ck_options.struct_size = sizeof(ck_options);
  ck_options.abi_version = RNS8_ABI_VERSION;
  ck_options.requested_backend = RNS8_BACKEND_CK;
  rns8_context* ck_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &ck_options, &ck_ctx) == RNS8_SUCCESS);

  rns8_device_info ck_device{};
  ck_device.struct_size = sizeof(ck_device);
  ck_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ck_ctx, &ck_device) == RNS8_SUCCESS);

  rns8_gemm_desc ck_desc{};
  ck_desc.struct_size = sizeof(ck_desc);
  ck_desc.abi_version = RNS8_ABI_VERSION;
  ck_desc.semantics = RNS8_BOUNDED_I64;
  ck_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  ck_desc.requested_backend = RNS8_BACKEND_CK;
  ck_desc.m = m;
  ck_desc.n = n;
  ck_desc.k = k;
  ck_desc.bound = static_cast<uint64_t>(k);
  ck_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* ck_plan = nullptr;
  REQUIRE(rns8_create_plan(ck_ctx, &ck_desc, &ck_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info ck_info{};
  ck_info.struct_size = sizeof(ck_info);
  ck_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(ck_plan, &ck_info) == RNS8_SUCCESS);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-ck-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << ck_info.autotune_key << "\","
          << "\"selected_backend\":\"ck\","
          << "\"selected_kernel\":\"" << ck_info.selected_kernel << "\","
          << "\"target_id\":\"" << ck_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << ck_info.accelerator_version << "\","
          << "\"semantic_contract\":\"bounded_i64\","
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=1;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << ck_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << ck_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << ck_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"reviewed_release_same_contract_fastest_windows_gfx1100\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-02T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(ck_plan);
  rns8_destroy_context(ck_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = ck_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_CK);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == ck_info.autotune_key);

  auto a_desc = bounded_matrix_desc(m, k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(k, n, RNS8_BOUNDED_I64);
  auto c_desc = bounded_matrix_desc(m, n, RNS8_BOUNDED_I64);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<int64_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<int64_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<int64_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_i64(auto_ctx, A, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(auto_ctx, B, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(auto_ctx, auto_plan, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(auto_ctx, auto_plan, C, c.data(), n) == RNS8_SUCCESS);
  for (int64_t value : c) {
    CHECK(value == k);
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}
#endif

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
TEST_CASE("AUTO plan consumes reviewed rocWMMA bounded cache entry with HIP resident matrices") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;

  rns8_context_options wmma_options{};
  wmma_options.struct_size = sizeof(wmma_options);
  wmma_options.abi_version = RNS8_ABI_VERSION;
  wmma_options.requested_backend = RNS8_BACKEND_WMMA;
  rns8_context* wmma_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &wmma_options, &wmma_ctx) == RNS8_SUCCESS);

  rns8_device_info wmma_device{};
  wmma_device.struct_size = sizeof(wmma_device);
  wmma_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(wmma_ctx, &wmma_device) == RNS8_SUCCESS);

  rns8_gemm_desc wmma_desc{};
  wmma_desc.struct_size = sizeof(wmma_desc);
  wmma_desc.abi_version = RNS8_ABI_VERSION;
  wmma_desc.semantics = RNS8_BOUNDED_I64;
  wmma_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  wmma_desc.requested_backend = RNS8_BACKEND_WMMA;
  wmma_desc.m = m;
  wmma_desc.n = n;
  wmma_desc.k = k;
  wmma_desc.bound = static_cast<uint64_t>(k);
  wmma_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* wmma_plan = nullptr;
  REQUIRE(rns8_create_plan(wmma_ctx, &wmma_desc, &wmma_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info wmma_info{};
  wmma_info.struct_size = sizeof(wmma_info);
  wmma_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(wmma_plan, &wmma_info) == RNS8_SUCCESS);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-wmma-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << wmma_info.autotune_key << "\","
          << "\"selected_backend\":\"wmma\","
          << "\"selected_kernel\":\"" << wmma_info.selected_kernel << "\","
          << "\"target_id\":\"" << wmma_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << wmma_info.accelerator_version << "\","
          << "\"semantic_contract\":\"bounded_i64\","
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=1;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << wmma_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << wmma_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << wmma_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"reviewed_release_same_contract_fastest_windows_gfx1100\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-03T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(wmma_plan);
  rns8_destroy_context(wmma_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = wmma_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_WMMA);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == wmma_info.autotune_key);

  auto a_desc = bounded_matrix_desc(m, k, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(k, n, RNS8_BOUNDED_I64);
  auto c_desc = bounded_matrix_desc(m, n, RNS8_BOUNDED_I64);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<int64_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<int64_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<int64_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_i64(auto_ctx, A, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(auto_ctx, B, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(auto_ctx, auto_plan, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(auto_ctx, auto_plan, C, c.data(), n) == RNS8_SUCCESS);
  for (int64_t value : c) {
    CHECK(value == k);
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}

TEST_CASE("AUTO plan consumes reviewed rocWMMA finite-u8 cache entry when modulus is plan keyed") {
  constexpr int64_t m = 64;
  constexpr int64_t n = 64;
  constexpr int64_t k = 64;

  rns8_context_options wmma_options{};
  wmma_options.struct_size = sizeof(wmma_options);
  wmma_options.abi_version = RNS8_ABI_VERSION;
  wmma_options.requested_backend = RNS8_BACKEND_WMMA;
  rns8_context* wmma_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &wmma_options, &wmma_ctx) == RNS8_SUCCESS);

  rns8_device_info wmma_device{};
  wmma_device.struct_size = sizeof(wmma_device);
  wmma_device.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(wmma_ctx, &wmma_device) == RNS8_SUCCESS);

  rns8_gemm_desc wmma_desc{};
  wmma_desc.struct_size = sizeof(wmma_desc);
  wmma_desc.abi_version = RNS8_ABI_VERSION;
  wmma_desc.semantics = RNS8_FINITE_RING_U8;
  wmma_desc.bound_kind = RNS8_BOUND_NONE;
  wmma_desc.requested_backend = RNS8_BACKEND_WMMA;
  wmma_desc.m = m;
  wmma_desc.n = n;
  wmma_desc.k = k;
  wmma_desc.finite_modulus = 251;

  rns8_plan* wmma_plan = nullptr;
  REQUIRE(rns8_create_plan(wmma_ctx, &wmma_desc, &wmma_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info wmma_info{};
  wmma_info.struct_size = sizeof(wmma_info);
  wmma_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(wmma_plan, &wmma_info) == RNS8_SUCCESS);
  CHECK(std::string(wmma_info.autotune_key).find(";finite_modulus=251;") != std::string::npos);
  CHECK(std::string(wmma_info.autotune_key).find(";epilogue=") != std::string::npos);

  const std::filesystem::path cache_path = unique_cache_fixture_path("rns8-auto-reviewed-wmma-finite-cache");
  {
    std::ofstream cache(cache_path, std::ios::trunc);
    cache << "{"
          << "\"schema_version\":1,"
          << "\"entries\":[{"
          << "\"key\":\"" << wmma_info.autotune_key << "\","
          << "\"selected_backend\":\"wmma\","
          << "\"selected_kernel\":\"" << wmma_info.selected_kernel << "\","
          << "\"target_id\":\"" << wmma_device.gcn_arch << "\","
          << "\"hip_sdk_or_library_version\":\"" << wmma_info.accelerator_version << "\","
          << "\"semantic_contract\":\"finite_ring_u8\","
          << "\"finite_modulus\":251,"
          << "\"shape\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << "},"
          << "\"layout\":\"row_major\","
          << "\"prefix_schedule_hash\":\"groups=0;adaptive_prefix=0;adaptive_skip=0\","
          << "\"k_block_size\":" << k << ","
          << "\"tile_m\":128,"
          << "\"tile_n\":128,"
          << "\"epilogue\":\"" << wmma_info.epilogue_mode << "\","
          << "\"kernel_family\":\"" << wmma_info.selected_kernel << "\","
          << "\"workspace_bytes\":" << wmma_info.workspace_required_bytes << ","
          << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
          << "\"performance_validated\":true,"
          << "\"validation_status\":\"reviewed_release_same_contract_fastest_windows_gfx1100\","
          << "\"schema_version\":1,"
          << "\"updated_utc\":\"2026-06-03T00:00:00Z\""
          << "}]}"
          << "\n";
  }
  rns8_destroy_plan(wmma_plan);
  rns8_destroy_context(wmma_ctx);

  ScopedAutotuneCachePath scoped_cache(cache_path);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &auto_options, &auto_ctx) == RNS8_SUCCESS);

  rns8_gemm_desc auto_desc = wmma_desc;
  auto_desc.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* auto_plan = nullptr;
  REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &auto_plan) == RNS8_SUCCESS);
  rns8_plan_backend_info auto_info{};
  auto_info.struct_size = sizeof(auto_info);
  auto_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(auto_plan, &auto_info) == RNS8_SUCCESS);
  REQUIRE(auto_info.backend == RNS8_BACKEND_WMMA);
  CHECK(auto_info.performance_validated == 1);
  CHECK(std::string(auto_info.autotune_key) == wmma_info.autotune_key);

  auto a_desc = finite_matrix_desc(m, k, RNS8_FINITE_RING_U8);
  auto b_desc = finite_matrix_desc(k, n, RNS8_FINITE_RING_U8);
  auto c_desc = finite_matrix_desc(m, n, RNS8_FINITE_RING_U8);
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &A) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &B) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &C) == RNS8_SUCCESS);

  std::vector<uint8_t> a(static_cast<std::size_t>(m * k), 1);
  std::vector<uint8_t> b(static_cast<std::size_t>(k * n), 1);
  std::vector<uint8_t> c(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_finite_u8(auto_ctx, A, 251, a.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(auto_ctx, B, 251, b.data(), n, 1) == RNS8_SUCCESS);

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(auto_ctx, auto_plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(auto_ctx, auto_plan, 251, A, B, C, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(auto_ctx, auto_plan, 251, C, c.data(), n) == RNS8_SUCCESS);
  for (uint8_t value : c) {
    CHECK(value == static_cast<uint8_t>(k % 251));
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(auto_plan);
  rns8_destroy_context(auto_ctx);
  std::filesystem::remove(cache_path);
}
#endif

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
