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

rns8_context* create_wrap64() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc gemm_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 1;
  desc.bound = 1;
  return desc;
}

rns8_gemm_desc bounded_looking_desc(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    rns8_backend_kind requested_backend = RNS8_BACKEND_CPU_REFERENCE) {
  auto desc = gemm_desc(semantics, bound_kind);
  desc.requested_backend = requested_backend;
  desc.bound = UINT64_MAX;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = 1;
  desc.cols = 1;
  desc.logical_ld = 1;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  return desc;
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t max_prefix = 0) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc bounded_looking_matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

}  // namespace

TEST_CASE("bounded semantics reject bound none explicitly") {
  rns8_context* ctx = create_cpu();
  {
    auto desc = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("unsupported semantic contracts do not fall through to bounded CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics :
       {RNS8_WRAP_U64_MOD_2_64, RNS8_FINITE_RING_U8, RNS8_FINITE_FIELD_U8}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide RNS output accepts only explicit unbounded semantics") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_destroy_plan(plan);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_SUCCESS);
    rns8_destroy_matrix(storage);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("unbounded semantic descriptors reject stale bound metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 1;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("global descriptors reject stray per-tile bound storage") {
  rns8_context* ctx = create_cpu();
  const uint64_t stale_bound = 100;
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.tile_bounds = &stale_bound;
  desc.tile_bounds_count = 1;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.bound = 0;
  exact.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  exact.tile_bounds_count = 1;
  CHECK(rns8_create_plan(ctx, &exact, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide semantics reject bounded-looking CRT metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    for (const rns8_bound_kind bound_kind :
         {RNS8_BOUND_GLOBAL_MAX_ABS,
          RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
          RNS8_BOUND_PER_TILE_MAX_ABS,
          RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
          RNS8_BOUND_INPUT_RANGE_AND_K}) {
      auto desc = bounded_looking_desc(semantics, bound_kind);
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);

      auto matrix = bounded_looking_matrix_desc(semantics, bound_kind);
      rns8_matrix* storage = nullptr;
      CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
      CHECK(storage == nullptr);
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("strict wraparound is not accepted as bounded odd-modulus CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_bound_kind bound_kind :
       {RNS8_BOUND_NONE,
        RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
        RNS8_BOUND_PER_TILE_MAX_ABS,
        RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
        RNS8_BOUND_INPUT_RANGE_AND_K}) {
    auto desc = bounded_looking_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = bounded_looking_matrix_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("wrap64 byte-limb context reports explicit CPU reference backend") {
  rns8_context* ctx = create_wrap64();
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ctx, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
  CHECK(info.device_id == -1);
  CHECK(info.hip_available == 0);
  rns8_destroy_context(ctx);
}

TEST_CASE("auto backend selection never routes across explicit semantic backends") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &auto_ctx) == RNS8_SUCCESS);

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(auto_ctx, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);

  auto wrap = gemm_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  wrap.bound = 0;
  wrap.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(auto_ctx, &wrap, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);
  rns8_destroy_context(auto_ctx);

  rns8_context* wrap_ctx = create_wrap64();
  CHECK(rns8_create_plan(wrap_ctx, &wrap, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);
  rns8_destroy_context(wrap_ctx);
}

TEST_CASE("future backend context kinds report unsupported status") {
  for (const rns8_backend_kind backend :
       {RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK, RNS8_BACKEND_WMMA}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);
  }
}

TEST_CASE("tile dimensions are powers of two from 64 to 512") {
  rns8_context* ctx = create_cpu();
  auto valid = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  valid.tile_m = 256;
  valid.tile_n = 512;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &valid, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);

  for (const uint32_t bad_tile : {32u, 96u, 513u}) {
    auto bad_plan = valid;
    bad_plan.tile_m = bad_tile;
    plan = nullptr;
    CHECK(rns8_create_plan(ctx, &bad_plan, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto bad_matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    bad_matrix.tile_n = bad_tile;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &bad_matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects incompatible matrix and workspace contracts") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 100;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  auto smaller_desc = desc;
  smaller_desc.k = 1;
  rns8_plan* smaller_plan = nullptr;
  rns8_workspace* wrong_workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &smaller_desc, &smaller_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, smaller_plan, &wrong_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, wrong_workspace) == RNS8_WORKSPACE_TOO_SMALL);

  rns8_matrix* wrong_b_shape = nullptr;
  auto wrong_b_shape_desc = matrix_desc(3, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_b_shape_desc, &wrong_b_shape) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, wrong_b_shape, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 8);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, wrong_c_prefix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_a_bound_kind = nullptr;
  auto wrong_a_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_a_bound_kind_desc, &wrong_a_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, wrong_a_bound_kind, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_a_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(wrong_b_shape);
  rns8_destroy_workspace(wrong_workspace);
  rns8_destroy_plan(smaller_plan);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects same-shape workspaces from different semantic contracts") {
  rns8_context* ctx = create_cpu();

  auto bounded = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  bounded.m = 2;
  bounded.n = 2;
  bounded.k = 1;
  bounded.bound = 100;
  bounded.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.m = bounded.m;
  exact.n = bounded.n;
  exact.k = bounded.k;
  exact.bound = 0;
  exact.max_prefix = bounded.max_prefix;

  const uint64_t tile_bound = 100;
  auto per_tile = bounded;
  per_tile.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  per_tile.bound = 0;
  per_tile.tile_m = 64;
  per_tile.tile_n = 64;
  per_tile.tile_bounds = &tile_bound;
  per_tile.tile_bounds_count = 1;

  rns8_plan* bounded_plan = nullptr;
  rns8_plan* exact_plan = nullptr;
  rns8_plan* per_tile_plan = nullptr;
  rns8_workspace* bounded_workspace = nullptr;
  rns8_workspace* exact_workspace = nullptr;
  rns8_workspace* per_tile_workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  REQUIRE(rns8_create_plan(ctx, &bounded, &bounded_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &exact, &exact_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &per_tile, &per_tile_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, exact_plan, &exact_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, per_tile_plan, &per_tile_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(2, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto b_desc = matrix_desc(1, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, bounded_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, exact_workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, per_tile_workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(per_tile_workspace);
  rns8_destroy_workspace(exact_workspace);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(per_tile_plan);
  rns8_destroy_plan(exact_plan);
  rns8_destroy_plan(bounded_plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded export rejects output matrices outside the plan contract") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 1;
  desc.bound = 100;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  uint64_t dst[4] = {};
  rns8_matrix* c_matrix = nullptr;
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, dst, 2) == RNS8_SUCCESS);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 8);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_prefix, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_bound_kind = nullptr;
  auto wrong_c_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_bound_kind_desc, &wrong_c_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_bound_kind, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_c_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("matrix creation rejects descriptor sizes that overflow owned storage") {
  {
    rns8_context* ctx = create_cpu();
    auto desc = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }

  {
    rns8_context* ctx = create_wrap64();
    auto desc = matrix_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = 0;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }
}
