#include <catch2/catch_test_macros.hpp>

#include <cstdint>

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
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
      CHECK(plan == nullptr);

      auto matrix = bounded_looking_matrix_desc(semantics, bound_kind);
      rns8_matrix* storage = nullptr;
      CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
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
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);

    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);

    auto matrix = bounded_looking_matrix_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
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
