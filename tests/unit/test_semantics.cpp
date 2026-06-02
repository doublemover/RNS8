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
       {RNS8_EXACT_WIDE_SIGNED,
        RNS8_EXACT_WIDE_UNSIGNED,
        RNS8_WRAP_U64_MOD_2_64,
        RNS8_FINITE_RING_U8,
        RNS8_FINITE_FIELD_U8}) {
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

TEST_CASE("future backend context kinds report unsupported status") {
  for (const rns8_backend_kind backend :
       {RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK, RNS8_BACKEND_WMMA, RNS8_BACKEND_WRAP64_BYTE_LIMB}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);
  }
}
