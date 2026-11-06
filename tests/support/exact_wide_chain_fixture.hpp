#pragma once
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>
#include <limits>
#include "core/api_internal.hpp"
#include "core/exact_range.hpp"
#include "rns8/rns8.h"
#include "../../benchmarks/exact_wide_magnitude.hpp"

namespace rns8::test_support {
struct ExactChainFixture {
  rns8_context* ctx = nullptr;
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  std::vector<rns8_matrix*> matrices;
  explicit ExactChainFixture(rns8_semantics semantics, int64_t size = 1, uint32_t prefix = 20)
      : ExactChainFixture(semantics, size, size, size, prefix) {}
  ExactChainFixture(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, uint32_t prefix,
                    rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE) {
    try {
      matrices.reserve(4);
      rns8_context_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = RNS8_ABI_VERSION;
      options.requested_backend = backend;
      REQUIRE(rns8_create_context(backend == RNS8_BACKEND_CPU_REFERENCE ? -1 : 0, &options, &ctx) ==
              RNS8_SUCCESS);
      rns8_gemm_desc desc{};
      desc.struct_size = sizeof(desc);
      desc.abi_version = RNS8_ABI_VERSION;
      desc.semantics = semantics;
      desc.bound_kind = RNS8_BOUND_NONE;
      desc.m = m;
      desc.n = n;
      desc.k = k;
      desc.requested_backend = backend;
      desc.max_prefix = prefix;
      desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
      const int64_t shapes[4][2] = {{m, k}, {k, n}, {m, n}, {m, n}};
      for (unsigned i = 0; i < 4; ++i) {
        rns8_matrix_desc md{};
        md.struct_size = sizeof(md);
        md.abi_version = RNS8_ABI_VERSION;
        md.rows = shapes[i][0];
        md.cols = shapes[i][1];
        md.logical_ld = md.cols;
        md.semantics = semantics;
        md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
        md.bound_kind = RNS8_BOUND_NONE;
        md.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
        rns8_matrix* matrix = nullptr;
        REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
        matrices.push_back(matrix);
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ExactChainFixture(const ExactChainFixture&) = delete;
  ExactChainFixture& operator=(const ExactChainFixture&) = delete;
  ~ExactChainFixture() { release(); }
  void release() noexcept {
    for (auto* matrix : matrices)
      rns8_destroy_matrix(matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_context(ctx);
    matrices.clear();
    workspace = nullptr;
    plan = nullptr;
    ctx = nullptr;
  }
  rns8_status multiply(unsigned a, unsigned b, unsigned c) {
    return rns8_gemm_rns(ctx, plan, matrices[a], matrices[b], matrices[c], workspace);
  }
};
inline std::vector<uint64_t> check_materialized_exact_wide_chain(rns8_backend_kind backend,
                                                                 bool signed_values) {
  using boost::multiprecision::cpp_int;
  const auto semantics = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
  ExactChainFixture fixture(semantics, 1, 2, 2, 20, backend);
  const int64_t signed_a[2] = {INT64_MAX, 0}, signed_b[4] = {2, 2, -2, -2};
  const uint64_t unsigned_a[2] = {UINT64_MAX, 0}, unsigned_b[4] = {0, 2, 0, 0};
  REQUIRE((signed_values ? rns8_pack_i64(fixture.ctx, fixture.matrices[0], signed_a, 2, 1)
                         : rns8_pack_u64(fixture.ctx, fixture.matrices[0], unsigned_a, 2, 1)) ==
          RNS8_SUCCESS);
  REQUIRE((signed_values ? rns8_pack_i64(fixture.ctx, fixture.matrices[1], signed_b, 2, 1)
                         : rns8_pack_u64(fixture.ctx, fixture.matrices[1], unsigned_b, 2, 1)) ==
          RNS8_SUCCESS);
  REQUIRE(fixture.multiply(0, 1, 2) == RNS8_SUCCESS);
  const auto export_view = [&](uint64_t* destination, int64_t ld, uint32_t limbs) {
    return signed_values
               ? rns8_export_exact_wide_signed_limbs(fixture.ctx, fixture.plan, fixture.matrices[2],
                                                     destination, ld, limbs)
               : rns8_export_exact_wide_unsigned_limbs(fixture.ctx, fixture.plan,
                                                       fixture.matrices[2], destination, ld, limbs);
  };
  std::vector<uint64_t> final_output(3, UINT64_C(0x5a5a5a5a5a5a5a5a));
  REQUIRE(export_view(final_output.data(), 3, 1) == RNS8_RANGE_ERROR);
  CHECK(final_output == std::vector<uint64_t>(3, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  const uint32_t limbs =
      rns8::bench::exact_wide_materialization_limb_count(rns8::detail::modulus_product(20));
  REQUIRE(limbs > 1);
  std::vector<uint64_t> intermediate(2 * limbs);
  REQUIRE(export_view(intermediate.data(), 2, limbs) == RNS8_SUCCESS);
  const auto b = [&](unsigned index) -> cpp_int {
    return signed_values ? cpp_int(signed_b[index]) : cpp_int(unsigned_b[index]);
  };
  const cpp_int a = signed_values ? cpp_int(signed_a[0]) : cpp_int(unsigned_a[0]);
  std::vector<cpp_int> expected_intermediate{a * b(0), a * b(1)};
  for (unsigned col = 0; col < 2; ++col) {
    cpp_int actual = 0;
    for (unsigned word = 0; word < limbs; ++word)
      actual += cpp_int(intermediate[col * limbs + word]) << (64u * word);
    if (signed_values && intermediate[(col + 1) * limbs - 1] >> 63u)
      actual -= cpp_int(1) << (64u * limbs);
    CHECK(actual == expected_intermediate[col]);
  }
  REQUIRE(rns8::detail::api::import_exact_wide_limbs(*fixture.ctx, *fixture.matrices[0],
                                                     intermediate.data(), 2, limbs, signed_values,
                                                     2) == RNS8_SUCCESS);
  REQUIRE(fixture.multiply(0, 1, 2) == RNS8_SUCCESS);
  REQUIRE(export_view(final_output.data(), 3, 1) == RNS8_SUCCESS);
  for (unsigned col = 0; col < 2; ++col) {
    const cpp_int expected =
        expected_intermediate[0] * b(col) + expected_intermediate[1] * b(2 + col);
    CHECK(expected == 0);
    CHECK(final_output[col] == expected.convert_to<uint64_t>());
  }
  CHECK(final_output[2] == UINT64_C(0x5a5a5a5a5a5a5a5a));
  return final_output;
}

} // namespace rns8::test_support
