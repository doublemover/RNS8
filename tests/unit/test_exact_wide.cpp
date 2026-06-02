#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"
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

void check_signed_residues(
    const rns8_matrix& matrix,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_i64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

void check_unsigned_residues(
    const rns8_matrix& matrix,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_u64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

}  // namespace

TEST_CASE("exact-wide signed CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 2;
  const int64_t A[] = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 17,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 31};
  const int64_t B[] = {
      std::numeric_limits<int64_t>::max() - 5,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 11,
      std::numeric_limits<int64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_signed_residues(*c_matrix, A, k, B, n, m, n, k);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 3};
  const uint64_t B[] = {
      std::numeric_limits<uint64_t>::max() - 5,
      std::numeric_limits<uint64_t>::max() - 7,
      std::numeric_limits<uint64_t>::max() - 11,
      std::numeric_limits<uint64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_unsigned_residues(*c_matrix, A, k, B, n, m, n, k);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
