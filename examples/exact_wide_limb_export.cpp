#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "example_common.hpp"

int main() {
  rns8_context* ctx = nullptr;
  auto options = cpu_context_options();
  if (const auto status = rns8_create_context(-1, &options, &ctx); status != RNS8_SUCCESS) {
    return fail_status("rns8_create_context", status);
  }

  auto desc = base_gemm_desc(RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, 1, 1, 2);
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  auto status = rns8_create_plan(ctx, &desc, &plan);
  if (status == RNS8_SUCCESS) {
    status = rns8_create_workspace(ctx, plan, &workspace);
  }
  const auto a_desc = row_major_matrix_desc(1, 2, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
  const auto b_desc = row_major_matrix_desc(2, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
  const auto c_desc = row_major_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
  if (status == RNS8_SUCCESS) {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  }

  const int64_t a[] = {1LL << 40, -3};
  const int64_t b[] = {1LL << 30, 5};
  uint64_t limbs[2] = {};

  if (status == RNS8_SUCCESS) {
    status = rns8_pack_i64(ctx, a_matrix, a, 2, 1);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8_pack_i64(ctx, b_matrix, b, 1, 1);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, 1, 2);
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);

  if (status != RNS8_SUCCESS) {
    return fail_status("exact-wide limb export workflow", status);
  }

  if (limbs[0] != 0xfffffffffffffff1ull || limbs[1] != 0x000000000000003full) {
    return fail_check("exact-wide limb export mismatch");
  }
  std::cout << "exact-wide signed limbs: 0x" << std::hex << limbs[1] << '_' << limbs[0] << std::dec << '\n';
  return EXIT_SUCCESS;
}
