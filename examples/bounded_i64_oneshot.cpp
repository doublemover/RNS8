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

  auto desc = base_gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, 2, 2, 3);
  desc.bound = 100;

  const int64_t a[] = {2, -3, 4, -5, 6, 7};
  const int64_t b[] = {1, 8, -2, 9, 3, -4};
  int64_t c[4] = {};

  const auto status = rns8_gemm_i64_oneshot(ctx, &desc, a, 3, b, 2, c, 2);
  rns8_destroy_context(ctx);
  if (status != RNS8_SUCCESS) {
    return fail_status("rns8_gemm_i64_oneshot", status);
  }

  const int64_t expected[] = {20, -27, 4, -14};
  for (int i = 0; i < 4; ++i) {
    if (c[i] != expected[i]) {
      return fail_check("bounded i64 result mismatch");
    }
  }
  std::cout << "bounded i64 oneshot: " << c[0] << ' ' << c[1] << ' ' << c[2] << ' ' << c[3] << '\n';
  return EXIT_SUCCESS;
}
