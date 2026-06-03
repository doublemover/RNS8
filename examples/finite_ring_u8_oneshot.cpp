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

  constexpr uint16_t modulus = 251;
  auto desc = base_gemm_desc(RNS8_FINITE_RING_U8, RNS8_BOUND_NONE, 2, 2, 3);
  desc.finite_modulus = modulus;

  const uint8_t a[] = {250, 2, 3, 4, 5, 6};
  const uint8_t b[] = {7, 8, 9, 10, 11, 12};
  uint8_t c[4] = {};

  const auto status = rns8_gemm_finite_ring_u8_oneshot(ctx, &desc, modulus, a, 3, b, 2, c, 2);
  rns8_destroy_context(ctx);
  if (status != RNS8_SUCCESS) {
    return fail_status("rns8_gemm_finite_ring_u8_oneshot", status);
  }

  const uint8_t expected[] = {44, 48, 139, 154};
  for (int i = 0; i < 4; ++i) {
    if (c[i] != expected[i]) {
      return fail_check("finite ring u8 result mismatch");
    }
  }
  std::cout << "finite ring u8 oneshot mod " << modulus << ": " << static_cast<int>(c[0]) << ' '
            << static_cast<int>(c[1]) << ' ' << static_cast<int>(c[2]) << ' ' << static_cast<int>(c[3]) << '\n';
  return EXIT_SUCCESS;
}
