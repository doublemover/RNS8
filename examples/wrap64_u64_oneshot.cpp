#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "example_common.hpp"

int main() {
  rns8_context* ctx = nullptr;
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  if (const auto status = rns8_create_context(-1, &options, &ctx); status != RNS8_SUCCESS) {
    return fail_status("rns8_create_context", status);
  }

  auto desc = base_gemm_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, 1, 2, 2);
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;

  const uint64_t a[] = {std::numeric_limits<uint64_t>::max(), 3};
  const uint64_t b[] = {5, 7, 11, 13};
  uint64_t c[2] = {};

  const auto status = rns8_gemm_wrap_u64_oneshot(ctx, &desc, a, 2, b, 2, c, 2);
  rns8_destroy_context(ctx);
  if (status != RNS8_SUCCESS) {
    return fail_status("rns8_gemm_wrap_u64_oneshot", status);
  }

  const uint64_t expected[] = {28, 32};
  if (c[0] != expected[0] || c[1] != expected[1]) {
    return fail_check("wrap64 result mismatch");
  }
  std::cout << "wrap64 u64 oneshot: " << c[0] << ' ' << c[1] << '\n';
  return EXIT_SUCCESS;
}
