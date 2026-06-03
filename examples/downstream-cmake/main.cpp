#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <rns8/rns8.h>

int main() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;

  rns8_context* ctx = nullptr;
  auto status = rns8_create_context(-1, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context failed: " << rns8_status_string(status) << '\n';
    return EXIT_FAILURE;
  }

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 2;
  desc.bound = 200;

  const uint64_t a[] = {6, 7};
  const uint64_t b[] = {8, 9};
  uint64_t c[] = {0};

  status = rns8_gemm_u64_oneshot(ctx, &desc, a, 2, b, 1, c, 1);
  rns8_destroy_context(ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_gemm_u64_oneshot failed: " << rns8_status_string(status) << '\n';
    return EXIT_FAILURE;
  }
  if (c[0] != 111) {
    std::cerr << "unexpected result: " << c[0] << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "downstream RNS8 smoke: " << c[0] << '\n';
  return EXIT_SUCCESS;
}
