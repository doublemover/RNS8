#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

rns8_context* create_cpu_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  return rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_context* create_hip_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx = nullptr;
  return rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_gemm_desc signed_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc signed_desc_for_backend(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  auto desc = signed_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

rns8_gemm_desc unsigned_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc unsigned_desc_for_backend(
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t bound,
    rns8_backend_kind backend) {
  auto desc = unsigned_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

bool verify_cpu() {
  if (rns8_validate_default_moduli() != RNS8_SUCCESS) {
    std::cerr << "default modulus ladder is not pairwise coprime\n";
    return false;
  }

  rns8_context* ctx = create_cpu_context();
  if (!ctx) {
    std::cerr << "failed to create CPU reference context\n";
    return false;
  }

  {
    const int64_t A[] = {7, -3, 5, -11, 13, 17};
    const int64_t B[] = {2, -5, 19, 23, -29, 31};
    int64_t C[4] = {};
    auto desc = signed_desc(2, 2, 3, 100000);
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 3, B, 2, C, 2);
    if (status != RNS8_SUCCESS || C[0] != -188 || C[1] != 51 || C[2] != -268 || C[3] != 881) {
      std::cerr << "bounded i64 verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = signed_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<int64_t>::max()) {
      std::cerr << "bounded i64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t C[] = {0};
    auto desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    const rns8_status status = rns8_gemm_u64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<uint64_t>::max()) {
      std::cerr << "bounded u64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  rns8_destroy_context(ctx);
  return true;
}

bool verify_hip_smoke() {
  if (!rns8::detail::hip_direct_compiled()) {
    std::cerr << "direct HIP backend was not compiled\n";
    return false;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const rns8_status probe = rns8::detail::hip_direct_probe(0, info);
  if (probe != RNS8_SUCCESS) {
    std::cerr << "direct HIP probe failed: " << rns8_status_string(probe) << "\n";
    return false;
  }

  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 4;
  const uint16_t modulus = 251;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, -13, 14, 15, -16};
  std::vector<int8_t> cpu(4, 0);
  std::vector<int8_t> gpu(4, 0);
  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  const rns8_status status =
      rns8::detail::hip_direct_ring_gemm_i8(0, A.data(), B.data(), gpu.data(), m, n, k, k, n, n, modulus);
  if (status != RNS8_SUCCESS || cpu != gpu) {
    std::cerr << "direct HIP ring GEMM smoke failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  const int64_t split_k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const uint64_t split_bound = static_cast<uint64_t>(split_k) * 127u * 127u;
  std::vector<int64_t> split_a(static_cast<std::size_t>(split_k), 127);
  std::vector<int64_t> split_b(static_cast<std::size_t>(split_k), 127);

  rns8_context* cpu_ctx = create_cpu_context();
  rns8_context* hip_ctx = create_hip_context();
  if (!cpu_ctx || !hip_ctx) {
    std::cerr << "failed to create CPU or direct HIP context for bounded smoke\n";
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  int64_t cpu_split_c[1] = {};
  int64_t hip_split_c[1] = {};
  auto cpu_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_split_status =
      rns8_gemm_i64_oneshot(cpu_ctx, &cpu_split_desc, split_a.data(), split_k, split_b.data(), 1, cpu_split_c, 1);
  const rns8_status hip_split_status =
      rns8_gemm_i64_oneshot(hip_ctx, &hip_split_desc, split_a.data(), split_k, split_b.data(), 1, hip_split_c, 1);
  if (cpu_split_status != RNS8_SUCCESS || hip_split_status != RNS8_SUCCESS || cpu_split_c[0] != hip_split_c[0] ||
      hip_split_c[0] != static_cast<int64_t>(split_bound)) {
    std::cerr << "direct HIP bounded i64 split smoke failed: CPU=" << rns8_status_string(cpu_split_status)
              << " HIP=" << rns8_status_string(hip_split_status) << "\n";
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const uint64_t u_a[] = {17, 3, 255, 9, 41, 5};
  const uint64_t u_b[] = {11, 7, 13, 19, 23, 29};
  uint64_t cpu_u_c[4] = {};
  uint64_t hip_u_c[4] = {};
  auto cpu_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_u_status = rns8_gemm_u64_oneshot(cpu_ctx, &cpu_u_desc, u_a, 3, u_b, 2, cpu_u_c, 2);
  const rns8_status hip_u_status = rns8_gemm_u64_oneshot(hip_ctx, &hip_u_desc, u_a, 3, u_b, 2, hip_u_c, 2);
  if (cpu_u_status != RNS8_SUCCESS || hip_u_status != RNS8_SUCCESS ||
      std::vector<uint64_t>(std::begin(cpu_u_c), std::end(cpu_u_c)) !=
          std::vector<uint64_t>(std::begin(hip_u_c), std::end(hip_u_c))) {
    std::cerr << "direct HIP bounded u64 smoke failed: CPU=" << rns8_status_string(cpu_u_status)
              << " HIP=" << rns8_status_string(hip_u_status) << "\n";
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  rns8_destroy_context(hip_ctx);
  rns8_destroy_context(cpu_ctx);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool hip_smoke = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hip-smoke") {
      hip_smoke = true;
    } else if (arg == "--help") {
      std::cout << "usage: rns8-verify [--hip-smoke]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (!verify_cpu()) {
    return 1;
  }
  std::cout << "CPU reference verification: PASS\n";

  if (hip_smoke) {
    if (!verify_hip_smoke()) {
      return 1;
    }
    std::cout << "Direct HIP ring and bounded GEMM smoke: PASS\n";
  }

  return 0;
}
