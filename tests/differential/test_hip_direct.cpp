#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"

TEST_CASE("direct HIP ring GEMM matches CPU reference for one modulus") {
  if (!rns8::detail::hip_direct_compiled()) {
    SKIP("direct HIP backend was not compiled");
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (rns8::detail::hip_direct_probe(0, info) != RNS8_SUCCESS) {
    SKIP("no HIP device available for direct HIP smoke");
  }

  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const uint16_t modulus = 255;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20};
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * n), 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  CHECK(rns8::detail::hip_direct_ring_gemm_i8(0, A.data(), B.data(), gpu.data(), m, n, k, k, n, n, modulus) ==
        RNS8_SUCCESS);
  CHECK(gpu == cpu);
}

