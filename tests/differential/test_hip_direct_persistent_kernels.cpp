#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "rns8/moduli.h"
#include "rns8/rns8.h"

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#include <hip/hip_runtime_api.h>

extern "C" int rns8_hip_direct_persistent_small_gemm_rns_device(
    const int8_t* a_residues, const int8_t* b_residues, int8_t* c_residues,
    int m, int n, int k, int prefix);

#define HIP_CHECK(call) do { hipError_t _e = (call); REQUIRE(_e == hipSuccess); } while(0)

static bool hip_available() {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  return err == hipSuccess && count > 0;
}
#endif

TEST_CASE("persistent small GEMM matches CPU reference for 16x16") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!hip_available()) { SKIP("no HIP device"); }

  constexpr int m = 16, n = 16, k = 16, prefix = 9;

  std::vector<int8_t> a_host(static_cast<size_t>(prefix * m * k));
  std::vector<int8_t> b_host(static_cast<size_t>(prefix * k * n));
  std::vector<int8_t> c_hip(static_cast<size_t>(prefix * m * n));
  std::vector<int8_t> c_cpu(static_cast<size_t>(prefix * m * n));

  for (int p = 0; p < prefix; ++p) {
    for (int row = 0; row < m; ++row)
      for (int ki = 0; ki < k; ++ki)
        a_host[static_cast<size_t>(p * m * k + row * k + ki)] =
            static_cast<int8_t>(((row + ki) % 128) - 64);
    for (int ki = 0; ki < k; ++ki)
      for (int col = 0; col < n; ++col)
        b_host[static_cast<size_t>(p * k * n + ki * n + col)] =
            static_cast<int8_t>(((ki + col) % 128) - 64);
  }

  // CPU reference
  for (int p = 0; p < prefix; ++p) {
    constexpr int mods[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167,163,157,151,149,139,137,131,127};
    const int mod = mods[p];
    const int8_t* ap = a_host.data() + static_cast<size_t>(p) * m * k;
    const int8_t* bp = b_host.data() + static_cast<size_t>(p) * k * n;
    int8_t* cp = c_cpu.data() + static_cast<size_t>(p) * m * n;
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        int32_t acc = 0;
        for (int ki = 0; ki < k; ++ki)
          acc += static_cast<int32_t>(ap[row * k + ki]) *
                 static_cast<int32_t>(bp[ki * n + col]);
        int32_t reduced = acc % mod;
        if (reduced < 0) reduced += mod;
        cp[row * n + col] = static_cast<int8_t>(
            reduced > mod / 2 ? reduced - mod : reduced);
      }
    }
  }

  // GPU path
  void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
  size_t ab = static_cast<size_t>(prefix) * m * k;
  size_t bb = static_cast<size_t>(prefix) * k * n;
  size_t cb = static_cast<size_t>(prefix) * m * n;
  HIP_CHECK(hipMalloc(&d_a, ab));
  HIP_CHECK(hipMalloc(&d_b, bb));
  HIP_CHECK(hipMalloc(&d_c, cb));
  HIP_CHECK(hipMemcpy(d_a, a_host.data(), ab, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, b_host.data(), bb, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_c, 0, cb));

  int code = rns8_hip_direct_persistent_small_gemm_rns_device(
      static_cast<const int8_t*>(d_a),
      static_cast<const int8_t*>(d_b),
      static_cast<int8_t*>(d_c),
      m, n, k, prefix);
  REQUIRE(code == static_cast<int>(hipSuccess));

  HIP_CHECK(hipMemcpy(c_hip.data(), d_c, cb, hipMemcpyDeviceToHost));

  for (int p = 0; p < prefix; ++p) {
    for (int cell = 0; cell < m * n; ++cell) {
      INFO("plane=" << p << " cell=" << cell);
      CHECK(c_hip[static_cast<size_t>(p * m * n + cell)] ==
            c_cpu[static_cast<size_t>(p * m * n + cell)]);
    }
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
#endif
}
