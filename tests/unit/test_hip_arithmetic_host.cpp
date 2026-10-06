#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <vector>

#include "backend_common/finite_u8_reducer.hpp"
#include "core/internal.hpp"

// Host execution of scalar arithmetic only. This does not validate HIP code
// generation, GPU memory behavior, wave operations, synchronization, or timing.
// The device helper currently requires unsigned __int128 (not MSVC).
namespace host_arithmetic {
#define __device__
#define __constant__
#define __global__
#define __builtin_nontemporal_load(pointer) (*(pointer))
constexpr int kRns8DefaultModulusCount = 28;
struct Index { int x = 0; int y = 0; int z = 0; };
Index blockIdx, blockDim, threadIdx;
#include "backend_hip_direct/hip_direct_device_common.cuh"
#include "backend_hip_direct/hip_direct_pack_kernels.cuh"
#undef __builtin_nontemporal_load
#undef __global__
#undef __constant__
#undef __device__
}

TEST_CASE("host-only HIP wide CRT arithmetic matches the multiprecision oracle", "[qualification][cpu]") {
  using boost::multiprecision::cpp_int;
  for (uint32_t prefix = 1; prefix <= RNS8_MAX_SUPPORTED_PREFIX; ++prefix) {
    const cpp_int product = rns8::detail::modulus_product(prefix);
    std::vector<cpp_int> values = {0, 1, 2, product / 2 - 1, product / 2, product - 1};
    if (prefix >= 9) {
      values.push_back(cpp_int(std::numeric_limits<uint64_t>::max()));
      values.push_back(product - (cpp_int(1) << 63));
    }
    for (const cpp_int& value : values) {
      INFO("prefix=" << prefix << " value=" << value);
      std::vector<int8_t> residues(prefix);
      for (uint32_t p = 0; p < prefix; ++p)
        residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
      host_arithmetic::rns8_u192_device result{}, modulus_product{};
      host_arithmetic::rns8_reconstruct_canonical_wide_device(
          residues.data(), 0, 1, static_cast<int>(prefix), &result, &modulus_product);
      auto integer = [](host_arithmetic::rns8_u192_device v) {
        return cpp_int(v.limb0) + (cpp_int(v.limb1) << 64) + (cpp_int(v.limb2) << 128);
      };
      CHECK(integer(result) == value);
      CHECK(integer(modulus_product) == product);
    }
  }
}

TEST_CASE("host-only HIP four-cell pack does not skip unaligned plane starts", "[qualification][cpu]") {
  const int64_t boundaries[] = {0, 1, -1, 127, 128, -128, -129,
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
  for (const int cells : {255, 256, 257, 258, 259, 4095, 4096}) {
    for (const int prefix : {1, 8, 9, 20}) {
      INFO("cells=" << cells << " prefix=" << prefix);
      std::vector<int64_t> input(cells);
      std::vector<int8_t> actual(cells * prefix, 99), expected(actual.size());
      for (int c = 0; c < cells; ++c) input[c] = boundaries[c % 9];
      for (int p = 0; p < prefix; ++p)
        for (int c = 0; c < cells; ++c)
          expected[p * cells + c] = rns8::detail::centered_residue(
              boost::multiprecision::cpp_int(input[c]), rns8::detail::kDefaultModuli[p]);
      host_arithmetic::blockDim.x = 256;
      for (int group = 0; group < (cells * prefix + 3) / 4; ++group) {
        host_arithmetic::blockIdx.x = group / 256;
        host_arithmetic::threadIdx.x = group % 256;
        host_arithmetic::rns8_pack_i64_4wide_coalesced_kernel(input.data(), actual.data(), 1, cells, prefix);
      }
      CHECK(actual == expected);
    }
  }
}
