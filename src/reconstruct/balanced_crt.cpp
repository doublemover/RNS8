#include "backend_common/balanced_crt.hpp"
#include "backend_common/garner_constants.hpp"
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <stdexcept>

namespace rns8::detail::balanced_crt {
namespace {
using boost::multiprecision::cpp_int;
Wide wide(cpp_int x) {
  if (x < 0 || x >= (cpp_int(1) << 160))
    throw std::logic_error("CRT plan width");
  Wide out;
  for (auto &limb : out.limb) {
    limb = static_cast<uint64_t>(x & UINT64_MAX);
    x >>= 64;
  }
  return out;
}
Plan make(uint32_t prefix) {
  Plan out;
  out.prefix = prefix;
  std::array<cpp_int, 20> products;
  std::array<uint32_t, 20> slots{}, ends{};
  for (uint32_t i = 0; i < prefix; ++i) {
    products[i] = garner::kConstants.moduli[i];
    slots[i] = i;
    ends[i] = i + 1;
  }
  for (uint32_t count = prefix; count > 1;) {
    uint32_t next = 0;
    for (uint32_t i = 0; i < count; i += 2) {
      const auto l = slots[i];
      if (i + 1 < count) {
        const auto r = slots[i + 1];
        // Build A^-1 mod B with the same small-modulus inverse owner as
        // Garner, followed by exact local Garner reconstruction over B.
        cpp_int inv = 0, product = 1;
        for (uint32_t p = r; p < ends[r]; ++p) {
          const auto m = garner::kConstants.moduli[p];
          const auto target =
              garner::inverse(static_cast<uint32_t>(products[l] % m), m);
          const auto current = static_cast<uint32_t>(inv % m);
          const auto local_inverse =
              garner::inverse(static_cast<uint32_t>(product % m), m);
          const auto coefficient =
              ((target + m - current) % m) * local_inverse % m;
          inv += product * coefficient;
          product *= m;
        }
        if (product != products[r] || (products[l] * inv) % product != 1)
          throw std::logic_error("CRT plan inverse");
        out.merges[out.count++] = {wide(products[l]), wide(products[r]),
                                   wide(inv), l, r};
        products[l] *= products[r];
        ends[l] = ends[r];
        (void)wide(products[l]);
      }
      slots[next++] = l;
    }
    count = next;
  }
  return out;
}
} // namespace
const Plan &plan(uint32_t prefix) {
  if (!prefix || prefix > 20)
    throw std::invalid_argument("CRT prefix");
  static const auto plans = [] {
    std::array<Plan, 21> result{};
    for (uint32_t p = 1; p <= 20; ++p)
      result[p] = make(p);
    return result;
  }();
  return plans[prefix];
}
} // namespace rns8::detail::balanced_crt
