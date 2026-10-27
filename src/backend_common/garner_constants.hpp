#ifndef RNS8_BACKEND_COMMON_GARNER_CONSTANTS_HPP
#define RNS8_BACKEND_COMMON_GARNER_CONSTANTS_HPP
#include <cstdint>

namespace rns8::detail::garner {
// Immutable compile-time data only. No allocation, device launch or host cache.
// Products through prefix 20 are below 256^20 = 2^160, hence three limbs suffice.
constexpr uint32_t kCount = 28;
constexpr uint32_t kProductCount = 21;
struct Constants {
  uint32_t moduli[kCount]{};
  uint32_t inverses[kCount]{};
  uint64_t products[kProductCount][3]{};
  uint32_t pair_moduli[10]{};
  uint32_t pair_inner_inverses[10]{};
  uint32_t pair_prefix_inverses[10]{};
};

constexpr uint32_t inverse(uint32_t a, uint32_t modulus) {
  int64_t t=0, next_t=1, r=modulus, next_r=a % modulus;
  while (next_r != 0) {
    const auto q=r/next_r;
    const auto new_t=t-q*next_t; t=next_t; next_t=new_t;
    const auto new_r=r-q*next_r; r=next_r; next_r=new_r;
  }
  if (r != 1) return 0;
  if (t < 0) t += modulus;
  return static_cast<uint32_t>(t);
}

constexpr Constants make_constants() {
  Constants out{};
  const uint32_t moduli[kCount] = {
      256,255,253,251,247,239,233,229,227,223,217,211,199,197,
      193,191,181,179,173,167,163,157,151,149,139,137,131,127};
  out.products[0][0]=1;
  for (uint32_t i=0;i<kCount;++i) {
    out.moduli[i]=moduli[i];
    uint32_t product_mod=1;
    for (uint32_t j=0;j<i;++j) product_mod=(product_mod*moduli[j]) % moduli[i];
    out.inverses[i]=inverse(product_mod,moduli[i]);
    if (i+1 < kProductCount) {
      // Base-2^32 schoolbook multiply avoids nonportable unsigned __int128.
      uint64_t carry=0;
      for (uint32_t limb=0;limb<3;++limb) {
        const uint64_t value=out.products[i][limb];
        const uint64_t lo=(value & 0xffffffffULL)*moduli[i]+carry;
        const uint64_t hi=(value >> 32)*moduli[i]+(lo >> 32);
        out.products[i+1][limb]=(lo & 0xffffffffULL) | (hi << 32);
        carry=hi >> 32;
      }
    }
  }
  for (uint32_t pair=0;pair<10;++pair) {
    const uint32_t lhs=moduli[2*pair], rhs=moduli[2*pair+1];
    out.pair_moduli[pair]=lhs*rhs;
    out.pair_inner_inverses[pair]=inverse(lhs % rhs,rhs);
    uint32_t prefix_mod=1;
    for (uint32_t j=0;j<2*pair;++j)
      prefix_mod=static_cast<uint32_t>((static_cast<uint64_t>(prefix_mod)*moduli[j]) % (lhs*rhs));
    out.pair_prefix_inverses[pair]=inverse(prefix_mod,lhs*rhs);
  }
  return out;
}
inline constexpr Constants kConstants=make_constants();
} // namespace rns8::detail::garner
#endif
