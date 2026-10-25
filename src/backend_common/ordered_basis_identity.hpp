#pragma once
#include "basis_packet.hpp"
#include "garner_constants.hpp"
#include <array>
namespace rns8::detail {
constexpr uint64_t ordered_basis_fingerprint(const uint16_t* moduli,uint32_t count) {
  uint64_t hash=UINT64_C(14695981039346656037);
  const auto add=[&](uint32_t value){for(unsigned i=0;i<4;++i)hash=(hash^uint8_t(value>>(8*i)))*UINT64_C(1099511628211);};
  add(1);add(count);for(uint32_t i=0;i<count;++i)add(moduli[i]);return hash;
}
constexpr std::array<uint16_t,20> default_ordered_basis() {
  std::array<uint16_t,20> values{};for(unsigned i=0;i<20;++i)values[i]=uint16_t(garner::kConstants.moduli[i]);return values;
}
struct OrderedBasisIdentity {
  uint32_t representation=1,count=20;
  std::array<uint16_t,20> moduli=default_ordered_basis();
  uint64_t fingerprint=ordered_basis_fingerprint(moduli.data(),count);
  bool operator==(const OrderedBasisIdentity& other) const {
    if(representation!=other.representation || count!=other.count)return false;
    // Full fixed content includes canonical unused zero slots; no hash-only hit.
    return moduli==other.moduli;
  }
};
inline bool valid_ordered_basis(const OrderedBasisIdentity& basis) {
  if(basis.representation!=1 || !basis.count || basis.count>20)return false;
  for(unsigned i=0;i<20;++i) {
    if(i>=basis.count){if(basis.moduli[i])return false;continue;}
    if(basis.moduli[i]<2 || basis.moduli[i]>256)return false;
    for(unsigned j=0;j<i;++j) {
      auto a=basis.moduli[i],b=basis.moduli[j];
      while(b){const auto remainder=uint16_t(a%b);a=b;b=remainder;}
      if(a!=1)return false;
    }
  }
  return basis.fingerprint==ordered_basis_fingerprint(basis.moduli.data(),basis.count);
}
inline OrderedBasisIdentity ordered_basis_identity(const BasisPacket& packet) {
  OrderedBasisIdentity identity;identity.count=packet.count;identity.moduli.fill(0);
  for(unsigned i=0;i<packet.count && i<20;++i)identity.moduli[i]=packet.moduli[i];
  identity.fingerprint=packet.identity;return identity;
}
}
