#pragma once
#include <cstdint>

namespace rns8::detail {
// Compact storage positions are never default-basis plane IDs. Carry both
// directions and the actual modulus into every transformed-product consumer.
struct IndexedModulusPlanes {
  uint32_t logical_count=0,eligible_count=0,fallback_count=0;
  uint8_t eligible[20]{},fallback[20]{},compact_index[20]{};
  uint16_t moduli[20]{};
};
template<class Modulus,class Eligible>
inline bool partition_modulus_planes(uint32_t count,const Modulus* moduli,Eligible eligible,
                                     IndexedModulusPlanes& output) {
  if(!count || count>20 || !moduli)return false;
  IndexedModulusPlanes result;result.logical_count=count;
  for(uint32_t plane=0;plane<count;++plane) {
    const auto q=moduli[plane];if(q<2 || q>256)return false;
    result.compact_index[plane]=UINT8_MAX;
    if(eligible(q,plane)) {
      result.compact_index[plane]=uint8_t(result.eligible_count);
      result.moduli[result.eligible_count]=uint16_t(q);result.eligible[result.eligible_count++]=uint8_t(plane);
    } else result.fallback[result.fallback_count++]=uint8_t(plane);
  }
  output=result;return true;
}
inline bool valid_indexed_modulus_planes(const IndexedModulusPlanes& planes) noexcept {
  if(!planes.logical_count || planes.logical_count>20 || planes.eligible_count>planes.logical_count ||
      planes.fallback_count!=planes.logical_count-planes.eligible_count)return false;
  uint32_t seen=0;
  for(uint32_t compact=0;compact<planes.eligible_count;++compact) {
    const auto logical=planes.eligible[compact];
    if(logical>=planes.logical_count || (seen&(1u<<logical)) || planes.compact_index[logical]!=compact ||
        planes.moduli[compact]<2 || planes.moduli[compact]>256)return false;
    seen|=1u<<logical;
  }
  for(uint32_t index=0;index<planes.fallback_count;++index) {
    const auto logical=planes.fallback[index];
    if(logical>=planes.logical_count || (seen&(1u<<logical)) || planes.compact_index[logical]!=UINT8_MAX)return false;
    seen|=1u<<logical;
  }
  return seen==((1u<<planes.logical_count)-1);
}
struct DefaultDenseModuli {static constexpr bool indexed=false;};
struct IndexedDenseModuli {
  static constexpr bool indexed=true;
  uint16_t moduli[20]{};
};
}
