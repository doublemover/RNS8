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
template<class Eligible>
inline bool partition_modulus_planes(uint32_t count,const uint16_t* moduli,Eligible eligible,
                                     IndexedModulusPlanes& output) {
  if(!count || count>20 || !moduli)return false;
  IndexedModulusPlanes result;result.logical_count=count;
  for(uint32_t plane=0;plane<count;++plane) {
    const auto q=moduli[plane];if(q<2 || q>256)return false;
    result.compact_index[plane]=UINT8_MAX;
    if(eligible(q,plane)) {
      result.compact_index[plane]=uint8_t(result.eligible_count);
      result.moduli[result.eligible_count]=q;result.eligible[result.eligible_count++]=uint8_t(plane);
    } else result.fallback[result.fallback_count++]=uint8_t(plane);
  }
  output=result;return true;
}
}
