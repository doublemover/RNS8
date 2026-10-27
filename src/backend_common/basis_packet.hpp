#pragma once
#include <cstdint>
namespace rns8::detail {
// Immutable ABI-1 packet copied to a plan-owned device allocation. No pointers,
// Boost objects or compiler-dependent bitfields cross the host/device boundary.
struct BasisPacket {
  uint32_t version=1,count=0;
  uint64_t identity=0;
  uint16_t moduli[20]{};
  uint16_t inverse_prefix[20]{};
  uint64_t products[21][3]{};
  uint64_t negative_thresholds[21][3]{};
};
struct BasisMagnitude {uint64_t words[3]{};};
}
