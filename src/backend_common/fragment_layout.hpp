#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define RNS8_FRAGMENT_HD __host__ __device__
#else
#define RNS8_FRAGMENT_HD
#endif

namespace rns8::detail::fragment_layout {
// Local 16x16 coordinates only: low four logical bits are row, high four
// column. Outer strides, basis planes and ragged directories stay separate.
enum class Role : uint32_t { a = 1, b = 2, accumulator = 3 };
struct Map {
  uint32_t version = 1;
  uint32_t instruction_version = 1; // RDNA3 iu8 16x16x16 wave32
  uint32_t wave_size = 32;
  Role role = Role::a;
  uint32_t physical_bits = 9;
  uint32_t element_bits = 8;
  uint32_t replicas = 2;
  bool masked_edges = true;
  // Physical bit order: lane[0:5], register[0:2], byte[0:2] for
  // inputs; lane[0:5], register[0:3] for the INT32 accumulator.
  uint16_t basis[9]{};

  RNS8_FRAGMENT_HD constexpr uint32_t logical(uint32_t physical) const {
    uint32_t result = 0;
#if defined(__HIPCC__)
#pragma unroll
#endif
    for (uint32_t bit = 0; bit < physical_bits; ++bit)
      if (physical & (1u << bit)) result ^= basis[bit];
    return result;
  }
};
inline constexpr Map a{1,1,32,Role::a,9,8,2,true,{1,2,4,8,0,64,128,16,32}};
inline constexpr Map b{1,1,32,Role::b,9,8,2,true,{16,32,64,128,0,4,8,1,2}};
inline constexpr Map c{1,1,32,Role::accumulator,8,32,1,true,{16,32,64,128,1,2,4,8,0}};

RNS8_FRAGMENT_HD constexpr uint32_t input_slot(uint32_t lane, uint32_t word, uint32_t byte) {
  return lane | (word << 5) | (byte << 7);
}
RNS8_FRAGMENT_HD constexpr uint32_t output_slot(uint32_t lane, uint32_t reg) {
  return lane | (reg << 5);
}
RNS8_FRAGMENT_HD constexpr uint32_t row(uint32_t logical) { return logical & 15; }
RNS8_FRAGMENT_HD constexpr uint32_t col(uint32_t logical) { return logical >> 4; }

RNS8_FRAGMENT_HD constexpr uint32_t a_coordinate(uint32_t lane, uint32_t word, uint32_t byte) {
  constexpr Map map = a;
  return map.logical(input_slot(lane,word,byte));
}
RNS8_FRAGMENT_HD constexpr uint32_t b_coordinate(uint32_t lane, uint32_t word, uint32_t byte) {
  constexpr Map map = b;
  return map.logical(input_slot(lane,word,byte));
}
RNS8_FRAGMENT_HD constexpr uint32_t c_coordinate(uint32_t lane, uint32_t reg) {
  constexpr Map map = c;
  return map.logical(output_slot(lane,reg));
}

// Structural equality is admission; a fingerprint alone never admits a map.
constexpr bool equal(const Map& x, const Map& y) {
  if (x.version!=y.version || x.instruction_version!=y.instruction_version ||
      x.wave_size!=y.wave_size || x.role!=y.role || x.physical_bits!=y.physical_bits ||
      x.element_bits!=y.element_bits || x.replicas!=y.replicas || x.masked_edges!=y.masked_edges) return false;
  for (uint32_t i=0;i<9;++i) if(x.basis[i]!=y.basis[i]) return false;
  return true;
}
constexpr bool admitted(const Map& map) {
  return equal(map,a) || equal(map,b) || equal(map,c);
}
constexpr uint32_t canonical_slot(const Map& map, uint32_t logical) {
  if (!admitted(map) || logical>=256) return UINT32_MAX;
  for (uint32_t slot=0;slot<(1u<<map.physical_bits);++slot)
    if(map.logical(slot)==logical) return slot;
  return UINT32_MAX;
}
constexpr bool active(const Map& map, uint32_t slot, uint32_t rows, uint32_t cols) {
  if(!admitted(map) || rows>16 || cols>16 || slot>=(1u<<map.physical_bits)) return false;
  const auto coordinate=map.logical(slot);
  return row(coordinate)<rows && col(coordinate)<cols;
}
// Conversion selects a canonical agreeing replica. It never treats repeated
// input fragment slots as independent output stores.
constexpr uint32_t source_slot(const Map& source, const Map& destination, uint32_t destination_slot) {
  if(!admitted(source) || !admitted(destination) || source.element_bits!=destination.element_bits ||
      destination_slot>=(1u<<destination.physical_bits)) return UINT32_MAX;
  const auto logical=destination.logical(destination_slot);
  // Keep an agreeing same-lane replica when possible, then choose the global
  // canonical replica. Classification below describes this actual selection.
  for(uint32_t slot=destination_slot&31;slot<(1u<<source.physical_bits);slot+=32)
    if(source.logical(slot)==logical) return slot;
  return canonical_slot(source,logical);
}
enum class Conversion : uint32_t { register_only, within_wave, unsupported };
constexpr Conversion conversion(const Map& source, const Map& destination) {
  if(!admitted(source) || !admitted(destination) || source.element_bits!=destination.element_bits)
    return Conversion::unsupported;
  bool exchange=false;
  for(uint32_t slot=0;slot<(1u<<destination.physical_bits);++slot) {
    const auto logical=destination.logical(slot);
    bool same_lane=false;
    for(uint32_t candidate=slot&31;candidate<(1u<<source.physical_bits);candidate+=32)
      if(source.logical(candidate)==logical) same_lane=true;
    exchange |= !same_lane;
  }
  return exchange ? Conversion::within_wave : Conversion::register_only;
}
} // namespace rns8::detail::fragment_layout
#undef RNS8_FRAGMENT_HD
