#ifndef RNS8_NATIVE_TILE_BOUND_HPP
#define RNS8_NATIVE_TILE_BOUND_HPP
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_BOUND_HD __host__ __device__
#else
#define RNS8_BOUND_HD
#endif
namespace rns8::detail {
struct NativeTileBound {
  uint64_t magnitude[3]{};
  uint32_t required_prefix = 0; // 0 proves zero; 21 means not admitted.
  uint32_t reserved = 0;
};
// Exact portable multiplication; no saturation or signed negation of INT64_MIN.
RNS8_BOUND_HD constexpr NativeTileBound native_tile_max_bound(uint64_t a, uint64_t b, uint32_t k) {
  NativeTileBound out{};
  const uint64_t al = static_cast<uint32_t>(a), ah = a >> 32;
  const uint64_t bl = static_cast<uint32_t>(b), bh = b >> 32;
  uint64_t low = al * bl, high = ah * bh;
  const uint64_t cross[] = {al * bh, ah * bl};
  for (int i = 0; i < 2; ++i) {
    const uint64_t old = low;
    low += cross[i] << 32;
    high += (cross[i] >> 32) + (low < old);
  }
  out.magnitude[0] = low;
  out.magnitude[1] = high;
  uint64_t carry = 0;
  for (int i = 0; i < 3; ++i) {
    const uint64_t x = static_cast<uint32_t>(out.magnitude[i]) * uint64_t{k} + carry;
    const uint64_t y = (out.magnitude[i] >> 32) * uint64_t{k} + (x >> 32);
    out.magnitude[i] = (y << 32) | static_cast<uint32_t>(x);
    carry = y >> 32;
  }
  return out;
}
// products has 21 rows including P0=1. Caller supplies the fixed basis table.
RNS8_BOUND_HD inline void select_native_bound_prefix(
    NativeTileBound& bound, bool signed_result, uint32_t capacity, const uint64_t products[21][3]) {
  if (!(bound.magnitude[0] | bound.magnitude[1] | bound.magnitude[2])) {
    bound.required_prefix = 0;
    return;
  }
  uint64_t threshold[3] = {bound.magnitude[0], bound.magnitude[1], bound.magnitude[2]};
  // Valid inputs are 64-bit magnitudes and K <= UINT32_MAX: bound < 2^160.
  if (signed_result) {
    threshold[2] = (threshold[2] << 1) | (threshold[1] >> 63);
    threshold[1] = (threshold[1] << 1) | (threshold[0] >> 63);
    threshold[0] <<= 1;
  }
  bound.required_prefix = 21;
  for (uint32_t prefix = 1; prefix <= capacity; ++prefix) {
    int compare = 0;
    for (int limb = 2; limb >= 0; --limb)
      if (products[prefix][limb] != threshold[limb]) {
        compare = products[prefix][limb] > threshold[limb] ? 1 : -1;
        break;
      }
    if (compare > 0) { bound.required_prefix = prefix; return; }
  }
}
} // namespace rns8::detail
#undef RNS8_BOUND_HD
#endif
