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
struct NativeAdaptiveWork {
  uint32_t compact_index = 0;
  uint32_t prefix = 0;
  uint32_t first_plane = 0;
};
// Offsets describe admitted prefix buckets 0..20. Zero tiles have no work.
// Widen before multiplying: UINT32_MAX tiles times twenty groups fits uint64.
RNS8_BOUND_HD constexpr uint64_t native_adaptive_work_count(
    const uint32_t* offsets, uint32_t capacity, uint32_t group) {
  if (!offsets || !capacity || capacity > 20 || (group != 1 && group != 2 && group != 4)) return 0;
  uint64_t count = 0;
  for (uint32_t prefix = 1; prefix <= capacity; ++prefix)
    count += uint64_t{offsets[prefix + 1] - offsets[prefix]} * ((prefix + group - 1) / group);
  return count;
}
RNS8_BOUND_HD constexpr NativeAdaptiveWork native_adaptive_work(
    const uint32_t* offsets, uint32_t capacity, uint32_t group, uint64_t ordinal) {
  if (!offsets || !capacity || capacity > 20 || (group != 1 && group != 2 && group != 4)) return {};
  for (uint32_t prefix = 1; prefix <= capacity; ++prefix) {
    const uint32_t groups = (prefix + group - 1) / group;
    const uint64_t count = uint64_t{offsets[prefix + 1] - offsets[prefix]} * groups;
    if (ordinal < count)
      return {offsets[prefix] + static_cast<uint32_t>(ordinal / groups), prefix,
              static_cast<uint32_t>(ordinal % groups) * group};
    ordinal -= count;
  }
  return {};
}
enum class NativeBoundMethod : uint32_t { tile_max = 0, axis_l1 = 1, panel_l1 = 2 };
struct NativeAxisSummary {
  uint64_t maximum = 0;
  uint64_t l1_low = 0;
  uint64_t l1_high = 0; // At most UINT32_MAX for at most UINT32_MAX native cells.
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
RNS8_BOUND_HD constexpr NativeTileBound native_l1_product_bound(uint64_t maximum, uint64_t low, uint64_t high) {
  auto result = native_tile_max_bound(maximum, low, 1);
  const auto upper = native_tile_max_bound(maximum, high, 1);
  const uint64_t old = result.magnitude[1];
  result.magnitude[1] += upper.magnitude[0];
  result.magnitude[2] = upper.magnitude[1] + (result.magnitude[1] < old);
  return result;
}
RNS8_BOUND_HD constexpr bool native_bound_less(const NativeTileBound& a, const NativeTileBound& b) {
  for (int limb = 2; limb >= 0; --limb)
    if (a.magnitude[limb] != b.magnitude[limb]) return a.magnitude[limb] < b.magnitude[limb];
  return false;
}
RNS8_BOUND_HD constexpr NativeTileBound native_axis_l1_bound(
    const NativeAxisSummary& a, const NativeAxisSummary& b, uint32_t k) {
  auto bound = native_tile_max_bound(a.maximum, b.maximum, k);
  const auto left = native_l1_product_bound(a.maximum, b.l1_low, b.l1_high);
  const auto right = native_l1_product_bound(b.maximum, a.l1_low, a.l1_high);
  if (native_bound_less(left, bound)) bound = left;
  if (native_bound_less(right, bound)) bound = right;
  return bound;
}
// Sum only disjoint K-panel bounds: total native K <= UINT32_MAX bounds the
// exact result below 2^160, so three limbs cannot overflow.
RNS8_BOUND_HD constexpr void add_native_panel_bound(NativeTileBound& total, const NativeTileBound& panel) {
  uint64_t carry = 0;
  for (int limb = 0; limb < 3; ++limb) {
    const uint64_t old = total.magnitude[limb];
    const uint64_t sum = old + panel.magnitude[limb];
    const uint64_t with_carry = sum + carry;
    carry = (sum < old) | (with_carry < sum);
    total.magnitude[limb] = with_carry;
  }
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
