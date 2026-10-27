#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define RNS8_CRT_HD __host__ __device__
#else
#define RNS8_CRT_HD
#endif
namespace rns8::detail::balanced_crt {
struct Wide {
  uint64_t limb[3]{};
};
struct Merge {
  Wide a, b, inverse;
  uint32_t left = 0, right = 0;
};
struct Plan {
  Merge merges[19]{};
  uint32_t prefix = 0, count = 0;
};
// Immutable default-basis plans, initialized once on the host. Leaves retain
// their original channel index; each merge overwrites its left child's slot.
#if !defined(__HIP_DEVICE_COMPILE__)
const Plan &plan(uint32_t prefix);
#endif
RNS8_CRT_HD inline int compare(Wide a, Wide b) {
  for (int i = 2; i >= 0; --i)
    if (a.limb[i] != b.limb[i])
      return a.limb[i] > b.limb[i] ? 1 : -1;
  return 0;
}
RNS8_CRT_HD inline Wide add(Wide a, Wide b) {
  Wide result;
  uint64_t carry = 0;
  for (int i = 0; i < 3; ++i) {
    const auto x = a.limb[i] + b.limb[i];
    const auto y = x + carry;
    carry = (x < a.limb[i]) || (y < x);
    result.limb[i] = y;
  }
  return result;
}
RNS8_CRT_HD inline Wide subtract(Wide a, Wide b) {
  Wide result;
  uint64_t borrow = 0;
  for (int i = 0; i < 3; ++i) {
    const auto x = a.limb[i] - b.limb[i];
    const auto y = x - borrow;
    borrow = (a.limb[i] < b.limb[i]) || (x < borrow);
    result.limb[i] = y;
  }
  return result;
}
RNS8_CRT_HD inline bool bit(Wide a, int i) {
  return (a.limb[i / 64] >> (i % 64)) & 1;
}
RNS8_CRT_HD inline int bits(Wide a) {
  for (int i = 191; i >= 0; --i)
    if (bit(a, i))
      return i + 1;
  return 0;
}
// All plan moduli are below 2^160. Doubling a reduced value and adding a
// bit stays below 2^161, so one subtraction suffices without a lost carry.
RNS8_CRT_HD inline Wide reduce(Wide a, Wide modulus) {
  Wide r;
  for (int i = bits(a) - 1; i >= 0; --i) {
    r = add(r, r);
    r.limb[0] |= static_cast<uint64_t>(bit(a, i));
    if (compare(r, modulus) >= 0)
      r = subtract(r, modulus);
  }
  return r;
}
RNS8_CRT_HD inline Wide multiply_mod(Wide a, Wide b, Wide modulus) {
  Wide r;
  // a is already reduced. Every addition is below 2*modulus < 2^161.
  for (int i = bits(b) - 1; i >= 0; --i) {
    r = add(r, r);
    if (compare(r, modulus) >= 0)
      r = subtract(r, modulus);
    if (bit(b, i)) {
      r = add(r, a);
      if (compare(r, modulus) >= 0)
        r = subtract(r, modulus);
    }
  }
  return r;
}
RNS8_CRT_HD inline Wide multiply_bounded(Wide a, Wide b) {
  Wide r;
  // Used only for A*t with t<B and A*B<2^160. Each Horner prefix is
  // nonnegative and no larger than the final product.
  for (int i = bits(b) - 1; i >= 0; --i) {
    r = add(r, r);
    if (bit(b, i))
      r = add(r, a);
  }
  return r;
}
RNS8_CRT_HD inline Wide merge(Wide left, Wide right, const Merge &node) {
  const auto reduced = reduce(left, node.b);
  const auto delta = compare(right, reduced) >= 0
                         ? subtract(right, reduced)
                         : subtract(node.b, subtract(reduced, right));
  return add(left, multiply_bounded(node.a,
                                    multiply_mod(delta, node.inverse, node.b)));
}
} // namespace rns8::detail::balanced_crt
#undef RNS8_CRT_HD
