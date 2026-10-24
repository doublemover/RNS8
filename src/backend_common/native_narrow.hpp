#ifndef RNS8_NATIVE_NARROW_HPP
#define RNS8_NATIVE_NARROW_HPP
#include <cstdint>
namespace rns8::detail {
// A magnitude envelope bounds every sequential or tree-reduction partial.
// Check divisions before multiplication; rejection leaves the output untouched.
inline bool native_partial_sum_bound(uint64_t a, uint64_t b, uint64_t k,
                                     uint64_t limit, uint64_t &bound) {
  if (!k)
    return false;
  if (!a || !b) {
    bound = 0;
    return true;
  }
  if (a > limit / b)
    return false;
  const uint64_t product = a * b;
  if (k > limit / product)
    return false;
  bound = product * k;
  return true;
}
} // namespace rns8::detail
#endif
