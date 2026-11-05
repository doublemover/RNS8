#ifndef RNS8_NATIVE_NARROW_HPP
#define RNS8_NATIVE_NARROW_HPP
#include <cstdint>
#include <rns8/rns8.h>
#include "native_domain.hpp"
namespace rns8::detail {
// A magnitude envelope bounds every sequential or tree-reduction partial.
// Check divisions before multiplication; rejection leaves the output untouched.
inline bool native_partial_sum_bound(uint64_t a, uint64_t b, uint64_t k, uint64_t limit,
                                     uint64_t& bound) {
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
inline rns8_status native_narrow_domain_admission(const NativeDomainRange& a,
                                                  const NativeDomainRange& b, uint64_t k,
                                                  bool byte_wmma, uint64_t& bound) {
  if (!k || !a.well_formed() || !b.well_formed() || a.unsigned_inputs != b.unsigned_inputs)
    return RNS8_INVALID_ARGUMENT;
  if (a.unsigned_inputs && !byte_wmma)
    return RNS8_UNSUPPORTED_BACKEND;
  if (byte_wmma && (!a.byte_domain() || !b.byte_domain()))
    return RNS8_RANGE_ERROR;
  uint64_t admitted = 0;
  if (!native_partial_sum_bound(a.maximum_magnitude(), b.maximum_magnitude(), k,
                                byte_wmma ? uint64_t(INT32_MAX) : uint64_t(INT64_MAX), admitted))
    return RNS8_RANGE_ERROR;
  bound = admitted;
  return RNS8_SUCCESS;
}
} // namespace rns8::detail
#endif
