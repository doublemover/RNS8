#pragma once
#include "native_summary_identity.hpp"
#include <algorithm>
#include <cstdint>

namespace rns8::detail {
// Inclusive native-value extrema, distinct from final-output/CRT range bounds.
// Separate signed/unsigned fields preserve the complete uint64_t domain.
struct NativeDomainRange {
  int64_t signed_minimum = 0, signed_maximum = 0;
  uint64_t unsigned_minimum = 0, unsigned_maximum = 0;
  bool valid = false, unsigned_inputs = false;

  static NativeDomainRange signed_bounds(int64_t minimum, int64_t maximum) noexcept {
    return {minimum, maximum, 0, 0, minimum <= maximum, false};
  }
  static NativeDomainRange unsigned_bounds(uint64_t minimum, uint64_t maximum) noexcept {
    return {0, 0, minimum, maximum, minimum <= maximum, true};
  }
  bool well_formed() const noexcept {
    return valid && (unsigned_inputs ? signed_minimum == 0 && signed_maximum == 0 &&
                                           unsigned_minimum <= unsigned_maximum
                                     : unsigned_minimum == 0 && unsigned_maximum == 0 &&
                                           signed_minimum <= signed_maximum);
  }
  bool merge(const NativeDomainRange& other) noexcept {
    if (!other.well_formed() ||
        (valid && (!well_formed() || unsigned_inputs != other.unsigned_inputs)))
      return false;
    if (!valid) {
      *this = other;
      return true;
    }
    if (unsigned_inputs) {
      unsigned_minimum = std::min(unsigned_minimum, other.unsigned_minimum);
      unsigned_maximum = std::max(unsigned_maximum, other.unsigned_maximum);
    } else {
      signed_minimum = std::min(signed_minimum, other.signed_minimum);
      signed_maximum = std::max(signed_maximum, other.signed_maximum);
    }
    return true;
  }
  uint64_t maximum_magnitude() const noexcept {
    if (unsigned_inputs)
      return unsigned_maximum;
    const auto magnitude = [](int64_t value) {
      return value < 0 ? uint64_t(0) - uint64_t(value) : uint64_t(value);
    };
    return std::max(magnitude(signed_minimum), magnitude(signed_maximum));
  }
  bool byte_domain() const noexcept {
    return well_formed() && (unsigned_inputs ? unsigned_maximum <= 255
                                             : signed_minimum >= -128 && signed_maximum <= 127);
  }
};

struct NativeDomainCertificate {
  NativeSummaryIdentity identity;
  NativeDomainRange range;
  bool matches(const NativeSummaryIdentity& desired) const noexcept {
    return identity == desired && range.well_formed() &&
           range.unsigned_inputs == desired.unsigned_inputs;
  }
};
} // namespace rns8::detail
