#pragma once

#include "backend_common/checked_arithmetic.hpp"
#include "backend_hipblaslt/matmul_metadata_plan.hpp"

namespace rns8::bench {
struct HipblasLtPackEventWork {
  uint64_t a_retained = 0, b_retained = 0, a_transient = 0, b_transient = 0;
  uint64_t submitted_planes = 0, timed_samples = 0;
};

// Successful backend pack submissions, independent of public residue reuse.
// Stage the complete result so a reset/overflow cannot publish a partial delta.
inline bool hipblaslt_pack_event_work(const detail::HipblasLtPrepackCounters& before,
                                      const detail::HipblasLtPrepackCounters& after,
                                      HipblasLtPackEventWork& out) noexcept {
  HipblasLtPackEventWork result;
  const auto delta = [](uint64_t first, uint64_t last, uint64_t& value) {
    if (last < first)
      return false;
    value = last - first;
    return true;
  };
  if (!delta(before.a_retained_planes_packed, after.a_retained_planes_packed, result.a_retained) ||
      !delta(before.b_retained_planes_packed, after.b_retained_planes_packed, result.b_retained) ||
      !delta(before.a_transient_planes_packed, after.a_transient_planes_packed,
             result.a_transient) ||
      !delta(before.b_transient_planes_packed, after.b_transient_planes_packed,
             result.b_transient) ||
      !detail::checked::add(result.a_retained, result.b_retained, result.submitted_planes) ||
      !detail::checked::add(result.submitted_planes, result.a_transient, result.submitted_planes) ||
      !detail::checked::add(result.submitted_planes, result.b_transient, result.submitted_planes))
    return false;
  out = result;
  return true;
}
} // namespace rns8::bench
