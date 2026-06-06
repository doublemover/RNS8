#include "core/backend_common.hpp"

#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"

#include <limits>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>
#endif

namespace rns8::detail {

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

bool round_up_aligned_u64(uint64_t value, uint64_t alignment, uint64_t& out) {
  if (value == 0 || alignment == 0) {
    return false;
  }
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    out = value;
    return true;
  }
  const uint64_t delta = alignment - remainder;
  if (value > std::numeric_limits<uint64_t>::max() - delta) {
    return false;
  }
  out = value + delta;
  return true;
}

int run_timed_device_code(const char* label, const std::function<int()>& fn) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!hip_direct_timing_enabled() || !label) {
    return fn();
  }

  hip_unique_event_pair events;
  hipError_t event_status = events.create_and_record_start();
  if (event_status != hipSuccess) {
    return fn();
  }

  const int code = fn();
  if (code == 0) {
    event_status = events.record_stop();
    if (event_status == hipSuccess) {
      hip_direct_timing_record_pending_event(label, events.release_start(), events.release_stop());
      return code;
    }
  }
  return code;
#else
  (void)label;
  return fn();
#endif
}

rns8_status status_from_device_code(int code) {
  switch (code) {
    case 0:
      return RNS8_SUCCESS;
    case 1:
      return RNS8_INVALID_ARGUMENT;
    case 2:
      return RNS8_UNSUPPORTED_BACKEND;
    case 3:
      return RNS8_BACKEND_FAILURE;
    case 4:
      return RNS8_RANGE_ERROR;
    default:
      return RNS8_BACKEND_FAILURE;
  }
}

}  // namespace rns8::detail
