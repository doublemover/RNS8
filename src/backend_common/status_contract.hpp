#pragma once

#include "rns8/status.h"

namespace rns8::detail {
// Fixed backing permits inspecting arbitrary C/FFI bits. Only named statuses
// have an error/ownership meaning at an external callback or evidence boundary.
inline constexpr bool public_status_valid(rns8_status status) noexcept {
  switch (status) {
  case RNS8_SUCCESS:
  case RNS8_INVALID_ARGUMENT:
  case RNS8_UNSUPPORTED_OS:
  case RNS8_UNSUPPORTED_ARCH:
  case RNS8_UNSUPPORTED_BACKEND:
  case RNS8_RANGE_ERROR:
  case RNS8_ACCUMULATION_OVERFLOW_RISK:
  case RNS8_WORKSPACE_TOO_SMALL:
  case RNS8_BACKEND_FAILURE:
  case RNS8_VERIFICATION_FAILED:
  case RNS8_INTERNAL_ERROR:
  case RNS8_BUSY:
  case RNS8_CANCELED:
  case RNS8_TIMEOUT:
    return true;
  }
  return false;
}
} // namespace rns8::detail
