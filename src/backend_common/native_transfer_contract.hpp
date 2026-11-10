#pragma once

#include "backend_common/abi_header.hpp"
#include "rns8/rns8.h"

namespace rns8::detail {
// Host and device transfer owners share the same public option contract.
// Admit it before backend routing, handle publication or owner inspection.
inline bool valid_native_transfer_options(const rns8_native_transfer_options* options) noexcept {
  return minimum_abi_header(options) && options->import_values <= 1 &&
         options->verify_input_bound <= 1 && options->flags == 0 &&
         (options->verify_input_bound || options->max_input_magnitude == 0) &&
         (options->import_values || !options->verify_input_bound);
}
} // namespace rns8::detail
