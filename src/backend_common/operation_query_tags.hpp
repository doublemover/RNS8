#pragma once

#include "backend_common/abi_header.hpp"
#include "rns8/rns8.h"

namespace rns8::detail {
// Validate the public tag envelope before routing or publishing output counts.
// Arithmetic, concrete method support and resources remain provider decisions.
inline bool operation_query_tags_valid(const rns8_operation_query* query) noexcept {
  return minimum_abi_header(query) && query->operation < RNS8_OP_COUNT && query->execution_modes &&
         !(query->execution_modes & ~127u) && !(query->proof_flags & ~15u) &&
         query->device_count <= 64 && query->output_limbs <= 32 && query->input_domain <= 5 &&
         query->output_domain <= 5 && query->orientation <= 1 &&
         query->source_prefix <= RNS8_MAX_SUPPORTED_PREFIX;
}
} // namespace rns8::detail
