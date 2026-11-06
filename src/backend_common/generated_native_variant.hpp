#pragma once

#include <cstdint>

namespace rns8::generated {
// Immutable generated launch metadata. IDs bind the full lowering/proof/source
// manifest; neither this table nor a callable address is execution qualification.
struct NativeVariant {
  const char* id;
  const char* target;
  const char* kernel_symbol;
  uint32_t group;
  bool unsigned_inputs;
  int (*launch)(const void*, const void*, int8_t*, int, int, int, int, int, int, int);
};
} // namespace rns8::generated
