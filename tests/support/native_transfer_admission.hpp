#pragma once

#include "rns8/rns8.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace native_transfer_admission {
// Run against a real retained buffer/lease. The caller checks owner and device
// accounting around this table, then continues with its successful roundtrip.
template <class Prepare>
void reject_options(const rns8_native_transfer_options& valid, Prepare prepare) {
  auto reject = [&](const rns8_native_transfer_options* options) {
    auto* const canary = reinterpret_cast<rns8_prepared_execution*>(uintptr_t(1));
    auto* output = canary;
    CHECK(prepare(options, &output) == RNS8_INVALID_ARGUMENT);
    CHECK(output == canary);
  };
  for (unsigned field = 0; field != 3; ++field)
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), field == 2 ? 1u : 2u}) {
      CAPTURE(field, bits);
      auto invalid = valid;
      if (field == 0)
        invalid.import_values = bits;
      else if (field == 1)
        invalid.verify_input_bound = bits;
      else
        invalid.flags = bits;
      reject(&invalid);
    }
  auto invalid = valid;
  invalid.import_values = 0;
  invalid.verify_input_bound = 1;
  reject(&invalid);
  invalid.verify_input_bound = 0;
  invalid.max_input_magnitude = 1;
  reject(&invalid);
  invalid.import_values = 1;
  reject(&invalid);
  for (unsigned field = 0; field != 2; ++field) {
    invalid = valid;
    if (field == 0)
      invalid.struct_size = sizeof(invalid) - 1;
    else
      invalid.abi_version = UINT32_MAX;
    reject(&invalid);
  }
  alignas(rns8_native_transfer_options) uint64_t prefix = sizeof(uint64_t);
  reject(reinterpret_cast<const rns8_native_transfer_options*>(&prefix));
  CHECK(prefix == sizeof(uint64_t));
}
} // namespace native_transfer_admission
