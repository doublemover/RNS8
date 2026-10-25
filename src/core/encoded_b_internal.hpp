#pragma once
#include "core/internal.hpp"
struct rns8_encoded_b {
  rns8_context* ctx=nullptr;
  const rns8_matrix* source=nullptr;
  uint64_t instance=0,generation=0;
  mutable uint64_t prepared_references=0;
  rns8::detail::hip_direct_device_buffer storage;
};
