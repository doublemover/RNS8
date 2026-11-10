#include "plan_descriptor_case.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  rns8_fuzz::descriptor_support::replay(data, size);
  return 0;
}
