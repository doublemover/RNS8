#pragma once

// Deliberately invalid analyzer-only code. The bounded analysis runner forces
// this first-party header into a TU; no runtime target includes or executes it.
namespace rns8_analysis_control {
#if defined(__HIP_DEVICE_COMPILE__)
__attribute__((global)) void device_null_probe(int* output) {
  int* pointer = nullptr;
  *output = *pointer; // RNS8_DEVICE_BRANCH
}
#else
int host_null_probe() {
  int* pointer = nullptr;
  return *pointer; // RNS8_HOST_BRANCH
}
#endif
} // namespace rns8_analysis_control
