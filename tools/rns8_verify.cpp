#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {
#include "rns8_verify_support.inc"
#include "rns8_verify_hip_ring_helpers.inc"
#include "rns8_verify_cpu_cases.inc"
#include "rns8_verify_hip_smoke.inc"
int main(int argc, char** argv) {
  bool hip_smoke = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hip-smoke") {
      hip_smoke = true;
    } else if (arg == "--help") {
      std::cout << "usage: rns8-verify [--hip-smoke]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (!verify_cpu()) {
    return 1;
  }
  std::cout << "CPU reference verification: PASS\n";

  if (hip_smoke) {
    if (!verify_hip_smoke()) {
      return 1;
    }
    std::cout << "Direct HIP pack, ring, bounded GEMM, adaptive bounded GEMM, finite u8, and wrap64 smoke: PASS\n";
  }

  return 0;
}
