#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
#  include <hip/hip_runtime_api.h>
#endif

namespace {
#include "test_rocwmma_support.inc"

}  // namespace

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
#include "test_rocwmma_bounded_reuse_cases.inc"
#include "test_rocwmma_adaptive_exact_finite_cases.inc"
#include "test_rocwmma_finite_wrap64_cases.inc"
#endif
