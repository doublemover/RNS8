#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_bounded_gemm_support.inc"

}  // namespace

#include "test_bounded_gemm_basic_cases.inc"
#include "test_bounded_gemm_schedule_cases.inc"
#include "test_bounded_gemm_per_tile_cases.inc"
