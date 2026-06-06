#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_wrap64_support.inc"

}  // namespace

#include "test_wrap64_oracle_cases.inc"
#include "test_wrap64_public_cases.inc"
#include "test_wrap64_residency_reject_cases.inc"
