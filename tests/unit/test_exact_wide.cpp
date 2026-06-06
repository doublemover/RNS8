#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_exact_wide_support.inc"
#include "test_exact_wide_limb_boundary_cases.inc"
#include "test_exact_wide_rns_contract_cases.inc"
#include "test_exact_wide_padded_export_cases.inc"
#include "test_exact_wide_error_cases.inc"
