#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <vector>

#include "../support/currentness_test_helpers.hpp"
#include "core/exact_range.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

// These textual fragments define helpers before their users.
// clang-format off
namespace {
#include "test_exact_wide_support.inc"
#include "test_exact_wide_limb_boundary_cases.inc"
#include "test_exact_wide_rns_contract_cases.inc"
#include "test_exact_wide_padded_export_cases.inc"
#include "test_exact_wide_error_cases.inc"
#include "test_exact_wide_chain_cases.inc"
#include "test_exact_wide_axis_cases.inc"
#include "test_exact_wide_lift_cases.inc"
#include "test_exact_wide_auto_cases.inc"
// clang-format on
