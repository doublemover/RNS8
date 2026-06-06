#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_semantics_support.inc"

}  // namespace

#include "test_semantics_validation_cases.inc"
#include "test_semantics_contract_selection_cases.inc"
#include "test_semantics_workspace_cases.inc"
#include "test_semantics_currentness_cases.inc"
