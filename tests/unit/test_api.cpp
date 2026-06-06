#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"
#include "rns8/rns8.hpp"

namespace {
#include "test_api_support.inc"

}  // namespace

#include "test_api_validation_cases.inc"
#include "test_api_exact_wide_cases.inc"
#include "test_api_backend_info_cases.inc"
#include "test_api_storage_prepack_cases.inc"
#include "test_api_plan_packing_cases.inc"
#include "test_api_auto_selector_cases.inc"
#include "test_api_status_cases.inc"
