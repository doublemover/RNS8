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

#include "core/api_internal.hpp"
#include "backend_common/complex_residue_plan.hpp"
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
#include "test_api_sparse_cases.inc"
#include "test_api_plan_packing_cases.inc"
#include "test_api_export_plan_cases.inc"
#include "test_api_result_cache_cases.inc"
#include "test_api_auto_selector_cases.inc"
#include "test_api_status_cases.inc"

#include "test_api_value_lifecycle_cases.inc"
#include "test_api_prepared_cases.inc"
#include "test_api_prepared_group_cases.inc"
#include "test_api_resident_graph_cases.inc"
#include "test_api_exact_epilogue_cases.inc"
#include "test_api_complex_residue_cases.inc"
#include "test_api_prepared_hip_cases.inc"
#include "test_api_plan_validation_cache_cases.inc"

#include "test_runtime_product_recipe_cases.inc"

#include "test_api_finite_verify_cases.inc"

#include "../support/ragged_owner_fixture.hpp"
TEST_CASE("public ragged owner preserves exact values across differently tiled consumers on CPU") {
  auto* ctx=create_cpu_context();
  for(bool u:{false,true}) ragged_owner_fixture::roundtrip(ctx,u);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

#include "../support/gram_fixture.hpp"
#include "test_api_gram_cases.inc"

#include "test_stream_panel_plan_cases.inc"
