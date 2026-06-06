#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/backend_common.hpp"
#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_hip_direct_support_env.inc"
#include "test_hip_direct_support_resident.inc"
#include "test_hip_direct_support_misc.inc"

}  // namespace

#include "test_hip_direct_ring_cases.inc"
#include "test_hip_direct_finite_native_cases.inc"
#include "test_hip_direct_finite_persistent_cases.inc"
#include "test_hip_direct_wrap64_private_cases.inc"
#include "test_hip_direct_wrap64_public_cases.inc"
#include "test_hip_direct_wrap64_colpair_cases.inc"
#include "test_hip_direct_rns_residue_cases.inc"
#include "test_hip_direct_vector_bridge_cases.inc"
#include "test_hip_direct_bounded_residency_cases.inc"
#include "test_hip_direct_bounded_schedule_cases.inc"
#include "test_hip_direct_exact_wide_residue_cases.inc"
#include "test_hip_direct_exact_wide_export_cases.inc"
#include "test_hip_direct_bounded_oneshot_cases.inc"
#include "test_hip_direct_per_tile_cases.inc"
