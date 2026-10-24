#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hip_direct/native_wmma_prepared.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/backend_common.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include "core/api_internal.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "../support/ragged_fixture.hpp"
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
#include "test_hip_direct_grouped_descriptor_cases.inc"
#include "test_hip_direct_bounded_residency_cases.inc"
#include "test_hip_direct_bounded_schedule_cases.inc"
#include "test_hip_direct_exact_wide_residue_cases.inc"
#include "test_hip_direct_exact_wide_export_cases.inc"
#include "test_hip_direct_bounded_oneshot_cases.inc"
#include "test_hip_direct_per_tile_cases.inc"
#include "test_hip_direct_result_cache_cases.inc"

#include "test_hip_direct_qualification_cases.inc"

#include "test_hip_direct_exact_lift_cases.inc"
#include "test_hip_direct_split_k_cases.inc"
#include "test_hip_direct_epilogue_cases.inc"
#include "test_hip_direct_complex_cases.inc"
#include "test_hip_direct_narrow_cases.inc"
#include "test_hip_direct_int4_cases.inc"
#if defined(RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA)
#include "native_variants.hpp"
#endif
#include "test_hip_direct_native_wmma_cases.inc"
#include "test_hip_direct_native_queue_cases.inc"
#include "test_hip_direct_bound_cases.inc"
#include "test_hip_direct_adaptive_native_cases.inc"
#include "test_hip_direct_ragged_cases.inc"
#include "test_hip_direct_block_sparse_cases.inc"
#include "test_hip_direct_layout_cases.inc"
#include "test_hip_direct_direct_crt_cases.inc"
#include "test_hip_direct_strassen_cases.inc"
#include "test_hip_direct_integrity_cases.inc"
