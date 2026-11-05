#include "rns8/generated/metadata_registry.hpp"
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
#include "backend_ck/arithmetic_contract.hpp"
#include "backend_ck/input_encoding.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include "backend_common/mxfp_modular.hpp"
#include "backend_common/resident_transfer_plan.hpp"
#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"
#include "rns8/rns8.hpp"
#include "../support/experimental_basis_fixture.hpp"

namespace {
#include "test_api_support.inc"

} // namespace

#include "test_api_validation_cases.inc"
#include "test_api_abi_header_cases.inc"
#include "test_api_enum_admission_cases.inc"
#include "test_api_narrow_summary_cases.inc"
#include "test_api_pipeline_recipe_cases.inc"
#include "test_api_exact_wide_cases.inc"
#include "test_api_host_address_cases.inc"
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
#include "test_api_folded_bounds_cases.inc"

#include "test_runtime_product_recipe_cases.inc"

#include "test_api_finite_verify_cases.inc"

#include "../support/ragged_owner_fixture.hpp"
TEST_CASE("public ragged owner preserves exact values across differently tiled consumers on CPU") {
  auto* ctx = create_cpu_context();
  for (bool u : {false, true}) {
    ragged_owner_fixture::roundtrip(ctx, u);
    ragged_owner_fixture::wide_roundtrip(ctx, u);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

#include "../support/gram_fixture.hpp"
#include "test_api_gram_cases.inc"

#include "test_stream_panel_plan_cases.inc"
#include "test_experimental_basis_cases.inc"

#include "test_fixed_conversion_cases.inc"

#include "test_word_mod_cases.inc"

#include "test_native_device_contract_cases.inc"

#include "../support/prepared_multi_dependency_fixture.hpp"
TEST_CASE("three prepared producer epochs feed one exact epilogue on CPU") {
  check_prepared_multi_dependencies(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("three HIP producer publications feed an exact epilogue in reverse observation order") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP dependency execution unavailable in this build");
#endif
}

#include "../support/product_recipe_rebind_fixture.hpp"
TEST_CASE("public CPU product recipe rebinds generations within its compiled proof envelope") {
  check_product_recipe_rebind(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("public HIP product recipe rebinds owners without reallocating its arena") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for (auto execution :
       {RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS, RNS8_PRODUCT_EXECUTION_ORDERED_HIP})
    check_product_recipe_rebind(RNS8_BACKEND_HIP_DIRECT, execution);
#else
  SKIP("HIP recipe execution unavailable in this build");
#endif
}

#include "test_api_split_k_cases.inc"

#include "test_instruction_roof_cases.inc"

#include "test_api_prepared_finite_hip_cases.inc"
#include "test_api_prepared_bounded_hip_cases.inc"

#include "test_api_resident_finite_cases.inc"

#include "test_api_matrix_cleanup_cases.inc"

#include "test_api_resident_bounded_cases.inc"

TEST_CASE("HIP event-dependent joins submit before producer observation",
          "[api][async][pipeline]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT, true, false);
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT, true, true);
#else
  SKIP("HIP event-dependent submission is unavailable in this build");
#endif
}

#include "test_api_pipeline_bounded_cases.inc"

#include "test_api_materialized_prefix_cases.inc"

#include "test_arithmetic_certificate_cases.inc"

#include "test_quarter_crt_cases.inc"
#include "test_checked_crt_cases.inc"

#include "test_api_operation_provider_cases.inc"

#include "test_api_int4_cases.inc"
#include "test_api_fp8_cases.inc"

#include "backend_common/direct_rns_route.hpp"
#include "backend_common/vector_small_n.hpp"
TEST_CASE("public uniform plan metadata and dispatch share one qualified route", "[api][skinny]") {
  using namespace rns8::detail;
  for (auto semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED})
    for (int64_t m : {1, 16, 64, 65, 128})
      for (int64_t n : {1, 2, 4, 8, 9})
        for (uint32_t prefix : {1u, 3u, 9u, 20u})
          for (bool parallel : {false, true}) {
            rns8_plan plan{};
            plan.backend = RNS8_BACKEND_HIP_DIRECT;
            plan.prefix = prefix;
            plan.desc.semantics = semantics;
            plan.desc.m = m;
            plan.desc.n = n;
            plan.desc.k = 65537;
            plan.desc.flags = parallel ? RNS8_PLAN_DIRECT_RNS_PARALLEL : 0;
            const auto route = direct_rns_route(m, n, 65537, 65537, n, n, prefix, parallel);
            CHECK(api::selected_kernel_for_plan(plan) == direct_rns_route_name(route));
            if (!parallel)
              CHECK(route ==
                    (m * n <= 64 ? DirectRnsRoute::persistent_small : DirectRnsRoute::grouped));
            if (parallel && n <= 8)
              CHECK(route == (n == 1 ? DirectRnsRoute::gemv_n1 : DirectRnsRoute::gemv_small_n));
          }
  CHECK(vector_small_n_declared_lds(2, true, true) == 12800);
  CHECK(vector_small_n_declared_lds(2, false, true) == 12288);
  CHECK(vector_small_n_declared_lds(2, true, false) == 51200);
  CHECK(vector_small_n_declared_lds(2, false, false) == 49152);
  for (uint32_t n = 2; n <= 8; ++n) {
    CHECK(vector_small_n_width(n, true) == n);
    CHECK(vector_small_n_width(n, false) == 8);
  }
}

TEST_CASE("narrow fused resource contracts omit native intermediate but retain proof scan",
          "[narrow][resources]") {
  using namespace rns8::detail;
  for (bool wmma : {false, true})
    for (uint32_t prefix : {1u, 2u, 9u, 20u}) {
      const auto staged = query_narrow_rns_support(17, 19, 31, 32, 21, prefix, wmma, UINT64_MAX);
      const auto fused =
          query_narrow_rns_support(17, 19, 31, 32, 21, prefix, wmma, UINT64_MAX, true);
      CHECK(staged.status == RNS8_SUCCESS);
      CHECK(fused.status == RNS8_SUCCESS);
      CHECK(staged.scratch_bytes - fused.scratch_bytes == 17 * 19 * 8);
      CHECK(fused.scratch_bytes == (wmma ? 32 : 16));
      CHECK(
          query_narrow_rns_support(17, 19, 31, 32, 21, prefix, wmma, fused.scratch_bytes - 1, true)
              .status == RNS8_WORKSPACE_TOO_SMALL);
    }
}
TEST_CASE("narrow split-K resources charge all native partials and the retained proof scan",
          "[narrow][resources]") {
  using namespace rns8::detail;
  for (int64_t k : {1, 4095, 4096, 4097, 65535, 65536, 65537}) {
    const auto parts = (uint64_t(k) + 4095) / 4096;
    const auto support =
        query_narrow_rns_support(3, 5, k, k + 1, 7, 20, false, UINT64_MAX, true, true);
    REQUIRE(support.status == RNS8_SUCCESS);
    CHECK(support.payload_bytes == 3 * 5 * parts * 8);
    CHECK(support.scratch_bytes == support.payload_bytes + 16);
    CHECK(query_narrow_rns_support(3, 5, k, k + 1, 7, 20, false, support.scratch_bytes, true, true)
              .status == RNS8_SUCCESS);
    CHECK(query_narrow_rns_support(3, 5, k, k + 1, 7, 20, false, support.scratch_bytes - 1, true,
                                   true)
              .status == RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(query_narrow_rns_support(3, 5, 4097, 4097, 5, 20, true, UINT64_MAX, true, true).status ==
        RNS8_INVALID_ARGUMENT);
  CHECK(query_narrow_rns_support(3, 5, 4097, 4097, 5, 20, false, UINT64_MAX, false, true).status ==
        RNS8_INVALID_ARGUMENT);
  CHECK(native_narrow_method_count == 5);
  CHECK(native_narrow_split(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS));
  CHECK(native_narrow_fused(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS));
  CHECK_FALSE(native_narrow_byte_wmma(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS));
}
TEST_CASE("narrow groups charge complete directories and reject malformed task contracts",
          "[api][narrow][group][host]") {
  using namespace rns8::detail;
  for (auto method : {RNS8_NARROW_SCALAR_FUSED_RNS, RNS8_NARROW_BYTE_WMMA_FUSED_RNS,
                      RNS8_NARROW_SPLIT_K_I64_FUSED_RNS})
    for (uint32_t count : {2u, 3u, 4096u}) {
      const auto task =
          query_narrow_rns_support(2, 3, 4097, 4097, 3, 20, native_narrow_byte_wmma(method),
                                   UINT64_MAX, true, native_narrow_split(method));
      const auto group = query_narrow_group_support(2, 3, 4097, 20, method, count, UINT64_MAX);
      REQUIRE(group.status == RNS8_SUCCESS);
      CHECK(group.payload_bytes == task.payload_bytes * count);
      CHECK(group.scratch_bytes == task.scratch_bytes * count + count * sizeof(NativeNarrowTask));
      CHECK(query_narrow_group_support(2, 3, 4097, 20, method, count, group.scratch_bytes - 1)
                .status == RNS8_WORKSPACE_TOO_SMALL);
    }
  for (uint32_t count : {0u, 1u, 4097u, UINT32_MAX})
    CHECK(query_narrow_group_support(2, 3, 4097, 20, 2, count, UINT64_MAX).status ==
          RNS8_INVALID_ARGUMENT);
  for (uint32_t method : {5u, UINT32_MAX, UINT32_C(0x80000000)})
    CHECK(query_narrow_group_support(2, 3, 4097, 20, method, 2, UINT64_MAX).status ==
          RNS8_INVALID_ARGUMENT);
  for (uint32_t method : {0u, 1u})
    CHECK(query_narrow_group_support(2, 3, 4097, 20, method, 2, UINT64_MAX).status ==
          RNS8_UNSUPPORTED_BACKEND);
  rns8_context ctx;
  rns8_plan plan;
  rns8_matrix a, b, c[2];
  rns8_workspace workspace[2];
  rns8_grouped_gemm_task tasks[2]{};
  for (unsigned i = 0; i < 2; ++i)
    tasks[i] = {sizeof(tasks[i]), RNS8_ABI_VERSION, &a, &b, &c[i], &workspace[i]};
  const auto run = [&] {
    return rns8_gemm_narrow_exact_grouped(&ctx, &plan, tasks, 2, RNS8_NARROW_SCALAR_FUSED_RNS,
                                          UINT64_MAX);
  };
  CHECK(run() == RNS8_UNSUPPORTED_BACKEND);
  for (unsigned bad = 0; bad < 5; ++bad) {
    const auto saved = tasks[1];
    if (bad == 0)
      tasks[1].struct_size = sizeof(uint64_t);
    if (bad == 1)
      ++tasks[1].abi_version;
    if (bad == 2)
      tasks[1].workspace = tasks[0].workspace;
    if (bad == 3)
      tasks[1].c = tasks[0].c;
    if (bad == 4)
      tasks[1].a = tasks[0].c;
    CHECK(run() == RNS8_INVALID_ARGUMENT);
    tasks[1] = saved;
  }
  CHECK_FALSE(workspace[0].in_flight);
  CHECK_FALSE(workspace[1].in_flight);
}
TEST_CASE("synchronous failure leases invalidate only written outputs and release clean failures",
          "[api][lifetime]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  rns8_matrix a{}, b{}, c{};
  rns8_workspace workspace{};
  c.host_residues_current = true;
  size_t token = 0;
  REQUIRE(begin_synchronous_lease(*ctx, a, b, c, workspace, token) == RNS8_SUCCESS);
  CHECK(a.in_flight_readers == 1);
  CHECK(c.in_flight_writer);
  CHECK(workspace.in_flight);
  REQUIRE(finish_synchronous_lease(*ctx, token, RNS8_RANGE_ERROR, false));
  CHECK(c.host_residues_current);
  CHECK_FALSE(c.in_flight_writer);
  REQUIRE(begin_synchronous_lease(*ctx, a, a, c, workspace, token) == RNS8_SUCCESS);
  CHECK(a.in_flight_readers == 1);
  REQUIRE(finish_synchronous_lease(*ctx, token, RNS8_RANGE_ERROR, true));
  CHECK_FALSE(c.host_residues_current);
  CHECK(a.in_flight_readers == 0);
  CHECK(ctx->failed_work_leases.empty());
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

#include "test_api_native_integrity_cases.inc"

#include "test_api_prepared_lift_cases.inc"

#include "../support/ragged_recipe_fixture.hpp"
TEST_CASE("ragged producer binds retained multi-output recipe consumers") {
  auto* ctx = create_cpu_context();
  for (bool u : {false, true})
    ragged_recipe_fixture::check(ctx, u);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

#include "test_api_pinned_host_cases.inc"

#include "test_api_value_bound_plan_cases.inc"

TEST_CASE("CPU materialized recipes consume and refresh differently tiled ragged producers") {
  auto* ctx = create_cpu_context();
  for (bool u : {false, true})
    for (bool experimental : {false, true})
      ragged_recipe_fixture::check(ctx, u, true, experimental);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

#include "test_api_prepared_concurrency_cases.inc"

TEST_CASE("reserved CPU publication preserves exclusivity metadata and one generation advance",
          "[api][lifetime]") {
  using namespace rns8::detail::api;
  for (bool existing : {false, true})
    for (bool finite : {false, true}) {
      auto* ctx = create_cpu_context();
      rns8_matrix a{}, b{}, c{};
      rns8_workspace work{};
      rns8_plan plan{};
      plan.prefix = 9;
      plan.desc.semantics = finite ? RNS8_FINITE_RING_U8 : RNS8_EXACT_WIDE_SIGNED;
      c.prefix = finite ? 0 : 9;
      c.residues = {1, 2};
      c.host_residues_current = existing;
      const auto generation = c.content_generation;
      size_t token = 0;
      REQUIRE(begin_synchronous_lease(*ctx, a, b, c, work, token) == RNS8_SUCCESS);
      {
        SynchronousLease lease(*ctx, token);
        const std::vector<int8_t> staged{3, 4};
        boost::multiprecision::cpp_int bound = 7;
        rns8_exact_matrix_ranges ranges;
        ranges.rows.resize(1);
        ranges.cols.resize(2);
        ranges.rows[0].max_magnitude = 7; // fallible metadata prepared before commit
        REQUIRE_THROWS_AS(require_matrix_writable(c), ResourceBusy);
        REQUIRE(lease.begin_write() == RNS8_SUCCESS);
        CHECK(c.in_flight_writer);
        CHECK_FALSE(c.host_residues_current);
        REQUIRE_THROWS_AS(require_matrix_writable(c), ResourceBusy);
        std::copy(staged.begin(), staged.end(), c.residues.begin());
        mark_output_host_residues_current(c, &plan);
        if (finite)
          c.finite_modulus = 255;
        else
          rns8::detail::commit_exact_range(c, plan, bound, &ranges);
        CHECK(c.in_flight_writer);
        CHECK(c.host_residues_current);
        CHECK(c.content_generation == generation + 1);
        if (finite)
          CHECK(c.finite_modulus == 255);
        else {
          CHECK(c.exact_max_magnitude == 7);
          CHECK(c.exact_range_prefix == 9);
          CHECK(c.exact_axis_ranges.rows.size() == 1);
        }
        REQUIRE(lease.finish(RNS8_SUCCESS) == RNS8_SUCCESS);
      }
      CHECK_FALSE(c.in_flight_writer);
      CHECK(a.in_flight_readers == 0);
      CHECK_FALSE(work.in_flight);
      const auto committed = c.content_generation;
      REQUIRE(begin_synchronous_lease(*ctx, a, b, c, work, token) == RNS8_SUCCESS);
      try {
        SynchronousLease lease(*ctx, token);
        throw std::bad_alloc{};
      } catch (const std::bad_alloc&) {
      }
      CHECK(c.host_residues_current);
      CHECK(c.content_generation == committed);
      CHECK(c.residues == std::vector<int8_t>{3, 4});
      REQUIRE(begin_synchronous_lease(*ctx, a, b, c, work, token) == RNS8_SUCCESS);
      {
        SynchronousLease lease(*ctx, token);
        REQUIRE(lease.begin_write() == RNS8_SUCCESS);
        c.residues[0] = 5;
        lease.finish(RNS8_RANGE_ERROR);
      }
      CHECK_FALSE(c.host_residues_current);
      CHECK_FALSE(c.in_flight_writer);
      CHECK(ctx->failed_work_leases.empty());
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}

#include "test_api_synchronous_dispatch_cases.inc"

#include "test_api_cpu_export_cases.inc"

#include "test_api_import_reuse_cases.inc"
#include "test_api_oneshot_cleanup_cases.inc"
