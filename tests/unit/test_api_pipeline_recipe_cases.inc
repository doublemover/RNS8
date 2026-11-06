#include "rns8/pipeline_recipe.h"
#include "core/hip_resources.hpp"
#include <array>
TEST_CASE("bound whole routes execute checked exports and reject stale or forged bindings",
          "[api][pipeline-recipe]") {
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled())
      continue;
    rns8_context* ctx = nullptr;
    if (hip) {
      rns8_context_options co{};
      co.struct_size = sizeof(co);
      co.abi_version = RNS8_ABI_VERSION;
      co.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      REQUIRE(rns8_create_context(0, &co, &ctx) == RNS8_SUCCESS);
    } else
      ctx = create_cpu_context();
    auto gd = exact_desc(RNS8_EXACT_WIDE_SIGNED, 2, 2, 3);
    if (hip)
      gd.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(ctx, &gd, &plan) == RNS8_SUCCESS);
    rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
    auto am = exact_matrix_desc(2, 3, gd.semantics), bm = exact_matrix_desc(3, 2, gd.semantics),
         cm = exact_matrix_desc(2, 2, gd.semantics);
    if (hip) {
      am.flags |= RNS8_MATRIX_RETAIN_NATIVE_INPUT;
      bm.flags |= RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    }
    REQUIRE(rns8_create_matrix(ctx, &am, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &bm, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &cm, &c) == RNS8_SUCCESS);
    int64_t av[] = {1, 2, 3, 4, 5, 6}, bv[] = {7, 8, 9, 10, 11, 12};
    REQUIRE(rns8_pack_i64(ctx, a, av, 3, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, bv, 2, 0) == RNS8_SUCCESS);
    rns8_pipeline_recipe_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = RNS8_ABI_VERSION;
    o.route_mask = 31;
    o.execution_mode = RNS8_EXEC_SYNC;
    o.output_mode = 1;
    o.output_limbs = 1;
    o.output_ld = 3;
    o.modulus_group = 4;
    o.workers = 2;
    o.fragments = 2;
    o.fragment_window = 1;
    o.locality_order_mask = 1;
    o.reuse_count = 10;
    o.max_extra_payload_bytes = 1 << 26;
    rns8_pipeline_recipe_candidate candidates[16]{};
    uint32_t count = 0;
    REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &o, nullptr, 0, candidates, 16,
                                        &count) == RNS8_SUCCESS);
    REQUIRE(count == 9);
    uint32_t chosen = 123;
    CHECK(rns8_select_pipeline_recipe(candidates, count, &chosen) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(chosen == 123);
    rns8_pipeline_recipe_prices price{};
    price.struct_size = sizeof(price);
    price.abi_version = RNS8_ABI_VERSION;
    price.known_mask = 7;
    price.origin = 1;
    std::strcpy(price.identity_sha256, candidates[0].cost_identity_sha256);
    std::strcpy(price.build_identity_sha256, candidates[0].build_identity_sha256);
    std::strcpy(price.target_id, candidates[0].target_id);
    std::strcpy(price.calibration_id, "explicit-fixture-model");
    for (int stage = 0; stage < 3; ++stage) {
      price.lower_ns[stage] = 10;
      price.upper_ns[stage] = 20;
    }
    REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &o, &price, 1, candidates, 16,
                                        &count) == RNS8_SUCCESS);
    REQUIRE(rns8_select_pipeline_recipe(candidates, count, &chosen) == RNS8_SUCCESS);
    CHECK(chosen == 0);
    CHECK(candidates[0].cost_origin == 1);
    const auto admission_counters = rns8::detail::hip_direct_allocation_counters_snapshot();
    const auto admission_references = ctx->prepared_references;
    const auto admission_generation = c->content_generation;
    std::array<unsigned char, sizeof(candidates)> candidate_bytes{};
    std::memcpy(candidate_bytes.data(), candidates, sizeof(candidates));
    for (unsigned field = 0; field < 2; ++field)
      for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), field == 0 ? 8u : 3u}) {
        CAPTURE(hip, field, bits);
        auto invalid = price;
        if (field == 0)
          invalid.known_mask = bits;
        else
          invalid.origin = bits;
        uint32_t required = 67;
        CHECK(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &o, &invalid, 1, candidates, 16,
                                          &required) == RNS8_INVALID_ARGUMENT);
        CHECK(required == 67);
        CHECK(std::memcmp(candidate_bytes.data(), candidates, sizeof(candidates)) == 0);
      }
    alignas(rns8_pipeline_recipe_candidate) uint64_t short_candidate = sizeof(uint64_t);
    const auto* short_record =
        reinterpret_cast<const rns8_pipeline_recipe_candidate*>(&short_candidate);
    uint32_t short_index = 79;
    CHECK(rns8_select_pipeline_recipe(short_record, 1, &short_index) == RNS8_INVALID_ARGUMENT);
    CHECK(short_index == 79);
    auto* const short_canary = reinterpret_cast<rns8_pipeline_recipe*>(uintptr_t(1));
    auto* short_owner = short_canary;
    CHECK(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, short_record, &short_owner) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(short_owner == short_canary);
    alignas(rns8_pipeline_recipe_options) uint64_t short_options = sizeof(uint64_t);
    const auto* short_request =
        reinterpret_cast<const rns8_pipeline_recipe_options*>(&short_options);
    uint32_t short_required = 83;
    CHECK(rns8_query_pipeline_recipes(ctx, plan, a, b, c, short_request, nullptr, 0, candidates, 16,
                                      &short_required) == RNS8_INVALID_ARGUMENT);
    CHECK(short_required == 83);
    CHECK(std::memcmp(candidate_bytes.data(), candidates, sizeof(candidates)) == 0);
    for (unsigned field = 0; field < 5; ++field)
      for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000),
                            field == 0   ? 32u
                            : field == 1 ? 3u
                            : field == 2 ? 6u
                            : field == 3 ? 5u
                                         : 16u}) {
        CAPTURE(hip, field, bits);
        auto invalid = o;
        if (field == 0)
          invalid.route_mask = bits;
        else if (field == 1)
          invalid.execution_mode = bits;
        else if (field == 2)
          invalid.output_mode = bits;
        else if (field == 3)
          invalid.export_algorithm = bits;
        else
          invalid.locality_order_mask = bits;
        uint32_t required = 71;
        CHECK(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &invalid, nullptr, 0, candidates, 16,
                                          &required) == RNS8_INVALID_ARGUMENT);
        CHECK(required == 71);
        CHECK(std::memcmp(candidate_bytes.data(), candidates, sizeof(candidates)) == 0);
        auto* const canary = reinterpret_cast<rns8_pipeline_recipe*>(uintptr_t(1));
        auto* rejected = canary;
        CHECK(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &invalid, &candidates[0],
                                           &rejected) == RNS8_INVALID_ARGUMENT);
        CHECK(rejected == canary);
      }
    for (unsigned field = 0; field < 4; ++field)
      for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000),
                            field == 0   ? 5u
                            : field == 1 ? 1u
                            : field == 2 ? 3u
                                         : 2u}) {
        CAPTURE(hip, field, bits);
        auto invalid = candidates[0];
        if (field == 0)
          invalid.route = bits;
        else if (field == 1)
          invalid.method = bits;
        else if (field == 2)
          invalid.execution_mode = bits;
        else
          invalid.cost_known = bits;
        uint32_t selected = 73;
        CHECK(rns8_select_pipeline_recipe(&invalid, 1, &selected) == RNS8_INVALID_ARGUMENT);
        CHECK(selected == 73);
        for (bool poisoned : {false, true}) {
          ctx->hip_failure_poisoned = poisoned;
          auto* const canary = reinterpret_cast<rns8_pipeline_recipe*>(uintptr_t(1));
          auto* rejected = canary;
          CHECK(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &invalid, &rejected) ==
                RNS8_INVALID_ARGUMENT);
          CHECK(rejected == canary);
          ctx->hip_failure_poisoned = false;
        }
      }
    // A valid but changed execution tag still contradicts the bound identity.
    auto changed_mode = candidates[0];
    changed_mode.execution_mode = RNS8_EXEC_OWNED_GRAPH;
    auto* const canary = reinterpret_cast<rns8_pipeline_recipe*>(uintptr_t(1));
    auto* rejected = canary;
    CHECK(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &changed_mode, &rejected) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rejected == canary);
    CHECK(ctx->prepared_references == admission_references);
    CHECK(c->content_generation == admission_generation);
    const auto rejected_counters = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(rejected_counters.allocate_calls == admission_counters.allocate_calls);
    CHECK(rejected_counters.free_calls == admission_counters.free_calls);
    CHECK(rejected_counters.h2d_calls == admission_counters.h2d_calls);
    CHECK(rejected_counters.d2h_calls == admission_counters.d2h_calls);
    auto forged = candidates[0];
    forged.identity_sha256[0] = forged.identity_sha256[0] == '0' ? '1' : '0';
    rns8_pipeline_recipe* p = nullptr;
    CHECK(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &forged, &p) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(p == nullptr);
    for (uint32_t i = 0; i < count; ++i)
      if (candidates[i].status == RNS8_SUCCESS)
        for (unsigned algorithm = 0; algorithm < 5; ++algorithm) {
          auto selected_options = o;
          selected_options.export_algorithm = algorithm;
          rns8_pipeline_recipe_candidate route[16]{};
          uint32_t total = 0;
          REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &selected_options, nullptr, 0,
                                              route, 16, &total) == RNS8_SUCCESS);
          REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &selected_options, &route[i],
                                               &p) == RNS8_SUCCESS);
          CHECK(rns8_destroy_plan(plan) == RNS8_BUSY);
          CHECK(rns8_destroy_matrix(a) == RNS8_BUSY);
          uint64_t output[6] = {0, 0, 0xfeed, 0, 0, 0xfeed};
          REQUIRE(rns8_execute_pipeline_recipe(p, output, 3) == RNS8_SUCCESS);
          CHECK(output[0] == 58);
          CHECK(output[1] == 64);
          CHECK(output[3] == 139);
          CHECK(output[4] == 154);
          CHECK(output[2] == 0xfeed);
          CHECK(output[5] == 0xfeed);
          REQUIRE(rns8_execute_pipeline_recipe(p, output, 3) ==
                  RNS8_SUCCESS); // Rebinds the retained checked export.
          REQUIRE(rns8_destroy_pipeline_recipe(p) == RNS8_SUCCESS);
          p = nullptr;
        }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hip)
      for (auto mode : {RNS8_EXEC_BORROWED_STREAM, RNS8_EXEC_OWNED_GRAPH}) {
        auto resident = o;
        resident.execution_mode = mode;
        resident.output_mode = 0;
        resident.output_limbs = 0;
        resident.output_ld = 0;
        rns8_pipeline_recipe_candidate routes[16]{};
        uint32_t total = 0;
        REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &resident, nullptr, 0, routes, 16,
                                            &total) == RNS8_SUCCESS);
        rns8::detail::hip_unique_stream stream;
        REQUIRE(stream.create_non_blocking() == hipSuccess);
        for (uint32_t i = 0; i < total; ++i)
          if (routes[i].status == RNS8_SUCCESS) {
            REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &resident, &routes[i], &p) ==
                    RNS8_SUCCESS);
            for (unsigned replay = 0; replay < 2; ++replay) {
              rns8_completion_token token{};
              REQUIRE(rns8_submit_pipeline_recipe_hip(p, stream.get(), &token) == RNS8_SUCCESS);
              CHECK(token.epoch > 0);
              rns8_completion_info info{};
              info.struct_size = sizeof(info);
              info.abi_version = RNS8_ABI_VERSION;
              REQUIRE(rns8_wait_pipeline_recipe_hip(p, token, &info) == RNS8_SUCCESS);
              uint64_t output[4]{};
              REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, output, 2, 1) ==
                      RNS8_SUCCESS);
              CHECK(output[0] == 58);
              CHECK(output[1] == 64);
              CHECK(output[2] == 139);
              CHECK(output[3] == 154);
            }
            REQUIRE(rns8_destroy_pipeline_recipe(p) == RNS8_SUCCESS);
            p = nullptr;
          }
      }
#endif
    REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &candidates[0], &p) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, av, 3, 0) == RNS8_SUCCESS);
    uint64_t unchanged[6] = {};
    CHECK(rns8_execute_pipeline_recipe(p, unchanged, 3) == RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_destroy_pipeline_recipe(p) == RNS8_SUCCESS);
    rns8_reap_pipeline_recipe_cleanup();
    for (auto* matrix : {c, b, a})
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}

TEST_CASE("bound uniform recipes preserve bounded finite and wrap contracts",
          "[api][pipeline-recipe]") {
  for (bool hip : {false, true})
    for (auto semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_FINITE_RING_U8,
                           RNS8_FINITE_FIELD_U8, RNS8_WRAP_U64_MOD_2_64}) {
      INFO("HIP=" << hip << ", semantics=" << semantics);
      if (hip && !rns8::detail::api::direct_hip_compiled())
        continue;
      const bool finite = semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
      const bool wrap = semantics == RNS8_WRAP_U64_MOD_2_64;
      rns8_context* ctx = nullptr;
      if (hip) {
        rns8_context_options d{};
        d.struct_size = sizeof(d);
        d.abi_version = RNS8_ABI_VERSION;
        d.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &d, &ctx) == RNS8_SUCCESS);
      } else
        ctx = wrap ? create_wrap_context() : create_cpu_context();
      const uint16_t modulus = semantics == RNS8_FINITE_FIELD_U8 ? 7 : 6;
      auto gd = finite ? finite_gemm_desc(semantics, 1, 1, 2, modulus)
                : wrap ? wrap_gemm_desc(1, 1, 2)
                       : bounded_gemm_desc(semantics, 1, 1, 2);
      gd.requested_backend = ctx->backend;
      if (!finite && !wrap)
        gd.bound = 32;
      rns8_plan* plan = nullptr;
      REQUIRE(rns8_create_plan(ctx, &gd, &plan) == RNS8_SUCCESS);
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
      auto md = [&](int64_t rows, int64_t cols) {
        return finite ? finite_matrix_desc(rows, cols, semantics)
               : wrap ? wrap_matrix_desc(rows, cols)
                      : bounded_matrix_desc(rows, cols, semantics);
      };
      auto am = md(1, 2), bm = md(2, 1), cm = md(1, 1);
      REQUIRE(rns8_create_matrix(ctx, &am, &a) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(ctx, &bm, &b) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(ctx, &cm, &c) == RNS8_SUCCESS);
      uint64_t av[]{wrap ? UINT64_MAX : 3, 2}, bv[]{2, 4};
      int64_t signed_a[]{-3, 2}, signed_b[]{2, 4};
      uint8_t finite_a[]{3, 2}, finite_b[]{2, 4};
      if (finite) {
        REQUIRE(rns8_pack_finite_u8(ctx, a, modulus, finite_a, 2, 0) == RNS8_SUCCESS);
        REQUIRE(rns8_pack_finite_u8(ctx, b, modulus, finite_b, 1, 0) == RNS8_SUCCESS);
      } else if (semantics == RNS8_BOUNDED_I64) {
        REQUIRE(rns8_pack_i64(ctx, a, signed_a, 2, 0) == RNS8_SUCCESS);
        REQUIRE(rns8_pack_i64(ctx, b, signed_b, 1, 0) == RNS8_SUCCESS);
      } else {
        REQUIRE(rns8_pack_u64(ctx, a, av, 2, 0) == RNS8_SUCCESS);
        REQUIRE(rns8_pack_u64(ctx, b, bv, 1, 0) == RNS8_SUCCESS);
      }
      rns8_pipeline_recipe_options o{};
      o.struct_size = sizeof(o);
      o.abi_version = RNS8_ABI_VERSION;
      o.route_mask = 1;
      o.execution_mode = RNS8_EXEC_SYNC;
      o.output_mode = finite ? 5 : 4;
      o.output_limbs = 1;
      o.output_ld = 2;
      o.modulus_group = 1;
      o.workers = o.fragments = o.fragment_window = 1;
      o.reuse_count = 1;
      o.max_extra_payload_bytes = 1 << 20;
      rns8_pipeline_recipe_candidate candidate{};
      uint32_t count = 0;
      REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &o, nullptr, 0, &candidate, 1,
                                          &count) == RNS8_SUCCESS);
      REQUIRE(candidate.status == RNS8_SUCCESS);
      rns8_pipeline_recipe* recipe = nullptr;
      REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &candidate, &recipe) ==
              RNS8_SUCCESS);
      uint64_t output[]{0, 0xfeed};
      REQUIRE(rns8_execute_pipeline_recipe(recipe, output, 2) == RNS8_SUCCESS);
      CHECK(output[0] == (finite                          ? 14 % modulus
                          : wrap                          ? 6
                          : semantics == RNS8_BOUNDED_I64 ? 2
                                                          : 14));
      CHECK(output[1] == 0xfeed);
      REQUIRE(rns8_destroy_pipeline_recipe(recipe) == RNS8_SUCCESS);
      for (auto* value : {a, b, c})
        REQUIRE(rns8_destroy_matrix(value) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}
TEST_CASE("bound recipes chain checked pinned export without prior output publication",
          "[api][pipeline-recipe][async]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for (bool sign : {false, true})
    for (auto mode : {RNS8_EXEC_BORROWED_STREAM, RNS8_EXEC_OWNED_GRAPH}) {
      rns8_context_options co{};
      co.struct_size = sizeof(co);
      co.abi_version = RNS8_ABI_VERSION;
      co.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      rns8_context* ctx = nullptr;
      REQUIRE(rns8_create_context(0, &co, &ctx) == RNS8_SUCCESS);
      const auto semantics = sign ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
      auto gd = exact_desc(semantics, 1, 1, 1);
      gd.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      rns8_plan* plan = nullptr;
      REQUIRE(rns8_create_plan(ctx, &gd, &plan) == RNS8_SUCCESS);
      auto md = exact_matrix_desc(1, 1, semantics);
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
      for (auto** value : {&a, &b, &c})
        REQUIRE(rns8_create_matrix(ctx, &md, value) == RNS8_SUCCESS);
      const int64_t negative = -3;
      const uint64_t positive = 3, factor = 4;
      REQUIRE((sign ? rns8_pack_i64(ctx, a, &negative, 1, 0)
                    : rns8_pack_u64(ctx, a, &positive, 1, 0)) == RNS8_SUCCESS);
      REQUIRE((sign ? rns8_pack_i64(ctx, b, reinterpret_cast<const int64_t*>(&factor), 1, 0)
                    : rns8_pack_u64(ctx, b, &factor, 1, 0)) == RNS8_SUCCESS);
      rns8_pipeline_recipe_options o{};
      o.struct_size = sizeof(o);
      o.abi_version = RNS8_ABI_VERSION;
      o.route_mask = 1;
      o.execution_mode = mode;
      o.output_mode = sign ? 1 : 2;
      o.output_limbs = 3;
      o.output_ld = 2;
      o.modulus_group = o.workers = o.fragments = o.fragment_window = 1;
      o.reuse_count = 3;
      o.max_extra_payload_bytes = 1 << 20;
      rns8_pipeline_recipe_candidate candidate{};
      uint32_t count = 0;
      REQUIRE(rns8_query_pipeline_recipes(ctx, plan, a, b, c, &o, nullptr, 0, &candidate, 1,
                                          &count) == RNS8_SUCCESS);
      REQUIRE(candidate.status == RNS8_SUCCESS);
      rns8_pipeline_recipe* recipe = nullptr;
      REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &candidate, &recipe) ==
              RNS8_SUCCESS);
      rns8::detail::hip_unique_stream stream;
      REQUIRE(stream.create_non_blocking() == hipSuccess);
      for (unsigned replay = 0; replay < 3; ++replay) {
        rns8_completion_token token{};
        REQUIRE(rns8_submit_pipeline_recipe_hip(recipe, stream.get(), &token) == RNS8_SUCCESS);
        const uint64_t* data = nullptr;
        uint64_t bytes = 0;
        CHECK(rns8_begin_pipeline_recipe_host_read(recipe, &data, &bytes) == RNS8_BUSY);
        if (replay == 1)
          REQUIRE(rns8_cancel_pipeline_recipe_hip(recipe, token) == RNS8_SUCCESS);
        rns8_completion_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = RNS8_ABI_VERSION;
        REQUIRE(rns8_wait_pipeline_recipe_hip(recipe, token, &info) == RNS8_SUCCESS);
        if (replay == 1) {
          CHECK(info.operation_status == RNS8_CANCELED);
          CHECK(rns8_begin_pipeline_recipe_host_read(recipe, &data, &bytes) ==
                RNS8_INVALID_ARGUMENT);
          continue;
        }
        REQUIRE(info.operation_status == RNS8_SUCCESS);
        REQUIRE(rns8_begin_pipeline_recipe_host_read(recipe, &data, &bytes) == RNS8_SUCCESS);
        REQUIRE(bytes == 48);
        CHECK(data[0] == (sign ? uint64_t(-12) : 12));
        CHECK(data[1] == (sign ? UINT64_MAX : 0));
        CHECK(data[2] == (sign ? UINT64_MAX : 0));
        CHECK(data[3] == 0);
        CHECK(data[4] == 0);
        CHECK(data[5] == 0);
        rns8_completion_token untouched{};
        CHECK(rns8_submit_pipeline_recipe_hip(recipe, stream.get(), &untouched) == RNS8_BUSY);
        CHECK(untouched.epoch == 0);
        CHECK(rns8_destroy_pipeline_recipe(recipe) == RNS8_BUSY);
        REQUIRE(rns8_end_pipeline_recipe_host_read(recipe) == RNS8_SUCCESS);
        // A failed close is cleanup-only; use a fresh recipe for further replay.
        REQUIRE(rns8_destroy_pipeline_recipe(recipe) == RNS8_SUCCESS);
        recipe = nullptr;
        if (replay < 2)
          REQUIRE(rns8_prepare_pipeline_recipe(ctx, plan, a, b, c, &o, &candidate, &recipe) ==
                  RNS8_SUCCESS);
      }
      REQUIRE(rns8_destroy_pipeline_recipe(recipe) == RNS8_SUCCESS);
      for (auto* value : {a, b, c})
        REQUIRE(rns8_destroy_matrix(value) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
#else
  SKIP("HIP asynchronous host recipe requires direct HIP");
#endif
}
