#include "core/hip_resources.hpp"
TEST_CASE("experimental complex API binds both proofs and publishes paired generations",
          "[api][complex][hip]") {
  using namespace rns8::detail;
  using namespace rns8::detail::api;
  for (bool hip : {false, true}) {
    if (hip && !direct_hip_compiled())
      continue;
    rns8_context *ctx = nullptr;
    if (hip) {
      rns8_context_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = RNS8_ABI_VERSION;
      options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    } else
      ctx = create_cpu_context();
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 2, 2, 2);
    desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
    if (hip)
      desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan *plan = nullptr;
    rns8_workspace *workspace = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    const auto md = exact_matrix_desc(2, 2, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix *ar = nullptr, *ai = nullptr, *br = nullptr, *bi = nullptr, *cr = nullptr,
                *ci = nullptr;
    for (auto **value : {&ar, &ai, &br, &bi, &cr, &ci})
      REQUIRE(rns8_create_matrix(ctx, &md, value) == RNS8_SUCCESS);
    std::array<std::vector<int64_t>, 4> values{{{INT64_MIN, INT64_MAX, 0, -7},
                                                {4, -9, INT64_MAX, INT64_MIN},
                                                {2, -3, 7, 11},
                                                {-5, 1, 0, -6}}};
    const std::array<rns8_matrix *, 4> inputs{ar, ai, br, bi};
    for (std::size_t i = 0; i < inputs.size(); ++i)
      REQUIRE(rns8_pack_i64(ctx, inputs[i], values[i].data(), 2, 1) == RNS8_SUCCESS);
    const auto exported = [&](rns8_matrix *matrix) {
      std::array<uint64_t, 12> limbs{};
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, limbs.data(), 2, 3) ==
              RNS8_SUCCESS);
      std::vector<cpp_int> result;
      for (std::size_t cell = 0; cell < 4; ++cell) {
        cpp_int value = cpp_int(limbs[cell * 3]) + (cpp_int(limbs[cell * 3 + 1]) << 64) +
                        (cpp_int(limbs[cell * 3 + 2]) << 128);
        if (limbs[cell * 3 + 2] >> 63)
          value -= cpp_int(1) << 192;
        result.push_back(value);
      }
      return result;
    };
    for (auto method : {ComplexProductMethod::three_real, ComplexProductMethod::four_real}) {
      for (int changed = 0; changed < 2; ++changed) {
        if (changed) {
          ++values[1][0];
          REQUIRE(rns8_pack_i64(ctx, ai, values[1].data(), 2, 0) == RNS8_SUCCESS);
        }
        const auto rg = cr->content_generation, ig = ci->content_generation;
        REQUIRE(rns8_gemm_complex_exact_signed(ctx, plan, ar, ai, br, bi, cr, ci,
                    method == ComplexProductMethod::three_real ? RNS8_COMPLEX_THREE_REAL_PRODUCTS
                                                              : RNS8_COMPLEX_FOUR_REAL_PRODUCTS,
                                          UINT64_MAX, workspace) == RNS8_SUCCESS);
        REQUIRE(cr->content_generation == rg + 1);
        REQUIRE(ci->content_generation == ig + 1);
        REQUIRE(cr->exact_range_prefix == plan->prefix);
        REQUIRE(ci->exact_range_prefix == plan->prefix);
        REQUIRE_FALSE(cr->in_flight_writer);
        REQUIRE_FALSE(ci->in_flight_writer);
        REQUIRE_FALSE(workspace->in_flight);
        for (auto *input : inputs)
          REQUIRE(input->in_flight_readers == 0);
        const auto real = exported(cr), imag = exported(ci);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int expected_r = 0, expected_i = 0;
            for (int q = 0; q < 2; ++q) {
              expected_r += cpp_int(values[0][row * 2 + q]) * values[2][q * 2 + col] -
                            cpp_int(values[1][row * 2 + q]) * values[3][q * 2 + col];
              expected_i += cpp_int(values[0][row * 2 + q]) * values[3][q * 2 + col] +
                            cpp_int(values[1][row * 2 + q]) * values[2][q * 2 + col];
            }
            REQUIRE(real[row * 2 + col] == expected_r);
            REQUIRE(imag[row * 2 + col] == expected_i);
            REQUIRE(boost::multiprecision::abs(expected_r) <= cr->exact_max_magnitude);
            REQUIRE(boost::multiprecision::abs(expected_i) <= ci->exact_max_magnitude);
          }
      }
    }
    // Both produced components participate in ordinary later graph arithmetic.
    const auto expected_real = exported(cr), expected_imag = exported(ci);
    rns8_matrix* chain = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &md, &chain) == RNS8_SUCCESS);
    rns8_resident_node nodes[2]{};
    for (auto& node : nodes) { node.struct_size = sizeof(node); node.abi_version = RNS8_ABI_VERSION; }
    nodes[0].kind = RNS8_RESIDENT_COMPLEX_MATMUL;
    nodes[0].plan = plan; nodes[0].workspace = workspace;
    nodes[0].a = ar; nodes[0].a_imag = ai; nodes[0].b = br; nodes[0].b_imag = bi;
    nodes[0].c = cr; nodes[0].c_imag = ci; nodes[0].max_staged_residue_bytes = UINT64_MAX;
    nodes[1].kind = RNS8_RESIDENT_MATMUL;
    nodes[1].plan = plan; nodes[1].workspace = workspace;
    nodes[1].a = cr; nodes[1].b = ci; nodes[1].c = chain;
    rns8_resident_graph* graph = nullptr;
    for (auto method : {RNS8_COMPLEX_THREE_REAL_PRODUCTS, RNS8_COMPLEX_FOUR_REAL_PRODUCTS}) {
      nodes[0].complex_method = method;
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(ci) == RNS8_BUSY);
      rns8_async_support_info support{};
      support.struct_size = sizeof(support); support.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_get_resident_graph_hip_support(graph, &support) == RNS8_SUCCESS);
      CHECK(support.support_status == (hip ? RNS8_SUCCESS : RNS8_UNSUPPORTED_BACKEND));
      rns8_graph_execution_info info{};
      info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      CHECK(info.completed_nodes == 2);
      CHECK(exported(cr) == expected_real);
      CHECK(exported(ci) == expected_imag);
      const auto actual = exported(chain);
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) {
          cpp_int expected = 0;
          for (int q = 0; q < 2; ++q)
            expected += expected_real[row * 2 + q] * expected_imag[q * 2 + col];
          CHECK(actual[row * 2 + col] == expected);
        }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      if (hip) {
        REQUIRE(rns8_prepare_resident_graph_hip(graph) == RNS8_SUCCESS);
        REQUIRE(rns8_capture_resident_graph_hip(graph) == RNS8_SUCCESS);
        REQUIRE(rns8_capture_resident_graph_hip(graph) == RNS8_SUCCESS);
        hip_unique_stream stream;
        REQUIRE(stream.create_non_blocking() == hipSuccess);
        for (bool replay : {false, true}) {
          rns8_graph_completion_token token{};
          const auto before = hip_direct_allocation_counters_snapshot();
          REQUIRE((replay ? rns8_replay_resident_graph_hip(graph, stream.get(), &token)
                          : rns8_submit_resident_graph_hip(graph, stream.get(), &token)) == RNS8_SUCCESS);
          CHECK(cr->in_flight_writer);
          CHECK(ci->in_flight_writer);
          CHECK(chain->in_flight_writer);
          CHECK(rns8_destroy_resident_graph(graph) == RNS8_BUSY);
          CHECK(rns8_pack_i64(ctx, ai, values[1].data(), 2, 0) == RNS8_BUSY);
          REQUIRE(rns8_wait_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
          REQUIRE(info.operation_status == RNS8_SUCCESS);
          CHECK(info.completed_nodes == 2);
          const auto after = hip_direct_allocation_counters_snapshot();
          CHECK(after.allocate_calls == before.allocate_calls);
          CHECK(after.free_calls == before.free_calls);
          CHECK(exported(cr) == expected_real);
          CHECK(exported(ci) == expected_imag);
          CHECK(exported(chain) == actual);
          CHECK_FALSE(cr->in_flight_writer);
          CHECK_FALSE(ci->in_flight_writer);
          for (auto* input : inputs) CHECK(input->in_flight_readers == 0);
          REQUIRE(rns8_replay_resident_graph_hip(graph, stream.get(), &token) == RNS8_SUCCESS);
          REQUIRE(rns8_cancel_resident_graph_hip(graph, token) == RNS8_SUCCESS);
          REQUIRE(rns8_wait_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
          REQUIRE(info.operation_status == RNS8_CANCELED);
          CHECK_FALSE(rns_residue_state_current_for_backend(*cr, ctx->backend));
          CHECK_FALSE(rns_residue_state_current_for_backend(*ci, ctx->backend));
          CHECK_FALSE(rns_residue_state_current_for_backend(*chain, ctx->backend));
        }
        auto changed_values = values[1];
        ++changed_values[0];
        REQUIRE(rns8_pack_i64(ctx, ai, changed_values.data(), 2, 0) == RNS8_SUCCESS);
        rns8_graph_completion_token changed_token{};
        CHECK(rns8_replay_resident_graph_hip(graph, stream.get(), &changed_token) == RNS8_INVALID_ARGUMENT);
        REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
        REQUIRE(rns8_replay_resident_graph_hip(graph, stream.get(), &changed_token) == RNS8_SUCCESS);
        REQUIRE(rns8_wait_resident_graph_hip(graph, changed_token, &info) == RNS8_SUCCESS);
        REQUIRE(info.operation_status == RNS8_SUCCESS);
        auto changed_real = expected_real, changed_imag = expected_imag;
        for (int col = 0; col < 2; ++col) {
          changed_real[col] -= values[3][col];
          changed_imag[col] += values[2][col];
        }
        CHECK(exported(cr) == changed_real);
        CHECK(exported(ci) == changed_imag);
        const auto changed_chain = exported(chain);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int expected = 0;
            for (int q = 0; q < 2; ++q)
              expected += changed_real[row * 2 + q] * changed_imag[q * 2 + col];
            CHECK(changed_chain[row * 2 + col] == expected);
          }
        REQUIRE(rns8_pack_i64(ctx, ai, values[1].data(), 2, 0) == RNS8_SUCCESS);
        REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
        // Synchronous use after preparation reuses the same arena and publication path.
        const auto before = hip_direct_allocation_counters_snapshot();
        REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
        CHECK(hip_direct_allocation_counters_snapshot().allocate_calls == before.allocate_calls);
        CHECK(exported(chain) == actual);
      }
#endif
      const auto generation = chain->content_generation;
      REQUIRE(rns8_pack_i64(ctx, ai, values[1].data(), 2, 0) == RNS8_SUCCESS);
      CHECK(rns8_execute_resident_graph(graph, &info) == RNS8_INVALID_ARGUMENT);
      CHECK(chain->content_generation == generation);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      graph = nullptr;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hip) {
      rns8_matrix* chain_imag = nullptr;
      REQUIRE(rns8_create_matrix(ctx, &md, &chain_imag) == RNS8_SUCCESS);
      const auto saved = nodes[1];
      nodes[1].kind = RNS8_RESIDENT_COMPLEX_MATMUL;
      nodes[1].a = cr; nodes[1].a_imag = ci;
      nodes[1].b = br; nodes[1].b_imag = bi; nodes[1].c_imag = chain_imag;
      nodes[1].max_staged_residue_bytes = UINT64_MAX;
      // Different 3M/4M scratch envelopes share one arena in both orders.
      for (auto first : {RNS8_COMPLEX_THREE_REAL_PRODUCTS, RNS8_COMPLEX_FOUR_REAL_PRODUCTS}) {
        nodes[0].complex_method = first;
        nodes[1].complex_method = first == RNS8_COMPLEX_THREE_REAL_PRODUCTS
            ? RNS8_COMPLEX_FOUR_REAL_PRODUCTS : RNS8_COMPLEX_THREE_REAL_PRODUCTS;
        REQUIRE(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_SUCCESS);
        REQUIRE(rns8_capture_resident_graph_hip(graph) == RNS8_SUCCESS);
        hip_unique_stream stream;
        REQUIRE(stream.create_non_blocking() == hipSuccess);
        rns8_graph_completion_token token{};
        REQUIRE(rns8_replay_resident_graph_hip(graph, stream.get(), &token) == RNS8_SUCCESS);
        rns8_graph_execution_info info{};
        info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
        REQUIRE(rns8_wait_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
        REQUIRE(info.operation_status == RNS8_SUCCESS);
        const auto real = exported(chain), imag = exported(chain_imag);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int er = 0, ei = 0;
            for (int q = 0; q < 2; ++q) {
              er += expected_real[row*2+q]*values[2][q*2+col] - expected_imag[row*2+q]*values[3][q*2+col];
              ei += expected_real[row*2+q]*values[3][q*2+col] + expected_imag[row*2+q]*values[2][q*2+col];
            }
            CHECK(real[row*2+col] == er);
            CHECK(imag[row*2+col] == ei);
          }
        REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
        graph = nullptr;
      }
      nodes[1] = saved;
      REQUIRE(rns8_destroy_matrix(chain_imag) == RNS8_SUCCESS);
    }
#endif
    nodes[0].max_staged_residue_bytes = 0;
    CHECK(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_WORKSPACE_TOO_SMALL);
    CHECK(graph == nullptr);
    nodes[0].max_staged_residue_bytes = UINT64_MAX;
    nodes[0].c_imag = cr;
    CHECK(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_INVALID_ARGUMENT);
    CHECK(graph == nullptr);
    REQUIRE(rns8_destroy_matrix(chain) == RNS8_SUCCESS);
    const auto old_r = exported(cr), old_i = exported(ci);
    const auto rg = cr->content_generation, ig = ci->content_generation;
    CHECK(rns8_gemm_complex_exact_signed(ctx, plan, ar, ai, br, bi, cr, ci,
        static_cast<rns8_complex_product_method>(UINT32_MAX), UINT64_MAX, workspace) ==
        RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_complex_exact_signed(ctx, plan, ar, ai, br, bi, cr, cr,
        RNS8_COMPLEX_THREE_REAL_PRODUCTS, UINT64_MAX, workspace) == RNS8_INVALID_ARGUMENT);
    const auto call = [&](uint64_t budget = UINT64_MAX) {
      return rns8_gemm_complex_exact_signed(ctx, plan, ar, ai, br, bi, cr, ci,
                                       RNS8_COMPLEX_THREE_REAL_PRODUCTS, budget, workspace);
    };
    CHECK(call(0) == RNS8_WORKSPACE_TOO_SMALL);
    for (auto *input : inputs) {
      CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {input, ci},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, input},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      input->in_flight_writer = true;
      CHECK(call() == RNS8_BUSY);
      input->in_flight_writer = false;
      input->in_flight_readers = UINT64_MAX;
      CHECK(call() == RNS8_BUSY);
      input->in_flight_readers = 0;
      const auto prefix = input->exact_range_prefix;
      input->exact_range_prefix = 0;
      CHECK(call() == RNS8_RANGE_ERROR);
      input->exact_range_prefix = prefix;
    }
    CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, cr},
                                    ComplexProductMethod::three_real, UINT64_MAX,
                                    workspace) == RNS8_INVALID_ARGUMENT);
    for (auto *output : {cr, ci}) {
      output->in_flight_readers = 1;
      CHECK(call() == RNS8_BUSY);
      output->in_flight_readers = 0;
      output->in_flight_writer = true;
      CHECK(call() == RNS8_BUSY);
      output->in_flight_writer = false;
      output->generation_exhausted = true;
      CHECK(call() == RNS8_RANGE_ERROR);
      output->generation_exhausted = false;
    }
    workspace->in_flight = true;
    CHECK(call() == RNS8_BUSY);
    workspace->in_flight = false;
    REQUIRE(exported(cr) == old_r);
    REQUIRE(exported(ci) == old_i);
    REQUIRE(cr->content_generation == rg);
    REQUIRE(ci->content_generation == ig);
    // Legal shared readers are reserved once, and real/imaginary bounds remain
    // independent even when both components reference one input handle.
    REQUIRE(complex_exact_signed_gemm(ctx, plan, {ar, ar}, {br, br}, {cr, ci},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_SUCCESS);
    REQUIRE(exported(cr) == std::vector<cpp_int>(4, 0));
    REQUIRE(ar->in_flight_readers == 0);
    REQUIRE(br->in_flight_readers == 0);
    // Produce genuine resident >64-bit inputs, then reject an unrepresentable
    // full complex expression without publishing either component.
    const std::vector<int64_t> large(4, int64_t{1} << 40);
    REQUIRE(rns8_pack_i64(ctx, ar, large.data(), 2, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, br, large.data(), 2, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, ar, br, ai, workspace) == RNS8_SUCCESS);
    REQUIRE(call() == RNS8_SUCCESS);
    const auto wide_r = exported(cr), wide_i = exported(ci);
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 2; ++col) {
        const cpp_int bi_sum = cpp_int(values[3][col]) + values[3][2 + col];
        REQUIRE(wide_r[row * 2 + col] == (cpp_int(1) << 81) - (cpp_int(1) << 81) * bi_sum);
        REQUIRE(wide_i[row * 2 + col] == (cpp_int(1) << 122) + (cpp_int(1) << 40) * bi_sum);
      }
    REQUIRE(rns8_gemm_rns(ctx, plan, ar, br, bi, workspace) == RNS8_SUCCESS);
    const auto before_r = exported(cr), before_i = exported(ci);
    CHECK(call() == RNS8_RANGE_ERROR);
    REQUIRE(exported(cr) == before_r);
    REQUIRE(exported(ci) == before_i);
    for (auto *matrix : {ar, ai, br, bi, cr, ci})
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}
