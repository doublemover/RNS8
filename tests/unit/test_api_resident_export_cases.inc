TEST_CASE(
    "resident exports preflight widths and preserve intermediate generations",
    "[api][graph][export]") {
  using boost::multiprecision::cpp_int;
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled())
      continue;
    for (bool signed_values : {false, true}) {
      rns8_context *ctx = nullptr;
      if (hip) {
        rns8_context_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = RNS8_ABI_VERSION;
        options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
      } else
        ctx = create_cpu_context();
      const auto semantics =
          signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
      auto desc = exact_desc(semantics, 2, 2, 2);
      if (hip)
        desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      auto md = exact_matrix_desc(2, 2, semantics);
      rns8_plan *plan = nullptr;
      rns8_workspace *work = nullptr;
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &work) == RNS8_SUCCESS);
      for (auto **m : {&a, &b, &c})
        REQUIRE(rns8_create_matrix(ctx, &md, m) == RNS8_SUCCESS);
      const auto pack = [&](rns8_matrix *m,
                            const std::vector<int64_t> &values) {
        if (signed_values)
          return rns8_pack_i64(ctx, m, values.data(), 2, 0);
        const std::vector<uint64_t> converted(values.begin(), values.end());
        return rns8_pack_u64(ctx, m, converted.data(), 2, 0);
      };
      const std::vector<int64_t> av = signed_values
                                          ? std::vector<int64_t>{-1, 2, 3, -4}
                                          : std::vector<int64_t>{1, 2, 3, 4};
      const std::vector<int64_t> bv{5, 6, 7, 8};
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      std::vector<uint64_t> result(18, 99), second(18, 99);
      rns8_resident_node nodes[2]{};
      for (auto &node : nodes) {
        node.struct_size = sizeof(node);
        node.abi_version = RNS8_ABI_VERSION;
        node.plan = plan;
        node.c = c;
      }
      nodes[0].kind = RNS8_RESIDENT_MATMUL;
      nodes[0].a = a;
      nodes[0].b = b;
      nodes[0].workspace = work;
      nodes[1].kind = RNS8_RESIDENT_EXPORT_LIMBS;
      nodes[1].export_limbs = result.data();
      nodes[1].export_ld = 3;
      nodes[1].export_limb_count = 3;
      rns8_resident_graph *graph = nullptr, *bad = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 2, &graph) ==
              RNS8_SUCCESS);
      REQUIRE(rns8_prepare_resident_graph_hip(graph) == RNS8_UNSUPPORTED_BACKEND);
      rns8_graph_execution_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      for (int repeat = 0; repeat < 2; ++repeat) {
        REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
        REQUIRE(info.completed_nodes == 2);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int expected = 0;
            for (int q = 0; q < 2; ++q)
              expected += cpp_int(av[row * 2 + q]) * bv[q * 2 + col];
            const auto pos = (row * 3 + col) * 3;
            cpp_int actual = cpp_int(result[pos]) +
                             (cpp_int(result[pos + 1]) << 64) +
                             (cpp_int(result[pos + 2]) << 128);
            if (signed_values && result[pos + 2] >> 63)
              actual -= cpp_int(1) << 192;
            REQUIRE(actual == expected);
          }
        for (int pos : {6, 7, 8, 15, 16, 17})
          REQUIRE(result[pos] == 99);
      }
      auto export_only = nodes[1];
      export_only.export_limbs = second.data();
      rns8_resident_graph *reader = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, &export_only, 1, &reader) ==
              RNS8_SUCCESS);
      const auto generation = c->content_generation;
      REQUIRE(rns8_execute_resident_graph(reader, &info) == RNS8_SUCCESS);
      REQUIRE(c->content_generation == generation);
      REQUIRE(second == result);
      REQUIRE(rns8_destroy_resident_graph(reader) == RNS8_SUCCESS);
      rns8_resident_node overlap[]{nodes[0], nodes[1], nodes[1]};
      REQUIRE(rns8_create_resident_graph(ctx, overlap, 3, &bad) ==
              RNS8_INVALID_ARGUMENT);
      std::swap(overlap[0], overlap[1]);
      REQUIRE(rns8_create_resident_graph(ctx, overlap, 2, &bad) ==
              RNS8_INVALID_ARGUMENT);
      auto malformed = nodes[1];
      malformed.workspace = work;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) ==
              RNS8_INVALID_ARGUMENT);
      malformed = nodes[1];
      malformed.export_ld = INT64_MAX;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) !=
              RNS8_SUCCESS);
      auto small_export = nodes[1];
      small_export.export_limb_count = 1;
      rns8_resident_node narrow[]{nodes[0], small_export};
      rns8_resident_graph *narrow_graph = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, narrow, 2, &narrow_graph) ==
              RNS8_SUCCESS);
      const std::vector<int64_t> huge(4, INT64_MAX);
      REQUIRE(pack(a, huge) == RNS8_SUCCESS);
      REQUIRE(pack(b, huge) == RNS8_SUCCESS);
      const auto previous = result;
      REQUIRE(rns8_refresh_resident_graph_inputs(narrow_graph) ==
              RNS8_RANGE_ERROR);
      REQUIRE(rns8_create_resident_graph(ctx, narrow, 2, &bad) ==
              RNS8_RANGE_ERROR);
      REQUIRE(c->content_generation == generation);
      REQUIRE(result == previous);
      REQUIRE(rns8_destroy_resident_graph(narrow_graph) == RNS8_SUCCESS);
      // The full-width export is legal and consumes the newly produced result.
      REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      REQUIRE(info.completed_nodes == 2);
      const cpp_int expected = cpp_int(2) * INT64_MAX * INT64_MAX;
      REQUIRE(cpp_int(result[0]) + (cpp_int(result[1]) << 64) +
                  (cpp_int(result[2]) << 128) ==
              expected);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      for (auto *m : {a, b, c})
        REQUIRE(rns8_destroy_matrix(m) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(work) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}
