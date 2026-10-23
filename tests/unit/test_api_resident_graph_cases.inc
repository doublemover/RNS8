TEST_CASE("resident exact DAG executes a diamond and rejects stale bindings before writes", "[api][graph]") {
  using boost::multiprecision::cpp_int;
  for (bool hip : {false,true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled()) continue;
    for (bool signed_values : {false,true}) {
      rns8_context* ctx = nullptr;
      if (hip) {
        rns8_context_options options{};
        options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
        options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
      } else ctx = create_cpu_context();
      const auto semantics = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
      auto desc = exact_desc(semantics, 2, 2, 2);
      if (hip) desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      auto md = exact_matrix_desc(2, 2, semantics);
      rns8_plan* plan = nullptr;
      rns8_workspace* workspace = nullptr;
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
      for (auto** out : {&a,&b,&c,&d,&e}) REQUIRE(rns8_create_matrix(ctx, &md, out) == RNS8_SUCCESS);
      auto pack = [&](rns8_matrix* matrix, const std::vector<int64_t>& values) {
        if (signed_values) return rns8_pack_i64(ctx, matrix, values.data(), 2, 0);
        std::vector<uint64_t> converted(values.begin(), values.end());
        return rns8_pack_u64(ctx, matrix, converted.data(), 2, 0);
      };
      std::vector<int64_t> av{1,2,3,4}, bv{5,6,7,8};
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      auto node = [&](const rns8_matrix* lhs, const rns8_matrix* rhs, rns8_matrix* output) {
        rns8_resident_matmul_node result{};
        result.struct_size = sizeof(result); result.abi_version = RNS8_ABI_VERSION;
        result.plan = plan; result.a = lhs; result.b = rhs; result.c = output; result.workspace = workspace;
        return result;
      };
      rns8_resident_matmul_node nodes[] = {node(a,b,c), node(c,a,d), node(c,b,e)};
      rns8_resident_graph* graph = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 3, &graph) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c) == RNS8_BUSY);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_BUSY);
      auto multiply = [](const std::vector<cpp_int>& lhs, const std::vector<cpp_int>& rhs) {
        std::vector<cpp_int> result(4);
        for (int row=0; row<2; ++row) for (int col=0; col<2; ++col)
          for (int k=0; k<2; ++k) result[row*2+col] += lhs[row*2+k]*rhs[k*2+col];
        return result;
      };
      auto check = [&](rns8_matrix* matrix, const std::vector<cpp_int>& expected) {
        uint64_t limbs[4]{};
        REQUIRE((signed_values ? rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, limbs, 2, 1)
                               : rns8_export_exact_wide_unsigned_limbs(ctx, plan, matrix, limbs, 2, 1)) == RNS8_SUCCESS);
        for (int i=0; i<4; ++i) REQUIRE(cpp_int(limbs[i]) == expected[i]);
      };
      rns8_graph_execution_info info{};
      info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
      for (int round=0; round<3; ++round) {
        if (round) {
          av = round == 1 ? std::vector<int64_t>{2,3,4,5} : std::vector<int64_t>(4,0);
          REQUIRE(pack(a, av) == RNS8_SUCCESS);
          const auto generation = c->content_generation;
          REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_INVALID_ARGUMENT);
          REQUIRE(info.completed_nodes == 0);
          REQUIRE(c->content_generation == generation);
          REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
        }
        REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
        REQUIRE(info.completed_nodes == 3);
        REQUIRE(info.failed_node == UINT32_MAX);
        const std::vector<cpp_int> ax(av.begin(), av.end()), bx(bv.begin(), bv.end());
        const auto cx = multiply(ax,bx);
        check(c,cx); check(d,multiply(cx,ax)); check(e,multiply(cx,bx));
      }
      rns8_resident_graph* bad = nullptr;
      auto forward = node(e,b,c);
      rns8_resident_matmul_node cycle[] = {forward,nodes[1],nodes[2]};
      REQUIRE(rns8_create_resident_graph(ctx, cycle, 3, &bad) == RNS8_INVALID_ARGUMENT);
      rns8_resident_matmul_node duplicate[] = {nodes[0],nodes[0]};
      REQUIRE(rns8_create_resident_graph(ctx, duplicate, 2, &bad) == RNS8_INVALID_ARGUMENT);
      const auto generation = c->content_generation;
      av.assign(4, INT64_MAX);
      REQUIRE(pack(a,av) == RNS8_SUCCESS);
      REQUIRE(pack(b,av) == RNS8_SUCCESS);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_RANGE_ERROR);
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 3, &bad) == RNS8_RANGE_ERROR);
      REQUIRE(bad == nullptr);
      REQUIRE(c->content_generation == generation);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      for (auto* matrix : {a,b,c,d,e}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}
