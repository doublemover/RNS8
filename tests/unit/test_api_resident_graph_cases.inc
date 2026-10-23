#include "test_api_resident_export_cases.inc"
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
        rns8_resident_node result{};
        result.struct_size = sizeof(result); result.abi_version = RNS8_ABI_VERSION;
        result.plan = plan; result.a = lhs; result.b = rhs; result.c = output; result.workspace = workspace;
        return result;
      };
      rns8_resident_node nodes[] = {node(a,b,c), node(c,a,d), node(c,b,e)};
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
      rns8_resident_node cycle[] = {forward,nodes[1],nodes[2]};
      REQUIRE(rns8_create_resident_graph(ctx, cycle, 3, &bad) == RNS8_INVALID_ARGUMENT);
      rns8_resident_node duplicate[] = {nodes[0],nodes[0]};
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

TEST_CASE("resident DAG explicitly extends intermediate materializations before wider consumers", "[api][graph]") {
  using boost::multiprecision::cpp_int;
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled()) continue;
    for (bool signed_values : {false, true}) {
      rns8_context* ctx = nullptr;
      if (hip) {
        rns8_context_options options{};
        options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
        options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
      } else ctx = create_cpu_context();
      const auto semantics = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
      auto desc = exact_desc(semantics, 2, 2, 2);
      desc.max_prefix = 20;
      if (hip) desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      rns8_plan *small = nullptr, *wide = nullptr;
      rns8_workspace *small_work = nullptr, *wide_work = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &small) == RNS8_SUCCESS);
      desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
      REQUIRE(rns8_create_plan(ctx, &desc, &wide) == RNS8_SUCCESS);
      REQUIRE(small->prefix < 20);
      REQUIRE(wide->prefix == 20);
      const uint64_t staged_bytes = 4 * (wide->prefix - small->prefix);
      REQUIRE(rns8_create_workspace(ctx, small, &small_work) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, wide, &wide_work) == RNS8_SUCCESS);
      auto md = exact_matrix_desc(2, 2, semantics);
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
      for (auto** output : {&a, &b, &c, &d}) REQUIRE(rns8_create_matrix(ctx, &md, output) == RNS8_SUCCESS);
      const std::vector<int64_t> av = signed_values ? std::vector<int64_t>{-2, 3, 4, -5} : std::vector<int64_t>{2, 3, 4, 5};
      const std::vector<int64_t> bv{5, 2, 3, 7};
      const auto pack = [&](rns8_matrix* matrix, const std::vector<int64_t>& values) {
        if (signed_values) return rns8_pack_i64(ctx, matrix, values.data(), 2, 0);
        const std::vector<uint64_t> unsigned_values(values.begin(), values.end());
        return rns8_pack_u64(ctx, matrix, unsigned_values.data(), 2, 0);
      };
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      rns8_resident_node nodes[3]{};
      for (auto& node : nodes) { node.struct_size = sizeof(node); node.abi_version = RNS8_ABI_VERSION; }
      nodes[0].plan = small; nodes[0].a = a; nodes[0].b = b; nodes[0].c = c; nodes[0].workspace = small_work;
      nodes[1].kind = RNS8_RESIDENT_BASE_EXTEND; nodes[1].c = c;
      nodes[1].target_prefix = 20; nodes[1].max_staged_residue_bytes = staged_bytes;
      nodes[2].plan = wide; nodes[2].a = c; nodes[2].b = b; nodes[2].c = d; nodes[2].workspace = wide_work;
      rns8_resident_graph* graph = nullptr;
      nodes[1].max_staged_residue_bytes = staged_bytes - 1;
      const auto initial_generation = c->content_generation;
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 3, &graph) == RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE(graph == nullptr);
      REQUIRE(c->content_generation == initial_generation);
      nodes[1].max_staged_residue_bytes = staged_bytes;
      auto malformed = nodes[1]; malformed.plan = small;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &graph) == RNS8_INVALID_ARGUMENT);
      malformed = nodes[1]; malformed.kind = static_cast<rns8_resident_node_kind>(UINT32_MAX);
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &graph) == RNS8_INVALID_ARGUMENT);
      rns8_resident_node forward[] = {nodes[1], nodes[0], nodes[2]};
      REQUIRE(rns8_create_resident_graph(ctx, forward, 3, &graph) == RNS8_INVALID_ARGUMENT);
      rns8_resident_node missing[] = {nodes[0], nodes[2]};
      REQUIRE(rns8_create_resident_graph(ctx, missing, 2, &graph) == RNS8_RANGE_ERROR);
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 3, &graph) == RNS8_SUCCESS);
      rns8_graph_execution_info info{};
      info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
      for (int round = 0; round < 2; ++round) {
        const auto generation = c->content_generation;
        REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
        REQUIRE(info.completed_nodes == 3);
        REQUIRE(c->exact_range_prefix == 20);
        REQUIRE(c->content_generation == generation + 1); // Lift itself preserves identity.
        std::vector<cpp_int> intermediate(4), expected(4);
        for (int r = 0; r < 2; ++r) for (int col = 0; col < 2; ++col)
          for (int q = 0; q < 2; ++q) intermediate[r * 2 + col] += cpp_int(av[r * 2 + q]) * bv[q * 2 + col];
        for (int r = 0; r < 2; ++r) for (int col = 0; col < 2; ++col)
          for (int q = 0; q < 2; ++q) expected[r * 2 + col] += intermediate[r * 2 + q] * bv[q * 2 + col];
        uint64_t limbs[4]{};
        REQUIRE((signed_values ? rns8_export_exact_wide_signed_limbs(ctx, wide, d, limbs, 2, 1)
                               : rns8_export_exact_wide_unsigned_limbs(ctx, wide, d, limbs, 2, 1)) == RNS8_SUCCESS);
        for (int i = 0; i < 4; ++i) {
          cpp_int actual = limbs[i];
          if (signed_values && (limbs[i] >> 63)) actual -= cpp_int(1) << 64;
          REQUIRE(actual == expected[i]);
        }
      }
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      // A materialization-only graph binds an external value and preserves its
      // generation even when actual new suffix planes are written.
      REQUIRE(rns8_gemm_rns(ctx, small, a, b, c, small_work) == RNS8_SUCCESS);
      REQUIRE(c->exact_range_prefix == small->prefix);
      nodes[1].target_prefix = 20; nodes[1].max_staged_residue_bytes = staged_bytes;
      REQUIRE(rns8_create_resident_graph(ctx, &nodes[1], 1, &graph) == RNS8_SUCCESS);
      const auto generation = c->content_generation;
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      REQUIRE(c->exact_range_prefix == 20);
      REQUIRE(c->content_generation == generation);
      REQUIRE(pack(c, av) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_INVALID_ARGUMENT);
      REQUIRE(info.completed_nodes == 0);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      for (auto* matrix : {a, b, c, d}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      for (auto* workspace : {small_work, wide_work}) REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
      for (auto* plan : {small, wide}) REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}


TEST_CASE("resident epilogue DAG proves produced addends before any writes", "[api][graph][epilogue]") {
  using boost::multiprecision::cpp_int;
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled()) continue;
    for (bool signed_values : {false, true}) {
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
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr;
      for (auto** matrix : {&a, &b, &c, &d, &e}) REQUIRE(rns8_create_matrix(ctx, &md, matrix) == RNS8_SUCCESS);
      const auto pack = [&](rns8_matrix* matrix, const std::vector<int64_t>& values) {
        if (signed_values) return rns8_pack_i64(ctx, matrix, values.data(), 2, 0);
        const std::vector<uint64_t> converted(values.begin(), values.end());
        return rns8_pack_u64(ctx, matrix, converted.data(), 2, 0);
      };
      const std::vector<int64_t> av = signed_values ? std::vector<int64_t>{-1, 2, 3, -4} : std::vector<int64_t>{1, 2, 3, 4};
      const std::vector<int64_t> bv{5, 6, 7, 8};
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      rns8_resident_node nodes[3]{};
      for (auto& node : nodes) {
        node.struct_size = sizeof(node); node.abi_version = RNS8_ABI_VERSION;
        node.plan = plan; node.workspace = workspace; node.a = a; node.b = b;
      }
      nodes[0].c = d;
      nodes[1].kind = RNS8_RESIDENT_EXACT_AXPBY; nodes[1].d = d; nodes[1].c = c;
      nodes[2].a = c; nodes[2].c = e;
      rns8_graph_execution_info info{};
      info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
      std::vector<std::pair<int64_t, int64_t>> coefficients{{0, 0}, {0, 1}, {2, 3}, {INT64_MAX, INT64_MAX}};
      if (signed_values) coefficients.emplace_back(INT64_MIN, -3);
      for (const auto [alpha, beta] : coefficients) {
        nodes[1].alpha = alpha; nodes[1].beta = beta;
        rns8_resident_graph* graph = nullptr;
        REQUIRE(rns8_create_resident_graph(ctx, nodes, 3, &graph) == RNS8_SUCCESS);
        REQUIRE(rns8_destroy_matrix(d) == RNS8_BUSY);
        for (int repeat = 0; repeat < 2; ++repeat) {
          REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
          REQUIRE(info.completed_nodes == 3);
          std::vector<cpp_int> product(4), expected(4);
          for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col)
            for (int q = 0; q < 2; ++q) product[row * 2 + col] += cpp_int(av[row * 2 + q]) * bv[q * 2 + col];
          for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col)
            for (int q = 0; q < 2; ++q)
              expected[row * 2 + col] += (cpp_int(alpha) + beta) * product[row * 2 + q] * bv[q * 2 + col];
          uint64_t limbs[12]{};
          REQUIRE((signed_values ? rns8_export_exact_wide_signed_limbs(ctx, plan, e, limbs, 2, 3)
                                 : rns8_export_exact_wide_unsigned_limbs(ctx, plan, e, limbs, 2, 3)) == RNS8_SUCCESS);
          for (int cell = 0; cell < 4; ++cell) {
            cpp_int actual = cpp_int(limbs[3 * cell]) + (cpp_int(limbs[3 * cell + 1]) << 64) +
                             (cpp_int(limbs[3 * cell + 2]) << 128);
            if (signed_values && limbs[3 * cell + 2] >> 63) actual -= cpp_int(1) << 192;
            REQUIRE(actual == expected[cell]);
          }
        }
        REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      }
      rns8_resident_graph* bad = nullptr;
      auto malformed = nodes[1]; malformed.d = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) == RNS8_INVALID_ARGUMENT);
      malformed = nodes[0]; malformed.alpha = 1;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) == RNS8_INVALID_ARGUMENT);
      malformed = nodes[1]; malformed.d = malformed.c;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) == RNS8_INVALID_ARGUMENT);
      malformed = nodes[1]; malformed.d = e;
      rns8_resident_node forward[]{nodes[0], malformed, nodes[2]};
      REQUIRE(rns8_create_resident_graph(ctx, forward, 3, &bad) == RNS8_INVALID_ARGUMENT);
      if (!signed_values) {
        malformed = nodes[1]; malformed.alpha = -1;
        REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &bad) == RNS8_INVALID_ARGUMENT);
      }
      // A valid first product followed by an inadmissible scaled expression:
      // refresh and execution must preserve every output generation.
      nodes[1].alpha = INT64_MAX; nodes[1].beta = 1;
      rns8_resident_graph* graph = nullptr;
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_SUCCESS);
      const auto cg = c->content_generation, dg = d->content_generation;
      REQUIRE(pack(a, std::vector<int64_t>(4, INT64_MAX)) == RNS8_SUCCESS);
      REQUIRE(pack(b, std::vector<int64_t>(4, INT64_MAX)) == RNS8_SUCCESS);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_RANGE_ERROR);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_INVALID_ARGUMENT);
      REQUIRE(info.completed_nodes == 0);
      REQUIRE(rns8_create_resident_graph(ctx, nodes, 2, &bad) == RNS8_RANGE_ERROR);
      REQUIRE(c->content_generation == cg);
      REQUIRE(d->content_generation == dg);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      // alpha=0 admits the expression without separately admitting AB.
      malformed = nodes[1]; malformed.alpha = 0; malformed.d = d;
      REQUIRE(rns8_create_resident_graph(ctx, &malformed, 1, &graph) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
      for (auto* matrix : {a, b, c, d, e}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}
