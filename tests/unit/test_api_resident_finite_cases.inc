#include <array>
#include <utility>
TEST_CASE("resident finite DAGs propagate moduli through unpublished intermediates", "[api][graph][finite]") {
  for(bool hip:{false,true}) {
    if(hip && !rns8::detail::api::direct_hip_compiled()) continue;
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    for(uint16_t modulus:{2,250,251,256}) {
      const auto semantics=modulus==251?RNS8_FINITE_FIELD_U8:RNS8_FINITE_RING_U8;
      constexpr int n=3;
      auto desc=finite_gemm_desc(semantics,n,n,n,modulus);desc.requested_backend=options.requested_backend;
      auto md=finite_matrix_desc(n,n,semantics);
      rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
      REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
      std::array<rns8_matrix*,5> matrices{};
      for(auto& matrix:matrices) REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
      std::vector<std::vector<uint8_t>> values(2,std::vector<uint8_t>(n*n));
      for(int i=0;i<2;++i) {
        for(int cell=0;cell<n*n;++cell) values[i][cell]=uint8_t((cell*7+i+modulus-1)%modulus);
        REQUIRE(rns8_pack_finite_u8(ctx,matrices[i],modulus,values[i].data(),n,0)==RNS8_SUCCESS);
      }
      const std::array<std::pair<int,int>,3> edges{{{0,1},{2,0},{2,1}}};
      std::array<rns8_resident_node,3> nodes{};
      for(int i=0;i<3;++i) {
        auto& node=nodes[i];node.struct_size=sizeof(node);node.abi_version=RNS8_ABI_VERSION;
        node.plan=plan;node.workspace=workspace;node.a=matrices[edges[i].first];node.b=matrices[edges[i].second];node.c=matrices[i+2];
        std::vector<uint8_t> expected(n*n);
        for(int r=0;r<n;++r) for(int c=0;c<n;++c) {
          uint64_t sum=0;for(int q=0;q<n;++q) sum+=uint64_t(values[edges[i].first][r*n+q])*values[edges[i].second][q*n+c];
          expected[r*n+c]=uint8_t(sum%modulus);
        }
        values.push_back(std::move(expected));
      }
      rns8_resident_graph* graph=nullptr;
      REQUIRE(rns8_create_resident_graph(ctx,nodes.data(),3,&graph)==RNS8_SUCCESS);
      const auto check=[&] {
        for(int i=2;i<5;++i) {
          std::vector<uint8_t> actual(n*n);
          REQUIRE(rns8_export_finite_u8(ctx,plan,modulus,matrices[i],actual.data(),n)==RNS8_SUCCESS);
          REQUIRE(actual==values[i]);REQUIRE(matrices[i]->exact_range_prefix==0);
        }
      };
      rns8_graph_execution_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);
      REQUIRE(info.completed_nodes==3);check();
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      if(hip) {
        rns8::detail::hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
        REQUIRE(rns8_capture_resident_graph_hip(graph)==RNS8_SUCCESS);
        for(bool replay:{false,true}) {
          rns8_graph_completion_token token{};
          REQUIRE((replay?rns8_replay_resident_graph_hip(graph,stream.get(),&token):
                          rns8_submit_resident_graph_hip(graph,stream.get(),&token))==RNS8_SUCCESS);
          for(int i=2;i<5;++i) REQUIRE_FALSE(matrices[i]->device_residues_current);
          REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_BUSY);
          REQUIRE(rns8_wait_resident_graph_hip(graph,token,&info)==RNS8_SUCCESS);
          REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.completed_nodes==3);check();
        }
      }
#endif
      std::array<uint64_t,3> generations{};for(int i=0;i<3;++i) generations[i]=matrices[i+2]->content_generation;
      std::vector<uint8_t> zeros(n*n,0);
      REQUIRE(rns8_pack_finite_u8(ctx,matrices[0],modulus==2?3:2,zeros.data(),n,0)==RNS8_SUCCESS);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_INVALID_ARGUMENT);
      REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_INVALID_ARGUMENT);
      for(int i=0;i<3;++i) CHECK(matrices[i+2]->content_generation==generations[i]);
      REQUIRE(rns8_pack_finite_u8(ctx,matrices[0],modulus,values[0].data(),n,0)==RNS8_SUCCESS);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);check();
      REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_SUCCESS);
      for(auto* matrix:matrices) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
