#include <array>
#include <utility>
TEST_CASE("resident wrap DAGs preserve unsigned low words through unpublished intermediates", "[api][graph][wrap]") {
  for(bool hip:{false,true}) {
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
    if(hip) continue;
#endif
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    {
      constexpr int n=3;
      auto desc=wrap_desc(n,n,n);desc.requested_backend=options.requested_backend;
      auto md=wrap_matrix_desc(n,n);
      rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
      REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
      std::array<rns8_matrix*,5> matrices{};
      for(auto& matrix:matrices) REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
      std::vector<std::vector<uint64_t>> values(2,std::vector<uint64_t>(n*n));
      for(int i=0;i<2;++i) {
        for(int cell=0;cell<n*n;++cell) values[i][cell]=UINT64_MAX-(uint64_t(cell*7+i)<<60);
        REQUIRE(rns8_pack_u64(ctx,matrices[i],values[i].data(),n,0)==RNS8_SUCCESS);
      }
      const std::array<std::pair<int,int>,3> edges{{{0,1},{2,0},{2,1}}};
      std::array<rns8_resident_node,3> nodes{};
      for(int i=0;i<3;++i) {
        auto& node=nodes[i];node.struct_size=sizeof(node);node.abi_version=RNS8_ABI_VERSION;
        node.plan=plan;node.workspace=workspace;node.a=matrices[edges[i].first];node.b=matrices[edges[i].second];node.c=matrices[i+2];
        std::vector<uint64_t> expected(n*n);
        for(int r=0;r<n;++r) for(int c=0;c<n;++c) {
          boost::multiprecision::cpp_int sum=0;for(int q=0;q<n;++q) sum+=boost::multiprecision::cpp_int(values[edges[i].first][r*n+q])*values[edges[i].second][q*n+c];
          expected[r*n+c]=low64(sum);
        }
        values.push_back(std::move(expected));
      }
      rns8_resident_graph* graph=nullptr;
      REQUIRE(rns8_create_resident_graph(ctx,nodes.data(),3,&graph)==RNS8_SUCCESS);
      const auto check=[&] {
        for(int i=2;i<5;++i) {
          std::vector<uint64_t> actual(n*n);
          REQUIRE(rns8_export_wrap_u64(ctx,plan,matrices[i],actual.data(),n)==RNS8_SUCCESS);
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
          for(int i=2;i<5;++i) REQUIRE_FALSE(matrices[i]->device_byte_limbs_current);
          REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_BUSY);
          REQUIRE(rns8_wait_resident_graph_hip(graph,token,&info)==RNS8_SUCCESS);
          REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.completed_nodes==3);check();
        }
      }
#endif
      std::array<uint64_t,3> generations{};for(int i=0;i<3;++i) generations[i]=matrices[i+2]->content_generation;
      REQUIRE(rns8_pack_u64(ctx,matrices[0],values[0].data(),n,0)==RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_INVALID_ARGUMENT);
      for(int i=0;i<3;++i) CHECK(matrices[i+2]->content_generation==generations[i]);
      REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
      REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);check();
      REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_SUCCESS);
      for(auto* matrix:matrices) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
