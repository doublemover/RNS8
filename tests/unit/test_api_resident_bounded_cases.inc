#include <array>
#include "core/hip_resources.hpp"
TEST_CASE("bounded resident DAGs preserve declared semantics through fanout", "[api][graph][bounded]") {
  using boost::multiprecision::cpp_int;
  for(bool hip:{false,true}) for(bool signed_values:{false,true}) for(bool narrow:{false,true}) {
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
    if(hip) continue;
#endif
    const auto semantics=signed_values?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    auto desc=bounded_gemm_desc(semantics,2,2,2);desc.requested_backend=options.requested_backend;
    desc.bound=narrow?31:(signed_values?uint64_t(INT64_MAX):UINT64_MAX);
    rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    auto md=bounded_matrix_desc(2,2,semantics);
    std::array<rns8_matrix*,5> matrices{};
    for(auto& m:matrices) REQUIRE(rns8_create_matrix(ctx,&md,&m)==RNS8_SUCCESS);
    const int64_t high=narrow?31:INT64_MAX;
    const uint64_t uhigh=narrow?31:UINT64_MAX;
    const int64_t a[]{high,-high,99,-1,1,99},identity[]{1,0,99,0,1,99};
    const uint64_t au[]{uhigh,0,99,1,uhigh,99},identity_u[]{1,0,99,0,1,99};
    const auto pack=[&] {
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrices[0],a,3,0):rns8_pack_u64(ctx,matrices[0],au,3,0))==RNS8_SUCCESS);
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrices[1],identity,3,0):rns8_pack_u64(ctx,matrices[1],identity_u,3,0))==RNS8_SUCCESS);
    };
    pack();
    std::array<rns8_resident_node,3> nodes{};
    for(int i=0;i<3;++i) {
      auto& node=nodes[i];node.struct_size=sizeof(node);node.abi_version=RNS8_ABI_VERSION;
      node.plan=plan;node.workspace=workspace;node.a=matrices[i==0?0:2];node.b=matrices[1];node.c=matrices[i+2];
    }
    if(narrow) {
      auto wider_desc=desc;wider_desc.bound=signed_values?uint64_t(INT64_MAX):UINT64_MAX;
      rns8_plan* wider=nullptr;rns8_workspace* wider_workspace=nullptr;
      REQUIRE(rns8_create_plan(ctx,&wider_desc,&wider)==RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx,wider,&wider_workspace)==RNS8_SUCCESS);
      nodes[1].plan=wider;nodes[1].workspace=wider_workspace;
      rns8_resident_graph* rejected=nullptr;
      REQUIRE(rns8_create_resident_graph(ctx,nodes.data(),3,&rejected)==RNS8_RANGE_ERROR);
      REQUIRE(rejected==nullptr);
      nodes[1].plan=plan;nodes[1].workspace=workspace;
      REQUIRE(rns8_destroy_workspace(wider_workspace)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(wider)==RNS8_SUCCESS);
    }
    rns8_resident_graph* graph=nullptr;
    REQUIRE(rns8_create_resident_graph(ctx,nodes.data(),3,&graph)==RNS8_SUCCESS);
    const auto check=[&] {
      for(int i=2;i<5;++i) {
        int64_t actual[6]{99,99,99,99,99,99};uint64_t actual_u[6]{99,99,99,99,99,99};
        REQUIRE((signed_values?rns8_export_i64(ctx,plan,matrices[i],actual,3):rns8_export_u64(ctx,plan,matrices[i],actual_u,3))==RNS8_SUCCESS);
        for(int r=0;r<2;++r) {
          for(int c=0;c<2;++c) {
            cpp_int expected=0;
            for(int q=0;q<2;++q) expected+=(signed_values?cpp_int(a[r*3+q]):cpp_int(au[r*3+q]))*(q==c?1:0);
            REQUIRE((signed_values?cpp_int(actual[r*3+c]):cpp_int(actual_u[r*3+c]))==expected);
          }
          REQUIRE(actual[r*3+2]==99);REQUIRE(actual_u[r*3+2]==99);
        }
        REQUIRE(matrices[i]->exact_range_prefix==0);
        REQUIRE(matrices[i]->materialized_rns_prefix==plan->prefix);
      }
    };
    rns8_graph_execution_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);REQUIRE(info.completed_nodes==3);check();
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
      rns8_graph_completion_token token{};
      REQUIRE(rns8_submit_resident_graph_hip(graph,stream.get(),&token)==RNS8_SUCCESS);
      REQUIRE(rns8_cancel_resident_graph_hip(graph,token)==RNS8_SUCCESS);
      REQUIRE(rns8_wait_resident_graph_hip(graph,token,&info)==RNS8_SUCCESS);
      REQUIRE(info.operation_status==RNS8_CANCELED);
      for(int i=2;i<5;++i) {
        REQUIRE_FALSE(matrices[i]->device_residues_current);
        REQUIRE(matrices[i]->materialized_rns_prefix==0);
      }
    }
#endif
    std::array<uint64_t,3> generations{};for(int i=0;i<3;++i) generations[i]=matrices[i+2]->content_generation;
    pack();REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_INVALID_ARGUMENT);
    for(int i=0;i<3;++i) CHECK(matrices[i+2]->content_generation==generations[i]);
    REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
    REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);check();
    REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_SUCCESS);
    for(auto* m:matrices) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
