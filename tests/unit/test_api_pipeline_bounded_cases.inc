#include <array>
#include "core/hip_resources.hpp"
TEST_CASE("bounded event-dependent GEMM reserves pending operands without host publication", "[api][async][pipeline][bounded]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for(bool signed_values:{false,true}) {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
    const auto semantics=signed_values?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
    auto desc=bounded_gemm_desc(semantics,2,2,2);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    desc.bound=signed_values?uint64_t(INT64_MAX):UINT64_MAX;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    std::array<rns8_matrix*,4> matrices{};std::array<rns8_workspace*,2> workspaces{};
    const auto md=bounded_matrix_desc(2,2,semantics);
    const int64_t values[]{INT64_MAX,-INT64_MAX,-1,1},identity[]{1,0,0,1},zero[4]{};
    const uint64_t values_u[]{UINT64_MAX,0,1,UINT64_MAX},identity_u[]{1,0,0,1},zero_u[4]{};
    for(size_t i=0;i<matrices.size();++i) {
      REQUIRE(rns8_create_matrix(ctx,&md,&matrices[i])==RNS8_SUCCESS);
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrices[i],i==0?values:i==1?identity:zero,2,0):
          rns8_pack_u64(ctx,matrices[i],i==0?values_u:i==1?identity_u:zero_u,2,0))==RNS8_SUCCESS);
    }
    for(auto& w:workspaces) REQUIRE(rns8_create_workspace(ctx,plan,&w)==RNS8_SUCCESS);
    rns8_prepared_execution *producer=nullptr,*consumer=nullptr;
    REQUIRE(rns8_prepare_gemm(ctx,plan,matrices[0],matrices[1],matrices[2],workspaces[0],&producer)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plan,matrices[2],matrices[1],matrices[3],workspaces[1],&consumer)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_hip_submission(producer)==RNS8_SUCCESS);REQUIRE(rns8_prepare_hip_submission(consumer)==RNS8_SUCCESS);
    rns8::detail::hip_unique_stream first,second;REQUIRE(first.create_non_blocking()==hipSuccess);
    REQUIRE(second.create_non_blocking()==hipSuccess);
    rns8_completion_token source{},result{};
    REQUIRE(rns8_submit_prepared_hip(producer,first.get(),&source)==RNS8_SUCCESS);
    REQUIRE(rns8_bind_prepared_dependency(consumer,producer,source)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_pipeline(consumer,second.get(),&result)==RNS8_SUCCESS);
    REQUIRE_FALSE(matrices[2]->device_residues_current);REQUIRE_FALSE(matrices[3]->device_residues_current);
    REQUIRE(matrices[2]->in_flight_writer);REQUIRE(matrices[2]->in_flight_readers==1);
    REQUIRE((signed_values?rns8_pack_i64(ctx,matrices[1],identity,2,0):
        rns8_pack_u64(ctx,matrices[1],identity_u,2,0))==RNS8_BUSY);
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(consumer,result,&info)==RNS8_SUCCESS);
    REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);
    REQUIRE(info.a_generation==matrices[2]->content_generation);
    REQUIRE(matrices[2]->in_flight_readers==0);REQUIRE_FALSE(matrices[2]->in_flight_writer);
    int64_t actual[4]{};uint64_t actual_u[4]{};
    REQUIRE((signed_values?rns8_export_i64(ctx,plan,matrices[3],actual,2):
        rns8_export_u64(ctx,plan,matrices[3],actual_u,2))==RNS8_SUCCESS);
    for(int i=0;i<4;++i) {
      using boost::multiprecision::cpp_int;
      REQUIRE((signed_values?cpp_int(actual[i]):cpp_int(actual_u[i]))==(signed_values?cpp_int(values[i]):cpp_int(values_u[i])));
    }
    REQUIRE(matrices[3]->exact_range_prefix==0);
    REQUIRE(rns8_destroy_prepared_execution(consumer)==RNS8_SUCCESS);REQUIRE(rns8_destroy_prepared_execution(producer)==RNS8_SUCCESS);
    for(auto* m:matrices) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    for(auto* w:workspaces) REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP event-dependent submission is unavailable in this build");
#endif
}
