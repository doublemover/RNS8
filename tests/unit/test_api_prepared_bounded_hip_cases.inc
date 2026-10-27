#include "core/hip_resources.hpp"
TEST_CASE("prepared bounded HIP preserves declared ranges and checked exports", "[api][async][bounded]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(auto backend:{RNS8_BACKEND_HIP_DIRECT,RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_AMDGPU_BUILTINS,RNS8_BACKEND_HIPBLASLT})
  for(bool signed_values:{false,true}) for(bool small_bound:{false,true}) {
    const auto semantics=signed_values?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=backend;
    rns8_context* ctx=nullptr;const auto created=rns8_create_context(0,&options,&ctx);
    if(created==RNS8_UNSUPPORTED_BACKEND || created==RNS8_UNSUPPORTED_ARCH)continue;REQUIRE(created==RNS8_SUCCESS);
    auto desc=bounded_gemm_desc(semantics,2,2,2);desc.requested_backend=backend;
    desc.bound=small_bound?31:(signed_values?uint64_t(INT64_MAX):UINT64_MAX);
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    rns8_workspace* workspace=nullptr;REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    auto md=bounded_matrix_desc(2,2,semantics);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    const int64_t high=small_bound?31:INT64_MAX;
    const uint64_t uhigh=small_bound?31:UINT64_MAX;
    const int64_t av[]{high,-high,99,-1,1,99},bv[]{1,0,99,0,1,99};
    const uint64_t au[]{uhigh,0,99,1,uhigh,99},bu[]{1,0,99,0,1,99};
    const auto pack=[&] {
      REQUIRE((signed_values?rns8_pack_i64(ctx,a,av,3,0):rns8_pack_u64(ctx,a,au,3,0))==RNS8_SUCCESS);
      REQUIRE((signed_values?rns8_pack_i64(ctx,b,bv,3,0):rns8_pack_u64(ctx,b,bu,3,0))==RNS8_SUCCESS);
    };
    pack();
    rns8_prepared_execution* prepared=nullptr;
    REQUIRE(rns8_prepare_gemm(ctx,plan,a,b,c,workspace,&prepared)==RNS8_SUCCESS);
    REQUIRE(rns8_capture_prepared_hip_graph(prepared)==RNS8_SUCCESS);
    hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
    const auto check=[&] {
      int64_t actual[6]{99,99,99,99,99,99};uint64_t actual_u[6]{99,99,99,99,99,99};
      REQUIRE((signed_values?rns8_export_i64(ctx,plan,c,actual,3):rns8_export_u64(ctx,plan,c,actual_u,3))==RNS8_SUCCESS);
      for(int row=0;row<2;++row) {
        for(int col=0;col<2;++col) {
          cpp_int expected=0;
          for(int q=0;q<2;++q) expected+=(signed_values?cpp_int(av[row*3+q]):cpp_int(au[row*3+q]))*
              (signed_values?cpp_int(bv[q*3+col]):cpp_int(bu[q*3+col]));
          REQUIRE((signed_values?cpp_int(actual[row*3+col]):cpp_int(actual_u[row*3+col]))==expected);
        }
        REQUIRE(actual[row*3+2]==99);REQUIRE(actual_u[row*3+2]==99);
      }
      REQUIRE(c->exact_range_prefix==0); // no fabricated exact-wide metadata
    };
    for(bool replay:{false,true}) {
      rns8_completion_token token{};
      REQUIRE((replay?rns8_submit_prepared_hip_graph(prepared,stream.get(),&token):
                      rns8_submit_prepared_hip(prepared,stream.get(),&token))==RNS8_SUCCESS);
      REQUIRE_FALSE(c->device_residues_current);
      rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(prepared,token,&info)==RNS8_SUCCESS);
      REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);check();
    }
    pack();rns8_completion_token token{77,88};
    REQUIRE(rns8_submit_prepared_hip_graph(prepared,stream.get(),&token)==RNS8_INVALID_ARGUMENT);
    REQUIRE(token.prepared_id==77);REQUIRE(token.epoch==88);
    REQUIRE(rns8_refresh_prepared_inputs(prepared)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph(prepared,stream.get(),&token)==RNS8_SUCCESS);
    REQUIRE(rns8_cancel_prepared_hip(prepared,token)==RNS8_SUCCESS);
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(prepared,token,&info)==RNS8_SUCCESS);
    REQUIRE(info.operation_status==RNS8_CANCELED);REQUIRE_FALSE(info.output_is_current);
    REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP support is not compiled");
#endif
}
