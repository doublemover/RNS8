#include "core/hip_resources.hpp"
#include <array>
TEST_CASE("prepared finite HIP streams and graphs publish explicit ring results", "[api][async][finite]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  for(auto backend:{RNS8_BACKEND_HIP_DIRECT,RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_AMDGPU_BUILTINS,RNS8_BACKEND_HIPBLASLT}) {
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=backend;
  rns8_context* ctx=nullptr;const auto created=rns8_create_context(0,&options,&ctx);
  if(created==RNS8_UNSUPPORTED_BACKEND || created==RNS8_UNSUPPORTED_ARCH)continue;REQUIRE(created==RNS8_SUCCESS);
  for(uint16_t modulus:{2,250,251,256}) for(int k:{17,65537}) {
    if((backend==RNS8_BACKEND_CK || backend==RNS8_BACKEND_ROCWMMA || backend==RNS8_BACKEND_AMDGPU_BUILTINS) && modulus!=251 && modulus!=256)continue;
    if(backend==RNS8_BACKEND_AMDGPU_BUILTINS && k>65536)continue;
    const auto semantics=modulus==251?RNS8_FINITE_FIELD_U8:RNS8_FINITE_RING_U8;
    const int m=k==17?3:1,n=k==17?5:2;
    auto desc=finite_gemm_desc(semantics,m,n,k,modulus);desc.requested_backend=backend;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr;std::array<rns8_matrix*,2> c{};
    auto md=finite_matrix_desc(m,k,semantics);REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
    md=finite_matrix_desc(k,n,semantics);REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    md=finite_matrix_desc(m,n,semantics);
    for(auto& output:c) REQUIRE(rns8_create_matrix(ctx,&md,&output)==RNS8_SUCCESS);
    std::vector<uint8_t> av(m*(k+1),255),bv(k*(n+2),255);
    for(int r=0;r<m;++r) for(int q=0;q<k;++q) av[r*(k+1)+q]=uint8_t((r*7+q*3+modulus-1)%modulus);
    for(int q=0;q<k;++q) for(int col=0;col<n;++col) bv[q*(n+2)+col]=uint8_t((q*5+col+modulus-1)%modulus);
    const auto pack=[&] {
      REQUIRE(rns8_pack_finite_u8(ctx,a,modulus,av.data(),k+1,0)==RNS8_SUCCESS);
      REQUIRE(rns8_pack_finite_u8(ctx,b,modulus,bv.data(),n+2,0)==RNS8_SUCCESS);
    };
    pack();
    std::array<rns8_workspace*,2> workspace{};
    std::array<rns8_prepared_execution*,2> prepared{};
    for(int i=0;i<2;++i) {
      REQUIRE(rns8_create_workspace(ctx,plan,&workspace[i])==RNS8_SUCCESS);
      REQUIRE(rns8_prepare_gemm(ctx,plan,a,b,c[i],workspace[i],&prepared[i])==RNS8_SUCCESS);
      rns8_async_support_info support{};support.struct_size=sizeof(support);support.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_hip_support(prepared[i],&support)==RNS8_SUCCESS);
      REQUIRE(support.support_status==RNS8_SUCCESS);
      REQUIRE(rns8_capture_prepared_hip_graph(prepared[i])==RNS8_SUCCESS);
    }
    rns8_prepared_group* group=nullptr;
    REQUIRE(rns8_create_prepared_group(prepared.data(),2,&group)==RNS8_SUCCESS);
    hip_unique_stream first,second;
    REQUIRE(first.create_non_blocking()==hipSuccess);REQUIRE(second.create_non_blocking()==hipSuccess);
    void* streams[]{first.get(),second.get()};
    for(bool graph:{false,true}) {
      std::array<rns8_completion_token,2> tokens{};uint32_t submitted=0;
      REQUIRE((graph?rns8_submit_prepared_group_hip_graph(group,streams,2,tokens.data(),2,&submitted):
                     rns8_submit_prepared_group_hip(group,streams,2,tokens.data(),2,&submitted))==RNS8_SUCCESS);
      REQUIRE(submitted==2);REQUIRE(a->in_flight_readers==2);REQUIRE(b->in_flight_readers==2);
      CHECK(rns8_pack_finite_u8(ctx,a,modulus,av.data(),k+1,0)==RNS8_BUSY);
      CHECK(rns8_destroy_prepared_group(group)==RNS8_BUSY);
      for(int i:{1,0}) {
        REQUIRE_FALSE(c[i]->device_residues_current);
        rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_wait_prepared_hip(prepared[i],tokens[i],&info)==RNS8_SUCCESS);
        REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);
        REQUIRE(c[i]->finite_modulus==modulus);REQUIRE(c[i]->exact_range_prefix==0);
        std::vector<uint8_t> actual(m*(n+1),99);
        REQUIRE(rns8_export_finite_u8(ctx,plan,modulus,c[i],actual.data(),n+1)==RNS8_SUCCESS);
        for(int r=0;r<m;++r) {
          for(int col=0;col<n;++col) {
            uint64_t expected=0;
            for(int q=0;q<k;++q) expected+=uint64_t(av[r*(k+1)+q])*bv[q*(n+2)+col];
            REQUIRE(actual[r*(n+1)+col]==expected%modulus);
          }
          REQUIRE(actual[r*(n+1)+n]==99);
        }
      }
      REQUIRE(a->in_flight_readers==0);REQUIRE(b->in_flight_readers==0);
    }
    REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);
    const auto generation=c[0]->content_generation;
    const bool limited=backend==RNS8_BACKEND_CK || backend==RNS8_BACKEND_ROCWMMA || backend==RNS8_BACKEND_AMDGPU_BUILTINS;
    const uint16_t other=limited?(modulus==251?256:251):(modulus==2?3:2);
    std::vector<uint8_t> zeros(m*(k+1),0);
    REQUIRE(rns8_pack_finite_u8(ctx,a,other,zeros.data(),k+1,0)==RNS8_SUCCESS);
    CHECK(rns8_refresh_prepared_inputs(prepared[0])==RNS8_INVALID_ARGUMENT);
    av[0]=uint8_t((av[0]+1)%modulus);pack();
    rns8_completion_token token{77,88};
    CHECK(rns8_submit_prepared_hip(prepared[0],first.get(),&token)==RNS8_INVALID_ARGUMENT);
    CHECK(token.prepared_id==77);CHECK(token.epoch==88);CHECK(c[0]->content_generation==generation);
    REQUIRE(rns8_refresh_prepared_inputs(prepared[0])==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph(prepared[0],first.get(),&token)==RNS8_SUCCESS);
    REQUIRE(rns8_cancel_prepared_hip(prepared[0],token)==RNS8_SUCCESS);
    rns8_completion_info canceled{};canceled.struct_size=sizeof(canceled);canceled.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(prepared[0],token,&canceled)==RNS8_SUCCESS);
    REQUIRE(canceled.operation_status==RNS8_CANCELED);REQUIRE_FALSE(canceled.output_is_current);
    REQUIRE_FALSE(c[0]->device_residues_current);
    for(int i=0;i<2;++i) {
      REQUIRE(rns8_destroy_prepared_execution(prepared[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c[i])==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP support is not compiled");
#endif
}
