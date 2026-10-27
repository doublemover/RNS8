#include "core/hip_resources.hpp"
#include <array>
TEST_CASE("prepared wrap HIP streams and graphs preserve full width byte semantics", "[api][async][wrap]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  for(int k:{17,4096,4097}) {
    const int m=k==17?3:1,n=k==17?5:2;
    auto desc=wrap_desc(m,n,k);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr;std::array<rns8_matrix*,2> c{};
    auto md=wrap_matrix_desc(m,k);REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
    md=wrap_matrix_desc(k,n);REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    md=wrap_matrix_desc(m,n);
    for(auto& output:c) REQUIRE(rns8_create_matrix(ctx,&md,&output)==RNS8_SUCCESS);
    std::vector<uint64_t> av(m*(k+1),99),bv(k*(n+2),99);
    const uint64_t values[]{UINT64_MAX,0,UINT64_C(1)<<63,1,UINT64_C(0x8080ff00fe017fff)};
    for(int r=0;r<m;++r) for(int q=0;q<k;++q) av[r*(k+1)+q]=values[(r*7+q*3)%5];
    for(int q=0;q<k;++q) for(int col=0;col<n;++col) bv[q*(n+2)+col]=values[(q*3+col)%5];
    const auto pack=[&] {
      REQUIRE(rns8_pack_u64(ctx,a,av.data(),k+1,0)==RNS8_SUCCESS);
      REQUIRE(rns8_pack_u64(ctx,b,bv.data(),n+2,0)==RNS8_SUCCESS);
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
    // Graphs must bind byte storage, even though every residue pointer is null.
    std::swap(c[0]->hip_byte_limbs,c[1]->hip_byte_limbs);
    rns8_completion_token rejected{77,88};
    CHECK(rns8_submit_prepared_hip_graph(prepared[0],first.get(),&rejected)==RNS8_UNSUPPORTED_BACKEND);
    CHECK(rejected.prepared_id==77);CHECK(rejected.epoch==88);
    CHECK_FALSE(c[0]->in_flight_writer);
    std::swap(c[0]->hip_byte_limbs,c[1]->hip_byte_limbs);
    for(bool graph:{false,true}) {
      std::array<rns8_completion_token,2> tokens{};uint32_t submitted=0;
      REQUIRE((graph?rns8_submit_prepared_group_hip_graph(group,streams,2,tokens.data(),2,&submitted):
                     rns8_submit_prepared_group_hip(group,streams,2,tokens.data(),2,&submitted))==RNS8_SUCCESS);
      REQUIRE(submitted==2);REQUIRE(a->in_flight_readers==2);REQUIRE(b->in_flight_readers==2);
      CHECK(rns8_pack_u64(ctx,a,av.data(),k+1,0)==RNS8_BUSY);
      CHECK(rns8_destroy_prepared_group(group)==RNS8_BUSY);
      for(int i:{1,0}) {
        REQUIRE_FALSE(c[i]->device_byte_limbs_current);
        rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_wait_prepared_hip(prepared[i],tokens[i],&info)==RNS8_SUCCESS);
        REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);
        REQUIRE_FALSE(c[i]->device_residues_current);REQUIRE(c[i]->finite_modulus==0);REQUIRE(c[i]->exact_range_prefix==0);
        std::vector<uint64_t> actual(m*(n+1),99);
        REQUIRE(rns8_export_wrap_u64(ctx,plan,c[i],actual.data(),n+1)==RNS8_SUCCESS);
        for(int r=0;r<m;++r) {
          for(int col=0;col<n;++col) {
            boost::multiprecision::cpp_int expected=0;
            for(int q=0;q<k;++q) expected+=boost::multiprecision::cpp_int(av[r*(k+1)+q])*bv[q*(n+2)+col];
            REQUIRE(actual[r*(n+1)+col]==low64(expected));
          }
          REQUIRE(actual[r*(n+1)+n]==99);
        }
      }
      REQUIRE(a->in_flight_readers==0);REQUIRE(b->in_flight_readers==0);
    }
    REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);
    const auto generation=c[0]->content_generation;
    av[0]^=UINT64_MAX;pack();
    rns8_completion_token token{77,88};
    CHECK(rns8_submit_prepared_hip(prepared[0],first.get(),&token)==RNS8_INVALID_ARGUMENT);
    CHECK(token.prepared_id==77);CHECK(token.epoch==88);CHECK(c[0]->content_generation==generation);
    REQUIRE(rns8_refresh_prepared_inputs(prepared[0])==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph(prepared[0],first.get(),&token)==RNS8_SUCCESS);
    REQUIRE(rns8_cancel_prepared_hip(prepared[0],token)==RNS8_SUCCESS);
    rns8_completion_info canceled{};canceled.struct_size=sizeof(canceled);canceled.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(prepared[0],token,&canceled)==RNS8_SUCCESS);
    REQUIRE(canceled.operation_status==RNS8_CANCELED);REQUIRE_FALSE(canceled.output_is_current);
    REQUIRE_FALSE(c[0]->device_byte_limbs_current);
    for(int i=0;i<2;++i) {
      REQUIRE(rns8_destroy_prepared_execution(prepared[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c[i])==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP support is not compiled");
#endif
}

TEST_CASE("wrap pending tokens compose byte limbs without CRT", "[api][async][wrap][pipeline]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for(bool canceled:{false,true}) {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
    auto desc=wrap_desc(2,2,2);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    auto md=wrap_matrix_desc(2,2);std::array<rns8_matrix*,3> matrix{};
    const uint64_t values[]{UINT64_MAX,UINT64_C(1)<<63,3,UINT64_MAX-1};
    for(auto& m:matrix){REQUIRE(rns8_create_matrix(ctx,&md,&m)==RNS8_SUCCESS);REQUIRE(rns8_pack_u64(ctx,m,values,2,0)==RNS8_SUCCESS);}
    std::array<rns8_workspace*,2> work{};std::array<rns8_prepared_execution*,2> prepared{};
    for(auto& w:work)REQUIRE(rns8_create_workspace(ctx,plan,&w)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plan,matrix[0],matrix[0],matrix[1],work[0],&prepared[0])==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plan,matrix[1],matrix[0],matrix[2],work[1],&prepared[1])==RNS8_SUCCESS);
    std::array<rns8::detail::hip_unique_stream,2> stream;
    for(size_t i=0;i<2;++i){REQUIRE(stream[i].create_non_blocking()==hipSuccess);REQUIRE(rns8_capture_prepared_hip_graph(prepared[i])==RNS8_SUCCESS);}
    rns8_completion_token p{},q{};REQUIRE(rns8_submit_prepared_hip(prepared[0],stream[0].get(),&p)==RNS8_SUCCESS);
    REQUIRE(rns8_bind_prepared_dependency(prepared[1],prepared[0],p)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph_pipeline(prepared[1],stream[1].get(),&q)==RNS8_SUCCESS);
    if(canceled)REQUIRE(rns8_cancel_prepared_hip(prepared[0],p)==RNS8_SUCCESS);
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(prepared[1],q,&info)==RNS8_SUCCESS);
    REQUIRE(info.operation_status==(canceled?RNS8_CANCELED:RNS8_SUCCESS));REQUIRE(bool(info.output_is_current)==!canceled);
    if(!canceled) {
      uint64_t result[4]{};REQUIRE(rns8_export_wrap_u64(ctx,plan,matrix[2],result,2)==RNS8_SUCCESS);
      for(unsigned r=0;r<2;++r)for(unsigned c=0;c<2;++c) {
        boost::multiprecision::cpp_int expected=0;
        for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)
          expected+=boost::multiprecision::cpp_int(values[r*2+a])*values[a*2+b]*values[b*2+c];
        REQUIRE(result[r*2+c]==low64(expected));
      }
      REQUIRE(matrix[2]->exact_range_prefix==0);REQUIRE_FALSE(matrix[2]->device_residues_current);
    }
    for(int i:{1,0})REQUIRE(rns8_destroy_prepared_execution(prepared[i])==RNS8_SUCCESS);
    for(auto* m:matrix)REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    for(auto* w:work)REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP wrap pipelines unavailable in this build");
#endif
}
