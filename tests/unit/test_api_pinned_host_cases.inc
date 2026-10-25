#include "backend_common/pinned_host_contract.hpp"
TEST_CASE("pinned native host rectangles and budgets are checked without HIP", "[api][host-transfer][contract]") {
  using namespace rns8::detail;rns8_pinned_host_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.element_type=RNS8_NATIVE_DEVICE_I64;d.access=RNS8_NATIVE_DEVICE_READ;d.rows=3;d.cols=5;d.ld=8;
  uint64_t bytes=99;REQUIRE(rns8_query_pinned_host_bytes(&d,&bytes)==RNS8_SUCCESS);CHECK(bytes==192);
  PinnedHostResources r;REQUIRE(query_pinned_host_contract(d,r)==RNS8_SUCCESS);CHECK(r.compact==120);
  uint64_t device=0,pinned=0,total=0;REQUIRE(pinned_host_transfer_bytes(r,20,true,true,device,pinned,total));
  CHECK(device==120+300+16);CHECK(pinned==196);CHECK(total==device+pinned+120);
  const uint64_t old_device=device,old_pinned=pinned,old_total=total;
  for(const auto invalid:{PinnedHostResources{192,120,0},PinnedHostResources{192,121,8},
      PinnedHostResources{100,120,8},PinnedHostResources{UINT64_MAX,UINT64_MAX-7,8},
      PinnedHostResources{UINT64_MAX-8,UINT64_MAX-15,1}}){
    CHECK_FALSE(pinned_host_transfer_bytes(invalid,20,true,true,device,pinned,total));
    CHECK(device==old_device);CHECK(pinned==old_pinned);CHECK(total==old_total);
  }
  d.ld=4;CHECK(rns8_query_pinned_host_bytes(&d,&bytes)==RNS8_INVALID_ARGUMENT);CHECK(bytes==192);
  d.ld=INT64_MAX;CHECK(rns8_query_pinned_host_bytes(&d,&bytes)==RNS8_RANGE_ERROR);
  d.ld=8;d.element_type=RNS8_NATIVE_DEVICE_SIGNED_LIMBS;d.limb_count=3;
  CHECK(rns8_query_pinned_host_bytes(&d,&bytes)==RNS8_INVALID_ARGUMENT);d.access=RNS8_NATIVE_DEVICE_WRITE;
  REQUIRE(rns8_query_pinned_host_bytes(&d,&bytes)==RNS8_SUCCESS);CHECK(bytes==576);
  auto* cpu=create_cpu_context();rns8_pinned_host_buffer* owner=reinterpret_cast<rns8_pinned_host_buffer*>(uintptr_t(1));
  CHECK(rns8_create_pinned_host_buffer_hip(cpu,&d,575,&owner)==RNS8_WORKSPACE_TOO_SMALL);CHECK(owner==nullptr);
  CHECK(rns8_create_pinned_host_buffer_hip(cpu,&d,576,&owner)==RNS8_UNSUPPORTED_BACKEND);CHECK(owner==nullptr);
  CHECK(rns8_destroy_pinned_host_buffer(nullptr)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(cpu)==RNS8_SUCCESS);
}
TEST_CASE("owned pinned imports and exports share prepared dependency and graph publication", "[api][hip][host-transfer]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;if(hip_direct_probe(0,device)!=RNS8_SUCCESS)SKIP("no HIP device");
  for(bool graph:{false,true}){
    rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&co,&ctx)==RNS8_SUCCESS);
    rns8_pinned_host_desc hd{};hd.struct_size=sizeof(hd);hd.abi_version=RNS8_ABI_VERSION;hd.rows=3;hd.cols=5;hd.ld=8;hd.element_type=RNS8_NATIVE_DEVICE_I64;hd.access=RNS8_NATIVE_DEVICE_READ;
    rns8_pinned_host_buffer *input=nullptr,*output=nullptr;REQUIRE(rns8_create_pinned_host_buffer_hip(ctx,&hd,192,&input)==RNS8_SUCCESS);
    hd.cols=2;hd.ld=7;hd.element_type=RNS8_NATIVE_DEVICE_SIGNED_LIMBS;hd.limb_count=3;hd.access=RNS8_NATIVE_DEVICE_WRITE;
    REQUIRE(rns8_create_pinned_host_buffer_hip(ctx,&hd,504,&output)==RNS8_SUCCESS);
    auto write=[&](uint64_t generation,bool bad){void* data=nullptr;uint64_t bytes=0;REQUIRE(rns8_begin_pinned_host_write(input,&data,&bytes)==RNS8_SUCCESS);CHECK(bytes==192);
      std::fill_n(static_cast<int64_t*>(data),24,99);for(int r=0;r<3;++r)for(int c=0;c<5;++c)static_cast<int64_t*>(data)[r*8+c]=bad && !r && !c?8:(r%2?-7:7);
      REQUIRE(rns8_end_pinned_host_write(input,generation)==RNS8_SUCCESS);};write(1,false);
    void* padded=nullptr;uint64_t padded_bytes=0;REQUIRE(rns8_begin_pinned_host_write(output,&padded,&padded_bytes)==RNS8_SUCCESS);std::fill_n(static_cast<uint64_t*>(padded),63,91);REQUIRE(rns8_end_pinned_host_write(output,1)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr,*copy=nullptr;auto md=exact_matrix_desc(3,5,RNS8_EXACT_WIDE_SIGNED);md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);md.flags=0;REQUIRE(rns8_create_matrix(ctx,&md,&copy)==RNS8_SUCCESS);md=exact_matrix_desc(5,2,RNS8_EXACT_WIDE_SIGNED);REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);md=exact_matrix_desc(3,2,RNS8_EXACT_WIDE_SIGNED);REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    int64_t twos[10];std::fill_n(twos,10,2);REQUIRE(rns8_pack_i64(ctx,b,twos,2,1)==RNS8_SUCCESS);
    rns8_native_transfer_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;o.import_values=1;o.verify_input_bound=1;o.max_input_magnitude=7;o.max_total_scratch_bytes=UINT64_MAX;
    rns8_prepared_execution *load=nullptr,*load_copy=nullptr,*compute=nullptr,*store=nullptr;
    o.max_total_scratch_bytes=751;CHECK(rns8_prepare_host_transfer_hip(ctx,a,input,&o,&load)==RNS8_WORKSPACE_TOO_SMALL);CHECK(load==nullptr);o.max_total_scratch_bytes=UINT64_MAX;
    REQUIRE(rns8_prepare_host_transfer_hip(ctx,a,input,&o,&load)==RNS8_SUCCESS);REQUIRE(rns8_prepare_host_transfer_hip(ctx,copy,input,&o,&load_copy)==RNS8_SUCCESS);
    rns8_completion_token initial{};REQUIRE(rns8_execute_prepared(load,&initial)==RNS8_SUCCESS);
    auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,3,2,5);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plan,a,b,c,work,&compute)==RNS8_SUCCESS);REQUIRE(rns8_execute_prepared(compute,&initial)==RNS8_SUCCESS);REQUIRE(rns8_prepare_hip_submission(compute)==RNS8_SUCCESS);
    o.import_values=0;o.verify_input_bound=0;o.max_input_magnitude=0;REQUIRE(rns8_prepare_host_transfer_hip(ctx,c,output,&o,&store)==RNS8_SUCCESS);
    hip_unique_stream s1,s2,s3;REQUIRE(s1.create_non_blocking()==hipSuccess);REQUIRE(s2.create_non_blocking()==hipSuccess);REQUIRE(s3.create_non_blocking()==hipSuccess);
    if(graph)for(auto* e:{load,compute,store})REQUIRE(rns8_capture_prepared_hip_graph(e)==RNS8_SUCCESS);
    for(bool bad:{false,true}){
      write(bad?3:2,bad);REQUIRE(rns8_refresh_prepared_inputs(load)==RNS8_SUCCESS);
      rns8_completion_token l{},g{},e{};REQUIRE((graph?rns8_submit_prepared_hip_graph(load,s1.get(),&l):rns8_submit_prepared_hip(load,s1.get(),&l))==RNS8_SUCCESS);
      void* rejected=reinterpret_cast<void*>(uintptr_t(1));uint64_t bytes=123;CHECK(rns8_begin_pinned_host_write(input,&rejected,&bytes)==RNS8_BUSY);CHECK(rejected==reinterpret_cast<void*>(uintptr_t(1)));CHECK(bytes==123);
      REQUIRE(rns8_bind_prepared_dependency(compute,load,l)==RNS8_SUCCESS);REQUIRE((graph?rns8_submit_prepared_hip_graph_pipeline(compute,s2.get(),&g):rns8_submit_prepared_hip_pipeline(compute,s2.get(),&g))==RNS8_SUCCESS);
      REQUIRE(rns8_bind_prepared_dependency(store,compute,g)==RNS8_SUCCESS);REQUIRE((graph?rns8_submit_prepared_hip_graph_pipeline(store,s3.get(),&e):rns8_submit_prepared_hip_pipeline(store,s3.get(),&e))==RNS8_SUCCESS);
      const void* result=nullptr;CHECK(rns8_begin_pinned_host_read(output,&result,&bytes)==RNS8_BUSY);CHECK(rns8_destroy_pinned_host_buffer(output)==RNS8_BUSY);CHECK(rns8_destroy_prepared_execution(store)==RNS8_BUSY);
      rns8_completion_info completed{};completed.struct_size=sizeof(completed);completed.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_wait_prepared_hip(store,e,&completed)==RNS8_SUCCESS);
      CHECK(completed.operation_status==(bad?RNS8_RANGE_ERROR:RNS8_SUCCESS));CHECK(bool(completed.output_is_current)==!bad);
      if(bad){CHECK(rns8_begin_pinned_host_read(output,&result,&bytes)==RNS8_INVALID_ARGUMENT);}
      else{REQUIRE(rns8_begin_pinned_host_read(output,&result,&bytes)==RNS8_SUCCESS);const auto* words=static_cast<const uint64_t*>(result);
        for(int r=0;r<3;++r)for(int col=0;col<7;++col)for(int limb=0;limb<3;++limb){const uint64_t expected=col>=2?91:limb==0?uint64_t(r%2?-70:70):r%2?UINT64_MAX:0;CHECK(words[(r*7+col)*3+limb]==expected);}
        CHECK(rns8_begin_pinned_host_write(output,&rejected,&bytes)==RNS8_BUSY);REQUIRE(rns8_end_pinned_host_read(output)==RNS8_SUCCESS);}
    }
    // Two independent H2D readers retain one pinned input through reverse waits.
    write(4,false);REQUIRE(rns8_refresh_prepared_inputs(load)==RNS8_SUCCESS);REQUIRE(rns8_refresh_prepared_inputs(load_copy)==RNS8_SUCCESS);
    rns8_completion_token l1{},l2{};REQUIRE(rns8_submit_prepared_hip(load,s1.get(),&l1)==RNS8_SUCCESS);REQUIRE(rns8_submit_prepared_hip(load_copy,s2.get(),&l2)==RNS8_SUCCESS);
    rns8_pinned_host_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_get_pinned_host_info(input,&info)==RNS8_SUCCESS);CHECK(info.device_readers==2);
    rns8_completion_info done{};done.struct_size=sizeof(done);done.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_wait_prepared_hip(load_copy,l2,&done)==RNS8_SUCCESS);CHECK(done.operation_status==RNS8_SUCCESS);
    REQUIRE(rns8_get_pinned_host_info(input,&info)==RNS8_SUCCESS);CHECK(info.device_readers==1);REQUIRE(rns8_wait_prepared_hip(load,l1,&done)==RNS8_SUCCESS);
    for(auto* e:{store,compute,load_copy,load})REQUIRE(rns8_destroy_prepared_execution(e)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_pinned_host_buffer(output)==RNS8_SUCCESS);REQUIRE(rns8_destroy_pinned_host_buffer(input)==RNS8_SUCCESS);
    for(auto* x:{copy,c,b,a})REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP not compiled");
#endif
}
