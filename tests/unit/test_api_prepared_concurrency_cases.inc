#include <array>
TEST_CASE("caller graph reserves every body and bounded slots retain unconsumed epochs", "[api][async][graph][pool]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,2,2);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  auto md=exact_matrix_desc(2,2,RNS8_EXACT_WIDE_SIGNED);
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  rns8_matrix* a=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
  const int64_t values[]{-2,3,4,-5};REQUIRE(rns8_pack_i64(ctx,a,values,2,0)==RNS8_SUCCESS);
  std::array<rns8_matrix*,3> output{};std::array<rns8_workspace*,3> work{};
  std::array<rns8_prepared_execution*,3> prepared{};
  for(size_t i=0;i<3;++i) {
    REQUIRE(rns8_create_matrix(ctx,&md,&output[i])==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&work[i])==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plan,a,a,output[i],work[i],&prepared[i])==RNS8_SUCCESS);
    REQUIRE(rns8_capture_prepared_hip_graph(prepared[i])==RNS8_SUCCESS);
  }
  std::array<rns8::detail::hip_unique_stream,3> streams;
  for(auto& stream:streams)REQUIRE(stream.create_non_blocking()==hipSuccess);
  rns8_caller_hip_graph* caller=nullptr;hipGraph_t graph=nullptr;
  REQUIRE(hipStreamBeginCapture(streams[0].get(),hipStreamCaptureModeThreadLocal)==hipSuccess);
  REQUIRE(rns8_capture_prepared_group_in_caller_hip_graph(prepared.data(),3,streams[0].get(),&caller)==RNS8_SUCCESS);
  REQUIRE(hipStreamEndCapture(streams[0].get(),&graph)==hipSuccess);
  REQUIRE(rns8_finalize_caller_hip_graph(caller,graph)==RNS8_SUCCESS);REQUIRE(hipGraphDestroy(graph)==hipSuccess);
  std::array<rns8_completion_token,3> tokens{};uint32_t submitted=99;
  REQUIRE(rns8_submit_caller_hip_graph_batch(caller,streams[0].get(),tokens.data(),2,&submitted)==RNS8_INVALID_ARGUMENT);
  REQUIRE_FALSE(a->in_flight_readers);
  REQUIRE(rns8_submit_caller_hip_graph_batch(caller,streams[0].get(),tokens.data(),3,&submitted)==RNS8_SUCCESS);
  REQUIRE(submitted==3);REQUIRE(a->in_flight_readers==3);
  REQUIRE(rns8_destroy_caller_hip_graph(caller)==RNS8_BUSY);
  rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  for(int i:{2,0,1}) {
    REQUIRE(rns8_wait_prepared_hip(prepared[i],tokens[i],&info)==RNS8_SUCCESS);
    REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);
  }
  REQUIRE(rns8_destroy_caller_hip_graph(caller)==RNS8_SUCCESS);
  for(unsigned mode:{0u,1u}) {
    rns8_prepared_pool* pool=nullptr;REQUIRE(rns8_create_prepared_pool(prepared.data(),3,mode,&pool)==RNS8_SUCCESS);
    for(unsigned cycle=0;cycle<2;++cycle) {
      for(uint32_t i=0;i<3;++i) {
        uint32_t slot=99;REQUIRE(rns8_submit_prepared_pool_hip(pool,streams[i].get(),&slot,&tokens[i])==RNS8_SUCCESS);
        REQUIRE(slot==i);REQUIRE(rns8_refresh_prepared_inputs(prepared[i])==RNS8_BUSY);
      }
      uint32_t rejected_slot=99;rns8_completion_token rejected{77,88};
      REQUIRE(rns8_submit_prepared_pool_hip(pool,streams[0].get(),&rejected_slot,&rejected)==RNS8_BUSY);
      REQUIRE(rejected_slot==99);REQUIRE(rejected.prepared_id==77);REQUIRE(rejected.epoch==88);
      REQUIRE(rns8_release_prepared_pool_slot(pool,tokens[0])==RNS8_BUSY);
      if(cycle==1)REQUIRE(rns8_cancel_prepared_pool_hip(pool,tokens[1])==RNS8_SUCCESS);
      for(int i:{2,1,0}) {
        REQUIRE(rns8_wait_prepared_hip(prepared[i],tokens[i],&info)==RNS8_SUCCESS);
        REQUIRE(info.operation_status==((cycle==1 && i==1)?RNS8_CANCELED:RNS8_SUCCESS));
        REQUIRE(rns8_query_prepared_pool_hip(pool,tokens[i],&info)==RNS8_SUCCESS);
      }
      // Observing all completions does not silently recycle any slot.
      REQUIRE(rns8_submit_prepared_pool_hip(pool,streams[0].get(),&rejected_slot,&rejected)==RNS8_BUSY);
      REQUIRE(rns8_destroy_prepared_pool(pool)==RNS8_BUSY);
      for(int i:{1,0,2}) {
        REQUIRE(rns8_release_prepared_pool_slot(pool,tokens[i])==RNS8_SUCCESS);
        REQUIRE(rns8_query_prepared_pool_hip(pool,tokens[i],&info)==RNS8_INVALID_ARGUMENT);
        REQUIRE(rns8_refresh_prepared_pool_slot(pool,i)==RNS8_SUCCESS);
      }
    }
    REQUIRE(rns8_destroy_prepared_pool(pool)==RNS8_SUCCESS);
  }
  // Verify each noncanceled result against the independent exact reference.
  for(int i:{0,2}) {
    uint64_t result[4]{};REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,output[i],result,2,1)==RNS8_SUCCESS);
    for(unsigned r=0;r<2;++r)for(unsigned c=0;c<2;++c) {
      boost::multiprecision::cpp_int expected=0;
      for(unsigned k=0;k<2;++k)expected+=boost::multiprecision::cpp_int(values[r*2+k])*values[k*2+c];
      if(expected<0)expected+=boost::multiprecision::cpp_int(1)<<64;
      REQUIRE(result[r*2+c]==expected.convert_to<uint64_t>());
    }
  }
  for(size_t i=0;i<3;++i) {
    REQUIRE(rns8_destroy_prepared_execution(prepared[i])==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(output[i])==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(work[i])==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP concurrency unavailable in this build");
#endif
}

TEST_CASE("finite token pipelines preserve modulus and cancellation publication", "[api][async][finite][pipeline]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for(uint16_t modulus:{2,250,251,256})for(bool canceled:{false,true}) {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
    const auto semantics=modulus==251?RNS8_FINITE_FIELD_U8:RNS8_FINITE_RING_U8;
    auto desc=finite_gemm_desc(semantics,2,2,2,modulus);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    auto md=finite_matrix_desc(2,2,semantics);std::array<rns8_matrix*,3> matrix{};
    const uint8_t values[]{1,uint8_t(modulus-1),0,1};
    for(auto& m:matrix){REQUIRE(rns8_create_matrix(ctx,&md,&m)==RNS8_SUCCESS);REQUIRE(rns8_pack_finite_u8(ctx,m,modulus,values,2,0)==RNS8_SUCCESS);}
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
      uint8_t result[4]{};REQUIRE(rns8_export_finite_u8(ctx,plan,modulus,matrix[2],result,2)==RNS8_SUCCESS);
      REQUIRE(result[0]==1);REQUIRE(result[1]==(3*(modulus-1))%modulus);REQUIRE(result[2]==0);REQUIRE(result[3]==1);
    }
    for(int i:{1,0})REQUIRE(rns8_destroy_prepared_execution(prepared[i])==RNS8_SUCCESS);
    for(auto* m:matrix)REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    for(auto* w:work)REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP pipelines unavailable in this build");
#endif
}
