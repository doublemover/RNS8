TEST_CASE("one completion system composes GEMM, value-preserving lift and dependent GEMM", "[api][hip][lift][pipeline]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;if(!hip_available())SKIP("no HIP device");
  rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&co,&ctx)==RNS8_SUCCESS);
  rns8_matrix *a=nullptr,*c=nullptr,*out=nullptr;auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);md.flags=0;
  REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&out)==RNS8_SUCCESS);
  int64_t two=2;REQUIRE(rns8_pack_i64(ctx,a,&two,1,1)==RNS8_SUCCESS);
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;desc.max_prefix=3;
  rns8_plan *p3=nullptr,*p9=nullptr;rns8_workspace *w3=nullptr,*w9=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&p3)==RNS8_SUCCESS);desc.max_prefix=9;REQUIRE(rns8_create_plan(ctx,&desc,&p9)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,p3,&w3)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,p9,&w9)==RNS8_SUCCESS);
  rns8_prepared_execution *producer=nullptr,*lift=nullptr,*consumer=nullptr;
  REQUIRE(rns8_prepare_gemm(ctx,p3,a,a,c,w3,&producer)==RNS8_SUCCESS);
  rns8_native_integrity_options integrity{};integrity.struct_size=sizeof(integrity);integrity.abi_version=RNS8_ABI_VERSION;integrity.mode=1;integrity.max_scratch_bytes=8;
  REQUIRE(rns8_set_prepared_native_integrity(producer,&integrity)==RNS8_SUCCESS);
  rns8_completion_token first{};REQUIRE(rns8_execute_prepared(producer,&first)==RNS8_SUCCESS);REQUIRE(c->materialized_rns_prefix==3);
  rns8_exact_lift_options lo{};lo.struct_size=sizeof(lo);lo.abi_version=RNS8_ABI_VERSION;lo.target_prefix=9;lo.algorithm=1;lo.max_total_scratch_bytes=14;
  REQUIRE(rns8_prepare_exact_lift_hip(ctx,c,&lo,&lift)==RNS8_SUCCESS);
  const auto generation=c->content_generation;REQUIRE(rns8_execute_prepared(lift,&first)==RNS8_SUCCESS);REQUIRE(c->content_generation==generation);
  REQUIRE(rns8_prepare_gemm(ctx,p9,c,a,out,w9,&consumer)==RNS8_SUCCESS);REQUIRE(rns8_prepare_hip_submission(consumer)==RNS8_SUCCESS);
  hip_unique_stream s1,s2,s3;REQUIRE(s1.create_non_blocking()==hipSuccess);REQUIRE(s2.create_non_blocking()==hipSuccess);REQUIRE(s3.create_non_blocking()==hipSuccess);
  for(bool corrupt:{false,true}) {
    std::vector<int8_t> input(a->prefix,corrupt?1:2);REQUIRE(hip_direct_copy_host_to_device(0,a->hip_residues,input.data(),input.size())==RNS8_SUCCESS);
    rns8_completion_token p{},l{},q{};REQUIRE(rns8_submit_prepared_hip(producer,s1.get(),&p)==RNS8_SUCCESS);
    REQUIRE(rns8_bind_prepared_dependency(lift,producer,p)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_pipeline(lift,s2.get(),&l)==RNS8_SUCCESS);
    REQUIRE(c->prepared_in_flight_writers==2);REQUIRE(c->in_flight_writer);
    REQUIRE(rns8_bind_prepared_dependency(consumer,lift,l)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_pipeline(consumer,s3.get(),&q)==RNS8_SUCCESS);
    REQUIRE(c->in_flight_readers==1);REQUIRE(rns8_destroy_prepared_execution(lift)==RNS8_BUSY);
    rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(consumer,q,&completion)==RNS8_SUCCESS);
    REQUIRE(completion.operation_status==(corrupt?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS));REQUIRE(bool(completion.output_is_current)==!corrupt);
    REQUIRE(!c->prepared_in_flight_writers);REQUIRE(!c->in_flight_writer);REQUIRE(!c->in_flight_readers);
    if(!corrupt){uint64_t limbs[3]{};REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,p9,out,limbs,1,3)==RNS8_SUCCESS);REQUIRE(limbs[0]==8);REQUIRE(limbs[1]==0);REQUIRE(limbs[2]==0);REQUIRE(c->exact_range_prefix==9);}
    else {REQUIRE(!c->device_residues_current);REQUIRE(!out->device_residues_current);}
  }
  REQUIRE(rns8_destroy_prepared_execution(consumer)==RNS8_SUCCESS);REQUIRE(rns8_destroy_prepared_execution(lift)==RNS8_SUCCESS);REQUIRE(rns8_destroy_prepared_execution(producer)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(out)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(w9)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(w3)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(p9)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(p3)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP not compiled");
#endif
}
