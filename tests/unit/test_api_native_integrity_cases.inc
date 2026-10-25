TEST_CASE("prepared native integrity is explicit and unavailable on the scalar provider", "[api][integrity]") {
  auto* ctx=create_cpu_context();auto d=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);
  rns8_plan* p=nullptr;rns8_workspace* w=nullptr;rns8_matrix *a=nullptr,*c=nullptr;
  REQUIRE(rns8_create_plan(ctx,&d,&p)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,p,&w)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(1,1,d.semantics);md.flags|=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
  const int64_t value=2;REQUIRE(rns8_pack_i64(ctx,a,&value,1,0)==RNS8_SUCCESS);
  rns8_prepared_execution* e=nullptr;REQUIRE(rns8_prepare_gemm(ctx,p,a,a,c,w,&e)==RNS8_SUCCESS);
  rns8_native_integrity_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;o.mode=1;o.max_scratch_bytes=7;
  REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_WORKSPACE_TOO_SMALL);
  o.max_scratch_bytes=8;REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_UNSUPPORTED_BACKEND);
  o.mode=0;REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(e)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(p)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("native sentinel gates ordinary, graph and native-group publication", "[api][hip][integrity]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  if(!hip_available())SKIP("no HIP device");
  for(bool u:{false,true}) {
    rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&co,&ctx)==RNS8_SUCCESS);
    auto d=exact_desc(u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED,1,1,1);d.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* p=nullptr;rns8_workspace* w=nullptr;rns8_matrix *a=nullptr,*c=nullptr;
    REQUIRE(rns8_create_plan(ctx,&d,&p)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,p,&w)==RNS8_SUCCESS);
    auto md=exact_matrix_desc(1,1,d.semantics);md.flags|=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    const uint64_t value=2;REQUIRE((u?rns8_pack_u64(ctx,a,&value,1,0):rns8_pack_i64(ctx,a,reinterpret_cast<const int64_t*>(&value),1,0))==RNS8_SUCCESS);
    rns8_prepared_execution* e=nullptr;REQUIRE(rns8_prepare_gemm(ctx,p,a,a,c,w,&e)==RNS8_SUCCESS);
    rns8_native_integrity_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;o.mode=1;o.max_scratch_bytes=8;
    REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_SUCCESS);
    REQUIRE(rns8_capture_prepared_hip_graph(e)==RNS8_SUCCESS);
    hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
    rns8_completion_token previous{};
    for(int mode=0;mode<3;++mode)for(bool corrupt:{false,true}) {
      // Authored inconsistent representation fixture: native value remains 2,
      // residue-domain value is 1. This is not OS/device fault injection.
      std::vector<int8_t> input(p->prefix,corrupt?1:2);
      REQUIRE(hip_direct_copy_host_to_device(0,a->hip_residues,input.data(),input.size())==RNS8_SUCCESS);
      rns8_completion_token token{};
      if(mode==0)REQUIRE(rns8_execute_prepared(e,&token)==(corrupt?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS));
      else {
        REQUIRE((mode==1?rns8_submit_prepared_hip(e,stream.get(),&token):rns8_submit_prepared_hip_graph(e,stream.get(),&token))==RNS8_SUCCESS);
        rns8_native_integrity_report busy{};busy.struct_size=sizeof(busy);busy.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_get_prepared_native_integrity_report(e,token,&busy)==RNS8_BUSY);
        REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_BUSY);
        rns8_completion_info ci{};ci.struct_size=sizeof(ci);ci.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_wait_prepared_hip(e,token,&ci)==RNS8_SUCCESS);
        REQUIRE(ci.operation_status==(corrupt?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS));REQUIRE(bool(ci.output_is_current)==!corrupt);
      }
      rns8_native_integrity_report r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
      if(previous.epoch)REQUIRE(rns8_get_prepared_native_integrity_report(e,previous,&r)==RNS8_INVALID_ARGUMENT);
      REQUIRE(rns8_get_prepared_native_integrity_report(e,token,&r)==RNS8_SUCCESS);
      REQUIRE(r.outcome==(corrupt?2u:1u));REQUIRE(r.failure_mask==(corrupt?12u:0u));
      REQUIRE(r.check_moduli[0]==257);REQUIRE(r.check_moduli[1]==263);REQUIRE(r.absolute_bound[0]==4);
      REQUIRE(r.a_generation==a->content_generation);REQUIRE(r.operation_id==token.prepared_id);REQUIRE(r.epoch==token.epoch);
      REQUIRE(bool(c->device_residues_current)==!corrupt);REQUIRE(!a->in_flight_readers);REQUIRE(!c->in_flight_writer);
      previous=token;
    }
    std::vector<int8_t> good(p->prefix,2);REQUIRE(hip_direct_copy_host_to_device(0,a->hip_residues,good.data(),good.size())==RNS8_SUCCESS);
    rns8_prepared_group* group=nullptr;REQUIRE(rns8_create_prepared_group(&e,1,&group)==RNS8_SUCCESS);
    rns8_native_group_binding binding{};binding.struct_size=sizeof(binding);binding.abi_version=RNS8_ABI_VERSION;binding.modulus_group=2;
    rns8_native_group_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.max_metadata_bytes=1<<20;gd.workers=1;gd.claim_chunk=1;
    const auto supported=rns8_prepare_native_group(group,&gd,&binding,1);
    if(supported==RNS8_SUCCESS) {
      rns8_completion_token token{};uint32_t submitted=0;
      REQUIRE(rns8_submit_native_group_hip(group,stream.get(),&token,1,&submitted)==RNS8_SUCCESS);REQUIRE(submitted==1);
      rns8_completion_info ci{};ci.struct_size=sizeof(ci);ci.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(e,token,&ci)==RNS8_SUCCESS);REQUIRE(ci.output_is_current);
      rns8_native_integrity_report r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_native_integrity_report(e,token,&r)==RNS8_SUCCESS);REQUIRE(r.outcome==1);
    } else REQUIRE(supported==RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);
    o.mode=0;REQUIRE(rns8_set_prepared_native_integrity(e,&o)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_execution(e)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(p)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
#else
  SKIP("HIP not compiled");
#endif
}
