namespace {
void check_basis_dispatch_custody(rns8_backend_kind backend) {
  for(const std::string operation:{"pack","gemm","convert","export"})for(bool sign:{false,true})
    for(unsigned phase=0;phase<5;++phase)for(bool throws:{false,true})for(bool uncertain:{false,true}){
      if(operation=="export" && phase>0 && phase<4)continue;
      CAPTURE(backend,operation,sign,phase,throws,uncertain);
      auto* ctx=create_ragged_custody_context(backend);
      const auto semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
      auto* ordered=experimental_basis_fixture::create(experimental_basis_fixture::moduli());
      auto* ordinary=experimental_basis_fixture::plan(ctx,nullptr,semantics,1,1,1,20);
      auto* converted=experimental_basis_fixture::plan(ctx,ordered,semantics,1,1,1,20);
      auto* plan=operation=="convert"?converted:ordinary;
      auto* a=experimental_basis_fixture::matrix(ctx,nullptr,semantics,1,1);
      auto* b=experimental_basis_fixture::matrix(ctx,nullptr,semantics,1,1);
      auto* c=experimental_basis_fixture::matrix(ctx,operation=="convert"?ordered:nullptr,semantics,1,1);
      const int64_t signed_value=INT64_MIN,signed_one=1,signed_prior=9;
      const uint64_t unsigned_value=UINT64_MAX,unsigned_one=1,unsigned_prior=9;
      REQUIRE((sign?rns8_pack_basis_i64(ordinary,a,&signed_value,1,1):rns8_pack_basis_u64(ordinary,a,&unsigned_value,1,1))==RNS8_SUCCESS);
      REQUIRE((sign?rns8_pack_basis_i64(ordinary,b,&signed_one,1,1):rns8_pack_basis_u64(ordinary,b,&unsigned_one,1,1))==RNS8_SUCCESS);
      REQUIRE((sign?rns8_pack_basis_i64(plan,c,&signed_prior,1,1):rns8_pack_basis_u64(plan,c,&unsigned_prior,1,1))==RNS8_SUCCESS);
      const auto generation=c->content_generation,source_generation=a->content_generation;
      const auto residues=c->residues,source_residues=a->residues;
      const std::string initial="basis_"+operation;
      const char* target=phase==0?initial.c_str():phase<3?"basis_commit":
          phase==3?"basis_commit_publish":"basis_transaction_publish";
      SynchronousDispatchFixture state;state.fail=true;state.throw_after_issue=throws;state.drain_fails=uncertain;
      state.target_operation=target;state.write_started_on_failure=phase==2;state.workspace_context=ctx;
      if(operation!="pack")state.readers={a};if(operation=="gemm")state.readers.push_back(b);
      if(operation!="export")state.writers={c};
      const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks=&hooks;std::array<uint64_t,8> output;output.fill(99);const auto original_output=output;
      auto execute=[&](){return operation=="pack"?(sign?rns8_pack_basis_i64(plan,c,&signed_value,1,2):rns8_pack_basis_u64(plan,c,&unsigned_value,1,2)):
          operation=="gemm"?rns8_gemm_basis(plan,a,b,c):operation=="convert"?rns8_convert_basis_values(plan,a,c):
          rns8_export_basis_limbs(plan,a,output.data()+1,3,3);};
      const auto status=execute();
      CHECK(status==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));CHECK(state.reservations_observed);
      CHECK(c->content_generation==generation);CHECK(a->content_generation==source_generation);CHECK(a->residues==source_residues);
      CHECK((a->host_residues_current || a->device_residues_current));
      CHECK((c->host_residues_current || c->device_residues_current)==(operation=="export" || phase<2));
      if(phase<3 || operation=="export")CHECK(c->residues==residues);
      CHECK(output==original_output);
      if(uncertain){
        CHECK(ctx->hip_failure_poisoned);CHECK(rns8_destroy_basis_plan(plan)==RNS8_BUSY);
        REQUIRE(ctx->failed_work_leases.size()==1);const auto& record=ctx->failed_work_leases.front();
        REQUIRE(record.workspace!=nullptr);CHECK(record.workspace->in_flight);
        REQUIRE(record.buffer_holds[0]!=nullptr);CHECK(record.buffer_holds[0]->pending());
        for(auto* source:state.readers)CHECK(rns8_destroy_matrix(const_cast<rns8_matrix*>(source))==RNS8_BUSY);
        if(operation!="export")CHECK(rns8_destroy_matrix(c)==RNS8_BUSY);
        state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
      }
      CHECK(ctx->failed_work_leases.empty());CHECK(a->in_flight_readers==0);CHECK(b->in_flight_readers==0);CHECK_FALSE(c->in_flight_writer);
      ctx->synchronous_test_hooks=nullptr;
      if(!uncertain){
        REQUIRE(execute()==RNS8_SUCCESS); // packet metadata must also recover after an exception
        REQUIRE(rns8_export_basis_limbs(plan,operation=="export"?a:c,output.data()+1,3,3)==RNS8_SUCCESS);
        CHECK(output[1]==(sign?static_cast<uint64_t>(INT64_MIN):UINT64_MAX));
        CHECK(output[2]==(sign?UINT64_MAX:0));CHECK(output[3]==(sign?UINT64_MAX:0));
        CHECK(output[0]==99);for(size_t index=4;index<output.size();++index)CHECK(output[index]==99);
      }
      REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_basis_plan(converted)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis_plan(ordinary)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_basis(ordered)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
    }
}
}
TEST_CASE("ordered-basis host operations retain staged words and shared completion custody", "[api][basis][lifetime]") {
  check_basis_dispatch_custody(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("ordered-basis HIP operations retain staged words and shared completion custody", "[api][basis][hip][lifetime]") {
  if(!rns8::detail::api::direct_hip_compiled())SKIP("direct HIP ordered-basis provider not compiled");
  check_basis_dispatch_custody(RNS8_BACKEND_HIP_DIRECT);
}
