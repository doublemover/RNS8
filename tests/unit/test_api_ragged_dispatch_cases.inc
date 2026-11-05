namespace {
rns8_context* create_ragged_custody_context(rns8_backend_kind backend) {
  if(backend==RNS8_BACKEND_CPU_REFERENCE)return create_cpu_context();
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=backend;rns8_context* ctx=nullptr;
  REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);return ctx;
}
void check_ragged_constructor_custody(rns8_backend_kind backend) {
  for(const char* phase:{"ragged_discover","ragged_compact","ragged_prepare_publish"})
    for(bool throws:{false,true})for(bool uncertain:{false,true}){
      CAPTURE(backend,phase,throws,uncertain);
      auto* ctx=create_ragged_custody_context(backend);
      auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);rns8_matrix* source=nullptr;
      REQUIRE(rns8_create_matrix(ctx,&md,&source)==RNS8_SUCCESS);
      const int64_t value=-7;REQUIRE(rns8_pack_i64(ctx,source,&value,1,1)==RNS8_SUCCESS);
      const auto generation=source->content_generation;
      SynchronousDispatchFixture state;state.readers={source};state.owner_context=ctx;
      state.target_operation=phase;state.fail=true;state.throw_after_issue=throws;state.drain_fails=uncertain;
      const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks=&hooks;rns8_ragged_materialization* owner=nullptr;
      CHECK(rns8_prepare_ragged_materialization(ctx,source,1,1,UINT64_MAX,&owner)==
          (throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));
      CHECK(owner==nullptr);CHECK(state.reservations_observed);CHECK(source->content_generation==generation);
      CHECK((source->host_residues_current || source->device_residues_current));
      if(uncertain){
        CHECK(ctx->hip_failure_poisoned);REQUIRE(ctx->failed_ragged_cleanup!=nullptr);
        CHECK(source->in_flight_readers==1);CHECK(source->prepared_references==1);
        CHECK(ctx->prepared_references==0); // unpublished ownership is on the context's cleanup list
        REQUIRE(ctx->failed_work_leases.size()==1);
        for(auto* buffer:ctx->failed_work_leases.front().buffer_holds){REQUIRE(buffer!=nullptr);CHECK(buffer->pending());}
        CHECK(rns8_destroy_matrix(source)==RNS8_BUSY);CHECK(rns8_destroy_context(ctx)==RNS8_BACKEND_FAILURE);
        state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
      }
      CHECK(ctx->failed_ragged_cleanup==nullptr);CHECK(ctx->failed_work_leases.empty());
      CHECK(source->prepared_references==0);CHECK(source->in_flight_readers==0);CHECK(ctx->prepared_references==0);
      ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_matrix(source)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
    }
}
void check_ragged_materialization_custody(rns8_backend_kind backend) {
  for(unsigned phase=0;phase<3;++phase)for(bool throws:{false,true})for(bool uncertain:{false,true}){
    CAPTURE(backend,phase,throws,uncertain);
    auto* ctx=create_ragged_custody_context(backend);
    auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);rns8_matrix *source=nullptr,*output=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&source)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&output)==RNS8_SUCCESS);
    const int64_t value=-7,prior=9;REQUIRE(rns8_pack_i64(ctx,source,&value,1,1)==RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx,output,&prior,1,1)==RNS8_SUCCESS);
    rns8_ragged_materialization* owner=nullptr;
    REQUIRE(rns8_prepare_ragged_materialization(ctx,source,1,1,UINT64_MAX,&owner)==RNS8_SUCCESS);
    const auto generation=output->content_generation;const auto old=output->residues;
    SynchronousDispatchFixture state;state.readers={source};state.writers={output};state.owner_context=ctx;
    state.target_operation=phase==2?"ragged_materialize_publish":"ragged_materialize";
    state.write_started_on_failure=phase==1;state.fail=true;state.throw_after_issue=throws;state.drain_fails=uncertain;
    const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
    ctx->synchronous_test_hooks=&hooks;
    CHECK(rns8_materialize_ragged_exact(ctx,owner,output,UINT64_MAX)==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));
    CHECK(state.reservations_observed);CHECK(output->content_generation==generation);
    CHECK((output->host_residues_current || output->device_residues_current)==(phase==0));
    if(phase<2)CHECK(output->residues==old);
    if(uncertain){
      CHECK(ctx->hip_failure_poisoned);CHECK(source->in_flight_readers==1);CHECK(output->in_flight_writer);
      CHECK(rns8_destroy_ragged_materialization(owner)==RNS8_BUSY);
      CHECK(rns8_destroy_matrix(source)==RNS8_BUSY);CHECK(rns8_destroy_matrix(output)==RNS8_BUSY);
      REQUIRE(ctx->failed_work_leases.size()==1);
      for(auto* buffer:ctx->failed_work_leases.front().buffer_holds){REQUIRE(buffer!=nullptr);CHECK(buffer->pending());}
      state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
    }
    CHECK(source->in_flight_readers==0);CHECK_FALSE(output->in_flight_writer);CHECK(ctx->failed_work_leases.empty());
    ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_ragged_materialization(owner)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(source)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(output)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
}
TEST_CASE("ragged host construction shares failure-retaining source and buffer custody", "[api][ragged][lifetime]") {
  check_ragged_constructor_custody(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("ragged host materialization retains its owner through publication", "[api][ragged][lifetime]") {
  check_ragged_materialization_custody(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("ragged HIP construction and materialization share failure-retaining custody", "[api][ragged][hip][lifetime]") {
  if(!rns8::detail::api::direct_hip_compiled())SKIP("direct HIP ragged provider not compiled");
  check_ragged_constructor_custody(RNS8_BACKEND_HIP_DIRECT);
  check_ragged_materialization_custody(RNS8_BACKEND_HIP_DIRECT);
}
