TEST_CASE("lease-owned prewrite checkpoint preserves or invalidates output across unwinding", "[api][lifetime]") {
  using namespace rns8::detail::api;
  for(bool late:{false,true})for(bool throws:{false,true})for(bool uncertain:{false,true}){
    CAPTURE(late,throws,uncertain);
    auto* ctx=create_cpu_context();auto md=bounded_matrix_desc(1,1,RNS8_BOUNDED_I64);
    rns8_matrix* output=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&output)==RNS8_SUCCESS);
    const int64_t value=7;REQUIRE(rns8_pack_i64(ctx,output,&value,1,1)==RNS8_SUCCESS);
    const auto generation=output->content_generation;const auto residues=output->residues;
    SynchronousDispatchFixture state;state.writers={output};state.fail=true;
    state.write_started_on_failure=late;state.throw_after_issue=throws;state.drain_fails=uncertain;
    const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
    ctx->synchronous_test_hooks=&hooks;
    const auto status=guard_api([&](){
      size_t token=0;auto admitted=begin_synchronous_output_lease(*ctx,*output,token);
      if(admitted!=RNS8_SUCCESS)return admitted;
      SynchronousLease lease(*ctx,token);
      return lease.finish(lease.checkpoint_before_write("host_prewrite"));
    });
    CHECK(status==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));CHECK(state.reservations_observed);
    CHECK(output->content_generation==generation);CHECK(output->residues==residues);
    CHECK(output->host_residues_current==!late);
    if(uncertain){
      CHECK(output->in_flight_writer);CHECK(rns8_destroy_matrix(output)==RNS8_BUSY);
      state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
    }
    CHECK_FALSE(output->in_flight_writer);CHECK(ctx->failed_work_leases.empty());
    ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_matrix(output)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}

TEST_CASE("public prepacked GEMM retains cache and workspace buffers across failed dispatch", "[api][prepack][hip][lifetime]") {
  if(!rns8::detail::rocwmma_compiled())SKIP("rocWMMA prepacked GEMM provider not compiled");
  for(bool late:{false,true})for(bool throws:{false,true})for(bool uncertain:{false,true}){
    CAPTURE(late,throws,uncertain);
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_ROCWMMA;rns8_context* ctx=nullptr;
    const auto created=rns8_create_context(0,&options,&ctx);
    if(created==RNS8_UNSUPPORTED_BACKEND || created==RNS8_UNSUPPORTED_ARCH)SKIP("no applicable rocWMMA device");
    REQUIRE(created==RNS8_SUCCESS);
    auto desc=bounded_gemm_desc(RNS8_BOUNDED_I64,2,2,2);desc.requested_backend=options.requested_backend;
    desc.bound=10000;rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    auto md=bounded_matrix_desc(2,2,desc.semantics);rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    for(auto** matrix:{&a,&b,&c})REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
    const int64_t av[]{1,2,3,4},bv[]{1,0,0,1};
    REQUIRE(rns8_pack_i64(ctx,a,av,2,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(ctx,b,bv,2,1)==RNS8_SUCCESS);
    rns8_prepack_cache* cache=nullptr;REQUIRE(rns8_create_prepack_cache(ctx,plan,b,RNS8_OPERAND_B,&cache)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_prepacked_b(ctx,plan,a,cache,c,workspace)==RNS8_SUCCESS);
    const auto generation=c->content_generation;
    SynchronousDispatchFixture state;state.readers={a};state.writers={c};state.workspace=workspace;
    state.private_owners={&cache->in_flight_readers};state.fail=true;state.write_started_on_failure=late;
    state.throw_after_issue=throws;state.drain_fails=uncertain;state.target_operation="gemm_prepacked_b";
    const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
    ctx->synchronous_test_hooks=&hooks;
    CHECK(rns8_gemm_rns_prepacked_b(ctx,plan,a,cache,c,workspace)==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));
    CHECK(state.calls==1);CHECK(state.reservations_observed);CHECK(c->content_generation==generation);
    CHECK(c->device_residues_current==!late);
    if(uncertain){
      CHECK(cache->data_owner.pending());CHECK(workspace->accelerator_workspace_owner.pending());
      CHECK(rns8_destroy_prepack_cache(cache)==RNS8_BUSY);CHECK(rns8_destroy_workspace(workspace)==RNS8_BUSY);
      CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);CHECK(rns8_destroy_matrix(c)==RNS8_BUSY);
      state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
    }
    CHECK(cache->in_flight_readers==0);CHECK_FALSE(cache->data_owner.pending());CHECK_FALSE(workspace->accelerator_workspace_owner.pending());
    ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_prepack_cache(cache)==RNS8_SUCCESS);
    for(auto* matrix:{a,b,c})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}

TEST_CASE("public incremental GEMM retains snapshot custody through each copy and region phase", "[api][result-cache][hip][lifetime]") {
  using namespace rns8::detail::api;
  if(!direct_hip_compiled())SKIP("direct HIP result-cache provider not compiled");
  for(bool finite:{false,true})for(unsigned phase=0;phase<9;++phase)for(bool throws:{false,true})for(bool uncertain:{false,true}){
    CAPTURE(finite,phase,throws,uncertain);
    const bool initial=phase<2,region=phase>=5;
    const char* target=initial?(phase==0?"incremental_snapshot":"incremental_snapshot_publish"):
        phase<4?"incremental_restore":phase==4?"incremental_restore_publish":
        phase==5?"incremental_region":phase==6?"incremental_region_compute":
        phase==7?"incremental_snapshot":"incremental_snapshot_publish";
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* ctx=nullptr;
    REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
    const auto semantics=finite?RNS8_FINITE_RING_U8:RNS8_BOUNDED_I64;
    auto desc=finite?finite_gemm_desc(semantics,2,2,2,251):bounded_gemm_desc(semantics,2,2,2);
    desc.requested_backend=options.requested_backend;if(!finite)desc.bound=10000;
    rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    auto md=finite?finite_matrix_desc(2,2,semantics):bounded_matrix_desc(2,2,semantics);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    for(auto** matrix:{&a,&b,&c})REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
    const int64_t av[]{1,2,3,4},bv[]{1,0,0,1};const uint8_t af[]{1,2,3,4},bf[]{1,0,0,1};
    REQUIRE((finite?rns8_pack_finite_u8(ctx,a,251,af,2,1):rns8_pack_i64(ctx,a,av,2,1))==RNS8_SUCCESS);
    REQUIRE((finite?rns8_pack_finite_u8(ctx,b,251,bf,2,1):rns8_pack_i64(ctx,b,bv,2,1))==RNS8_SUCCESS);
    rns8_result_cache_desc cd{};cd.struct_size=sizeof(cd);cd.abi_version=RNS8_ABI_VERSION;
    cd.flags=RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES|RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY|
        RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE|RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
    rns8_result_cache* cache=nullptr;REQUIRE(rns8_create_result_cache(ctx,plan,&cd,&cache)==RNS8_SUCCESS);
    rns8_dirty_region dirty{};dirty.struct_size=sizeof(dirty);dirty.abi_version=RNS8_ABI_VERSION;dirty.row_extent=dirty.col_extent=1;
    auto execute=[&](bool partial){return finite?
        rns8_gemm_finite_u8_incremental(ctx,plan,251,a,b,c,workspace,cache,partial?&dirty:nullptr,partial?1:0):
        rns8_gemm_rns_incremental(ctx,plan,a,b,c,workspace,cache,partial?&dirty:nullptr,partial?1:0);};
    if(!initial)REQUIRE(execute(false)==RNS8_SUCCESS);
    const auto generation=c->content_generation;
    SynchronousDispatchFixture state;state.fail=true;state.throw_after_issue=throws;state.drain_fails=uncertain;
    state.target_operation=target;state.private_owners={&cache->in_flight_uses};state.write_started_on_failure=phase==3;
    if(initial)state.readers={c};else{state.readers={a,b};state.writers={c};state.workspace=workspace;}
    const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
    ctx->synchronous_test_hooks=&hooks;
    CHECK(execute(region)==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));CHECK(state.reservations_observed);
    const bool preserved=initial || phase==2 || phase==5;
    CHECK(c->device_residues_current==preserved);
    CHECK(c->content_generation==generation+(initial || phase>=7));
    CHECK_FALSE(cache->last_cache_hit);
    if(initial || phase==8)CHECK_FALSE(cache->initialized);
    if(uncertain){
      CHECK(ctx->hip_failure_poisoned);CHECK(cache->in_flight_uses==1);
      if(phase!=5)CHECK(cache->snapshot_owner.pending());
      CHECK(rns8_destroy_result_cache(cache)==RNS8_BUSY);CHECK(rns8_invalidate_result_cache(cache)==RNS8_BUSY);
      CHECK(rns8_destroy_matrix(c)==RNS8_BUSY);
      if(!initial){CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);CHECK(rns8_destroy_workspace(workspace)==RNS8_BUSY);}
      state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
    }
    CHECK(cache->in_flight_uses==0);CHECK_FALSE(cache->snapshot_owner.pending());CHECK(ctx->failed_work_leases.empty());
    ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_result_cache(cache)==RNS8_SUCCESS);
    for(auto* matrix:{a,b,c})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
