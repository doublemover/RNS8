namespace {
struct SynchronousDispatchFixture {
  std::vector<const rns8_matrix*> readers;
  std::vector<const rns8_matrix*> writers;
  const rns8_workspace* workspace=nullptr;
  unsigned calls=0,drains=0;
  bool fail=false,throw_after_issue=false,drain_fails=false,reservations_observed=false;
  static rns8_status dispatch(void* opaque,const char*,bool* started) {
    auto& s=*static_cast<SynchronousDispatchFixture*>(opaque);++s.calls;
    s.reservations_observed=true;
    for(auto* matrix:s.readers)s.reservations_observed&=matrix->in_flight_readers>0;
    for(auto* matrix:s.writers)s.reservations_observed&=matrix->in_flight_writer;
    if(s.workspace)s.reservations_observed&=s.workspace->in_flight;
    if(s.fail && started)*started=true; // simulated issue; no faulty device work
    if(s.throw_after_issue)throw std::bad_alloc{};
    return s.fail?RNS8_BACKEND_FAILURE:RNS8_SUCCESS;
  }
  static rns8_status drain(void* opaque) noexcept {
    auto& s=*static_cast<SynchronousDispatchFixture*>(opaque);++s.drains;
    return s.drain_fails?RNS8_BACKEND_FAILURE:RNS8_SUCCESS;
  }
};
}

TEST_CASE("public synchronous dispatch retains every dependency after uncertain completion", "[api][lifetime]") {
  for(const std::string operation:{"rns","finite","wrap64","gram","axpby","complex","verify"})
    for(unsigned failure=0;failure<5;++failure) {
      CAPTURE(operation,failure);
      const bool wrap=operation=="wrap64",finite=operation=="finite" || operation=="verify";
      auto* ctx=wrap?create_wrap_context():create_cpu_context();
      auto desc=wrap?wrap_gemm_desc(1,1,1):finite?finite_gemm_desc(RNS8_FINITE_RING_U8,1,1,1,251):exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);
      auto md=wrap?wrap_matrix_desc(1,1):finite?finite_matrix_desc(1,1,RNS8_FINITE_RING_U8):exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);
      rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;
      REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
      rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr,*ai=nullptr,*ci=nullptr;
      for(auto** value:{&a,&b,&c,&d,&ai,&ci}) {
        REQUIRE(rns8_create_matrix(ctx,&md,value)==RNS8_SUCCESS);
        const int64_t one=1;const uint64_t uone=1;const uint8_t byte=1;
        REQUIRE((wrap?rns8_pack_u64(ctx,*value,&uone,1,1):finite?rns8_pack_finite_u8(ctx,*value,251,&byte,1,1):rns8_pack_i64(ctx,*value,&one,1,1))==RNS8_SUCCESS);
      }
      rns8_finite_verify_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
      options.repetition_count=2;options.max_workspace_bytes=UINT64_MAX;options.sampler=RNS8_VERIFY_SPLITMIX64_BINARY_V1;
      rns8_finite_verify_report report{};report.struct_size=sizeof(report);report.abi_version=RNS8_ABI_VERSION;
      auto execute=[&]() {
        if(operation=="finite")return rns8_gemm_finite_u8(ctx,plan,251,a,b,c,work);
        if(operation=="verify")return rns8_verify_finite_product(ctx,a,b,c,&options,&report);
        if(operation=="wrap64")return rns8_gemm_wrap_u64(ctx,plan,a,b,c,work);
        if(operation=="gram")return rns8_gram_residues(ctx,a,c,RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,UINT64_MAX);
        if(operation=="axpby")return rns8_gemm_exact_axpby(ctx,plan,a,b,d,c,1,1,work);
        if(operation=="complex")return rns8_gemm_complex_exact_signed(ctx,plan,a,ai,b,d,c,ci,RNS8_COMPLEX_THREE_REAL_PRODUCTS,UINT64_MAX,work);
        return rns8_gemm_rns(ctx,plan,a,b,c,work);
      };
      REQUIRE(execute()==RNS8_SUCCESS);
      const auto generation=c->content_generation,imag_generation=ci->content_generation;
      SynchronousDispatchFixture state;
      state.readers=operation=="complex"?std::vector<const rns8_matrix*>{a,ai,b,d}:
          operation=="axpby"?std::vector<const rns8_matrix*>{a,b,d}:
          operation=="verify"?std::vector<const rns8_matrix*>{a,b,c}:
          operation=="gram"?std::vector<const rns8_matrix*>{a}:std::vector<const rns8_matrix*>{a,b};
      if(operation!="verify")state.writers={c};if(operation=="complex")state.writers.push_back(ci);
      if(operation!="verify" && operation!="gram")state.workspace=work;
      state.fail=failure!=0;state.throw_after_issue=failure>=3;state.drain_fails=failure==2 || failure==3;
      const rns8_synchronous_test_hooks hooks{&state,&SynchronousDispatchFixture::dispatch,&SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks=&hooks;
      auto* saved=ctx;ctx=nullptr;CHECK(execute()==RNS8_INVALID_ARGUMENT);ctx=saved;
      CHECK(state.calls==0);CHECK(c->content_generation==generation);
      report.passed=99;
      const auto expected=failure==0?RNS8_SUCCESS:failure>=3?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE;
      CHECK(execute()==expected);CHECK(state.calls==1);CHECK(state.reservations_observed);
      CHECK(c->content_generation==generation+(failure==0 && operation!="verify"));
      if(operation=="complex")CHECK(ci->content_generation==imag_generation+(failure==0));
      if(operation=="verify")CHECK(report.passed==(failure==0?1:99));
      if(failure==0)CHECK(state.drains==0); // no added success-path device wait
      if(state.drain_fails) {
        CHECK(ctx->hip_failure_poisoned);CHECK_FALSE(ctx->failed_work_leases.empty());
        for(auto* matrix:state.readers) {
          CHECK(matrix->in_flight_readers>0);
          CHECK(rns8_destroy_matrix(const_cast<rns8_matrix*>(matrix))==RNS8_BUSY);
        }
        for(auto* matrix:state.writers)CHECK(matrix->in_flight_writer);
        state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
        CHECK(ctx->hip_failure_poisoned); // draining memory does not revive authority
      }
      for(auto* matrix:{a,b,c,d,ai,ci}) {CHECK(matrix->in_flight_readers==0);CHECK_FALSE(matrix->in_flight_writer);}
      CHECK_FALSE(work->in_flight);CHECK(ctx->failed_work_leases.empty());
      ctx->synchronous_test_hooks=nullptr;
      for(auto* matrix:{a,b,c,d,ai,ci})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
    }
}

TEST_CASE("failed synchronous work retains private owner counters until one successful drain", "[api][lifetime]") {
  using namespace rns8::detail::api;
  auto* ctx=create_cpu_context();rns8_matrix a{},b{},c{};rns8_workspace work{};
  SynchronousDispatchFixture state;state.drain_fails=true;
  const rns8_synchronous_test_hooks hooks{&state,nullptr,&SynchronousDispatchFixture::drain};
  ctx->synchronous_test_hooks=&hooks;
  uint64_t first=0,second=0;size_t token=0;
  REQUIRE(begin_synchronous_lease(*ctx,a,b,c,work,token)==RNS8_SUCCESS);
  SynchronousLease lease(*ctx,token);
  REQUIRE(add_synchronous_owner_hold(*ctx,token,first)==RNS8_SUCCESS);
  REQUIRE(add_synchronous_owner_hold(*ctx,token,first)==RNS8_SUCCESS);
  REQUIRE(add_synchronous_owner_hold(*ctx,token,second)==RNS8_SUCCESS);
  CHECK(first==1);CHECK(second==1);
  CHECK(lease.finish(RNS8_BACKEND_FAILURE)==RNS8_BACKEND_FAILURE);
  CHECK(first==1);CHECK(second==1);CHECK(ctx->hip_failure_poisoned);
  state.drain_fails=false;REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);
  CHECK(first==0);CHECK(second==0);CHECK(ctx->failed_work_leases.empty());
  REQUIRE(rns8_drain_failed_work(ctx)==RNS8_SUCCESS);CHECK(first==0);CHECK(second==0);
  ctx->synchronous_test_hooks=nullptr;REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
