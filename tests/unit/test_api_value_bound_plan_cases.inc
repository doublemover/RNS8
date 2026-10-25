TEST_CASE("explicit exact value-bound capacity preserves default native admission", "[api][exact][planning]") {
  using namespace rns8::detail;auto* ctx=create_cpu_context();
  for(auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED})for(uint32_t prefix=1;prefix<=20;++prefix){
    auto d=exact_desc(semantics,1,1,2);d.max_prefix=prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* p=nullptr;const bool full=modulus_product(prefix)>(boost::multiprecision::cpp_int(2)<<(semantics==RNS8_EXACT_WIDE_SIGNED?127:128));
    CHECK(rns8_create_plan(ctx,&d,&p)==(full?RNS8_SUCCESS:RNS8_RANGE_ERROR));REQUIRE(rns8_destroy_plan(p)==RNS8_SUCCESS);p=nullptr;
    d.flags|=RNS8_PLAN_EXACT_VALUE_BOUND;REQUIRE(rns8_create_plan(ctx,&d,&p)==RNS8_SUCCESS);
    CHECK(p->prefix==prefix);CHECK(p->schedule_range_bit_length==0);CHECK(p->schedule_min_required_prefix==prefix);
    CHECK(p->backend_autotune_key.find("range_admission=current_value_fixed_basis_v1")!=std::string::npos);
    CHECK(p->arithmetic_guard->certificate.instruction.lowering_identity.find("range_admission=current_value_fixed_basis_v1")!=std::string::npos);
    CHECK_FALSE(p->arithmetic_guard->certificate.bound_known);CHECK(api::revalidate_plan_schedule_contract(*p));
    REQUIRE(rns8_destroy_plan(p)==RNS8_SUCCESS);
  }
  auto d=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);d.flags=RNS8_PLAN_EXACT_VALUE_BOUND;rns8_plan* invalid=nullptr;
  CHECK(rns8_create_plan(ctx,&d,&invalid)==RNS8_INVALID_ARGUMENT);CHECK(invalid==nullptr);
  d=bounded_gemm_desc(RNS8_BOUNDED_I64,1,1,1);d.flags=RNS8_PLAN_EXACT_VALUE_BOUND|RNS8_PLAN_FORCE_FIXED_PREFIX;
  CHECK(rns8_create_plan(ctx,&d,&invalid)==RNS8_INVALID_ARGUMENT);CHECK(invalid==nullptr);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("short exact plans reprove current values before ordinary or prepared writes", "[api][exact][planning]") {
  for(auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED}){
    auto* ctx=create_cpu_context();auto d=exact_desc(semantics,1,1,1);d.max_prefix=1;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
    rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr,*addend=nullptr;
    REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    auto md=exact_matrix_desc(1,1,semantics);for(auto** out:{&a,&b,&c,&addend})REQUIRE(rns8_create_matrix(ctx,&md,out)==RNS8_SUCCESS);
    uint64_t version=0;const auto pack=[&](rns8_matrix* matrix,uint64_t value){++version;REQUIRE((semantics==RNS8_EXACT_WIDE_SIGNED?
        rns8_pack_i64(ctx,matrix,reinterpret_cast<const int64_t*>(&value),1,version):rns8_pack_u64(ctx,matrix,&value,1,version))==RNS8_SUCCESS);};
    pack(a,2);pack(b,3);pack(addend,2);REQUIRE(rns8_gemm_rns(ctx,plan,a,b,c,work)==RNS8_SUCCESS);
    uint64_t words[3]{};const auto export_value=[&](){REQUIRE((semantics==RNS8_EXACT_WIDE_SIGNED?
        rns8_export_exact_wide_signed_limbs(ctx,plan,c,words,1,3):rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,words,1,3))==RNS8_SUCCESS);};
    export_value();CHECK(words[0]==6);CHECK(words[1]==0);CHECK(words[2]==0);
    rns8_prepared_execution* prepared=nullptr;REQUIRE(rns8_prepare_gemm(ctx,plan,a,b,c,work,&prepared)==RNS8_SUCCESS);
    const auto generation=c->content_generation;pack(a,20);pack(b,20);
    CHECK(rns8_gemm_rns(ctx,plan,a,b,c,work)==RNS8_RANGE_ERROR);CHECK(c->content_generation==generation);CHECK(c->host_residues_current);
    rns8_completion_token token{123,456};CHECK(rns8_execute_prepared(prepared,&token)==RNS8_RANGE_ERROR);CHECK(token.prepared_id==123);CHECK(token.epoch==456);
    CHECK(rns8_refresh_prepared_inputs(prepared)==RNS8_RANGE_ERROR);export_value();CHECK(words[0]==6);
    // The complete expression, not an unobserved AB intermediate, is admitted.
    rns8_prepared_execution* epilogue=nullptr;REQUIRE(rns8_prepare_exact_axpby(ctx,plan,a,b,addend,c,0,1,work,&epilogue)==RNS8_SUCCESS);
    REQUIRE(rns8_execute_prepared(epilogue,&token)==RNS8_SUCCESS);export_value();CHECK(words[0]==2);
    REQUIRE(rns8_destroy_prepared_execution(epilogue)==RNS8_SUCCESS);
    pack(a,2);pack(b,3);REQUIRE(rns8_refresh_prepared_inputs(prepared)==RNS8_SUCCESS);REQUIRE(rns8_execute_prepared(prepared,&token)==RNS8_SUCCESS);export_value();CHECK(words[0]==6);
    REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);
    for(auto* matrix:{addend,c,b,a})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
