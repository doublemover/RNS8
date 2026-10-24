TEST_CASE("resident writers reject aliases and publish mutation identities", "[api][value-lifecycle]") {
  for (bool hip : {false, true}) {
  if (hip && !rns8::detail::api::direct_hip_compiled()) continue;
  for (auto semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED,
                         RNS8_EXACT_WIDE_UNSIGNED, RNS8_FINITE_RING_U8, RNS8_FINITE_FIELD_U8,
                         RNS8_WRAP_U64_MOD_2_64}) {
    const bool wrap = semantics == RNS8_WRAP_U64_MOD_2_64;
    const bool finite = semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
    const bool exact = rns8::detail::exact_wide_semantics(semantics);
    rns8_context* ctx=nullptr;
    if (hip) {
      rns8_context_options options{}; options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
      options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
      REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
    } else ctx=wrap ? create_wrap_context() : create_cpu_context();
    auto desc = wrap ? wrap_gemm_desc(2,2,2) : finite ? finite_gemm_desc(semantics,2,2,2,251) :
                exact ? exact_desc(semantics,2,2,2) : bounded_gemm_desc(semantics,2,2,2);
    if (hip) desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    if (!wrap && !finite && !exact) desc.bound = 10000;
    auto md = wrap ? wrap_matrix_desc(2,2) : finite ? finite_matrix_desc(2,2,semantics) :
              exact ? exact_matrix_desc(2,2,semantics) : bounded_matrix_desc(2,2,semantics);
    rns8_plan* plan = nullptr; rns8_workspace* ws = nullptr;
    rns8_matrix *a=nullptr, *b=nullptr, *c=nullptr, *d=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&ws) == RNS8_SUCCESS);
    for (auto** out : {&a,&b,&c,&d}) REQUIRE(rns8_create_matrix(ctx,&md,out) == RNS8_SUCCESS);
    auto pack = [&](rns8_matrix* matrix, int first, uint64_t version) {
      int64_t si[4]; uint64_t ui[4]; uint8_t fi[4];
      for (int i=0;i<4;++i) { si[i]=first+i; ui[i]=first+i; fi[i]=static_cast<uint8_t>(first+i); }
      return finite ? rns8_pack_finite_u8(ctx,matrix,251,fi,2,version) :
             (semantics==RNS8_BOUNDED_I64 || semantics==RNS8_EXACT_WIDE_SIGNED) ?
               rns8_pack_i64(ctx,matrix,si,2,version) : rns8_pack_u64(ctx,matrix,ui,2,version);
    };
    auto gemm = [&](rns8_matrix* lhs, rns8_matrix* rhs, rns8_matrix* out) {
      return wrap ? rns8_gemm_wrap_u64(ctx,plan,lhs,rhs,out,ws) :
             finite ? rns8_gemm_finite_u8(ctx,plan,251,lhs,rhs,out,ws) : rns8_gemm_rns(ctx,plan,lhs,rhs,out,ws);
    };
    REQUIRE(pack(a,1,0)==RNS8_SUCCESS); REQUIRE(pack(b,5,0)==RNS8_SUCCESS);
    const auto old_a=*a, old_b=*b;
    CHECK(gemm(a,b,a)==RNS8_INVALID_ARGUMENT); CHECK(gemm(a,b,b)==RNS8_INVALID_ARGUMENT);
    CHECK(a->residues==old_a.residues); CHECK(b->residues==old_b.residues);
    CHECK(a->byte_limbs==old_a.byte_limbs); CHECK(b->byte_limbs==old_b.byte_limbs);
    CHECK(a->content_generation==old_a.content_generation); CHECK(b->source_version==old_b.source_version);
    REQUIRE(gemm(a,b,c)==RNS8_SUCCESS);
    const auto preserved=*c;
    for (int conflict=0; conflict<4; ++conflict) {
      if (conflict==0) a->in_flight_writer=true;
      if (conflict==1) b->in_flight_writer=true;
      if (conflict==2) c->in_flight_writer=true;
      if (conflict==3) c->in_flight_readers=1;
      CHECK(gemm(a,b,c)==RNS8_BUSY);
      CHECK(c->content_generation==preserved.content_generation);
      CHECK(c->source_version==preserved.source_version);
      CHECK(c->residues==preserved.residues);
      CHECK(c->byte_limbs==preserved.byte_limbs);
      CHECK(c->host_residues_current==preserved.host_residues_current);
      CHECK(c->device_residues_current==preserved.device_residues_current);
      CHECK(c->host_byte_limbs_current==preserved.host_byte_limbs_current);
      CHECK(c->device_byte_limbs_current==preserved.device_byte_limbs_current);
      a->in_flight_writer=b->in_flight_writer=c->in_flight_writer=false;
      c->in_flight_readers=0;
    }
    a->in_flight_readers=b->in_flight_readers=1;
    REQUIRE(gemm(a,b,c)==RNS8_SUCCESS);
    CHECK(a->in_flight_readers==1);
    CHECK(b->in_flight_readers==1);
    a->in_flight_readers=b->in_flight_readers=0;
    const auto first=c->source_version, generation=c->content_generation;
    REQUIRE(pack(a,2,0)==RNS8_SUCCESS); REQUIRE(gemm(a,b,c)==RNS8_SUCCESS);
    CHECK(c->source_version!=first); CHECK(c->content_generation>generation);
    REQUIRE(pack(a,1,1)==RNS8_SUCCESS); REQUIRE(pack(d,2,1)==RNS8_SUCCESS);
    REQUIRE(gemm(a,b,c)==RNS8_SUCCESS); const auto second=c->source_version;
    REQUIRE(gemm(d,b,c)==RNS8_SUCCESS); CHECK(c->source_version!=second);
    REQUIRE(gemm(a,a,c)==RNS8_SUCCESS);
    // Saturation never wraps into a formerly eligible identity.
    c->content_generation=std::numeric_limits<uint64_t>::max();
    REQUIRE(gemm(a,b,c)==RNS8_SUCCESS); CHECK(c->source_version==0); CHECK(c->generation_exhausted);
    for (auto* matrix : {a,b,c,d}) rns8_destroy_matrix(matrix);
    rns8_destroy_workspace(ws); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
  }
}

}

TEST_CASE("wide limb import rebuilds proof from padded signed and unsigned values", "[api][value-lifecycle]") {
  using namespace rns8::detail;
  for (const bool signed_values : {false,true}) {
    auto* ctx=create_cpu_context();
    const auto semantics=signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
    auto md=exact_matrix_desc(2,2,semantics); rns8_matrix* matrix=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    std::vector<uint64_t> limbs(12,999); // 2 rows, padded LD=3, 2 limbs per cell
    for (int round=0;round<3;++round) {
      std::vector<cpp_int> values;
      for (int r=0;r<2;++r) for (int c=0;c<2;++c) {
        cpp_int value=round==0 ? cpp_int(1)<<80 : cpp_int(round==1 ? 3 : 1);
        if (signed_values && c==1) value=-value;
        values.push_back(value);
        cpp_int bits=value; if (bits<0) bits+=cpp_int(1)<<128;
        limbs[(r*3+c)*2]=static_cast<uint64_t>(bits & ((cpp_int(1)<<64)-1));
        limbs[(r*3+c)*2+1]=static_cast<uint64_t>(bits>>64);
      }
      REQUIRE(api::import_exact_wide_limbs(*ctx,*matrix,limbs.data(),3,2,signed_values,round+1)==RNS8_SUCCESS);
      cpp_int maximum; auto expected=native_exact_ranges(values.data(),2,2,2,maximum);
      CHECK(matrix->exact_max_magnitude==maximum);
      CHECK(matrix->exact_axis_ranges.rows[0].sum==expected.rows[0].sum);
      CHECK(matrix->exact_axis_ranges.cols[1].squared_sum==expected.cols[1].squared_sum);
      CHECK(matrix->exact_range_prefix==20);
    }
    const auto old=matrix->residues;
    std::vector<uint64_t> too_large(16,0); too_large[3]=512; // 2^201
    CHECK(api::import_exact_wide_limbs(*ctx,*matrix,too_large.data(),2,4,signed_values,9)==RNS8_RANGE_ERROR);
    CHECK(matrix->residues==old);
    rns8_destroy_matrix(matrix); rns8_destroy_context(ctx);
  }
}

TEST_CASE("scalar first proof is opt in and refines before range rejection", "[api][value-lifecycle]") {
  using namespace rns8::detail;
  auto* ctx=create_cpu_context();
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,2);
  desc.flags=RNS8_PLAN_EXACT_SCALAR_FIRST | RNS8_PLAN_FORCE_FIXED_PREFIX;
  desc.max_prefix=20;
  rns8_plan* plan=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  rns8_plan narrow=*plan; narrow.validated_contract.reset(); narrow.prefix=1; narrow.modulus_product=256;
  rns8_matrix a,b; a.exact_range_prefix=b.exact_range_prefix=1;
  const int64_t av[]={100,100},bv[]={1,-1};
  a.exact_axis_ranges=native_exact_ranges(av,1,2,2,a.exact_max_magnitude);
  b.exact_axis_ranges=native_exact_ranges(bv,2,1,1,b.exact_max_magnitude);
  cpp_int bound; rns8_exact_matrix_ranges ranges;
  REQUIRE(exact_gemm_range(narrow,a,b,bound,&ranges)==RNS8_SUCCESS);
  CHECK(bound==0); // scalar 200 cannot fit signed prefix 1; cancellation must refine
  CHECK_FALSE(ranges.rows.empty());
  a.exact_max_magnitude=1; b.exact_max_magnitude=1;
  REQUIRE(exact_gemm_range(narrow,a,b,bound,&ranges)==RNS8_SUCCESS);
  CHECK(bound==2); CHECK(ranges.rows.empty());
  rns8_destroy_plan(plan);
  desc=bounded_gemm_desc(RNS8_BOUNDED_I64,1,1,2);desc.flags=RNS8_PLAN_EXACT_SCALAR_FIRST;
  CHECK(rns8_create_plan(ctx,&desc,&plan)==RNS8_INVALID_ARGUMENT);
  rns8_destroy_context(ctx);
}

TEST_CASE("reimported exact chains match Boost at every step and reject CRT overflow", "[api][value-lifecycle]") {
  using namespace rns8::detail;
  for(bool hip:{false,true}) {
    if(hip && !api::direct_hip_compiled()) continue;
    for(bool signed_values:{false,true}) {
      auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
      rns8_context_options options{}; options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
      options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
      rns8_context* ctx=nullptr;
      REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
      auto desc=exact_desc(semantics,1,1,1); desc.requested_backend=options.requested_backend;
      desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX; desc.max_prefix=20;
      rns8_plan* plan=nullptr; rns8_workspace* ws=nullptr;
      REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
      auto md=exact_matrix_desc(1,1,semantics);
      rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
      for(auto** m:{&a,&b,&c}) REQUIRE(rns8_create_matrix(ctx,&md,m)==RNS8_SUCCESS);
      uint64_t initial[4]={2,0,0,0};
      REQUIRE(api::import_exact_wide_limbs(*ctx,*a,initial,1,4,signed_values,1)==RNS8_SUCCESS);
      int64_t factor=-2; uint64_t unsigned_factor=2;
      REQUIRE((signed_values?rns8_pack_i64(ctx,b,&factor,1,1):rns8_pack_u64(ctx,b,&unsigned_factor,1,1))==RNS8_SUCCESS);
      cpp_int expected=2; bool rejected=false;
      for(int step=0;step<200;++step) {
        expected*=signed_values?-2:2;
        const cpp_int magnitude=expected<0?-expected:expected;
        const bool fits=plan->modulus_product > magnitude*(signed_values?2:1);
        auto status=rns8_gemm_rns(ctx,plan,a,b,c,ws);
        if(!fits) { CHECK(status==RNS8_RANGE_ERROR); rejected=true; break; }
        REQUIRE(status==RNS8_SUCCESS);
        uint64_t limbs[4]{};
        auto export_value=[&](uint32_t count) {return signed_values?
          rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs,1,count):
          rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs,1,count);};
        REQUIRE(export_value(4)==RNS8_SUCCESS);
        cpp_int decoded=0;
        for(int j=3;j>=0;--j) { decoded<<=64; decoded+=limbs[j]; }
        if(signed_values && (limbs[3]>>63)) decoded-=cpp_int(1)<<256;
        CHECK(decoded==expected);
        REQUIRE(api::import_exact_wide_limbs(*ctx,*a,limbs,1,4,signed_values,step+2)==RNS8_SUCCESS);
        if(magnitude >= (cpp_int(1)<<(signed_values?63:64)) && expected!=(cpp_int(-1)<<63))
          CHECK(export_value(1)==RNS8_RANGE_ERROR);
      }
      CHECK(rejected);
      for(auto* m:{a,b,c}) rns8_destroy_matrix(m);
      rns8_destroy_workspace(ws); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
    }
  }
}

TEST_CASE("prepacked operation preflight reports reservation conflicts before backend admission") {
  using namespace rns8::detail::api;
  // CPU-only metadata fixture: no device objects or kernel execution are claimed.
  rns8_context context;
  rns8_plan plan;
  rns8_matrix a,c;
  rns8_prepack_cache b;
  for (int conflict=0; conflict<3; ++conflict) {
    a.in_flight_writer=conflict==0;
    c.in_flight_writer=conflict==1;
    c.in_flight_readers=conflict==2 ? 1 : 0;
    CHECK(validate_rns_gemm_prepacked_b_operands(context,plan,a,b,c)==RNS8_BUSY);
  }
  a.in_flight_writer=c.in_flight_writer=false;
  c.in_flight_readers=0;
  a.in_flight_readers=1;
  CHECK(validate_rns_gemm_prepacked_b_operands(context,plan,a,b,c)!=RNS8_BUSY);
  CHECK(a.in_flight_readers==1);
}
