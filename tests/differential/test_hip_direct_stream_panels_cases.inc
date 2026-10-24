TEST_CASE("forced-budget serial streaming completes one product across all panel tails") {
  if(!hip_available()) SKIP("no HIP device for out-of-core panels");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool signed_values:{false,true}) {
    rns8_stream_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.semantics=signed_values?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
    d.m=17;d.n=19;d.k=33;d.lda=35;d.ldb=21;d.ldc=23;
    d.panel_m=3;d.panel_n=4;d.panel_k=5;d.prefix=9;d.max_abs_a=5;d.max_abs_b=3;
    d.device_budget_bytes=d.host_budget_bytes=UINT64_MAX;
    rns8_stream_gemm_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_query_stream_gemm(&d,&r)==RNS8_SUCCESS);
    d.device_budget_bytes=r.device_bytes;d.host_budget_bytes=r.host_bytes;
    REQUIRE(r.device_bytes<uint64_t(d.m*d.k+d.k*d.n+d.m*d.n)*d.prefix);
    std::vector<uint64_t> a(d.m*d.lda,UINT64_MAX),b(d.k*d.ldb,UINT64_MAX),c(d.m*d.ldc,UINT64_C(0xabcdef));
    for(int row=0;row<d.m;++row) for(int q=0;q<d.k;++q) {
      const int64_t value=signed_values?(row+q)%11-5:(row+q)%6;
      a[row*d.lda+q]=uint64_t(value);
    }
    for(int q=0;q<d.k;++q) for(int col=0;col<d.n;++col) {
      const int64_t value=signed_values?(q+col)%7-3:(q+col)%4;
      b[q*d.ldb+col]=uint64_t(value);
    }
    rns8_stream_gemm_plan* plan=nullptr;
    REQUIRE(rns8_create_stream_gemm(ctx,&d,&plan)==RNS8_SUCCESS);
    CHECK(rns8_destroy_context(ctx)==RNS8_BUSY);
    rns8_stream_gemm_result result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
    for(uint64_t generation:{1,2}) {
      if(generation==2) a[0]=1;
      REQUIRE(rns8_execute_stream_gemm(plan,a.data(),b.data(),c.data(),11,generation,12,1,&result)==RNS8_SUCCESS);
      CHECK(result.output_valid==1);CHECK(result.output_started==1);CHECK(result.quarantined==0);
      CHECK(result.epoch==generation);CHECK(result.completed_tiles==r.output_tiles);
      CHECK(result.completed_panels==r.output_tiles*r.k_panels_per_tile);
      for(int row=0;row<d.m;++row) for(int col=0;col<d.n;++col) {
        boost::multiprecision::cpp_int expected=0;
        for(int q=0;q<d.k;++q) {
          const auto av=a[row*d.lda+q],bv=b[q*d.ldb+col];
          expected+=(signed_values?boost::multiprecision::cpp_int(static_cast<int64_t>(av)):
                                   boost::multiprecision::cpp_int(av))*
                    (signed_values?boost::multiprecision::cpp_int(static_cast<int64_t>(bv)):
                                   boost::multiprecision::cpp_int(bv));
        }
        if(expected<0) expected+=boost::multiprecision::cpp_int(1)<<64;
        REQUIRE(c[row*d.ldc+col]==expected.convert_to<uint64_t>());
      }
      for(int row=0;row<d.m;++row) for(int col=d.n;col<d.ldc;++col) REQUIRE(c[row*d.ldc+col]==UINT64_C(0xabcdef));
    }
    const auto previous=c;const auto epoch=result.epoch;
    a[0]=6;
    CHECK(rns8_execute_stream_gemm(plan,a.data(),b.data(),c.data(),11,3,12,1,&result)==RNS8_RANGE_ERROR);
    CHECK(c==previous);CHECK(result.epoch==epoch);
    CHECK(rns8_execute_stream_gemm(plan,a.data(),b.data(),a.data(),11,3,12,1,&result)==RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_destroy_stream_gemm(plan)==RNS8_SUCCESS);
    --d.device_budget_bytes;
    plan=nullptr;
    CHECK(rns8_create_stream_gemm(ctx,&d,&plan)==RNS8_WORKSPACE_TOO_SMALL);CHECK(plan==nullptr);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
