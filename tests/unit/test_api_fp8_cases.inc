TEST_CASE("public FP8 finite research is explicit and unsupported on CPU", "[api][fp8]") {
  auto* ctx=create_cpu_context();auto gd=finite_gemm_desc(RNS8_FINITE_RING_U8,2,3,17,256);
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);
  rns8_fp8_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.format=1;d.max_preparation_bytes=4;
  rns8_fp8_recipe* owner=nullptr;
  CHECK(rns8_prepare_fp8_recipe(ctx,plan,&d,&owner)==RNS8_UNSUPPORTED_BACKEND);CHECK(owner==nullptr);
  d.flags=RNS8_FP8_ASSUME_NAMED_INSTRUCTION_MODEL;
  CHECK(rns8_prepare_fp8_recipe(ctx,plan,&d,&owner)==RNS8_UNSUPPORTED_BACKEND);CHECK(owner==nullptr);
  d.flags=2;CHECK(rns8_prepare_fp8_recipe(ctx,plan,&d,&owner)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("named FP8 finite research matches independent complete padded products", "[api][fp8][hip]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* ctx=nullptr;
  if(rns8_create_context(0,&options,&ctx)!=RNS8_SUCCESS)SKIP("direct HIP unavailable");
  const uint32_t format=std::strncmp(ctx->device_info.gcn_arch,"gfx942",6)==0?1:std::strncmp(ctx->device_info.gcn_arch,"gfx1201",7)==0?2:0;
  if(!format){REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);SKIP("gfx942 FNUZ or gfx1201 OCP required");}
  for(unsigned method:{0u,1u,2u,3u})for(unsigned modulus=2;modulus<=256;++modulus)for(int64_t k:{31,32,33,65535,65536,65537}) {
    if(k!=33 && modulus!=2 && modulus!=16 && modulus!=127 && modulus!=251 && modulus!=255 && modulus!=256)continue;
    auto gd=finite_gemm_desc(RNS8_FINITE_RING_U8,2,3,k,modulus);gd.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;auto am=finite_matrix_desc(2,k,RNS8_FINITE_RING_U8);
    auto bm=finite_matrix_desc(k,3,RNS8_FINITE_RING_U8),cm=finite_matrix_desc(2,3,RNS8_FINITE_RING_U8);
    REQUIRE(rns8_create_matrix(ctx,&am,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&bm,&b)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&cm,&c)==RNS8_SUCCESS);
    std::vector<uint8_t> av(2*(k+2),99),bv(k*5,99),out(10,0xa5);
    for(int row=0;row<2;++row)for(int64_t q=0;q<k;++q)av[row*(k+2)+q]=uint8_t(q*37+row*131);
    for(int64_t q=0;q<k;++q)for(int col=0;col<3;++col)bv[q*5+col]=uint8_t(q*71+col*127);
    REQUIRE(rns8_pack_finite_u8(ctx,a,modulus,av.data(),k+2,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_finite_u8(ctx,b,modulus,bv.data(),5,1)==RNS8_SUCCESS);
    rns8_fp8_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.format=format;d.flags=1;d.method=method;d.max_preparation_bytes=4+((method&1)?(method>=2?12:16)*2*3:0);
    rns8_fp8_recipe* owner=nullptr;--d.max_preparation_bytes;CHECK(rns8_prepare_fp8_recipe(ctx,plan,&d,&owner)==RNS8_WORKSPACE_TOO_SMALL);CHECK(owner==nullptr);++d.max_preparation_bytes;REQUIRE(rns8_prepare_fp8_recipe(ctx,plan,&d,&owner)==RNS8_SUCCESS);
    CHECK(rns8_destroy_plan(plan)==RNS8_BUSY);
    rns8_fp8_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_fp8_info(owner,&info)==RNS8_SUCCESS);CHECK(info.format==format);CHECK(info.method==method);CHECK(info.retained_device_bytes==d.max_preparation_bytes);CHECK(info.model_proven);CHECK_FALSE(info.target_qualified);
    REQUIRE(rns8_gemm_fp8_research(owner,a,b,c,work)==RNS8_SUCCESS);REQUIRE(rns8_export_finite_u8(ctx,plan,modulus,c,out.data(),5)==RNS8_SUCCESS);
    for(int row=0;row<2;++row)for(int col=0;col<5;++col) {
      boost::multiprecision::cpp_int exact=0;
      if(col<3)for(int64_t q=0;q<k;++q)exact+=boost::multiprecision::cpp_int(av[row*(k+2)+q])*bv[q*5+col];
      CHECK(out[row*5+col]==(col<3?(exact%modulus).convert_to<unsigned>():0xa5));
    }
    const auto generation=c->content_generation;CHECK(rns8_gemm_fp8_research(owner,a,b,a,work)==RNS8_INVALID_ARGUMENT);CHECK(c->content_generation==generation);
    REQUIRE(rns8_destroy_fp8_recipe(owner)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP unavailable in this build");
#endif
}
