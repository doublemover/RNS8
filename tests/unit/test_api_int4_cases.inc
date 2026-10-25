#include "backend_common/int4_encoding.hpp"
TEST_CASE("signed nibble words preserve every finite alphabet and numerical zero tails", "[int4][layout]") {
  using namespace rns8::detail;
  for(int modulus=2;modulus<=16;++modulus)for(int a=-128;a<=127;++a)for(int b=-128;b<=127;++b){
    const uint32_t packed=finite_int4_nibble(int8_t(a),modulus)|(finite_int4_nibble(int8_t(b),modulus)<<4);
    CHECK(signed_nibble(packed,0)==centered_int4_residue(int8_t(a),modulus));CHECK(signed_nibble(packed,1)==centered_int4_residue(int8_t(b),modulus));
    CHECK((signed_nibble(packed,0)*signed_nibble(packed,1)-a*b)%modulus==0);
  }
  for(unsigned k:{1,7,8,15,16,17,31,32,33}){const unsigned n=19;std::vector<uint32_t> packed(((k+15)/16)*n*2,UINT32_MAX);
    for(unsigned block=0;block<(k+15)/16;++block)for(unsigned col=0;col<n;++col)for(unsigned word=0;word<2;++word){uint32_t value=0;
      for(unsigned nibble=0;nibble<8;++nibble){unsigned q=block*16+word*8+nibble;if(q<k)value|=finite_int4_nibble(int8_t((q+col)%16),16)<<(4*nibble);}
      packed[int4_b_word_index(block,col,n,word)]=value;}
    for(unsigned col=0;col<n;++col)for(unsigned q=k;q<((k+15)/16)*16;++q)CHECK(signed_nibble(packed[int4_b_word_index(q/16,col,n,(q%16)/8)],q%8)==0);
  }
}
TEST_CASE("public INT4 rejects CPU recipes before touching resident finite values", "[api][int4]") {
  auto* ctx=create_cpu_context();auto desc=finite_gemm_desc(RNS8_FINITE_RING_U8,1,1,1,16);rns8_plan* plan=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);auto md=finite_matrix_desc(1,1,RNS8_FINITE_RING_U8);rns8_matrix* b=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);const uint8_t value=15;REQUIRE(rns8_pack_finite_u8(ctx,b,16,&value,1,1)==RNS8_SUCCESS);
  const auto before=b->residues;const auto generation=b->content_generation;rns8_int4_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.mode=1;d.max_preparation_bytes=UINT64_MAX;rns8_int4_recipe* owner=nullptr;
  CHECK(rns8_prepare_int4_recipe(ctx,plan,b,&d,&owner)==RNS8_UNSUPPORTED_BACKEND);CHECK(owner==nullptr);CHECK(b->residues==before);CHECK(b->content_generation==generation);
  d.mode=2;CHECK(rns8_prepare_int4_recipe(ctx,plan,b,&d,&owner)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("public finite INT4 preparation owns generation-bound B and preserves stale rejection", "[api][hip][int4]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* ctx=nullptr;
  if(rns8_create_context(0,&options,&ctx)!=RNS8_SUCCESS)SKIP("direct HIP unavailable");
  for(uint32_t modulus=2;modulus<=16;++modulus)for(uint32_t mode:{0u,1u})for(int64_t k:{1,17,65535,65536,65537}){
    auto desc=finite_gemm_desc(RNS8_FINITE_RING_U8,2,3,k,modulus);desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;auto am=finite_matrix_desc(2,k,RNS8_FINITE_RING_U8),bm=finite_matrix_desc(k,3,RNS8_FINITE_RING_U8),cm=finite_matrix_desc(2,3,RNS8_FINITE_RING_U8);
    REQUIRE(rns8_create_matrix(ctx,&am,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&bm,&b)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&cm,&c)==RNS8_SUCCESS);
    std::vector<uint8_t> av(2*k,modulus-1),bv(k*3,modulus/2),result(2*5,0xa5);REQUIRE(rns8_pack_finite_u8(ctx,a,modulus,av.data(),k,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_finite_u8(ctx,b,modulus,bv.data(),3,1)==RNS8_SUCCESS);
    rns8_int4_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.mode=mode;d.max_preparation_bytes=UINT64_MAX;rns8_int4_recipe* owner=nullptr;
    const auto status=rns8_prepare_int4_recipe(ctx,plan,b,&d,&owner);
    if(status==RNS8_UNSUPPORTED_BACKEND){REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);SKIP("actual code object lacks signed INT4 WMMA");}
    REQUIRE(status==RNS8_SUCCESS);CHECK(rns8_destroy_matrix(b)==RNS8_BUSY);
    rns8_int4_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_get_int4_info(owner,&info)==RNS8_SUCCESS);CHECK(info.source_is_current);CHECK(info.safe_k==65536);CHECK(info.arithmetic_proven);CHECK_FALSE(info.target_qualified);
    REQUIRE(rns8_gemm_int4(owner,a,c,work)==RNS8_SUCCESS);REQUIRE(rns8_export_finite_u8(ctx,plan,modulus,c,result.data(),5)==RNS8_SUCCESS);
    for(unsigned row=0;row<2;++row)for(unsigned col=0;col<5;++col)CHECK(result[row*5+col]==(col<3?uint64_t(k)*(modulus-1)*(modulus/2)%modulus:0xa5));
    const auto output_generation=c->content_generation;REQUIRE(rns8_pack_finite_u8(ctx,b,modulus,bv.data(),3,2)==RNS8_SUCCESS);
    CHECK(rns8_gemm_int4(owner,a,c,work)==RNS8_INVALID_ARGUMENT);CHECK(c->content_generation==output_generation);REQUIRE(rns8_get_int4_info(owner,&info)==RNS8_SUCCESS);CHECK_FALSE(info.source_is_current);
    REQUIRE(rns8_destroy_int4_recipe(owner)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP unavailable in this build");
#endif
}
