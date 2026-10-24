#include "backend_common/split_k_plan.hpp"
TEST_CASE("public split-K query proves fragment caps and exact scratch budgets") {
  rns8_split_k_resources out{};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
  for(uint32_t fragments:{1u,2u,3u,127u,128u,129u,4096u}) {
    REQUIRE(rns8_query_split_k(3,5,65536,20,fragments,UINT64_MAX,&out)==RNS8_SUCCESS);
    CHECK(out.partial_bytes==uint64_t(300)*fragments);
    CHECK(out.merge_bytes==uint64_t(300)*((fragments+127)/128));
    CHECK(out.scratch_bytes==out.partial_bytes+out.merge_bytes);
    CHECK(out.merge_levels==(fragments==1?0u:fragments<=128?1u:2u));
    const auto bytes=out.scratch_bytes;
    CHECK(rns8_query_split_k(3,5,65536,20,fragments,bytes-1,&out)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(out.scratch_bytes==bytes);
    CHECK(rns8_query_split_k(3,5,65536,20,fragments,bytes,&out)==RNS8_SUCCESS);
  }
  CHECK(rns8_query_split_k(1,1,65537,20,1,UINT64_MAX,&out)==RNS8_RANGE_ERROR);
  CHECK(rns8_query_split_k(1,1,65537,20,2,UINT64_MAX,&out)==RNS8_SUCCESS);
  CHECK(rns8_query_split_k(1,1,1,20,2,UINT64_MAX,&out)==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_query_split_k(INT64_MAX,1,1,20,1,UINT64_MAX,&out)==RNS8_UNSUPPORTED_BACKEND);
  for(uint32_t window:{1u,2u,127u,128u,129u}) {
    REQUIRE(rns8_query_split_k_windowed(3,5,65537,20,129,window,UINT64_MAX,&out)==RNS8_SUCCESS);
    CHECK(out.fragment_window==window);CHECK(out.window_count==(129+window-1)/window);
    CHECK(out.partial_bytes==uint64_t(300)*window);
    CHECK(out.running_bytes==(window==129?0u:300u));
    CHECK(out.scratch_bytes==out.partial_bytes+out.merge_bytes+out.running_bytes);
    const auto bytes=out.scratch_bytes;
    CHECK(rns8_query_split_k_windowed(3,5,65537,20,129,window,bytes-1,&out)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(out.scratch_bytes==bytes);
  }
  CHECK(rns8_query_split_k_windowed(1,1,17,20,3,0,UINT64_MAX,&out)==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_query_split_k_windowed(1,1,17,20,3,4,UINT64_MAX,&out)==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_destroy_split_k(nullptr)==RNS8_SUCCESS);
  auto* ctx=create_cpu_context();auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=20;
  rns8_matrix* a=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
  rns8_split_k* owner=nullptr;
  CHECK(rns8_create_split_k(ctx,a,a,1,UINT64_MAX,&owner)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(owner==nullptr);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("public HIP split-K owner reuses scratch and publishes complete exact products") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  for(bool unsigned_values:{false,true}) for(int k:{1,17,65537}) {
    const int m=2,n=3;const auto semantics=unsigned_values?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
    auto md=exact_matrix_desc(m,k,semantics);md.max_prefix=20;
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
    md.rows=k;md.cols=n;md.logical_ld=n;REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    md.rows=m;REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    std::vector<int64_t> av(m*(k+2),99),bv(k*(n+2),99);
    std::vector<uint64_t> au(av.size(),UINT64_MAX),bu(bv.size(),UINT64_MAX);
    for(int i=0;i<m;++i) for(int q=0;q<k;++q) {
      av[i*(k+2)+q]=(q%2?-128:127)+i;au[i*(k+2)+q]=uint64_t(q%256+i);
    }
    for(int q=0;q<k;++q) for(int j=0;j<n;++j) {
      bv[q*(n+2)+j]=j==0?1:((q%3)-1)*127;bu[q*(n+2)+j]=uint64_t(q%251+j);
    }
    if(k==17) {av[0]=INT64_MIN;bv[0]=INT64_MAX;au[0]=UINT64_MAX;bu[0]=UINT64_MAX;}
    const auto pack=[&]() {
      if(unsigned_values) {
        REQUIRE(rns8_pack_u64(ctx,a,au.data(),k+2,0)==RNS8_SUCCESS);
        REQUIRE(rns8_pack_u64(ctx,b,bu.data(),n+2,0)==RNS8_SUCCESS);
      } else {
        REQUIRE(rns8_pack_i64(ctx,a,av.data(),k+2,0)==RNS8_SUCCESS);
        REQUIRE(rns8_pack_i64(ctx,b,bv.data(),n+2,0)==RNS8_SUCCESS);
      }
    };pack();
    std::vector<cpp_int> reference(m*n);
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) for(int q=0;q<k;++q)
      reference[i*n+j]+=(unsigned_values?cpp_int(au[i*(k+2)+q]):cpp_int(av[i*(k+2)+q]))*
          (unsigned_values?cpp_int(bu[q*(n+2)+j]):cpp_int(bv[q*(n+2)+j]));
    const std::vector<uint32_t> counts=k==1?std::vector<uint32_t>{1}:k==17?std::vector<uint32_t>{1,3,17}:
        std::vector<uint32_t>{2,3,129,4096};
    for(auto fragments:counts) for(bool windowed:{false,true}) {
      const uint32_t window=fragments>128?128:fragments>1?fragments-1:1;
      rns8_split_k_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
      REQUIRE((windowed?rns8_query_split_k_windowed(m,n,k,20,fragments,window,UINT64_MAX,&resources):
          rns8_query_split_k(m,n,k,20,fragments,UINT64_MAX,&resources))==RNS8_SUCCESS);
      rns8_split_k* owner=nullptr;
      const auto create=[&](uint64_t budget) {return windowed?
          rns8_create_split_k_windowed(ctx,a,b,fragments,window,budget,&owner):
          rns8_create_split_k(ctx,a,b,fragments,budget,&owner);};
      CHECK(create(resources.scratch_bytes-1)==RNS8_WORKSPACE_TOO_SMALL);
      CHECK(owner==nullptr);
      REQUIRE(create(resources.scratch_bytes)==RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);CHECK(rns8_destroy_context(ctx)==RNS8_BUSY);
      CHECK(rns8_execute_split_k(owner,a)==RNS8_INVALID_ARGUMENT);
      for(int repeat=0;repeat<2;++repeat) {
        REQUIRE(rns8_execute_split_k(owner,c)==RNS8_SUCCESS);
        std::vector<int8_t> result(m*n*20);
        REQUIRE(hip_direct_copy_device_to_host(0,result.data(),c->hip_residues,result.size())==RNS8_SUCCESS);
        for(unsigned p=0;p<20;++p) for(int i=0;i<m*n;++i)
          CHECK(result[p*m*n+i]==centered_residue(reference[i],kDefaultModuli[p]));
        CHECK(a->in_flight_readers==0);CHECK(b->in_flight_readers==0);CHECK_FALSE(c->in_flight_writer);
        CHECK(c->prepared_references==0);CHECK(c->exact_range_prefix==20);
      }
      const auto generation=c->content_generation;pack();
      CHECK(rns8_execute_split_k(owner,c)==RNS8_INVALID_ARGUMENT);CHECK(c->content_generation==generation);
      REQUIRE(rns8_destroy_split_k(owner)==RNS8_SUCCESS);
    }
    for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  }
  {
    auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=20;
    rns8_matrix *a=nullptr,*c=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    int64_t value=-3;REQUIRE(rns8_pack_i64(ctx,a,&value,1,0)==RNS8_SUCCESS);
    rns8_split_k* owner=nullptr;REQUIRE(rns8_create_split_k(ctx,a,a,1,UINT64_MAX,&owner)==RNS8_SUCCESS);
    CHECK(a->prepared_references==1);REQUIRE(rns8_execute_split_k(owner,c)==RNS8_SUCCESS);
    CHECK(a->in_flight_readers==0);std::vector<int8_t> result(20);
    REQUIRE(hip_direct_copy_device_to_host(0,result.data(),c->hip_residues,20)==RNS8_SUCCESS);
    for(unsigned p=0;p<20;++p) CHECK(result[p]==centered_residue(cpp_int(9),kDefaultModuli[p]));
    REQUIRE(rns8_destroy_split_k(owner)==RNS8_SUCCESS);CHECK(a->prepared_references==0);
    REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP split-K execution unavailable in this build");
#endif
}
