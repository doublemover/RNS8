#include <catch2/catch_test_macros.hpp>
#include "backend_common/cpu_i8mm_layout.hpp"
#include "rns8/cpu_i8mm.h"
#include "core/internal.hpp"
#include <array>
#include <vector>
namespace {
template<class T>T i8mm_abi(){T t{};t.struct_size=sizeof(t);t.abi_version=RNS8_ABI_VERSION;return t;}
}
TEST_CASE("I8MM layout proves signed byte columns padding and payload arithmetic") {
  using namespace rns8::detail;
  CpuI8mmLayout layout;REQUIRE(cpu_i8mm_layout(3,5,9,20,layout));
  CHECK(layout.packed==20*3*2*16);CHECK(layout.output==20*3*5);
  CHECK_FALSE(cpu_i8mm_layout(INT64_MAX,INT64_MAX,INT64_MAX,20,layout));
  CHECK_FALSE(cpu_i8mm_layout(1,1,1,0,layout));
  for(int64_t k:{1,7,8,9,63,64,65})for(int64_t n:{1,2,3,5}) {
    std::vector<int8_t> source(size_t(k)*(n+3));
    for(size_t i=0;i<source.size();++i)source[i]=int8_t(int(i%256)-128);
    for(int64_t col=0;col<n;col+=2)for(int64_t kb=0;kb<k;kb+=8) {
      std::array<int8_t,16> packed;packed.fill(99);cpu_i8mm_pack_b(source.data(),k,n,n+3,kb,col,packed.data());
      for(int r=0;r<8;++r)for(int c=0;c<2;++c)
        CHECK(packed[c*8+r]==((kb+r<k && col+c<n)?source[(kb+r)*(n+3)+col+c]:0));
      for(int a:{-128,-1,0,127})for(int c=0;c<2;++c) {
        int32_t physical=0,logical=0;
        for(int q=0;q<8;++q){physical+=a*packed[c*8+q];if(kb+q<k && col+c<n)logical+=a*source[(kb+q)*(n+3)+col+c];}
        CHECK(physical==logical);
      }
    }
  }
  auto info=i8mm_abi<rns8_cpu_i8mm_info>();REQUIRE(rns8_query_cpu_i8mm(&info)==RNS8_SUCCESS);
  CHECK(info.safe_k==65536);CHECK(info.admitted==uint32_t(info.compiled && info.linux_aarch64 && info.os_asimd && info.os_i8mm));
}
TEST_CASE("Arm SMMLA exact products match scalar Boost and retain B generations", "[cpu-i8mm][hardware]") {
  auto info=i8mm_abi<rns8_cpu_i8mm_info>();REQUIRE(rns8_query_cpu_i8mm(&info)==RNS8_SUCCESS);
  if(!info.admitted)SKIP("requires configured Linux AArch64 I8MM and OS capabilities");
  auto options=i8mm_abi<rns8_context_options>();options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(-1,&options,&ctx)==RNS8_SUCCESS);
  for(bool sign:{false,true})for(int64_t k:{1,7,8,9,65535,65536,65537}) {
    const int64_t m=3,n=5;const auto semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    auto d=i8mm_abi<rns8_gemm_desc>();d.requested_backend=RNS8_BACKEND_CPU_REFERENCE;d.semantics=semantics;
    d.bound_kind=RNS8_BOUND_NONE;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;d.max_prefix=20;d.m=m;d.n=n;d.k=k;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);
    std::array<rns8_matrix*,3> values{};
    for(unsigned i=0;i<3;++i){auto md=i8mm_abi<rns8_matrix_desc>();md.semantics=semantics;md.max_prefix=20;
      md.rows=i==1?k:m;md.cols=i==0?k:n;md.logical_ld=md.cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.bound_kind=RNS8_BOUND_NONE;
      REQUIRE(rns8_create_matrix(ctx,&md,&values[i])==RNS8_SUCCESS);}
    std::vector<uint64_t> a(size_t(m)*k),b(size_t(k)*n);
    const uint64_t edges[]{0,1,UINT64_MAX,UINT64_C(1)<<63,UINT64_C(0x7fffffffffffffff)};
    for(size_t i=0;i<a.size();++i)a[i]=edges[i%5];for(size_t i=0;i<b.size();++i)b[i]=edges[(i+2)%5];
    const auto pack=[&](rns8_matrix* value,const std::vector<uint64_t>& data,int64_t ld){return sign?
      rns8_pack_i64(ctx,value,reinterpret_cast<const int64_t*>(data.data()),ld,0):rns8_pack_u64(ctx,value,data.data(),ld,0);};
    REQUIRE(pack(values[0],a,k)==RNS8_SUCCESS);REQUIRE(pack(values[1],b,n)==RNS8_SUCCESS);
    rns8_prepared_cpu_i8mm* owner=nullptr;
    CHECK(rns8_prepare_cpu_i8mm(ctx,plan,values[0],values[1],values[2],0,&owner)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(rns8_prepare_cpu_i8mm(ctx,plan,values[0],values[1],values[2],UINT64_MAX,&owner)==RNS8_SUCCESS);
    CHECK(rns8_destroy_matrix(values[0])==RNS8_BUSY);
    REQUIRE(rns8_execute_cpu_i8mm(owner)==RNS8_SUCCESS);const auto result=values[2]->residues;
    rns8_workspace* workspace=nullptr;REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx,plan,values[0],values[1],values[2],workspace)==RNS8_SUCCESS);CHECK(values[2]->residues==result);
    using boost::multiprecision::cpp_int;
    for(int64_t r=0;r<m;++r)for(int64_t c=0;c<n;++c){cpp_int expected=0;
      for(int64_t q=0;q<k;++q){cpp_int left=a[size_t(r*k+q)],right=b[size_t(q*n+c)];
        if(sign && a[size_t(r*k+q)]>>63)left-=cpp_int(1)<<64;if(sign && b[size_t(q*n+c)]>>63)right-=cpp_int(1)<<64;expected+=left*right;}
      for(unsigned p=0;p<20;++p)CHECK(result[(uint64_t(p)*m+r)*n+c]==rns8::detail::centered_residue(expected,rns8::detail::kDefaultModuli[p]));}
    REQUIRE(rns8_execute_cpu_i8mm(owner)==RNS8_SUCCESS);
    auto before=i8mm_abi<rns8_cpu_i8mm_info>();REQUIRE(rns8_get_cpu_i8mm_info(owner,&before)==RNS8_SUCCESS);CHECK(before.b_pack_count==1);
    REQUIRE(pack(values[1],b,n)==RNS8_SUCCESS);CHECK(rns8_execute_cpu_i8mm(owner)==RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_refresh_cpu_i8mm(owner)==RNS8_SUCCESS);REQUIRE(rns8_execute_cpu_i8mm(owner)==RNS8_SUCCESS);
    REQUIRE(rns8_get_cpu_i8mm_info(owner,&info)==RNS8_SUCCESS);CHECK(info.b_pack_count==2);
    REQUIRE(rns8_destroy_cpu_i8mm(owner)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
    for(auto* value:values)REQUIRE(rns8_destroy_matrix(value)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("Arm SMMLA covers every byte product in prime composite and even rings", "[cpu-i8mm][hardware]") {
  auto info=i8mm_abi<rns8_cpu_i8mm_info>();REQUIRE(rns8_query_cpu_i8mm(&info)==RNS8_SUCCESS);
  if(!info.admitted)SKIP("requires configured Linux AArch64 I8MM and OS capabilities");
  auto options=i8mm_abi<rns8_context_options>();options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(-1,&options,&ctx)==RNS8_SUCCESS);
  for(uint16_t modulus:std::array<uint16_t,6>{2,16,167,253,255,256}) {
    auto d=i8mm_abi<rns8_gemm_desc>();d.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
    d.semantics=RNS8_FINITE_RING_U8;d.finite_modulus=modulus;d.m=d.n=256;d.k=1;d.bound_kind=RNS8_BOUND_NONE;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);
    std::array<rns8_matrix*,3> values{};
    for(unsigned i=0;i<3;++i){auto md=i8mm_abi<rns8_matrix_desc>();md.rows=i==1?1:256;md.cols=i==0?1:256;
      md.logical_ld=md.cols;md.semantics=RNS8_FINITE_RING_U8;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.bound_kind=RNS8_BOUND_NONE;
      REQUIRE(rns8_create_matrix(ctx,&md,&values[i])==RNS8_SUCCESS);}
    std::array<uint8_t,256> bytes{};for(unsigned i=0;i<256;++i)bytes[i]=uint8_t(i);
    REQUIRE(rns8_pack_finite_u8(ctx,values[0],modulus,bytes.data(),1,0)==RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(ctx,values[1],modulus,bytes.data(),256,0)==RNS8_SUCCESS);
    rns8_prepared_cpu_i8mm* owner=nullptr;
    REQUIRE(rns8_prepare_cpu_i8mm(ctx,plan,values[0],values[1],values[2],UINT64_MAX,&owner)==RNS8_SUCCESS);
    REQUIRE(rns8_execute_cpu_i8mm(owner)==RNS8_SUCCESS);
    for(unsigned r=0;r<256;++r)for(unsigned c=0;c<256;++c) {
      int expected=int((r*c)%modulus);if(expected>=int((modulus+1)/2))expected-=modulus;
      CHECK(values[2]->residues[r*256+c]==expected);
    }
    REQUIRE(rns8_destroy_cpu_i8mm(owner)==RNS8_SUCCESS);
    for(auto* value:values)REQUIRE(rns8_destroy_matrix(value)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
