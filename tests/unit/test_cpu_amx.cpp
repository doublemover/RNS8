#include <catch2/catch_test_macros.hpp>
#include "core/api_internal.hpp"
#include "cpu/cpu_amx.hpp"
#include <array>
#include <vector>
#include <limits>
using boost::multiprecision::cpp_int;
namespace {
template<class T>T abi(){T value{};value.struct_size=sizeof(value);value.abi_version=RNS8_ABI_VERSION;return value;}
struct CpuOwner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  ~CpuOwner(){rns8_destroy_workspace(work);rns8_destroy_matrix(c);rns8_destroy_matrix(b);rns8_destroy_matrix(a);rns8_destroy_plan(plan);rns8_destroy_context(ctx);}
  bool create(rns8_backend_kind backend,int64_t m,int64_t n,int64_t k,rns8_semantics semantics,uint32_t prefix,uint16_t modulus=0){
    auto options=abi<rns8_context_options>();options.requested_backend=backend;
    const auto status=rns8_create_context(-1,&options,&ctx);
    if(status==RNS8_UNSUPPORTED_BACKEND){CHECK(ctx==nullptr);return false;}REQUIRE(status==RNS8_SUCCESS);
    auto d=abi<rns8_gemm_desc>();d.requested_backend=backend;d.m=m;d.n=n;d.k=k;d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=prefix;d.finite_modulus=modulus;d.flags=modulus?0:RNS8_PLAN_FORCE_FIXED_PREFIX;d.max_prefix=modulus?0:prefix;
    REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    auto matrix=[&](int64_t rows,int64_t cols,rns8_matrix** out){auto md=abi<rns8_matrix_desc>();md.rows=rows;md.cols=cols;md.logical_ld=cols+3;
      md.semantics=semantics;md.bound_kind=RNS8_BOUND_NONE;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=modulus?0:prefix;REQUIRE(rns8_create_matrix(ctx,&md,out)==RNS8_SUCCESS);};
    matrix(m,k,&a);matrix(k,n,&b);matrix(m,n,&c);return true;
  }
};
void compare_finite(CpuOwner& amx,int64_t m,int64_t n,int64_t k,uint16_t q,const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){
  CpuOwner scalar;REQUIRE(scalar.create(RNS8_BACKEND_CPU_REFERENCE,m,n,k,RNS8_FINITE_RING_U8,1,q));
  for(auto* o:{&amx,&scalar}){
    REQUIRE(rns8_pack_finite_u8(o->ctx,o->a,q,a.data()+1,k+3,0)==RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(o->ctx,o->b,q,b.data()+1,n+5,0)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(o->ctx,o->plan,q,o->a,o->b,o->c,o->work)==RNS8_SUCCESS);
  }
  CHECK(amx.c->residues==scalar.c->residues);
  std::vector<uint8_t> output(size_t(m)*(n+7),91);
  REQUIRE(rns8_export_finite_u8(amx.ctx,amx.plan,q,amx.c,output.data(),n+7)==RNS8_SUCCESS);
  for(int64_t r=0;r<m;++r)for(int64_t c=0;c<n+7;++c){
    if(c>=n){CHECK(output[r*(n+7)+c]==91);continue;}
    cpp_int expected=0;for(int64_t x=0;x<k;++x)expected+=cpp_int(a[1+r*(k+3)+x])*b[1+x*(n+5)+c];
    CHECK(output[r*(n+7)+c]==(expected%q).convert_to<uint8_t>());
  }
}
}
TEST_CASE("AMX admission needs compiled tile INT8 OS permission and xstate", "[cpu-amx][contract]") {
  auto good=abi<rns8_cpu_amx_info>();good.compiled=good.cpuid_tile=good.cpuid_int8=good.osxsave=good.palette_supported=good.os_supported=good.permission_enabled=good.xcr0_enabled=1;
  CHECK(rns8::detail::cpu_amx_admitted(good));
  for(auto member:{&rns8_cpu_amx_info::compiled,&rns8_cpu_amx_info::cpuid_tile,&rns8_cpu_amx_info::cpuid_int8,&rns8_cpu_amx_info::osxsave,
      &rns8_cpu_amx_info::palette_supported,&rns8_cpu_amx_info::os_supported,&rns8_cpu_amx_info::permission_enabled,&rns8_cpu_amx_info::xcr0_enabled}){
    auto missing=good;missing.*member=0;CHECK_FALSE(rns8::detail::cpu_amx_admitted(missing));}
  auto observed=abi<rns8_cpu_amx_info>();REQUIRE(rns8_query_cpu_amx(0,&observed)==RNS8_SUCCESS);
  CHECK(observed.admitted==uint32_t(rns8::detail::cpu_amx_admitted(observed)));
  CHECK(rns8_query_cpu_amx(2,&observed)==RNS8_INVALID_ARGUMENT);
  uint64_t bytes=0;CHECK(rns8::detail::cpu_amx_workspace_bytes(17,19,65,20,4,bytes));CHECK(bytes>20*17*19);
  CHECK_FALSE(rns8::detail::cpu_amx_workspace_bytes(INT64_MAX,INT64_MAX,INT64_MAX,20,4,bytes));
  CHECK_FALSE(rns8::detail::cpu_amx_workspace_bytes(1,1,1,20,0,bytes));
}
TEST_CASE("AMX k4-interleaved B layout has independent scalar dot semantics", "[cpu-amx][contract]") {
  for(int64_t k:{1,3,4,63,64,65,127})for(int64_t n:{1,15,16,17,31}){
    std::vector<int8_t> b(size_t(k)*(n+5));for(size_t i=0;i<b.size();++i)b[i]=int8_t(i%256-128);
    std::array<int8_t,1024> tile{};
    for(int64_t kb=0;kb<k;kb+=64)for(int64_t nb=0;nb<n;nb+=16){
      rns8::detail::cpu_amx_pack_b_tile(b.data(),k,n,n+5,kb,nb,tile.data());
      for(int col=0;col<16;++col){int64_t via_tile=0,logical=0;
        for(int x=0;x<64;++x){const int lhs=x*3-128;via_tile+=lhs*int(tile[(x/4)*64+col*4+x%4]);
          if(kb+x<k && nb+col<n)logical+=lhs*int(b[(kb+x)*(n+5)+nb+col]);else CHECK(tile[(x/4)*64+col*4+x%4]==0);}
        CHECK(via_tile==logical);}
    }
  }
}
TEST_CASE("AMX exhaustive byte products agree with independent scalar and Boost", "[cpu-amx][hardware]") {
  for(uint16_t q:{2,16,167,253,254,255,256}){
    CpuOwner o;if(!o.create(RNS8_BACKEND_CPU_AMX,256,256,1,RNS8_FINITE_RING_U8,1,q)){SUCCEED("AMX target or OS permission unavailable");return;}
    std::vector<uint8_t> a(1+256*4),b(1+261);for(int i=0;i<256;++i){a[1+i*4]=i;b[1+i]=i;}
    compare_finite(o,256,256,1,q,a,b);
  }
}
TEST_CASE("AMX tails K segmentation centering thread counts and cache generations", "[cpu-amx][hardware]") {
  struct Reset {uint32_t requested=rns8::detail::cpu_parallel_requested_threads(),maximum=rns8::detail::cpu_parallel_max_threads();uint64_t threshold=rns8::detail::cpu_parallel_threshold_ops();bool progress=rns8::detail::cpu_parallel_progress_enabled();
    ~Reset(){rns8::detail::configure_cpu_parallel(maximum,threshold,progress);rns8::detail::configure_cpu_parallel(requested,threshold,progress);}} reset;
  for(uint32_t threads:{1,2,4})for(int64_t k:{63,64,65,65535,65536,65537,131071,131072,131073}){
    rns8::detail::configure_cpu_parallel(threads,0,false);CpuOwner o;
    const int64_t m=k>1000?1:17,n=k>1000?3:19;const uint16_t q=k%2?255:256;
    if(!o.create(RNS8_BACKEND_CPU_AMX,m,n,k,RNS8_FINITE_RING_U8,1,q)){SUCCEED("AMX unavailable");return;}
    std::vector<uint8_t>a(1+size_t(m)*(k+3),128),b(1+size_t(k)*(n+5),128);
    compare_finite(o,m,n,k,q,a,b);
    auto first=abi<rns8_cpu_amx_workspace_info>();REQUIRE(rns8_get_cpu_amx_workspace_info(o.work,&first)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(o.ctx,o.plan,q,o.a,o.b,o.c,o.work)==RNS8_SUCCESS);
    auto reused=abi<rns8_cpu_amx_workspace_info>();REQUIRE(rns8_get_cpu_amx_workspace_info(o.work,&reused)==RNS8_SUCCESS);
    CHECK(reused.b_pack_count==first.b_pack_count);CHECK(reused.b_reuse_count==first.b_reuse_count+1);
    CHECK(reused.owned_payload_bytes==first.owned_payload_bytes);CHECK(reused.safe_k==65536);
    b[1]=3;REQUIRE(rns8_pack_finite_u8(o.ctx,o.b,q,b.data()+1,n+5,0)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(o.ctx,o.plan,q,o.a,o.b,o.c,o.work)==RNS8_SUCCESS);
    auto changed=abi<rns8_cpu_amx_workspace_info>();REQUIRE(rns8_get_cpu_amx_workspace_info(o.work,&changed)==RNS8_SUCCESS);
    CHECK(changed.b_pack_count==first.b_pack_count+1);CHECK(changed.b_generation!=first.b_generation);
    compare_finite(o,m,n,k,q,a,b);
  }
}
TEST_CASE("AMX exact wide public CRT and prepared generations remain independent", "[cpu-amx][hardware]") {
  for(bool u:{false,true})for(int64_t n:{1,15,16,17,31}){
    CpuOwner o;const int64_t m=17,k=1;
    if(!o.create(RNS8_BACKEND_CPU_AMX,m,n,k,u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED,20)){SUCCEED("AMX unavailable");return;}
    std::vector<uint64_t>a(m*4),b(n+5);for(int64_t r=0;r<m;++r)a[r*4]=u?UINT64_MAX:(r%2?uint64_t(INT64_MAX):uint64_t(INT64_MIN));
    for(int64_t c=0;c<n;++c)b[c]=u?UINT64_MAX:(c%2?uint64_t(INT64_MAX):uint64_t(INT64_MIN));
    auto pack=[&](){if(u){REQUIRE(rns8_pack_u64(o.ctx,o.a,a.data(),4,0)==RNS8_SUCCESS);REQUIRE(rns8_pack_u64(o.ctx,o.b,b.data(),n+5,0)==RNS8_SUCCESS);}
      else{REQUIRE(rns8_pack_i64(o.ctx,o.a,reinterpret_cast<int64_t*>(a.data()),4,0)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(o.ctx,o.b,reinterpret_cast<int64_t*>(b.data()),n+5,0)==RNS8_SUCCESS);}};
    pack();rns8_prepared_execution* prepared=nullptr;REQUIRE(rns8_prepare_gemm(o.ctx,o.plan,o.a,o.b,o.c,o.work,&prepared)==RNS8_SUCCESS);
    rns8_prepared_group* group=nullptr;REQUIRE(rns8_create_prepared_group(&prepared,1,&group)==RNS8_SUCCESS);
    auto check=[&](){rns8_completion_token token{};uint32_t executed=0;REQUIRE(rns8_execute_prepared_group(group,&token,1,&executed)==RNS8_SUCCESS);CHECK(executed==1);
      std::vector<uint64_t> out(m*(n+3)*3,91);REQUIRE((u?rns8_export_exact_wide_unsigned_limbs(o.ctx,o.plan,o.c,out.data(),n+3,3):rns8_export_exact_wide_signed_limbs(o.ctx,o.plan,o.c,out.data(),n+3,3))==RNS8_SUCCESS);
      for(int64_t r=0;r<m;++r)for(int64_t c=0;c<n+3;++c){cpp_int v=0;if(c<n)v=(u?cpp_int(a[r*4]):cpp_int(int64_t(a[r*4])))*(u?cpp_int(b[c]):cpp_int(int64_t(b[c])));
        if(c<n)for(unsigned p=0;p<20;++p)CHECK(o.c->residues[p*m*n+r*n+c]==rns8::detail::centered_residue(v,rns8_default_modulus(p)));
        if(v<0)v+=cpp_int(1)<<192;for(unsigned l=0;l<3;++l)CHECK(out[(r*(n+3)+c)*3+l]==(c<n?uint64_t((v>>(64*l))&UINT64_MAX):91));}};
    check();b[0]=7;pack();rns8_completion_token token{};uint32_t executed=99;CHECK(rns8_execute_prepared_group(group,&token,1,&executed)==RNS8_INVALID_ARGUMENT);CHECK(executed==0);
    REQUIRE(rns8_refresh_prepared_group_inputs(group)==RNS8_SUCCESS);check();REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);
  }
}
TEST_CASE("AMX bounded adaptive tiles preserve active-prefix and zero contracts", "[cpu-amx][hardware]") {
  auto options=abi<rns8_context_options>();options.requested_backend=RNS8_BACKEND_CPU_AMX;CpuOwner o;
  const auto status=rns8_create_context(-1,&options,&o.ctx);if(status==RNS8_UNSUPPORTED_BACKEND){SUCCEED("AMX unavailable");return;}REQUIRE(status==RNS8_SUCCESS);
  constexpr int64_t m=65,n=67,k=17;const uint64_t bounds[]{0,0,0,17};
  auto d=abi<rns8_gemm_desc>();d.requested_backend=RNS8_BACKEND_CPU_AMX;d.semantics=RNS8_BOUNDED_I64;d.bound_kind=RNS8_BOUND_PER_TILE_MAX_ABS;
  d.m=m;d.n=n;d.k=k;d.max_prefix=9;d.tile_m=d.tile_n=64;d.tile_bounds=bounds;d.tile_bounds_count=4;d.flags=RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  REQUIRE(rns8_create_plan(o.ctx,&d,&o.plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(o.ctx,o.plan,&o.work)==RNS8_SUCCESS);
  auto matrix=[&](int64_t rows,int64_t cols,rns8_matrix** out){auto md=abi<rns8_matrix_desc>();md.rows=rows;md.cols=cols;md.logical_ld=cols;
    md.semantics=d.semantics;md.bound_kind=d.bound_kind;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=9;md.tile_m=md.tile_n=64;REQUIRE(rns8_create_matrix(o.ctx,&md,out)==RNS8_SUCCESS);};
  matrix(m,k,&o.a);matrix(k,n,&o.b);matrix(m,n,&o.c);
  std::vector<int64_t>a(m*k),b(k*n),out(m*(n+3),91);for(int x=0;x<k;++x){a[64*k+x]=1;for(int c=64;c<n;++c)b[x*n+c]=c%2?1:-1;}
  REQUIRE(rns8_pack_i64(o.ctx,o.a,a.data(),k,0)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(o.ctx,o.b,b.data(),n,0)==RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(o.ctx,o.plan,o.a,o.b,o.c,o.work)==RNS8_SUCCESS);REQUIRE(rns8_export_i64(o.ctx,o.plan,o.c,out.data(),n+3)==RNS8_SUCCESS);
  for(int r=0;r<m;++r)for(int c=0;c<n+3;++c)CHECK(out[r*(n+3)+c]==(c>=n?91:r==64 && c>=64?(c%2?17:-17):0));
  CHECK(o.c->materialized_rns_schedule!=nullptr);
}
