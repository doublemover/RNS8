#include <rns8/rns8.h>
#include "noninteractive_errors.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(rns8_status status,const char* operation) {
  if(status!=RNS8_SUCCESS) throw std::runtime_error(std::string(operation)+": "+rns8_status_string(status));
}
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
}
int main(int argc,char** argv) {
  rns8_example::configure_noninteractive_errors();
  rns8_context* context=nullptr;rns8_stream_gemm_plan* plan=nullptr;
  try {
    const bool hip=argc==2 && std::strcmp(argv[1],"--hip")==0;
    require(argc==1 || hip,"usage: rns8-downstream-stream-bounded-i64 [--hip]");
    rns8_stream_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.semantics=RNS8_BOUNDED_I64;d.m=17;d.n=19;d.k=33;d.lda=35;d.ldb=21;d.ldc=23;
    d.panel_m=3;d.panel_n=4;d.panel_k=5;d.prefix=RNS8_DEFAULT_BOUNDED_PREFIX;
    d.max_abs_a=5;d.max_abs_b=3;
    rns8_stream_gemm_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    require(rns8_query_stream_gemm(&d,&resources)==RNS8_WORKSPACE_TOO_SMALL,"zero budgets must report required capacity");
    require(resources.minimum_device_bytes && resources.minimum_host_bytes,"minimum capacity missing");
    d.device_budget_bytes=resources.device_bytes;d.host_budget_bytes=resources.host_bytes;
    check(rns8_query_stream_gemm(&d,&resources),"query exact budgets");
    const uint64_t resident_residues=uint64_t(d.m*d.k+d.k*d.n+d.m*d.n)*d.prefix;
    require(resources.device_bytes<resident_residues,"fixture must force capacity below complete resident RNS operands");
    std::cout<<"recipe=stream-bounded-i64-v1 mode="<<(hip?"hip":"query-only")
      <<" device_payload_bytes="<<resources.device_bytes<<" host_payload_bytes="<<resources.host_bytes
      <<" full_resident_residue_bytes="<<resident_residues<<" tiles="<<resources.output_tiles
      <<" k_panels_per_tile="<<resources.k_panels_per_tile<<'\n';
    if(!hip) return EXIT_SUCCESS;
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0,&options,&context),"create direct HIP context");
    check(rns8_create_stream_gemm(context,&d,&plan),"create fixed-budget plan");
    require(rns8_destroy_context(context)==RNS8_BUSY,"stream plan must retain context");
    std::vector<int64_t> a(std::size_t(d.m*d.lda),99),b(std::size_t(d.k*d.ldb),99),c(std::size_t(d.m*d.ldc),99),expected(c);
    for(int64_t row=0;row<d.m;++row) for(int64_t q=0;q<d.k;++q) a[std::size_t(row*d.lda+q)]=(row+q)%11-5;
    for(int64_t q=0;q<d.k;++q) for(int64_t col=0;col<d.n;++col) b[std::size_t(q*d.ldb+col)]=(q+col)%7-3;
    rns8_stream_gemm_result result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
    for(uint64_t generation=1;generation<=2;++generation) {
      if(generation==2) a[0]=1;
      // Independently bounded native oracle: every partial magnitude <=33*5*3.
      for(int64_t row=0;row<d.m;++row) for(int64_t col=0;col<d.n;++col) {
        int64_t sum=0;for(int64_t q=0;q<d.k;++q) sum+=a[std::size_t(row*d.lda+q)]*b[std::size_t(q*d.ldb+col)];
        require(sum>=-495 && sum<=495,"oracle envelope violated");expected[std::size_t(row*d.ldc+col)]=sum;
      }
      check(rns8_execute_stream_gemm(plan,a.data(),b.data(),c.data(),101,generation,202,1,&result),"stream product");
      require(c==expected,"full product or padding mismatch");
      require(result.output_valid && !result.quarantined && result.epoch==generation,"completion identity mismatch");
      require(result.completed_tiles==resources.output_tiles &&
        result.completed_panels==resources.output_tiles*resources.k_panels_per_tile,"incomplete panel coverage");
    }
    const auto previous=c;const auto epoch=result.epoch;a[0]=6;
    require(rns8_execute_stream_gemm(plan,a.data(),b.data(),c.data(),101,3,202,1,&result)==RNS8_RANGE_ERROR,"violated source bound accepted");
    require(c==previous && result.epoch==epoch,"preflight failure published output");
    check(rns8_destroy_stream_gemm(plan),"destroy stream plan");plan=nullptr;
    check(rns8_destroy_context(context),"destroy context");context=nullptr;
    std::cout<<"full_output_verified=1 padding_verified=1 generations=2\n";
    return EXIT_SUCCESS;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    // A failed drain/free deliberately keeps the owner alive until process exit.
    if(plan && rns8_destroy_stream_gemm(plan)!=RNS8_SUCCESS) return EXIT_FAILURE;
    if(context) (void)rns8_destroy_context(context);
    return EXIT_FAILURE;
  }
}
