#include "explicit_sparse_common.hpp"
#include <cstdlib>

namespace {
void run() {
  using namespace sparse_example;
  constexpr size_t m=6,n=3,k=8,ald=10,bld=5,cld=5;
  constexpr uint16_t modulus=250; // Composite ring: nonzero need not mean invertible.
  auto ctx=context();
  auto selected=plan(ctx.get(),RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,m,n,k,0,modulus);
  auto work=workspace(ctx.get(),selected.get());
  auto dense_a=matrix(ctx.get(),m,k,RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,0);
  auto b=matrix(ctx.get(),k,n,RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,0);
  auto output=matrix(ctx.get(),m,n,RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,0);
  auto dense_output=matrix(ctx.get(),m,n,RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,0);
  auto sparse_desc=descriptor(m,k,RNS8_FINITE_RING_U8,RNS8_BOUND_NONE,0,modulus);
  auto a=sparse(ctx.get(),sparse_desc);

  std::vector<uint8_t> a_values(m*ald,0xA5),b_values(k*bld,0xA5);
  for(size_t row=0;row<m;++row) {
    std::fill_n(a_values.begin()+row*ald,k,uint8_t(0));
    for(size_t group=0;group<k/4;++group) {
      const auto& pattern=pairs[(row+group)%6]; // Every legal pair is exercised.
      a_values[row*ald+group*4+pattern[0]]=125;
      a_values[row*ald+group*4+pattern[1]]=uint8_t(3*(row+group+1));
    }
  }
  for(size_t q=0;q<k;++q) for(size_t col=0;col<n;++col)
    b_values[q*bld+col]=uint8_t((q*13+col*17+2)%modulus);
  b_values[0]=2; // 125*2 is zero in this ring, without dropping either input.
  const auto imported=compress_import(sparse_desc,a_values,ald);
  check(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx.get(),a.get(),imported.data(),ald,101),"pack resident finite sparse A");
  verify_resident_rollback(ctx.get(),a.get(),imported,ald);
  check(rns8_pack_finite_u8(ctx.get(),dense_a.get(),modulus,a_values.data(),ald,101),"pack dense finite control A");
  check(rns8_pack_finite_u8(ctx.get(),b.get(),modulus,b_values.data(),bld,202),"pack dense finite B");

  check(rns8_gemm_finite_u8_sparse_a(ctx.get(),selected.get(),modulus,a.get(),b.get(),output.get(),work.get()),"finite sparse GEMM");
  check(rns8_gemm_finite_u8(ctx.get(),selected.get(),modulus,dense_a.get(),b.get(),dense_output.get(),work.get()),"finite dense control GEMM");
  std::vector<uint8_t> observed(m*cld,0xA5),dense_result(observed),expected(observed);
  check(rns8_export_finite_u8(ctx.get(),selected.get(),modulus,output.get(),observed.data(),cld),"export sparse finite output");
  check(rns8_export_finite_u8(ctx.get(),selected.get(),modulus,dense_output.get(),dense_result.data(),cld),"export dense finite control");
  // Independent ordinary integer sum, then one remainder. At most 8*249^2,
  // so uint64_t cannot overflow; no RNS/CRT or library result is the oracle.
  for(size_t row=0;row<m;++row) for(size_t col=0;col<n;++col) {
    uint64_t sum=0;
    for(size_t q=0;q<k;++q) sum+=uint64_t(a_values[row*ald+q])*b_values[q*bld+col];
    expected[row*cld+col]=uint8_t(sum%modulus);
  }
  require(observed==expected && dense_result==expected,"finite sparse/dense result or export padding differs from scalar oracle");
  require(rns8_gemm_finite_u8_sparse_a(ctx.get(),selected.get(),251,a.get(),b.get(),output.get(),work.get())==RNS8_INVALID_ARGUMENT,
          "mismatched finite modulus was accepted");
  std::vector<uint8_t> preserved(expected.size(),0xA5);
  check(rns8_export_finite_u8(ctx.get(),selected.get(),modulus,output.get(),preserved.data(),cld),"export after rejected modulus");
  require(preserved==expected,"rejected modulus changed the previous result");
  const auto info=storage(a.get());
  require(info.backend==RNS8_BACKEND_CPU_REFERENCE && info.host_current==1 && info.device_current==0,
          "example did not execute the explicitly requested CPU sparse route");
  std::cout<<"recipe=sparse-finite-ring-v1 modulus="<<modulus<<" planes=1 groups="<<info.group_count<<'\n';
  print_result("explicit finite Sparse-A",observed,m,n,cld);
}
}
int main() {
  rns8_example::configure_noninteractive_errors();
  try {run();return sparse_example::cleanup_failed?EXIT_FAILURE:EXIT_SUCCESS;}
  catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return EXIT_FAILURE;}
}
