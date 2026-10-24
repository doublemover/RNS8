#include "explicit_sparse_common.hpp"
#include <cstdlib>

namespace {
uint8_t centered_byte(int64_t value,uint16_t modulus) {
  int64_t residue=value%modulus;
  if(residue<0) residue+=modulus;
  if(residue>=(modulus+1)/2) residue-=modulus;
  return static_cast<uint8_t>(residue); // Defined modulo-256 conversion preserves signed byte bits.
}
void run() {
  using namespace sparse_example;
  constexpr size_t m=6,n=3,k=8,ald=10,bld=5,cld=5;
  constexpr uint32_t prefix=RNS8_DEFAULT_BOUNDED_PREFIX;
  constexpr uint64_t bound=4096;
  auto ctx=context();
  auto selected=plan(ctx.get(),RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,m,n,k,bound);
  auto work=workspace(ctx.get(),selected.get());
  auto dense_a=matrix(ctx.get(),m,k,RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,prefix);
  auto b=matrix(ctx.get(),k,n,RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,prefix);
  auto output=matrix(ctx.get(),m,n,RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,prefix);
  auto dense_output=matrix(ctx.get(),m,n,RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,prefix);
  const auto sparse_desc=descriptor(m,k,RNS8_BOUNDED_I64,RNS8_BOUND_GLOBAL_MAX_ABS,prefix);
  auto a=sparse(ctx.get(),sparse_desc);
  check(rns8_validate_default_moduli(),"validate public default basis");
  require(rns8_default_modulus_count()>=prefix,"default basis is too short");
  std::vector<int64_t> a_values(m*ald,99),b_values(k*bld,99);
  for(size_t row=0;row<m;++row) {
    std::fill_n(a_values.begin()+row*ald,k,int64_t(0));
    for(size_t group=0;group<k/4;++group) {
      const auto& pattern=pairs[(row+group)%6];
      const int64_t magnitude=static_cast<int64_t>(row+group+1);
      a_values[row*ald+group*4+pattern[0]]=magnitude;
      a_values[row*ald+group*4+pattern[1]]=-(magnitude+1);
    }
  }
  for(size_t q=0;q<k;++q) for(size_t col=0;col<n;++col)
    b_values[q*bld+col]=static_cast<int64_t>((q+col*2)%7)-3;
  std::vector<uint8_t> planes(size_t(prefix)*m*ald,0xA5);
  for(uint32_t p=0;p<prefix;++p) {
    const auto modulus=rns8_default_modulus(p);
    require(modulus>8 && modulus<=256,"recipe needs nonzero magnitudes <=8 in every selected channel");
    for(size_t row=0;row<m;++row) for(size_t q=0;q<k;++q)
      planes[(size_t(p)*m+row)*ald+q]=centered_byte(a_values[row*ald+q],modulus);
  }
  const auto imported=compress_import(sparse_desc,planes,ald);
  check(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx.get(),a.get(),imported.data(),ald,101),"pack per-plane bounded sparse A");
  verify_resident_rollback(ctx.get(),a.get(),imported,ald);

  // An integer nonzero need not remain a stored nonzero in every channel.
  // Re-encode one native value as modulus[0]; its first-plane zero breaks V1.
  auto invalid=planes;
  const int64_t mathematical_nonzero=rns8_default_modulus(0);
  for(uint32_t p=0;p<prefix;++p) invalid[size_t(p)*m*ald]=centered_byte(mathematical_nonzero,rns8_default_modulus(p));
  require(mathematical_nonzero!=0 && invalid[0]==0,"nonzero-integer/zero-residue demonstration failed");
  require(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx.get(),a.get(),invalid.data(),ald,102)==RNS8_INVALID_ARGUMENT,
          "integer-only sparsity was incorrectly accepted as a per-plane proof");
  require(storage(a.get()).source_version==101,"rejected residue pattern replaced the source identity");

  check(rns8_pack_i64(ctx.get(),dense_a.get(),a_values.data(),ald,101),"pack dense signed control A");
  check(rns8_pack_i64(ctx.get(),b.get(),b_values.data(),bld,202),"pack dense signed B");
  check(rns8_gemm_rns_sparse_a(ctx.get(),selected.get(),a.get(),b.get(),output.get(),work.get()),"bounded sparse GEMM");
  check(rns8_gemm_rns(ctx.get(),selected.get(),dense_a.get(),b.get(),dense_output.get(),work.get()),"bounded dense control GEMM");
  std::vector<int64_t> observed(m*cld,999),dense_result(observed),expected(observed);
  check(rns8_export_i64(ctx.get(),selected.get(),output.get(),observed.data(),cld),"export sparse bounded output");
  check(rns8_export_i64(ctx.get(),selected.get(),dense_output.get(),dense_result.data(),cld),"export dense bounded control");
  // |every partial| <= K*8*3 =192, independently below INT64_MAX and bound4096.
  // This scalar signed oracle does not reconstruct from the residue inputs.
  for(size_t row=0;row<m;++row) for(size_t col=0;col<n;++col) {
    int64_t sum=0;
    for(size_t q=0;q<k;++q) sum+=a_values[row*ald+q]*b_values[q*bld+col];
    require(sum>=-192 && sum<=192,"recipe exceeds its independently proven oracle bound");
    expected[row*cld+col]=sum;
  }
  require(observed==expected && dense_result==expected,"bounded sparse/dense result or export padding differs from scalar oracle");
  const auto info=storage(a.get());
  require(info.backend==RNS8_BACKEND_CPU_REFERENCE && info.host_current==1 && info.device_current==0 && info.max_prefix==prefix,
          "example did not preserve its explicitly selected CPU/default-prefix contract");
  std::cout<<"recipe=sparse-bounded-i64-v1 bound="<<bound<<" planes="<<prefix<<" groups="<<info.group_count<<'\n';
  print_result("explicit bounded Sparse-A",observed,m,n,cld);
}
}
int main() {
  rns8_example::configure_noninteractive_errors();
  try {run();return sparse_example::cleanup_failed?EXIT_FAILURE:EXIT_SUCCESS;}
  catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return EXIT_FAILURE;}
}
