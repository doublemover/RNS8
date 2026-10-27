#pragma once
#include "address_domain.hpp"
namespace rns8::detail {
struct CpuI8mmLayout {
  uint64_t mt=0,nt=0,kt=0,packed=0,output=0,total=0,jobs=0;
};
inline bool cpu_i8mm_layout(int64_t m,int64_t n,int64_t k,uint32_t prefix,CpuI8mmLayout& out) {
  AddressDomain a,b,c;
  if(!prefix || prefix>20 || !checked_host_plane_address(m,k,k,prefix,1,a) ||
      !checked_host_plane_address(k,n,n,prefix,1,b) || !checked_host_plane_address(m,n,n,prefix,1,c))return false;
  CpuI8mmLayout s;s.mt=(uint64_t(m)+1)/2;s.nt=(uint64_t(n)+1)/2;s.kt=(uint64_t(k)+7)/8;
  const auto limit=uint64_t(PTRDIFF_MAX);
  if(s.nt>limit/s.kt || s.nt*s.kt>limit/(16*uint64_t(prefix)) || s.mt>limit/s.nt || s.mt*s.nt>limit/prefix)return false;
  s.packed=s.nt*s.kt*16*prefix;s.output=c.byte_extent;s.jobs=s.mt*s.nt*prefix;
  if(s.packed>limit-s.output)return false;s.total=s.packed+s.output;out=s;return true;
}
// SMMLA's B operand is two columns of eight signed bytes, not the row-major
// interleaving used by AMX. All physical tails are numerical zero.
inline void cpu_i8mm_pack_b(const int8_t* source,int64_t k,int64_t n,int64_t ld,int64_t kb,int64_t col,int8_t* output) noexcept {
  for(int c=0;c<2;++c)for(int q=0;q<8;++q)
    output[c*8+q]=(col+c<n && kb+q<k)?source[(kb+q)*ld+col+c]:0;
}
}
