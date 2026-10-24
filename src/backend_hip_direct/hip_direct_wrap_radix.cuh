#include "backend_common/wrap64_signed_radix.hpp"
// Version 1: canonical compact cells -> tile-local signed radix256 view ->
// canonical compact output. Each MMA starts at zero and covers exactly 16 K
// terms, including masked zeros, then folds into the unsigned 64-bit ring.
template<bool Signed>
__global__ void rns8_wrap_radix_wmma_kernel(const uint8_t* a,const uint8_t* b,uint8_t* c,
                                         int m,int n,int k) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  using fragment=int __attribute__((ext_vector_type(4)));
  using accumulator=int __attribute__((ext_vector_type(8)));
  __shared__ uint64_t ap[256],bp[256];
  const int lane=threadIdx.x,local=lane&15;
  const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  uint64_t result[8]{};
  for(int64_t base=0;base<k;base+=16) {
    for(int cell=lane;cell<256;cell+=32) {
      const int r=cell/16,q=cell%16;
      uint64_t av=0,bv=0;
      if(rowbase+r<m && base+q<k)
        for(int byte=0;byte<8;++byte) av|=uint64_t(a[((rowbase+r)*k+base+q)*8+byte])<<(8*byte);
      if(base+r<k && colbase+q<n)
        for(int byte=0;byte<8;++byte) bv|=uint64_t(b[((base+r)*n+colbase+q)*8+byte])<<(8*byte);
      if constexpr(Signed) {
        av=rns8::detail::wrap64_signed_recode(av);
        bv=rns8::detail::wrap64_signed_recode(bv);
      }
      ap[cell]=av; bp[cell]=bv;
    }
    __syncthreads();
    for(int i=0;i<8;++i) for(int j=0;i+j<8;++j) {
      fragment av{},bv{};
      for(int word=0;word<4;++word) {
        uint32_t aw=0,bw=0;
        for(int byte=0;byte<4;++byte) {
          const int q=word*4+byte;
          aw|=uint32_t((ap[local*16+q]>>(8*i))&255)<<(8*byte);
          bw|=uint32_t((bp[q*16+local]>>(8*j))&255)<<(8*byte);
        }
        // Bit-preserving conversion: packed words may exceed signed INT_MAX.
        av[word]=__builtin_bit_cast(int,aw); bv[word]=__builtin_bit_cast(int,bw);
      }
      accumulator acc{};
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(Signed,av,Signed,bv,acc,false);
      for(int reg=0;reg<8;++reg) result[reg]+=uint64_t(int64_t(acc[reg]))<<(8*(i+j));
    }
    __syncthreads();
  }
  for(int reg=0;reg<8;++reg) {
    const int64_t row=rowbase+2*reg+(lane>>4),col=colbase+local;
    if(row<m && col<n)
      for(int byte=0;byte<8;++byte) c[(row*n+col)*8+byte]=uint8_t(result[reg]>>(8*byte));
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;
#endif
}
extern "C" int rns8_hip_wrap_radix_wmma(const uint8_t* a,const uint8_t* b,uint8_t* c,
                                       int m,int n,int k,int signed_digits) {
  const dim3 grid((uint64_t(n)+15)/16,(uint64_t(m)+15)/16);
  if(signed_digits) hipLaunchKernelGGL((rns8_wrap_radix_wmma_kernel<true>),grid,dim3(32),0,nullptr,a,b,c,m,n,k);
  else hipLaunchKernelGGL((rns8_wrap_radix_wmma_kernel<false>),grid,dim3(32),0,nullptr,a,b,c,m,n,k);
  return int(hipGetLastError());
}
