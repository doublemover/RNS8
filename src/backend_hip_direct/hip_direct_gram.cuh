#include "backend_common/gram_plan.hpp"
__device__ __forceinline__ int8_t rns8_gram_load(const int8_t* x,int64_t plane,int64_t index,
    int64_t q,int rows,int cols,int ld,bool transposed) {
  return x[transposed?(plane*rows+q)*ld+index:(plane*rows+index)*ld+q];
}
template<bool Wmma>
__global__ void rns8_gram_kernel(const int8_t* x,int8_t* c,const rns8::detail::GramTile* jobs,
                                int rows,int cols,int ld,int ldc,int modulus,bool transposed) {
  const auto job=jobs[blockIdx.x];
  const int64_t n=transposed?cols:rows,k=transposed?rows:cols,plane=blockIdx.y;
  const int qmod=modulus?modulus:rns8_default_moduli_device[plane];
  if constexpr(!Wmma) {
    const int64_t row=int64_t(job.row)*16+threadIdx.x/16,col=int64_t(job.col)*16+threadIdx.x%16;
    if(row>=n || col>=n || row>col) return;
    int sum=0;
    for(int64_t q=0;q<k;++q)
      sum=(sum+int(rns8_gram_load(x,plane,row,q,rows,cols,ld,transposed))*
                   int(rns8_gram_load(x,plane,col,q,rows,cols,ld,transposed)))%qmod;
    if(sum<0) sum+=qmod;if(sum>=(qmod+1)/2) sum-=qmod;
    c[(plane*n+row)*ldc+col]=int8_t(sum);
    if(row!=col) c[(plane*n+col)*ldc+row]=int8_t(sum);
  } else {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
    using fragment=int __attribute__((ext_vector_type(4)));
    using accumulator=int __attribute__((ext_vector_type(8)));
    const int lane=threadIdx.x,local=lane&15;
    const int64_t ar=int64_t(job.row)*16+local,bc=int64_t(job.col)*16+local;
    accumulator acc{};
    for(int64_t base=0;base<k;base+=16) {
      fragment av{},bv{};
      for(int word=0;word<4;++word) {
        uint32_t aw=0,bw=0;
        for(int byte=0;byte<4;++byte) {
          const int64_t q=base+word*4+byte;
          if(ar<n && q<k) aw|=uint32_t(uint8_t(rns8_gram_load(x,plane,ar,q,rows,cols,ld,transposed)))<<(8*byte);
          if(bc<n && q<k) bw|=uint32_t(uint8_t(rns8_gram_load(x,plane,bc,q,rows,cols,ld,transposed)))<<(8*byte);
        }
        av[word]=__builtin_bit_cast(int,aw);bv[word]=__builtin_bit_cast(int,bw);
      }
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,acc,false);
      for(int reg=0;reg<8;++reg) acc[reg]%=qmod;
    }
    for(int reg=0;reg<8;++reg) {
      const int64_t row=int64_t(job.row)*16+2*reg+(lane>>4),col=bc;
      if(row>=n || col>=n || row>col) continue;
      int sum=acc[reg];if(sum<0) sum+=qmod;if(sum>=(qmod+1)/2) sum-=qmod;
      c[(plane*n+row)*ldc+col]=int8_t(sum);
      if(row!=col) c[(plane*n+col)*ldc+row]=int8_t(sum);
    }
#endif
  }
}
extern "C" int rns8_hip_gram(const int8_t* x,int8_t* c,const rns8::detail::GramTile* jobs,
    int count,int rows,int cols,int ld,int ldc,int prefix,int modulus,int transposed,int wmma) {
  const dim3 grid(count,prefix);
  if(wmma) hipLaunchKernelGGL((rns8_gram_kernel<true>),grid,dim3(32),0,nullptr,
      x,c,jobs,rows,cols,ld,ldc,modulus,transposed!=0);
  else hipLaunchKernelGGL((rns8_gram_kernel<false>),grid,dim3(256),0,nullptr,
      x,c,jobs,rows,cols,ld,ldc,modulus,transposed!=0);
  return int(hipGetLastError());
}
