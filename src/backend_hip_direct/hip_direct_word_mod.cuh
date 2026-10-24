#include "backend_common/word_mod_arithmetic.hpp"
__global__ void rns8_word_mod_validate(const uint64_t* a,const uint64_t* b,int m,int n,int k,int lda,int ldb,uint64_t modulus,int* invalid) {
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index<int64_t(m)*k && a[(index/k)*lda+index%k]>=modulus) atomicExch(invalid,1);
  if(index<int64_t(k)*n && b[(index/n)*ldb+index%n]>=modulus) atomicExch(invalid,1);
}
__global__ void rns8_word_mod_gemm(const uint64_t* a,const uint64_t* b,uint64_t* c,int m,int n,int k,int lda,int ldb,int ldc,uint64_t modulus) {
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=int64_t(m)*n) return;
  const int row=index/n,col=index%n;uint64_t sum=0;
  for(int h=0;h<k;++h) sum=rns8::detail::word_mod::add(sum,
      rns8::detail::word_mod::multiply(a[int64_t(row)*lda+h],b[int64_t(h)*ldb+col],modulus),modulus);
  c[int64_t(row)*ldc+col]=sum;
}
extern "C" int rns8_hip_word_mod_validate(const void* a,const void* b,int m,int n,int k,int lda,int ldb,uint64_t modulus,int* invalid) {
  const uint64_t ac=uint64_t(m)*k,bc=uint64_t(k)*n;
  hipLaunchKernelGGL(rns8_word_mod_validate,dim3(((ac>bc?ac:bc)+255)/256),dim3(256),0,nullptr,
      static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),m,n,k,lda,ldb,modulus,invalid);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_word_mod_gemm(const void* a,const void* b,void* c,int m,int n,int k,int lda,int ldb,int ldc,uint64_t modulus) {
  hipLaunchKernelGGL(rns8_word_mod_gemm,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,
      static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),static_cast<uint64_t*>(c),m,n,k,lda,ldb,ldc,modulus);
  return int(hipGetLastError());
}

#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && __has_builtin(__builtin_amdgcn_udot4)
#define RNS8_WORD_DOT4 1
#else
#define RNS8_WORD_DOT4 0
#endif
__global__ void rns8_word_mod_dot4_capability(int* flag) {*flag=RNS8_WORD_DOT4;}
__global__ void rns8_word_mod_dot4_kernel(const uint64_t* a,const uint64_t* b,uint64_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,uint64_t modulus,rns8::detail::word_mod::DigitWeights weights) {
#if RNS8_WORD_DOT4
  using namespace rns8::detail::word_mod;
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=int64_t(m)*n) return;
  const int row=index/n,col=index%n;uint64_t result=0;
  for(int64_t begin=0;begin<k;begin+=digit_slab_k) {
    const int64_t end=begin+digit_slab_k<k?begin+digit_slab_k:k;
    uint32_t diagonal[15]{};
    for(int64_t q=begin;q<end;q+=4) {
      uint32_t left[8]{},right[8]{};
      for(unsigned lane=0;lane<4;++lane) {
        const uint64_t av=q+lane<end?a[int64_t(row)*lda+q+lane]:0;
        const uint64_t bv=q+lane<end?b[(q+lane)*ldb+col]:0;
        for(unsigned digit=0;digit<8;++digit) {
          left[digit]|=uint32_t((av>>(8*digit))&255)<<(8*lane);
          right[digit]|=uint32_t((bv>>(8*digit))&255)<<(8*lane);
        }
      }
      for(unsigned i=0;i<8;++i) for(unsigned j=0;j<8;++j)
        diagonal[i+j]=__builtin_amdgcn_udot4(left[i],right[j],diagonal[i+j],false);
    }
    // Each diagonal has at most eight products per K term; the fixed 4096
    // slab keeps even the largest one below INT32_MAX. Reduce before folding.
    for(unsigned d=0;d<15;++d)
      result=add(result,multiply(weights.values[d],diagonal[d]%modulus,modulus),modulus);
  }
  c[int64_t(row)*ldc+col]=result;
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)modulus;(void)weights;
#endif
}
extern "C" int rns8_hip_word_mod_dot4_capability(int* flag) {
  hipLaunchKernelGGL(rns8_word_mod_dot4_capability,dim3(1),dim3(1),0,nullptr,flag);return int(hipGetLastError());
}
extern "C" int rns8_hip_word_mod_dot4(const void* a,const void* b,void* c,int m,int n,int k,int lda,int ldb,int ldc,
    uint64_t modulus,rns8::detail::word_mod::DigitWeights weights) {
  hipLaunchKernelGGL(rns8_word_mod_dot4_kernel,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,
      static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),static_cast<uint64_t*>(c),m,n,k,lda,ldb,ldc,modulus,weights);
  return int(hipGetLastError());
}
#undef RNS8_WORD_DOT4
