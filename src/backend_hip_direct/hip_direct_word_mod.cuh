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
