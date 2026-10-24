#pragma once
#include "backend_common/finite_verify.hpp"
__global__ void rns8_finite_br_kernel(const int8_t* b, uint64_t n, uint64_t k,
    uint32_t q, const uint8_t* r, uint32_t* br, uint32_t* flag) {
  uint32_t errors = 0;
  for (uint64_t row = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; row<k;
       row += uint64_t(gridDim.x)*blockDim.x) {
    uint32_t value = 0;
    for (uint64_t col=0; col<n; ++col)
      value = (value + rns8::detail::finite_verify::decode(b[row*n+col],q,errors)*r[col])%q;
    br[row]=value;
  }
  if (errors) atomicOr(flag,errors);
}
__global__ void rns8_finite_compare_kernel(const int8_t* a, const int8_t* c,
    uint64_t m,uint64_t n,uint64_t k,uint32_t q,const uint8_t* r,const uint32_t* br,uint32_t* flag) {
  uint32_t errors=0;
  for (uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<m;
       row+=uint64_t(gridDim.x)*blockDim.x) {
    uint32_t abr=0,cr=0;
    for(uint64_t j=0;j<k;++j)
      abr=(abr+rns8::detail::finite_verify::decode(a[row*k+j],q,errors)*br[j])%q;
    for(uint64_t col=0;col<n;++col)
      cr=(cr+rns8::detail::finite_verify::decode(c[row*n+col],q,errors)*r[col])%q;
    if(abr!=cr) errors|=1;
  }
  if(errors) atomicOr(flag,errors);
}
extern "C" int rns8_hip_finite_verify(const int8_t* a,const int8_t* b,const int8_t* c,
    uint64_t m,uint64_t n,uint64_t k,uint32_t q,const uint8_t* r,uint32_t* br,uint32_t* flag) {
  const unsigned kb=static_cast<unsigned>(k<262144?(k+255)/256:1024);
  const unsigned mb=static_cast<unsigned>(m<262144?(m+255)/256:1024);
  hipLaunchKernelGGL(rns8_finite_br_kernel,dim3(kb),dim3(256),0,nullptr,b,n,k,q,r,br,flag);
  auto error=hipGetLastError();
  if(error!=hipSuccess) return static_cast<int>(error);
  hipLaunchKernelGGL(rns8_finite_compare_kernel,dim3(mb),dim3(256),0,nullptr,a,c,m,n,k,q,r,br,flag);
  return static_cast<int>(hipGetLastError());
}
