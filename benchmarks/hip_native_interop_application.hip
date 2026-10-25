// External application work. Deliberately uses no private RNS8 storage or kernels.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ uint64_t interop_value(uint64_t i,uint32_t seed,bool unsigned_values,bool fullwidth) {
  const uint64_t x=(i*37+seed)%15;
  if(!fullwidth)return unsigned_values?x:uint64_t(int64_t(x)-7);
  return unsigned_values?(x%3==0?UINT64_MAX:x%3==1?(UINT64_C(1)<<63):x):
      (x%3==0?uint64_t(INT64_MIN):x%3==1?uint64_t(INT64_MAX):uint64_t(-int64_t(x)));
}
__global__ void interop_produce(uint64_t* a,uint64_t* b,uint64_t* c,uint64_t ac,uint64_t bc,uint64_t cc,uint32_t seed,bool u,bool full) {
  const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<ac)a[i]=interop_value(i,seed,u,full);if(i<bc)b[i]=interop_value(i+11,seed,u,full);
  if(i<cc)c[i]=UINT64_C(0x5a5a5a5a5a5a5a5a);
}
__global__ void interop_consume(const uint64_t* c,uint64_t* digest,uint64_t count) {
  const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)atomicXor(reinterpret_cast<unsigned long long*>(digest),static_cast<unsigned long long>(c[i]^(i*UINT64_C(0x9e3779b97f4a7c15))));
}
extern "C" int rns8_interop_application_produce(void* a,void* b,void* c,uint64_t ac,uint64_t bc,uint64_t cc,uint32_t seed,bool u,bool full,void* stream) {
  const uint64_t count=ac>bc?(ac>cc?ac:cc):(bc>cc?bc:cc);
  hipLaunchKernelGGL(interop_produce,dim3(unsigned((count+255)/256)),dim3(256),0,static_cast<hipStream_t>(stream),
      static_cast<uint64_t*>(a),static_cast<uint64_t*>(b),static_cast<uint64_t*>(c),ac,bc,cc,seed,u,full);return int(hipGetLastError());
}
extern "C" int rns8_interop_application_consume(const void* c,void* digest,uint64_t count,void* stream) {
  hipLaunchKernelGGL(interop_consume,dim3(unsigned((count+255)/256)),dim3(256),0,static_cast<hipStream_t>(stream),
      static_cast<const uint64_t*>(c),static_cast<uint64_t*>(digest),count);return int(hipGetLastError());
}
