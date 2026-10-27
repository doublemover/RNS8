#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void produce(int64_t* a,int64_t* b,int64_t* c) {
  const int i=threadIdx.x;
  if(i<22) a[i]=99;if(i<21) b[i]=99;if(i<13) c[i]=99;
  __syncthreads();
  if(i<15) {const int row=i/5,q=i%5;a[1+row*7+q]=(row+q)%11-5;}
  if(i<10) {const int q=i/2,col=i%2;b[1+q*4+col]=(q+col)%7-3;}
}
__global__ void consume(const int64_t* c,int* failure) {
  const int i=threadIdx.x;if(i>=13) return;
  int64_t expected=99;
  if(i>=1 && i<11 && (i-1)%4<2) {
    const int row=(i-1)/4,col=(i-1)%4;expected=0;
    for(int q=0;q<5;++q) expected+=int64_t((row+q)%11-5)*int64_t((q+col)%7-3);
  }
  if(c[i]!=expected) atomicOr(failure,1);
}
extern "C" int native_produce(void* a,void* b,void* c) {
  hipLaunchKernelGGL(produce,dim3(1),dim3(32),0,nullptr,static_cast<int64_t*>(a),static_cast<int64_t*>(b),static_cast<int64_t*>(c));
  return int(hipGetLastError());
}
extern "C" int native_consume(const void* c,void* failure) {
  hipLaunchKernelGGL(consume,dim3(1),dim3(32),0,nullptr,static_cast<const int64_t*>(c),static_cast<int*>(failure));
  return int(hipGetLastError());
}

extern "C" int native_produce_on_stream(void* a,void* b,void* c,void* stream) {
  hipLaunchKernelGGL(produce,dim3(1),dim3(32),0,static_cast<hipStream_t>(stream),static_cast<int64_t*>(a),static_cast<int64_t*>(b),static_cast<int64_t*>(c));
  return int(hipGetLastError());
}
extern "C" int native_consume_on_stream(const void* c,void* failure,void* stream) {
  hipLaunchKernelGGL(consume,dim3(1),dim3(32),0,static_cast<hipStream_t>(stream),static_cast<const int64_t*>(c),static_cast<int*>(failure));
  return int(hipGetLastError());
}
