#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void rns8_system_roof_copy(const uint32_t* source,uint32_t* destination,uint64_t words) {
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=uint64_t(gridDim.x)*blockDim.x)
    destination[i]=source[i];
}
template<bool Write>
__global__ void rns8_system_roof_lds(const uint32_t* source,uint32_t* destination,uint32_t iterations,uint32_t stride) {
  extern __shared__ uint32_t storage[];
  volatile uint32_t* shared=storage;
  const uint32_t lane=threadIdx.x;
  shared[lane*stride]=source[uint64_t(blockIdx.x)*128+lane];
  __syncthreads();
  uint32_t sum=0;
  for(uint32_t iteration=0;iteration<iterations;++iteration) {
    const uint32_t value=shared[((lane+iteration)&127)*stride];sum+=value;
    if constexpr(Write) {
      __syncthreads(); // every reader finishes before any overwritten source
      shared[lane*stride]=value+iteration+lane;
      __syncthreads();
    }
  }
  destination[uint64_t(blockIdx.x)*128+lane]=sum;
}
__global__ void rns8_system_roof_launch(const uint32_t* source,uint32_t* destination) {
  if(threadIdx.x==0)destination[0]=source[0]+1;
}
__global__ void rns8_system_roof_displace(volatile uint32_t* data,uint64_t words) {
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=uint64_t(gridDim.x)*blockDim.x)
    data[i]=data[i]^UINT32_C(0xa5a5a5a5);
}
extern "C" int rns8_hip_system_roof(uint32_t mode,const uint32_t* source,uint32_t* destination,
    uint64_t words,uint32_t blocks,uint32_t iterations,uint32_t stride,void* stream) {
  const auto s=static_cast<hipStream_t>(stream);
  switch(mode) {
    case 0:hipLaunchKernelGGL(rns8_system_roof_copy,dim3(blocks),dim3(128),0,s,source,destination,words);break;
    case 1:hipLaunchKernelGGL((rns8_system_roof_lds<false>),dim3(blocks),dim3(128),128*stride*4,s,source,destination,iterations,stride);break;
    case 2:hipLaunchKernelGGL((rns8_system_roof_lds<true>),dim3(blocks),dim3(128),128*stride*4,s,source,destination,iterations,stride);break;
    case 3:hipLaunchKernelGGL(rns8_system_roof_launch,dim3(1),dim3(128),0,s,source,destination);break;
    default:return int(hipErrorInvalidValue);
  }
  return int(hipGetLastError());
}
extern "C" int rns8_hip_system_roof_displace(uint32_t* data,uint64_t words,void* stream) {
  hipLaunchKernelGGL(rns8_system_roof_displace,dim3(4096),dim3(128),0,static_cast<hipStream_t>(stream),data,words);
  return int(hipGetLastError());
}
