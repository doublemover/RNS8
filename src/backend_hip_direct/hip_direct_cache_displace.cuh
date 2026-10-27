// An explicitly observable full-buffer line walk, not a claim about replacement
// policy, physical XCD assignment or guaranteed eviction on any architecture.
__global__ void rns8_cache_displace_kernel(volatile uint32_t* data,uint64_t lines,uint32_t pass){
  const uint64_t line=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(line<lines){const uint64_t word=line*16;const uint32_t value=data[word];data[word]=value+UINT32_C(0x9e3779b9)+pass;}
}
extern "C" int rns8_hip_cache_displace(void* storage,uint64_t bytes,uint32_t passes){
  for(uint32_t pass=0;pass<passes;++pass){hipLaunchKernelGGL(rns8_cache_displace_kernel,dim3((bytes/64+255)/256),dim3(256),0,nullptr,
      static_cast<volatile uint32_t*>(storage),bytes/64,pass);const auto status=hipGetLastError();if(status!=hipSuccess)return int(status);}
  return int(hipSuccess);
}
