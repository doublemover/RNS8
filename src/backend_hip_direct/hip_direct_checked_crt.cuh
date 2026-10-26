#include "backend_common/checked_crt_export.hpp"

__global__ void rns8_checked_crt_decode_kernel(const uint64_t* roots,uint64_t* output,
    int cells,rns8::detail::checked_crt::Decode contract,uint32_t* status) {
  const int cell=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=cells)return;
  rns8::detail::balanced_crt::Wide root{{roots[int64_t(cell)*3],roots[int64_t(cell)*3+1],roots[int64_t(cell)*3+2]}};
  if(!rns8::detail::checked_crt::decode(root,contract,output+int64_t(cell)*contract.limbs))
    atomicOr(status,1u);
}
extern "C" int rns8_hip_checked_crt_decode(const uint64_t* roots,uint64_t* output,int cells,
    const rns8::detail::checked_crt::Decode* contract,uint32_t* status) {
  if(!roots || !output || !contract || !status || cells<=0 || cells>INT32_MAX-255)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_checked_crt_decode_kernel,dim3((cells+255)/256),dim3(256),0,nullptr,
      roots,output,cells,*contract,status);
  return int(hipGetLastError());
}
