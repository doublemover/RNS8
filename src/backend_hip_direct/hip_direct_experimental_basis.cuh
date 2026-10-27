// Explicit experimental ordered-basis kernels. Default-ladder launch admission
// remains unchanged; these entry points consume separately owned basis packets.
namespace {
using rns8::detail::BasisPacket;
using rns8::detail::BasisMagnitude;
__global__ void basis_pack_native_kernel(const uint64_t* native,int8_t* residues,uint64_t cells,
    const BasisPacket* basis,uint32_t prefix,bool signed_values) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const uint32_t plane=blockIdx.y;if(cell>=cells)return;
  const uint32_t modulus=basis->moduli[plane];const uint64_t word=native[cell];
  const bool negative=signed_values && (word>>63);const auto magnitude=negative?uint64_t(0)-word:word;
  uint32_t residue=uint32_t(magnitude%modulus);if(negative && residue)residue=modulus-residue;
  residues[uint64_t(plane)*cells+cell]=rns8::detail::basis_center(residue,modulus);
  (void)prefix;
}
__device__ bool basis_checked_value(const int8_t* residues,uint64_t cells,uint64_t cell,const BasisPacket& basis,
    uint32_t prefix,bool signed_values,BasisMagnitude bound,uint64_t* value,int* status) {
  if(!rns8::detail::basis_reconstruct_words(residues+cell,cells,prefix,basis,signed_values,value)) {
    atomicMax(status,int(RNS8_INVALID_ARGUMENT));return false;
  }
  uint64_t magnitude[3]{value[0],value[1],value[2]};
  if(signed_values && value[2]>>63)rns8::detail::basis_negate_words(magnitude);
  if(rns8::detail::basis_compare_words(magnitude,bound.words)>0) {
    atomicMax(status,int(RNS8_RANGE_ERROR));return false;
  }
  return true;
}
__global__ void basis_export_kernel(const int8_t* residues,uint64_t* destination,uint64_t cells,
    const BasisPacket* basis,uint32_t prefix,bool signed_values,uint32_t limbs,BasisMagnitude bound,int* status) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=cells)return;
  uint64_t value[3]{};if(!basis_checked_value(residues,cells,cell,*basis,prefix,signed_values,bound,value,status))return;
  const bool negative=signed_values && (value[2]>>63);const uint64_t extension=negative?UINT64_MAX:0;
  bool fits=!signed_values || bool(value[limbs-1]>>63)==negative;
  for(uint32_t i=limbs;i<3;++i)fits= fits && value[i]==extension;
  if(!fits){atomicMax(status,int(RNS8_RANGE_ERROR));return;}
  for(uint32_t i=0;i<limbs;++i)destination[cell*limbs+i]=value[i];
}
__global__ void basis_convert_kernel(const int8_t* source,int8_t* destination,uint64_t cells,
    const BasisPacket* from,const BasisPacket* to,uint32_t source_prefix,uint32_t target_prefix,
    bool signed_values,BasisMagnitude bound,int* status) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=cells)return;
  uint64_t value[3]{};if(!basis_checked_value(source,cells,cell,*from,source_prefix,signed_values,bound,value,status))return;
  const bool negative=signed_values && value[2]>>63;if(negative)rns8::detail::basis_negate_words(value);
  for(uint32_t i=0;i<target_prefix;++i) {
    const auto modulus=uint32_t(to->moduli[i]);auto residue=rns8::detail::basis_words_modulo(value,modulus);
    if(negative && residue)residue=modulus-residue;
    destination[uint64_t(i)*cells+cell]=rns8::detail::basis_center(residue,modulus);
  }
}
}
extern "C" int rns8_hip_basis_pack(const uint64_t* native,int8_t* residues,uint64_t cells,
    const rns8::detail::BasisPacket* basis,uint32_t prefix,bool signed_values) {
  if(!native || !residues || !basis || !cells || cells>INT32_MAX || !prefix || prefix>20)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(basis_pack_native_kernel,dim3((cells+255)/256,prefix),dim3(256),0,nullptr,native,residues,cells,basis,prefix,signed_values);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_basis_export(const int8_t* residues,uint64_t* destination,uint64_t cells,
    const rns8::detail::BasisPacket* basis,uint32_t prefix,bool signed_values,uint32_t limbs,
    rns8::detail::BasisMagnitude bound,int* status) {
  if(!residues || !destination || !basis || !status || !cells || cells>INT32_MAX || !prefix || prefix>20 || !limbs || limbs>3)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(basis_export_kernel,dim3((cells+255)/256),dim3(256),0,nullptr,residues,destination,cells,basis,prefix,signed_values,limbs,bound,status);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_basis_convert(const int8_t* source,int8_t* destination,uint64_t cells,
    const rns8::detail::BasisPacket* from,const rns8::detail::BasisPacket* to,uint32_t source_prefix,uint32_t target_prefix,
    bool signed_values,rns8::detail::BasisMagnitude bound,int* status) {
  if(!source || !destination || !from || !to || !status || !cells || cells>INT32_MAX || !source_prefix || source_prefix>20 || !target_prefix || target_prefix>20)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(basis_convert_kernel,dim3((cells+255)/256),dim3(256),0,nullptr,source,destination,cells,from,to,source_prefix,target_prefix,signed_values,bound,status);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_basis_gemm_stream(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,
    const uint16_t* host_moduli,uint32_t prefix,void* stream) {
  if(!a || !b || !c || !host_moduli || m<=0 || n<=0 || k<=0 || !prefix || prefix>20)return int(hipErrorInvalidValue);
  const dim3 block(kRns8HipTileN,kRns8HipTileM),grid((uint32_t(n)+block.x-1)/block.x,(uint32_t(m)+block.y-1)/block.y);
  for(uint32_t p=0;p<prefix;++p) {
    const auto modulus=host_moduli[p];if(modulus<2 || modulus>256)return int(hipErrorInvalidValue);
    for(int offset=0;offset<k;) {
      const int depth=k-offset>65536?65536:k-offset;
      hipLaunchKernelGGL(rns8_ring_gemm_i8_i32_tiled_kernel,grid,block,0,static_cast<hipStream_t>(stream),
          a+uint64_t(p)*m*k,b+uint64_t(p)*k*n,c+uint64_t(p)*m*n,
          m,n,offset,depth,k,n,n,int(modulus),rns8_modulus_reciprocal_host(modulus),offset!=0);
      const auto error=hipGetLastError();if(error!=hipSuccess)return int(error);offset+=depth;
    }
  }
  return int(hipSuccess);
}
extern "C" int rns8_hip_basis_gemm(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,
    const uint16_t* host_moduli,uint32_t prefix) {
  return rns8_hip_basis_gemm_stream(a,b,c,m,n,k,host_moduli,prefix,nullptr);
}
