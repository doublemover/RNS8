#include "backend_common/fixed_conversion.hpp"
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && __has_builtin(__builtin_amdgcn_udot4)
#define RNS8_FIXED_DOT4 1
#else
#define RNS8_FIXED_DOT4 0
#endif
__device__ __constant__ rns8::detail::fixed_conversion::Constants rns8_fixed_conversion_table=
    rns8::detail::fixed_conversion::make_constants();
__global__ void rns8_fixed_conversion_capability_kernel(int* flag) {*flag=RNS8_FIXED_DOT4;}
__global__ void rns8_fixed_import_kernel(const uint64_t* source,int8_t* residues,int cells,int prefix,bool signed_values) {
#if RNS8_FIXED_DOT4
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=int64_t(cells)*prefix) return;
  const int plane=index/cells,cell=index%cells;
  const uint64_t value=source[cell];const uint32_t modulus=rns8_default_moduli_device[plane];
  uint32_t sum=__builtin_amdgcn_udot4(uint32_t(value),rns8_fixed_conversion_table.import_weights[plane][0],0,false);
  sum=__builtin_amdgcn_udot4(uint32_t(value>>32),rns8_fixed_conversion_table.import_weights[plane][1],sum,false);
  sum%=modulus;
  if(signed_values && (value>>63)) sum=(sum+modulus-rns8_fixed_conversion_table.sign_correction[plane])%modulus;
  residues[index]=int8_t(sum>=(modulus+1)/2?int(sum)-int(modulus):int(sum));
#else
  (void)source;(void)residues;(void)cells;(void)prefix;(void)signed_values;
#endif
}
__global__ void rns8_fixed_crt_kernel(const int8_t* residues,uint64_t* limbs,int cells,int prefix) {
#if RNS8_FIXED_DOT4
  const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells) return;
  uint32_t coefficients[5]{};
  for(int p=0;p<prefix;++p) {
    const uint32_t modulus=rns8_default_moduli_device[p];
    const auto residue=rns8_canonical_from_centered_device(residues[int64_t(p)*cells+cell],modulus);
    const uint32_t value=residue*rns8_direct_crt_table.inverses[prefix][p]%modulus;
    coefficients[p/4]|=value<<(8*(p%4));
  }
  uint64_t words[3]{};uint32_t carry=0;
  for(int byte=0;byte<24;++byte) {
    uint32_t sum=carry;
    for(int group=0;group<(prefix+3)/4;++group)
      sum=__builtin_amdgcn_udot4(coefficients[group],rns8_fixed_conversion_table.crt_weights[prefix][byte][group],sum,false);
    words[byte/8]|=uint64_t(sum&255)<<(8*(byte%8));carry=sum>>8;
  }
  rns8_u192_device sum{words[0],words[1],words[2]};
  const auto* product_words=rns8_direct_crt_table.products[prefix];
  const rns8_u192_device product{product_words[0],product_words[1],product_words[2]};
  while(rns8_u192_compare_device(sum,product)>=0) sum=rns8_u192_sub_device(sum,product);
  limbs[int64_t(cell)*3]=sum.limb0;limbs[int64_t(cell)*3+1]=sum.limb1;limbs[int64_t(cell)*3+2]=sum.limb2;
#else
  (void)residues;(void)limbs;(void)cells;(void)prefix;
#endif
}
extern "C" int rns8_hip_fixed_conversion_capability(int* flag) {
  hipLaunchKernelGGL(rns8_fixed_conversion_capability_kernel,dim3(1),dim3(1),0,nullptr,flag);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_fixed_conversion(const void* source,void* destination,int cells,int prefix,int mode) {
  if(mode==2) hipLaunchKernelGGL(rns8_fixed_crt_kernel,dim3((uint64_t(cells)+255)/256),dim3(256),0,nullptr,
      static_cast<const int8_t*>(source),static_cast<uint64_t*>(destination),cells,prefix);
  else hipLaunchKernelGGL(rns8_fixed_import_kernel,dim3((uint64_t(cells)*prefix+255)/256),dim3(256),0,nullptr,
      static_cast<const uint64_t*>(source),static_cast<int8_t*>(destination),cells,prefix,mode==0);
  return int(hipGetLastError());
}
#undef RNS8_FIXED_DOT4
