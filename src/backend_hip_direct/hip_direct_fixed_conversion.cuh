#include "backend_common/fixed_conversion.hpp"
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && __has_builtin(__builtin_amdgcn_udot4)
#define RNS8_FIXED_DOT4 1
#else
#define RNS8_FIXED_DOT4 0
#endif
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define RNS8_FIXED_WMMA 1
#else
#define RNS8_FIXED_WMMA 0
#endif
__device__ __constant__ rns8::detail::fixed_conversion::Constants rns8_fixed_conversion_table=
    rns8::detail::fixed_conversion::make_constants();
__global__ void rns8_fixed_conversion_capability_kernel(int* flag,int mode) {*flag=mode>=3?RNS8_FIXED_WMMA:RNS8_FIXED_DOT4;}
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
// One wave computes 16 native cells x 16 modulus columns. Both fragment halves
// duplicate the same rows/columns as required by RDNA3 wave32 WMMA. K=8 is padded
// with exact zeros to K=16; unsigned byte operands need no signed-bias correction.
__global__ __launch_bounds__(32) void rns8_fixed_import_wmma_kernel(
    const uint64_t* source,int8_t* residues,int cells,int prefix,bool signed_values) {
#if RNS8_FIXED_WMMA
  using fragment=int __attribute__((ext_vector_type(4)));
  using accumulator=int __attribute__((ext_vector_type(8)));
  const int lane=threadIdx.x,cell=int(blockIdx.x)*16+(lane&15);
  const int plane=int(blockIdx.y)*16+(lane&15);
  const uint64_t raw=cell<cells?source[cell]:0;
  fragment av{},bv{};
  av[0]=int(uint32_t(raw));av[1]=int(uint32_t(raw>>32));
  if(plane<prefix) {
    bv[0]=int(rns8_fixed_conversion_table.import_weights[plane][0]);
    bv[1]=int(rns8_fixed_conversion_table.import_weights[plane][1]);
  }
  accumulator acc{};
  acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,av,false,bv,acc,false);
  if(plane<prefix) for(int reg=0;reg<8;++reg) {
    const int row=int(blockIdx.x)*16+2*reg+(lane>>4);
    if(row>=cells) continue;
    const uint32_t modulus=rns8_default_moduli_device[plane];
    uint32_t value=uint32_t(acc[reg])%modulus; // <=8*255^2 before reduction
    if(signed_values && (source[row]>>63))
      value=(value+modulus-rns8_fixed_conversion_table.sign_correction[plane])%modulus;
    residues[int64_t(plane)*cells+row]=int8_t(value>=(modulus+1)/2?int(value)-int(modulus):int(value));
  }
#else
  (void)source;(void)residues;(void)cells;(void)prefix;(void)signed_values;
#endif
}
// 16 cells x 24 byte columns, two K=16 fragments covering at most 20 moduli.
// LDS retains exact column sums for cross-column base256 carries. There is no
// global INT32 staging and no approximate quotient in the canonical correction.
__global__ __launch_bounds__(32) void rns8_fixed_crt_wmma_kernel(
    const int8_t* residues,uint64_t* limbs,int cells,int prefix) {
#if RNS8_FIXED_WMMA
  using fragment=int __attribute__((ext_vector_type(4)));
  using accumulator=int __attribute__((ext_vector_type(8)));
  __shared__ uint32_t columns[16][24];
  const int lane=threadIdx.x,cell=int(blockIdx.x)*16+(lane&15);
  fragment coefficients[2]{};
  for(int p=0;p<prefix;++p) {
    const uint32_t modulus=rns8_default_moduli_device[p];
    const uint32_t residue=cell<cells?rns8_canonical_from_centered_device(residues[int64_t(p)*cells+cell],modulus):0;
    const uint32_t coefficient=residue*rns8_direct_crt_table.inverses[prefix][p]%modulus;
    coefficients[p/16][(p%16)/4]|=int(coefficient<<(8*(p%4)));
  }
  for(int tile=0;tile<2;++tile) {
    const int column=tile*16+(lane&15);accumulator acc{};
    for(int slab=0;slab<(prefix+15)/16;++slab) {
      fragment weights{};
      if(column<24) for(int word=0;word<4;++word) {
        const int group=slab*4+word;
        if(group<5) weights[word]=int(rns8_fixed_conversion_table.crt_weights[prefix][column][group]);
      }
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,coefficients[slab],false,weights,acc,false);
    }
    if(column<24) for(int reg=0;reg<8;++reg)
      columns[2*reg+(lane>>4)][column]=uint32_t(acc[reg]); // <=20*255^2
  }
  __syncthreads();
  if(lane<16 && cell<cells) {
    uint64_t words[3]{};uint32_t carry=0;
    for(int byte=0;byte<24;++byte) {
      const uint32_t sum=columns[lane][byte]+carry;
      words[byte/8]|=uint64_t(sum&255)<<(8*(byte%8));carry=sum>>8;
    }
    rns8_u192_device sum{words[0],words[1],words[2]};
    const auto* pw=rns8_direct_crt_table.products[prefix];
    const rns8_u192_device product{pw[0],pw[1],pw[2]};
    while(rns8_u192_compare_device(sum,product)>=0) sum=rns8_u192_sub_device(sum,product);
    limbs[int64_t(cell)*3]=sum.limb0;limbs[int64_t(cell)*3+1]=sum.limb1;limbs[int64_t(cell)*3+2]=sum.limb2;
  }
#else
  (void)residues;(void)limbs;(void)cells;(void)prefix;
#endif
}
extern "C" int rns8_hip_fixed_conversion_capability(int* flag,int mode) {
  hipLaunchKernelGGL(rns8_fixed_conversion_capability_kernel,dim3(1),dim3(1),0,nullptr,flag,mode);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_fixed_conversion(const void* source,void* destination,int cells,int prefix,int mode) {
  if(mode==5) hipLaunchKernelGGL(rns8_fixed_crt_wmma_kernel,dim3((uint64_t(cells)+15)/16),dim3(32),0,nullptr,
      static_cast<const int8_t*>(source),static_cast<uint64_t*>(destination),cells,prefix);
  else if(mode>=3) hipLaunchKernelGGL(rns8_fixed_import_wmma_kernel,dim3((uint64_t(cells)+15)/16,(prefix+15)/16),dim3(32),0,nullptr,
      static_cast<const uint64_t*>(source),static_cast<int8_t*>(destination),cells,prefix,mode==3);
  else if(mode==2) hipLaunchKernelGGL(rns8_fixed_crt_kernel,dim3((uint64_t(cells)+255)/256),dim3(256),0,nullptr,
      static_cast<const int8_t*>(source),static_cast<uint64_t*>(destination),cells,prefix);
  else hipLaunchKernelGGL(rns8_fixed_import_kernel,dim3((uint64_t(cells)*prefix+255)/256),dim3(256),0,nullptr,
      static_cast<const uint64_t*>(source),static_cast<int8_t*>(destination),cells,prefix,mode==0);
  return int(hipGetLastError());
}
#undef RNS8_FIXED_DOT4

#undef RNS8_FIXED_WMMA
