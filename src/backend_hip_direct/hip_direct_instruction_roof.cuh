// Calibration only: isolate dependent or independent WMMA accumulator chains.
// Every wave writes all 256 matrix cells for every chain; no output is discarded.
template<int Chains,bool Reduce>
__global__ void rns8_instruction_roof_kernel(const int8_t* a,const int8_t* b,
                                            int32_t* output,int iterations,int waves) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  using fragment=int __attribute__((ext_vector_type(4)));
  using accumulator=int __attribute__((ext_vector_type(8)));
  const int wave=int(blockIdx.x)*4+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(wave>=waves) return;
  fragment av[Chains]{},bv[Chains]{};accumulator acc[Chains]{};
#pragma unroll
  for(int chain=0;chain<Chains;++chain) {
    const uint32_t lhs=uint32_t(uint8_t(a[chain]))*0x01010101u;
    const uint32_t rhs=uint32_t(uint8_t(b[chain]))*0x01010101u;
#pragma unroll
    for(int word=0;word<4;++word) {av[chain][word]=int(lhs);bv[chain][word]=int(rhs);}
  }
  for(int iteration=0;iteration<iterations;++iteration) {
#pragma unroll
    for(int chain=0;chain<Chains;++chain) {
      acc[chain]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av[chain],true,bv[chain],acc[chain],false);
      if constexpr(Reduce) {
#pragma unroll
        for(int reg=0;reg<8;++reg) {
          int value=acc[chain][reg]%251;
          if(value<0) value+=251;if(value>=126) value-=251;
          acc[chain][reg]=value;
        }
      }
    }
  }
#pragma unroll
  for(int chain=0;chain<Chains;++chain) {
#pragma unroll
    for(int reg=0;reg<8;++reg) {
      const int row=2*reg+(lane>>4),col=lane&15;
      output[((wave*Chains+chain)*16+row)*16+col]=acc[chain][reg];
    }
  }
#else
  (void)a;(void)b;(void)output;(void)iterations;(void)waves;
#endif
}
extern "C" int rns8_hip_instruction_roof(const int8_t* a,const int8_t* b,int32_t* output,
    int chains,int iterations,int waves,bool reduce) {
#define RNS8_ROOF_CASE(C) case C: \
  if(reduce) hipLaunchKernelGGL((rns8_instruction_roof_kernel<C,true>),dim3((waves+3)/4),dim3(128),0,nullptr,a,b,output,iterations,waves); \
  else hipLaunchKernelGGL((rns8_instruction_roof_kernel<C,false>),dim3((waves+3)/4),dim3(128),0,nullptr,a,b,output,iterations,waves); break
  switch(chains) {RNS8_ROOF_CASE(1);RNS8_ROOF_CASE(2);RNS8_ROOF_CASE(4);RNS8_ROOF_CASE(8);
    default:return int(hipErrorInvalidValue);}
#undef RNS8_ROOF_CASE
  return int(hipGetLastError());
}
