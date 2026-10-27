#pragma once
#include <cstdint>
#include "finite_u8_reducer.hpp"

#if defined(__HIPCC__)
#define RNS8_EPILOGUE_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_EPILOGUE_INLINE inline
#endif

namespace rns8::detail {
struct ResidueAxpbyCoefficients {
  uint32_t modulus=0,reciprocal=0;
  int32_t alpha=0,beta=0;
};
struct ResidueAxpbyTable {
  static constexpr uint32_t abi_version=1;
  ResidueAxpbyCoefficients channel[20]{};
};
// Integer coefficients, including INT64_MIN, are reduced before multiplication.
// Runtime kernels receive this by-value table: no coefficient allocation, upload
// kernel, value cache, or full-width coefficient division per output cell.
RNS8_EPILOGUE_INLINE ResidueAxpbyCoefficients lower_residue_axpby(
    int64_t alpha,int64_t beta,uint32_t modulus) {
  int32_t a=static_cast<int32_t>(alpha%static_cast<int64_t>(modulus));
  int32_t b=static_cast<int32_t>(beta%static_cast<int64_t>(modulus));
  if(a<0)a+=static_cast<int32_t>(modulus);
  if(b<0)b+=static_cast<int32_t>(modulus);
  return {modulus,finite_u8::modulus_reciprocal_u32(modulus),a,b};
}
template<class Modulus>
inline bool lower_residue_axpby_table(int64_t alpha,int64_t beta,const Modulus* moduli,
    uint32_t prefix,ResidueAxpbyTable& out) {
  if(!moduli || !prefix || prefix>20)return false;
  ResidueAxpbyTable table;
  for(uint32_t p=0;p<prefix;++p){
    if(moduli[p]<2 || moduli[p]>256)return false;
    table.channel[p]=lower_residue_axpby(alpha,beta,uint32_t(moduli[p]));
  }
  out=table;return true;
}
RNS8_EPILOGUE_INLINE int8_t centered_residue_axpby_reduced(
    int8_t product,int8_t addend,const ResidueAxpbyCoefficients& c) {
  // At most two terms of 255*128: absolute value <=65280, safely INT32.
  // The producer is responsible for omitting dead operand loads/dot products.
  return finite_u8::reduce_to_centered_i32(c.alpha*int32_t(product)+c.beta*int32_t(addend),c.modulus,c.reciprocal);
}
RNS8_EPILOGUE_INLINE int8_t centered_residue_axpby(
    int8_t product,int8_t addend,int64_t alpha,int64_t beta,uint32_t modulus) {
  return centered_residue_axpby_reduced(product,addend,lower_residue_axpby(alpha,beta,modulus));
}
}
#undef RNS8_EPILOGUE_INLINE
