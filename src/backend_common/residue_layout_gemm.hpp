#pragma once
#include "residue_layout.hpp"

namespace rns8::detail {
// This consumer operates on explicitly described centered byte residues. Its
// output is a residue representation; integer uniqueness belongs to its owner.
inline bool compatible_layout_gemm(const ResidueLayout& a,const ResidueLayout& b,
    const ResidueLayout& c,uint32_t prefix) {
  return valid_residue_layout(a) && valid_residue_layout(b) && valid_residue_layout(c) &&
      a.basis==b.basis && a.basis==c.basis && a.cols==b.rows && c.rows==a.rows &&
      c.cols==b.cols && prefix && prefix<=a.prefix && prefix<=b.prefix && prefix<=c.prefix;
}
inline bool residue_layout_gemm_host(const int8_t* a,const ResidueLayout& al,
    const int8_t* b,const ResidueLayout& bl,int8_t* c,const ResidueLayout& cl,uint32_t prefix) {
  if(!compatible_layout_gemm(al,bl,cl,prefix) ||
      !nonoverlapping_byte_ranges(a,al.bytes,c,cl.bytes) ||
      !nonoverlapping_byte_ranges(b,bl.bytes,c,cl.bytes))return false;
  for(uint32_t p=0;p<prefix;++p) {
    const int64_t modulus=al.basis.moduli[p];
    for(uint64_t row=0;row<al.rows;++row)for(uint64_t col=0;col<bl.cols;++col) {
      int64_t sum=0;
      for(uint64_t q=0;q<al.cols;++q)
        sum=(sum+int64_t(a[al.offset(p,row,q)])*b[bl.offset(p,q,col)])%modulus;
      if(sum<0)sum+=modulus;
      if(sum>=(modulus+1)/2)sum-=modulus;
      c[cl.offset(p,row,col)]=static_cast<int8_t>(sum);
    }
  }
  return true;
}
// Flat kernel argument: no standard-library host member calls in device code.
struct ResidueLayoutModuli {uint16_t values[20]{};};
} // namespace rns8::detail
