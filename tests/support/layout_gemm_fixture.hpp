#pragma once
#include "backend_common/residue_layout_gemm.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <vector>
#include <stdexcept>
#include <utility>

namespace rns8::test {
inline std::vector<std::pair<uint64_t,uint32_t>> layout_gemm_cases() {
  std::vector<std::pair<uint64_t,uint32_t>> cases;
  for(uint32_t prefix=1;prefix<=20;++prefix)cases.push_back({17,prefix});
  for(uint64_t k:{1,7,31,32,33,65535,65536,65537})cases.push_back({k,k<64?uint32_t(k%20+1):20});
  return cases;
}
struct LayoutGemmFixture {
  detail::ResidueLayout a,b,packed,c;
  std::vector<int8_t> av,bv,packed_expected,expected;
  uint32_t active;
  static int8_t center(const boost::multiprecision::cpp_int& value,unsigned modulus) {
    int residue=(value%modulus).convert_to<int>();
    if(residue<0)residue+=modulus;
    if(residue>=int((modulus+1)/2))residue-=modulus;
    return static_cast<int8_t>(residue);
  }
  LayoutGemmFixture(uint64_t k,uint32_t prefix,bool reordered,bool tiled_output=false):active(prefix) {
    using namespace detail;using boost::multiprecision::cpp_int;
    const uint64_t m=k<64?17:1,n=k<64?19:3;
    OrderedBasisIdentity basis;
    if(reordered)std::swap(basis.moduli[0],basis.moduli[1]);
    basis.fingerprint=ordered_basis_fingerprint(basis.moduli.data(),basis.count);
    if(!make_residue_layout(ResidueLayoutKind::modulus_major,m,k,k+3,20,0,0,a,basis) ||
       !make_residue_layout(ResidueLayoutKind::modulus_major,k,n,n+3,20,0,0,b,basis) ||
       !make_residue_layout(ResidueLayoutKind::tile_major,k,n,n,20,16,16,packed,basis) ||
       !make_residue_layout(tiled_output?ResidueLayoutKind::tile_major:ResidueLayoutKind::modulus_major,
          m,n,tiled_output?n:n+3,20,tiled_output?4:0,tiled_output?8:0,c,basis))throw std::runtime_error("layout fixture admission");
    av.assign(a.bytes,77);bv.assign(b.bytes,77);packed_expected.assign(packed.bytes,99);expected.assign(c.bytes,77);
    std::vector<cpp_int> native_a(m*k),native_b(k*n),gold(m*n);
    for(uint64_t row=0;row<m;++row)for(uint64_t q=0;q<k;++q)
      native_a[row*k+q]=(row+q)%3==0?cpp_int(INT64_MIN):(row+q)%3==1?cpp_int(INT64_MAX):cpp_int(-128);
    for(uint64_t q=0;q<k;++q)for(uint64_t col=0;col<n;++col)
      native_b[q*n+col]=(q+col)%3==0?cpp_int(INT64_MAX):(q+col)%3==1?cpp_int(INT64_MIN):cpp_int(127);
    for(uint64_t row=0;row<m;++row)for(uint64_t col=0;col<n;++col)
      for(uint64_t q=0;q<k;++q)gold[row*n+col]+=native_a[row*k+q]*native_b[q*n+col];
    for(uint32_t p=0;p<20;++p) {
      const auto modulus=basis.moduli[p];
      for(uint64_t row=0;row<m;++row)for(uint64_t q=0;q<k;++q)
        av[a.offset(p,row,q)]=center(native_a[row*k+q],modulus);
      for(uint64_t q=0;q<k;++q)for(uint64_t col=0;col<n;++col) {
        const auto value=center(native_b[q*n+col],modulus);bv[b.offset(p,q,col)]=value;
        if(p<active)packed_expected[packed.offset(p,q,col)]=value;
      }
      if(p<active)for(uint64_t row=0;row<m;++row)for(uint64_t col=0;col<n;++col)
        expected[c.offset(p,row,col)]=center(gold[row*n+col],modulus);
    }
  }
};
} // namespace rns8::test
