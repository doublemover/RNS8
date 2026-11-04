#pragma once
#include "backend_common/residue_layout_gemm.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <vector>
#include <stdexcept>
#include <utility>

namespace rns8::test {
struct LayoutGemmCase {
  uint64_t k;uint32_t prefix;bool reordered=false,tiled_output=false;
  uint16_t power_two=256;bool tiled_a=false;uint32_t packed_cols=16;bool worst_positive=false;
};
inline std::vector<LayoutGemmCase> layout_gemm_cases() {
  std::vector<LayoutGemmCase> cases;
  for(uint32_t prefix=1;prefix<=20;++prefix)cases.push_back({17,prefix,true,true});
  for(uint64_t k:{1,7,31,32,33,65535,65536,65537,131071,131072,131073})
    cases.push_back({k,k<64?uint32_t(k%20+1):20,k%2!=0,k<32});
  for(uint16_t q:{2,16}) {
    for(uint64_t k:{65535,65536,65537,131073})cases.push_back({k,20,true,true,q});
    cases.push_back({33,20,true,true,q,true,5});
  }
  cases.push_back({131073,20,false,false,256,false,16,true});
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
  explicit LayoutGemmFixture(const LayoutGemmCase& test):active(test.prefix) {
    using namespace detail;using boost::multiprecision::cpp_int;
    const auto k=test.k;const auto prefix=test.prefix;
    const uint64_t m=k<64?17:1,n=k<64?19:3;
    OrderedBasisIdentity basis;
    basis.moduli[0]=test.power_two;
    if(test.reordered)std::swap(basis.moduli[0],basis.moduli[1]);
    basis.fingerprint=ordered_basis_fingerprint(basis.moduli.data(),basis.count);
    if(!make_residue_layout(test.tiled_a?ResidueLayoutKind::tile_major:ResidueLayoutKind::modulus_major,
        m,k,test.tiled_a?k:k+3,20,test.tiled_a?4:0,test.tiled_a?8:0,a,basis) ||
       !make_residue_layout(ResidueLayoutKind::modulus_major,k,n,n+3,20,0,0,b,basis) ||
       !make_residue_layout(ResidueLayoutKind::tile_major,k,n,n,20,16,test.packed_cols,packed,basis) ||
       !make_residue_layout(test.tiled_output?ResidueLayoutKind::tile_major:ResidueLayoutKind::modulus_major,
          m,n,test.tiled_output?n:n+3,20,test.tiled_output?4:0,test.tiled_output?8:0,c,basis))throw std::runtime_error("layout fixture admission");
    av.assign(a.bytes,77);bv.assign(b.bytes,77);packed_expected.assign(packed.bytes,99);expected.assign(c.bytes,77);
    std::vector<cpp_int> native_a(m*k),native_b(k*n),gold(m*n);
    for(uint64_t row=0;row<m;++row)for(uint64_t q=0;q<k;++q)
      native_a[row*k+q]=test.worst_positive?cpp_int(128):(row+q)%3==0?cpp_int(INT64_MIN):(row+q)%3==1?cpp_int(INT64_MAX):cpp_int(-128);
    for(uint64_t q=0;q<k;++q)for(uint64_t col=0;col<n;++col)
      native_b[q*n+col]=test.worst_positive?cpp_int(128):(q+col)%3==0?cpp_int(INT64_MAX):(q+col)%3==1?cpp_int(INT64_MIN):cpp_int(127);
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
