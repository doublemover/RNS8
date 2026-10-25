#pragma once
#include "basis_packet.hpp"
#if defined(__HIPCC__)
#define RNS8_BASIS_HD __host__ __device__
#else
#define RNS8_BASIS_HD
#endif
namespace rns8::detail {
RNS8_BASIS_HD inline int basis_compare_words(const uint64_t* a,const uint64_t* b) {
  for(int i=2;i>=0;--i)if(a[i]!=b[i])return a[i]>b[i]?1:-1;
  return 0;
}
RNS8_BASIS_HD inline void basis_subtract_words(uint64_t* a,const uint64_t* b) {
  uint64_t borrow=0;
  for(unsigned i=0;i<3;++i) {
    const uint64_t prior=a[i],difference=prior-b[i],underflow=prior<b[i];
    a[i]=difference-borrow;borrow=underflow|(difference<borrow);
  }
}
RNS8_BASIS_HD inline void basis_negate_words(uint64_t* a) {
  uint64_t carry=1;
  for(unsigned i=0;i<3;++i){a[i]=~a[i]+carry;carry=carry && !a[i];}
}
RNS8_BASIS_HD inline uint32_t basis_words_modulo(const uint64_t* a,uint32_t modulus) {
  uint64_t remainder=0;
  for(int i=2;i>=0;--i) {
    remainder=((remainder<<32)+(a[i]>>32))%modulus;
    remainder=((remainder<<32)+uint32_t(a[i]))%modulus;
  }
  return uint32_t(remainder);
}
RNS8_BASIS_HD inline void basis_add_scaled_words(uint64_t* a,const uint64_t* product,uint32_t digit) {
  uint64_t carry=0;
  for(unsigned i=0;i<3;++i) {
    const uint64_t low=uint64_t(uint32_t(product[i]))*digit+carry;
    const uint64_t high=(product[i]>>32)*digit+(low>>32);
    const uint64_t term=(high<<32)|uint32_t(low),prior=a[i];
    a[i]=prior+term;carry=(high>>32)+(a[i]<prior);
  }
}
RNS8_BASIS_HD inline int8_t basis_center(uint32_t residue,uint32_t modulus) {
  return static_cast<int8_t>(int32_t(residue)-(residue>=(modulus+1)/2?int32_t(modulus):0));
}
// Exact mixed-radix reconstruction into three words. Every admitted 20-channel
// byte basis has product <=2^160, so this representation cannot overflow.
RNS8_BASIS_HD inline bool basis_reconstruct_words(const int8_t* source,uint64_t stride,uint32_t prefix,
    const BasisPacket& basis,bool signed_values,uint64_t* value) {
  value[0]=value[1]=value[2]=0;
  for(uint32_t i=0;i<prefix;++i) {
    const auto modulus=uint32_t(basis.moduli[i]);const auto centered=int32_t(source[uint64_t(i)*stride]);
    if(centered < -int32_t(modulus/2) || centered>int32_t((modulus-1)/2))return false;
    const auto canonical=uint32_t(centered<0?centered+int32_t(modulus):centered);
    const auto remainder=basis_words_modulo(value,modulus);
    const auto digit=((canonical+modulus-remainder)*basis.inverse_prefix[i])%modulus;
    basis_add_scaled_words(value,basis.products[i],digit);
  }
  if(signed_values && basis_compare_words(value,basis.negative_thresholds[prefix])>=0)
    basis_subtract_words(value,basis.products[prefix]);
  return true;
}
}
#undef RNS8_BASIS_HD
