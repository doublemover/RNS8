#pragma once
#include "direct_crt_constants.hpp"
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_QUARTER_HD __host__ __device__
#else
#define RNS8_QUARTER_HD
#endif
namespace rns8::detail::quarter_crt {
// This table is W_i=(M/m_i)*inverse_i, not the older M_i table.
struct Constants {uint16_t weights[21][20][12]{};uint16_t products[21][12]{};uint32_t fractions[21][20]{};};
constexpr uint32_t ratio_bits(uint32_t numerator,uint32_t denominator) {
  int exponent=0;while((uint64_t(numerator)<<(-exponent))<denominator)--exponent;
  const uint64_t scaled=uint64_t(numerator)<<(23-exponent);
  uint32_t significand=static_cast<uint32_t>(scaled/denominator);const auto remainder=scaled%denominator;
  if(remainder*2>denominator || (remainder*2==denominator && (significand&1)))++significand;
  if(significand==(1u<<24)){significand>>=1;++exponent;}
  return (uint32_t(exponent+127)<<23)|(significand-(1u<<23));
}
constexpr Constants make_constants() {
  Constants out{};
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    for(uint32_t digit=0;digit<12;++digit)out.products[prefix][digit]=uint16_t(direct_crt::kConstants.products[prefix][digit/4]>>(16*(digit%4)));
    for(uint32_t p=0;p<prefix;++p) {
      const auto inverse=direct_crt::kConstants.inverses[prefix][p];uint32_t carry=0;
      for(uint32_t digit=0;digit<12;++digit) {
        const uint32_t part=uint16_t(direct_crt::kConstants.weights[prefix][p][digit/4]>>(16*(digit%4)));
        const uint32_t value=part*inverse+carry;out.weights[prefix][p][digit]=uint16_t(value);carry=value>>16;
      }
      out.fractions[prefix][p]=ratio_bits(inverse,garner::kConstants.moduli[p]);
    }
  }
  return out;
}
inline constexpr Constants kConstants=make_constants();
// z<5121<8192. Each FMA rounding error <=2^-12; each table
// coefficient error <=2^-25, multiplied by r<=255. Exact bound <1/64.
constexpr uint32_t kErrorNumerator=20*(255+8192),kErrorDenominator=1u<<25;
static_assert(uint64_t(kErrorNumerator)*64<kErrorDenominator);
constexpr uint32_t kQuotientMax=5121;
static_assert(20u*255u*65535u+kQuotientMax*65535u+8192u<INT32_MAX);
RNS8_QUARTER_HD inline int32_t euclidean_carry(int32_t value) {
  return value>=0?value/65536:-static_cast<int32_t>((static_cast<uint32_t>(-value)+65535u)/65536u);
}
RNS8_QUARTER_HD inline void delayed_digits(const Constants& table,const uint32_t* residues,uint32_t prefix,
    uint32_t quotient,uint32_t digits,uint16_t* out) {
  int32_t carry=0;
  for(uint32_t digit=0;digit<digits;++digit) {
    uint32_t sum=0;for(uint32_t p=0;p<prefix;++p)sum+=residues[p]*uint32_t(table.weights[prefix][p][digit]);
    const int32_t value=static_cast<int32_t>(sum)-static_cast<int32_t>(quotient*uint32_t(table.products[prefix][digit]))+carry;
    carry=euclidean_carry(value);out[digit]=static_cast<uint16_t>(value-carry*65536);
  }
}
RNS8_QUARTER_HD inline void canonicalize(const Constants& table,uint32_t prefix,uint16_t* digits) {
  if(!(digits[11]&0x8000))return;
  uint32_t carry=0;for(uint32_t d=0;d<12;++d){const uint32_t value=uint32_t(digits[d])+table.products[prefix][d]+carry;digits[d]=uint16_t(value);carry=value>>16;}
}
// Independent host FMA evaluator: all values in this certificate are exact
// multiples of 2^-32, and the unreduced FMA numerator fits 45 unsigned bits.
inline uint64_t fixed32(uint32_t bits) {
  if(!bits)return 0;const uint32_t exponent=(bits>>23)&255,significand=(bits&0x7fffff)|(1u<<23);
  const int shift=static_cast<int>(exponent)-118;
  return shift>=0?uint64_t(significand)<<shift:uint64_t(significand)>>(-shift);
}
inline uint32_t round_fixed32(uint64_t value) {
  if(!value)return 0;unsigned top=0;for(uint64_t x=value;x>>=1;)++top;
  uint64_t significand;
  if(top<=23)significand=value<<(23-top);
  else {const unsigned shift=top-23;significand=value>>shift;const auto remainder=value&((uint64_t(1)<<shift)-1),half=uint64_t(1)<<(shift-1);
    if(remainder>half || (remainder==half && (significand&1)))++significand;
    if(significand==(uint64_t(1)<<24)){significand>>=1;++top;}}
  return ((top+95)<<23)|uint32_t(significand-(1u<<23));
}
inline uint32_t host_quotient(const Constants& table,const uint32_t* residues,uint32_t prefix) {
  uint32_t z=0;for(uint32_t p=0;p<prefix;++p)z=round_fixed32(fixed32(z)+uint64_t(residues[p])*fixed32(table.fractions[prefix][p]));
  const uint64_t value=fixed32(z);uint32_t q=static_cast<uint32_t>(value>>32);const uint32_t remainder=static_cast<uint32_t>(value);
  return q+(remainder>0x80000000u || (remainder==0x80000000u && (q&1)));
}
}
#undef RNS8_QUARTER_HD
