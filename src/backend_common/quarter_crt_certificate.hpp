#pragma once
#include "quarter_crt.hpp"
#include "arithmetic_certificate.hpp"
namespace rns8::detail::quarter_crt {
using Integer=boost::multiprecision::cpp_int;
struct Certificate {
  uint32_t prefix=0,digits=0,limbs=0,mode=0; // 0 signed, 1 unsigned, 2 canonical
  uint64_t source_instance=0,source_generation=0;
  Integer bound=0;std::string target,build,plan_hash,sha256,failure;
  bool admitted=false;
};
inline Integer from_digits(const uint16_t* words) {Integer value=0;for(int d=11;d>=0;--d){value<<=16;value+=words[d];}return value;}
inline Certificate certify(Certificate c,const Constants& table=kConstants) {
  const auto fail=[&](const char* reason){c.failure=reason;c.admitted=false;return c;};
  if(!c.prefix || c.prefix>20 || !c.limbs || c.limbs>32 || c.mode>2 || !c.source_instance || !c.source_generation || c.bound<0 || c.target.empty() || c.build.empty() || c.plan_hash.size()!=64)return fail("quarter_structure");
  Integer product=1;for(uint32_t p=0;p<c.prefix;++p)product*=garner::kConstants.moduli[p];
  if(4*c.bound>product)return fail("quarter_range_requires_four_B_le_M");
  if(c.mode==2) {if(product>(Integer(1)<<(64*c.limbs)))return fail("canonical_destination_width");c.digits=12;}
  else {
    if(c.bound>=(Integer(1)<<(64*c.limbs-(c.mode==0?1:0))))return fail("integer_destination_width");
    c.digits=1;while(c.bound>=(Integer(1)<<(16*c.digits-1)))++c.digits;
    if(c.digits>12)return fail("delayed_digit_width");
  }
  if(from_digits(table.products[c.prefix])!=product)return fail("product_table_corruption");
  std::string artifact="quarter-crt-v1;fixed_order_rne_fma;canonical_representatives;error=168940/33554432";
  for(uint32_t p=0;p<c.prefix;++p) {
    const auto m=garner::kConstants.moduli[p],y=direct_crt::kConstants.inverses[c.prefix][p];
    const Integer partial=product/m;
    if((partial*y)%m!=1 || from_digits(table.weights[c.prefix][p])!=partial*y)return fail("full_weight_table_corruption");
    const auto bits=table.fractions[c.prefix][p],e=(bits>>23)&255;
    if(bits>>31 || e<119 || e>126 || bits!=ratio_bits(y,m))return fail("rne_fraction_table_corruption");
    const Integer denominator=Integer(1)<<(150-e),numerator=(bits&0x7fffff)|(1u<<23);
    Integer error=numerator*m-Integer(y)*denominator;if(error<0)error=-error;
    if((error<<25)>Integer(m)*denominator)return fail("fraction_error_bound");
    artifact+=';'+std::to_string(m)+':'+std::to_string(bits)+':'+from_digits(table.weights[c.prefix][p]).str();
  }
  artifact+=";M="+product.str()+";B="+c.bound.str()+";mode="+std::to_string(c.mode)+";digits="+std::to_string(c.digits)+";limbs="+std::to_string(c.limbs)+
    ";instance="+std::to_string(c.source_instance)+";generation="+std::to_string(c.source_generation)+";plan="+c.plan_hash;
  artifact+=';'+std::to_string(c.target.size())+':'+c.target+';'+std::to_string(c.build.size())+':'+c.build;
  c.sha256=rns8::detail::sha256(artifact);c.admitted=true;return c;
}
inline bool reconstruct_host(const Certificate& c,const uint32_t* residues,uint64_t* output) {
  if(!c.admitted)return false;for(uint32_t p=0;p<c.prefix;++p)if(residues[p]>=garner::kConstants.moduli[p])return false;
  const auto q=host_quotient(kConstants,residues,c.prefix);if(q>kQuotientMax)return false;
  uint16_t digits[12]{};delayed_digits(kConstants,residues,c.prefix,q,c.digits,digits);
  bool negative=(digits[c.digits-1]&0x8000)!=0;
  if(c.mode==1 && negative)return false;
  if(c.mode==2){canonicalize(kConstants,c.prefix,digits);negative=false;}
  for(uint32_t limb=0;limb<c.limbs;++limb){uint64_t value=0;for(uint32_t part=0;part<4;++part){const auto d=limb*4+part;
      value|=uint64_t(d<c.digits?digits[d]:(negative?65535:0))<<(16*part);}output[limb]=value;}
  return true;
}
}
