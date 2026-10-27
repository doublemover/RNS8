#pragma once
#include "direct_crt_constants.hpp"
#include <array>
namespace rns8::detail::fixed_conversion {
struct Constants {
  uint32_t import_weights[20][2]{};
  uint32_t sign_correction[20]{};
  uint32_t crt_weights[21][24][5]{};
};
constexpr Constants make_constants() {
  Constants out{};
  for(unsigned p=0;p<20;++p) {
    const auto modulus=garner::kConstants.moduli[p];uint32_t power=1;
    for(unsigned byte=0;byte<8;++byte) {
      out.import_weights[p][byte/4]|=power<<(8*(byte%4));
      power=power*256%modulus;
    }
    out.sign_correction[p]=power;
  }
  for(unsigned prefix=1;prefix<=20;++prefix) for(unsigned byte=0;byte<24;++byte)
    for(unsigned p=0;p<prefix;++p) {
      const uint32_t digit=uint32_t((direct_crt::kConstants.weights[prefix][p][byte/8]>>(8*(byte%8)))&255);
      out.crt_weights[prefix][byte][p/4]|=digit<<(8*(p%4));
    }
  return out;
}
inline constexpr Constants kConstants=make_constants();
inline constexpr uint32_t dot4(uint32_t a,uint32_t b,uint32_t prior=0) {
  for(unsigned byte=0;byte<4;++byte) prior+=((a>>(8*byte))&255)*((b>>(8*byte))&255);
  return prior;
}
inline constexpr uint32_t import_canonical(uint64_t value,unsigned plane,bool signed_value) {
  const uint32_t modulus=garner::kConstants.moduli[plane];
  uint32_t sum=dot4(uint32_t(value),kConstants.import_weights[plane][0]);
  sum=dot4(uint32_t(value>>32),kConstants.import_weights[plane][1],sum)%modulus;
  if(signed_value && (value>>63)) sum=(sum+modulus-kConstants.sign_correction[plane])%modulus;
  return sum;
}
// Returns unreduced S=sum(v_i*(P/m_i)); caller retains the exact canonical
// correction. All prefix padding has zero coefficients and zero weights.
inline std::array<uint64_t,3> crt_unreduced(const uint32_t* canonical,unsigned prefix) {
  uint32_t packed[5]{};
  for(unsigned p=0;p<prefix;++p) {
    const auto modulus=garner::kConstants.moduli[p];
    const auto coefficient=canonical[p]*direct_crt::kConstants.inverses[prefix][p]%modulus;
    packed[p/4]|=coefficient<<(8*(p%4));
  }
  std::array<uint64_t,3> result{};uint32_t carry=0;
  for(unsigned byte=0;byte<24;++byte) {
    uint32_t sum=carry;
    for(unsigned group=0;group<(prefix+3)/4;++group)
      sum=dot4(packed[group],kConstants.crt_weights[prefix][byte][group],sum);
    result[byte/8]|=uint64_t(sum&255)<<(8*(byte%8));carry=sum>>8;
  }
  return result;
}
static_assert(8*255*255==520200);
static_assert(20*255*255==1300500);
}
