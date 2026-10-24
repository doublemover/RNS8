#pragma once
#include <cstdint>
#include <initializer_list>
#if defined(__HIPCC__)
#define RNS8_WORD_HD __host__ __device__
#else
#define RNS8_WORD_HD
#endif
namespace rns8::detail::word_mod {
struct DigitWeights {uint64_t values[15]{};};
// Kernel argument POD; constructed only after the exact host bound admits it.
struct FloatingSchedule {
  uint64_t alpha=0,beta=0;
  uint32_t left_digits=0,right_digits=0,slab_k=0,reserved=0;
  uint64_t weights[16]{};
};
inline constexpr uint32_t digit_slab_k=4096;
static_assert(uint64_t(digit_slab_k)*8*255*255<=INT32_MAX,
              "eight byte-pair terms per diagonal must fit signed INT32");
// Preconditions: modulus >= 2 and both operands canonical. No overflowing
// sum or product is formed, including at modulus UINT64_MAX.
RNS8_WORD_HD inline uint64_t add(uint64_t a,uint64_t b,uint64_t modulus) {
  return a>=modulus-b ? a-(modulus-b) : a+b;
}
RNS8_WORD_HD inline uint64_t multiply(uint64_t a,uint64_t b,uint64_t modulus) {
  uint64_t result=0;
  while(b) {if(b&1) result=add(result,a,modulus);b>>=1;if(b) a=add(a,a,modulus);}
  return result;
}
inline DigitWeights digit_weights(uint64_t modulus) {
  DigitWeights weights;weights.values[0]=1;
  for(unsigned d=1;d<15;++d) weights.values[d]=multiply(weights.values[d-1],256%modulus,modulus);
  return weights;
}
inline uint64_t digit_product(uint64_t a,uint64_t b,uint64_t modulus) {
  const auto weights=digit_weights(modulus);uint32_t diagonal[15]{};
  for(unsigned i=0;i<8;++i) for(unsigned j=0;j<8;++j)
    diagonal[i+j]+=uint32_t((a>>(8*i))&255)*uint32_t((b>>(8*j))&255);
  uint64_t sum=0;for(unsigned d=0;d<15;++d) sum=add(sum,multiply(weights.values[d],diagonal[d]%modulus,modulus),modulus);
  return sum;
}
inline uint64_t power(uint64_t a,uint64_t exponent,uint64_t modulus) {
  uint64_t result=1;
  while(exponent) {if(exponent&1) result=multiply(result,a,modulus);
    exponent>>=1;if(exponent) a=multiply(a,a,modulus);}
  return result;
}
// Deterministic Miller-Rabin witness set for the full unsigned 64-bit domain.
inline bool prime(uint64_t n) {
  if(n<2) return false;
  for(uint64_t p:{2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL})
    if(n%p==0) return n==p;
  uint64_t d=n-1;unsigned shifts=0;while(!(d&1)) {d>>=1;++shifts;}
  for(uint64_t witness:{2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL}) {
    const uint64_t a=witness%n;if(!a) continue;
    uint64_t x=power(a,d,n);if(x==1 || x==n-1) continue;
    bool passed=false;
    for(unsigned s=1;s<shifts;++s) {x=multiply(x,x,n);if(x==n-1) {passed=true;break;}}
    if(!passed) return false;
  }
  return true;
}
} // namespace rns8::detail::word_mod
#undef RNS8_WORD_HD
