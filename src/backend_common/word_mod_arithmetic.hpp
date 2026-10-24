#pragma once
#include <cstdint>
#include <initializer_list>
#if defined(__HIPCC__)
#define RNS8_WORD_HD __host__ __device__
#else
#define RNS8_WORD_HD
#endif
namespace rns8::detail::word_mod {
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
