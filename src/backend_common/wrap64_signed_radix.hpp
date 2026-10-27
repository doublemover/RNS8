#ifndef RNS8_WRAP64_SIGNED_RADIX_HPP
#define RNS8_WRAP64_SIGNED_RADIX_HPP
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_RADIX_HD __host__ __device__
#else
#define RNS8_RADIX_HD
#endif
namespace rns8::detail {
inline constexpr uint64_t wrap64_radix_bias=UINT64_C(0x8080808080808080);
inline constexpr uint32_t wrap64_signed_pair_k_cap=131071;
inline constexpr uint32_t wrap64_unsigned_pair_k_cap=33025;
RNS8_RADIX_HD inline constexpr uint64_t wrap64_signed_recode(uint64_t value) {
  return (value+wrap64_radix_bias)^wrap64_radix_bias;
}
RNS8_RADIX_HD inline constexpr int32_t wrap64_signed_digit(uint64_t recoded,uint32_t index) {
  const uint32_t byte=uint32_t((recoded>>(8*index))&255);
  return byte<128 ? int32_t(byte) : int32_t(byte)-256;
}
inline constexpr uint64_t wrap64_signed_product(uint64_t a,uint64_t b) {
  a=wrap64_signed_recode(a); b=wrap64_signed_recode(b);
  uint64_t sum=0;
  for(uint32_t i=0;i<8;++i) for(uint32_t j=0;i+j<8;++j)
    sum+=uint64_t(int64_t(wrap64_signed_digit(a,i))*wrap64_signed_digit(b,j))<<(8*(i+j));
  return sum;
}
static_assert(uint64_t(wrap64_signed_pair_k_cap)*16384<=INT32_MAX);
static_assert(uint64_t(wrap64_signed_pair_k_cap+1)*16384>INT32_MAX);
static_assert(uint64_t(wrap64_unsigned_pair_k_cap)*65025<=INT32_MAX);
static_assert(uint64_t(wrap64_unsigned_pair_k_cap+1)*65025>INT32_MAX);
static_assert(wrap64_signed_recode(0)==0);
static_assert(wrap64_signed_recode(UINT64_MAX)==255);
}
#undef RNS8_RADIX_HD
#endif
