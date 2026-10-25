#pragma once
#include <rns8/rns8.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>

namespace rns8::bench::capacity {
inline constexpr int64_t m=65537,n=3,k=65537,lda=k+3,ldb=n+3,ldc=n+3;
inline constexpr uint32_t prefix=9,panel_m=256,panel_n=4,panel_k=4096;
inline constexpr int64_t native_base=int64_t{1}<<20,maximum=native_base+7;
inline constexpr uint64_t bound=uint64_t(k)*maximum*maximum;
static_assert(bound<=INT64_MAX,"capacity fixture must retain bounded signed64 output");
inline constexpr uint64_t a_bytes=uint64_t(m)*lda*8,b_bytes=uint64_t(k)*ldb*8;
inline constexpr uint64_t c_bytes=uint64_t(m)*ldc*8,oracle_bytes=uint64_t(m)*n*8;
inline constexpr uint64_t pattern_bytes=15*uint64_t(lda)*8;
inline constexpr uint64_t caller_bytes=a_bytes+b_bytes+c_bytes+oracle_bytes+pattern_bytes;
inline constexpr uint64_t whole_a_residue_bytes=uint64_t(m)*k*prefix;
inline constexpr uint64_t host_reserve_bytes=uint64_t{8}<<30;

inline rns8_stream_gemm_desc problem(rns8_stream_gemm_schedule schedule,
    uint64_t device_budget,uint64_t host_budget) {
  rns8_stream_gemm_desc d{};
  d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.semantics=RNS8_BOUNDED_I64;d.m=m;d.n=n;d.k=k;d.lda=lda;d.ldb=ldb;d.ldc=ldc;
  d.panel_m=panel_m;d.panel_n=panel_n;d.panel_k=panel_k;d.prefix=prefix;
  d.max_abs_a=d.max_abs_b=maximum;d.device_budget_bytes=device_budget;d.host_budget_bytes=host_budget;
  d.schedule=schedule;return d;
}

// Independent period-sum reference; it never reads the generated input table.
inline int64_t reference(int64_t row,int64_t col,int64_t inner=k) {
  using boost::multiprecision::cpp_int;
  cpp_int period=0,tail=0;
  for(int64_t q=0;q<15;++q) {
    const cpp_int product=cpp_int(native_base+(3*(row%15)+7*q)%15-7)*
        (col%2?1:-1)*(native_base+(11*q+5*col+3)%15-7);
    period+=product;if(q<inner%15)tail+=product;
  }
  return (period*(inner/15)+tail).convert_to<int64_t>();
}
} // namespace rns8::bench::capacity
