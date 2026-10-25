#pragma once
#include "word_mod_arithmetic.hpp"
#include "address_domain.hpp"
#include "rns8/rns8.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <vector>
#include <stdexcept>
#include <limits>
namespace rns8::detail::word_mod {
enum class Domain : uint32_t { ring=0,prime_field=1 };
struct Problem {
  uint32_t version=1;
  Domain domain=Domain::ring;
  uint64_t modulus=0;
  int64_t m=0,n=0,k=0,lda=0,ldb=0,ldc=0;
};
struct Support {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t a_bytes=0,b_bytes=0,c_bytes=0;
};
inline Support query(const Problem& p,bool gpu=false) {
  Support s;AddressDomain a,b,c;
  if(p.version!=1 || p.modulus<2 || (p.domain!=Domain::ring && p.domain!=Domain::prime_field) ||
     p.m<=0 || p.n<=0 || p.k<=0 || p.lda<p.k || p.ldb<p.n || p.ldc<p.n ||
     !checked_host_plane_address(p.m,p.k,p.lda,1,8,a) ||
     !checked_host_plane_address(p.k,p.n,p.ldb,1,8,b) ||
     !checked_host_plane_address(p.m,p.n,p.ldc,1,8,c)) return s;
  if(p.domain==Domain::prime_field && !prime(p.modulus)) return s;
  if(gpu && !qualified_i32_gemm_address(p.m,p.n,p.k,p.lda,p.ldb,p.ldc)) {
    s.status=RNS8_UNSUPPORTED_BACKEND;return s;
  }
  s.status=RNS8_SUCCESS;s.a_bytes=a.byte_extent;s.b_bytes=b.byte_extent;s.c_bytes=c.byte_extent;return s;
}
inline rns8_status gemm_cpu(const uint64_t* a,const uint64_t* b,uint64_t* c,const Problem& p) {
  const auto s=query(p);if(s.status!=RNS8_SUCCESS) return s.status;
  if(!a || !b || !c || !nonoverlapping_byte_ranges(a,s.a_bytes,c,s.c_bytes) ||
     !nonoverlapping_byte_ranges(b,s.b_bytes,c,s.c_bytes)) return RNS8_INVALID_ARGUMENT;
  for(int64_t i=0;i<p.m;++i) for(int64_t h=0;h<p.k;++h) if(a[i*p.lda+h]>=p.modulus) return RNS8_INVALID_ARGUMENT;
  for(int64_t h=0;h<p.k;++h) for(int64_t j=0;j<p.n;++j) if(b[h*p.ldb+j]>=p.modulus) return RNS8_INVALID_ARGUMENT;
  for(int64_t i=0;i<p.m;++i) for(int64_t j=0;j<p.n;++j) {
    uint64_t sum=0;for(int64_t h=0;h<p.k;++h) sum=add(sum,multiply(a[i*p.lda+h],b[h*p.ldb+j],p.modulus),p.modulus);
    c[i*p.ldc+j]=sum;
  }
  return RNS8_SUCCESS;
}
// Exact integer root, no floating conversion. Compare in unbounded integers.
inline uint64_t ceil_root(uint64_t p,unsigned degree) {
  if(p<2 || degree<1 || degree>4) return 0;
  uint64_t lo=1,hi=p;
  while(lo<hi) {
    const uint64_t mid=lo+(hi-lo)/2;boost::multiprecision::cpp_int value=1;
    for(unsigned i=0;i<degree;++i) value*=mid;
    if(value>=p) hi=mid;else lo=mid+1;
  }
  return lo;
}
struct Decomposition {
  uint32_t left_digits=0,right_digits=0,mantissa_bits=0;
  uint64_t alpha=0,beta=0,lambda=0;
  // Sufficient (deliberately conservative) condition; not a hardware certificate.
  bool exact_bound=false;
};
inline Decomposition decomposition(uint64_t p,unsigned u,unsigned v,uint64_t lambda,unsigned bits) {
  Decomposition d{u,v,bits,ceil_root(p,u),ceil_root(p,v),lambda,false};
  if(!d.alpha || !d.beta || !lambda || bits<1 || bits>113) return d;
  using boost::multiprecision::cpp_int;
  d.exact_bound=cpp_int(lambda)*d.alpha*d.beta+p-1 <= (cpp_int(1)<<bits);
  return d;
}
inline std::vector<Decomposition> candidates(uint64_t p,uint64_t lambda,unsigned bits) {
  std::vector<Decomposition> out;
  for(const auto pair:std::array<std::array<unsigned,2>,6>{{{1,1},{1,2},{1,3},{1,4},{2,2},{2,3}}}) {
    out.push_back(decomposition(p,pair[0],pair[1],lambda,bits));
    if(pair[0]!=pair[1]) out.push_back(decomposition(p,pair[1],pair[0],lambda,bits));
  }
  return out;
}
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53,
              "word-mod floating schedules require IEEE binary64");
// 0=fused output tile; 1=word-major expanded axes; 2=cell-major expanded axes.
struct FloatingRequest {uint32_t left_digits=0,right_digits=0,slab_k=0,concatenation=0;};
inline rns8_status floating_schedule(uint64_t modulus,const FloatingRequest& request,FloatingSchedule& out) {
  if(!request.slab_k || request.slab_k>65536 || !request.left_digits || request.left_digits>4 ||
     !request.right_digits || request.right_digits>4 || request.concatenation>2 || modulus<2) return RNS8_INVALID_ARGUMENT;
  const auto proof=decomposition(modulus,request.left_digits,request.right_digits,request.slab_k,53);
  if(!proof.exact_bound) return RNS8_RANGE_ERROR;
  FloatingSchedule result;result.alpha=proof.alpha;result.beta=proof.beta;
  result.left_digits=request.left_digits;result.right_digits=request.right_digits;result.slab_k=request.slab_k;result.reserved=request.concatenation;
  uint64_t ap=1;
  for(unsigned i=0;i<request.left_digits;++i) {
    uint64_t bp=1;
    for(unsigned j=0;j<request.right_digits;++j) {
      result.weights[i*4+j]=multiply(ap,bp,modulus);bp=multiply(bp,proof.beta%modulus,modulus);
    }
    ap=multiply(ap,proof.alpha%modulus,modulus);
  }
  out=result;return RNS8_SUCCESS;
}
// Independent exact digit-splitting/recombination reference. It never divides
// by a residue or assumes a nonzero ring element has an inverse.
inline uint64_t decomposed_product(uint64_t a,uint64_t b,uint64_t p,unsigned u,unsigned v) {
  if(p<2 || a>=p || b>=p || u<1 || u>4 || v<1 || v>4) throw std::invalid_argument("word-mod decomposition");
  const uint64_t alpha=ceil_root(p,u),beta=ceil_root(p,v);
  std::array<uint64_t,4> left{},right{};
  for(unsigned i=0;i<u;++i) {left[i]=a%alpha;a/=alpha;}
  for(unsigned j=0;j<v;++j) {right[j]=b%beta;b/=beta;}
  using boost::multiprecision::cpp_int;
  cpp_int sum=0,ap=1;
  for(unsigned i=0;i<u;++i) {cpp_int bp=1;for(unsigned j=0;j<v;++j) {
    sum+=cpp_int(left[i])*right[j]*ap*bp;bp*=beta;
  } ap*=alpha;}
  return (sum%p).convert_to<uint64_t>();
}
} // namespace rns8::detail::word_mod
