#pragma once
#include "backend_common/native_tile_bound.hpp"
#include "backend_common/garner_constants.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rns8::bench {
inline bool native_dataset_profile(const std::string& name) {
  return name=="full-width-v1" || name=="uniform-small-v2" || name=="mixed-tiles-v2" ||
         name=="zero-tiles-v2" || name=="localized-outlier-v2";
}
inline uint64_t dataset_word(uint64_t value) {
  value+=0x9e3779b97f4a7c15ULL;
  value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
  value=(value^(value>>27))*0x94d049bb133111ebULL;
  return value^(value>>31);
}
inline std::string native_dataset_identity(const std::string& name,uint32_t seed) {
  if(name=="full-width-v1" && seed==3135)return "full_width_native64_mt19937_64_seed3135_forced_extrema_v1";
  return "native_distribution_v2:"+name+":"+std::to_string(seed);
}
inline void fill_native_dataset(std::vector<uint64_t>& a,std::vector<uint64_t>& b,
    uint32_t m,uint32_t n,uint32_t k,bool unsigned_inputs,const std::string& profile,
    uint32_t seed,std::mt19937_64& rng) {
  if(!native_dataset_profile(profile) || a.size()!=uint64_t{m}*k || b.size()!=uint64_t{k}*n || !m || !n || !k)
    throw std::invalid_argument("invalid native dataset shape/profile");
  if(profile=="full-width-v1") {
    for(auto& x:a)x=rng();for(auto& x:b)x=rng();
    a.front()=uint64_t{1}<<63;b.front()=UINT64_MAX;return;
  }
  auto sample=[&](uint64_t index,uint32_t axis,bool rhs) {
    const uint64_t word=dataset_word(index+(uint64_t{seed}<<32)+(rhs?0x123456789abcdef0ULL:0));
    uint64_t magnitude=1+word%7;
    if(profile=="mixed-tiles-v2" && (axis/16)%2)magnitude=(uint64_t{1}<<48)+(word&65535);
    if(profile=="zero-tiles-v2" && (axis/16)%2)magnitude=0;
    return !unsigned_inputs && (word>>63)?uint64_t{0}-magnitude:magnitude;
  };
  for(uint32_t row=0;row<m;++row)for(uint32_t q=0;q<k;++q)
    a[uint64_t{row}*k+q]=sample(uint64_t{row}*k+q,row,false);
  for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col)
    b[uint64_t{q}*n+col]=sample(uint64_t{q}*n+col,col,true);
  if(profile=="localized-outlier-v2") {
    a[uint64_t{m/2}*k+k/2]=uint64_t{1}<<60;
    b[uint64_t{k/2}*n+n/2]=uint64_t{1}<<60;
  }
}
struct NativeDatasetModel {
  std::array<uint64_t,21> required_histogram{}, executed_histogram{};
  uint64_t required_channels=0,executed_channels=0,group_work_items=0,fixed_capacity_group_work_items=0;
  uint64_t input_hash=1469598103934665603ULL,zero_tiles=0;
  uint32_t globally_required_prefix=0;
};
inline NativeDatasetModel model_native_dataset(const std::vector<uint64_t>& a,const std::vector<uint64_t>& b,
    uint32_t m,uint32_t n,uint32_t k,bool unsigned_inputs,uint32_t group,const std::string& profile) {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail;
  if(a.size()!=uint64_t{m}*k || b.size()!=uint64_t{k}*n || !m || !n || !k ||
      (group!=1 && group!=2 && group!=4))throw std::invalid_argument("invalid dataset model");
  NativeDatasetModel result;
  std::vector<NativeAxisSummary> rows((m+15)/16),cols((n+15)/16);
  auto magnitude=[&](uint64_t bits){return !unsigned_inputs && (bits>>63)?uint64_t{0}-bits:bits;};
  auto scan=[&](bool rhs) {
    auto& summaries=rhs?cols:rows;const auto& values=rhs?b:a;const uint32_t count=rhs?n:m;
    for(uint32_t axis=0;axis<count;++axis) {
      cpp_int sum=0;uint64_t maximum=0;
      for(uint32_t q=0;q<k;++q) {
        const uint64_t bits=values[rhs?uint64_t{q}*n+axis:uint64_t{axis}*k+q];
        const uint64_t mag=magnitude(bits);sum+=mag;maximum=std::max(maximum,mag);
      }
      auto& summary=summaries[axis/16];summary.maximum=std::max(summary.maximum,maximum);
      const cpp_int previous=cpp_int(summary.l1_low)+(cpp_int(summary.l1_high)<<64);
      if(sum>previous) {summary.l1_low=static_cast<uint64_t>(sum & ((cpp_int(1)<<64)-1));summary.l1_high=static_cast<uint64_t>(sum>>64);}
    }
  };
  scan(false);scan(true);
  for(const auto* values:{&a,&b})for(uint64_t bits:*values)for(int byte=0;byte<8;++byte) {
    result.input_hash^=(bits>>(byte*8))&255;result.input_hash*=1099511628211ULL;
  }
  for(const auto& lhs:rows)for(const auto& rhs:cols) {
    const cpp_int al1=cpp_int(lhs.l1_low)+(cpp_int(lhs.l1_high)<<64);
    const cpp_int bl1=cpp_int(rhs.l1_low)+(cpp_int(rhs.l1_high)<<64);
    cpp_int exact=cpp_int(lhs.maximum)*rhs.maximum*k;
    exact=std::min(exact,cpp_int(lhs.maximum*bl1));exact=std::min(exact,cpp_int(rhs.maximum*al1));
    auto production=native_axis_l1_bound(lhs,rhs,k);
    const cpp_int actual=cpp_int(production.magnitude[0])+(cpp_int(production.magnitude[1])<<64)+(cpp_int(production.magnitude[2])<<128);
    if(actual!=exact)throw std::logic_error("axis bound differs from independent exact model");
    uint32_t required=0;cpp_int product=1;const cpp_int threshold=exact*(unsigned_inputs?1:2);
    if(exact!=0) {do {if(required==20)throw std::invalid_argument("full output basis does not certify dataset");product*=garner::kConstants.moduli[required++];}while(product<=threshold);}
    select_native_bound_prefix(production,!unsigned_inputs,20,garner::kConstants.products);
    if(production.required_prefix!=required)throw std::logic_error("strict prefix selection differs from exact oracle");
    ++result.required_histogram[required];++result.executed_histogram[required];
    result.required_channels+=required;result.executed_channels+=required;
    result.group_work_items+=(required+group-1)/group;
    result.globally_required_prefix=std::max(result.globally_required_prefix,required);
  }
  result.zero_tiles=result.required_histogram[0];
  result.fixed_capacity_group_work_items=uint64_t{rows.size()}*cols.size()*((20+group-1)/group);
  uint32_t buckets=0;for(uint32_t p=1;p<=20;++p)buckets+=result.required_histogram[p]!=0;
  if((profile=="mixed-tiles-v2" && buckets<3) || (profile=="localized-outlier-v2" && buckets<2) ||
     (profile=="zero-tiles-v2" && (!result.zero_tiles || !buckets)))
    throw std::invalid_argument("dataset shape degenerated its advertised heterogeneous/zero buckets");
  if(profile!="full-width-v1" && result.group_work_items>=result.fixed_capacity_group_work_items)
    throw std::invalid_argument("savings fixture has no modeled group-work savings");
  return result;
}
} // namespace rns8::bench
