#pragma once
#include "rns8/rns8.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>
namespace rns8::detail {
inline uint32_t locality_target_id(const std::string& arch) {
  const auto target=arch.substr(0,arch.find(':'));
  return target=="gfx942"?942:target=="gfx950"?950:target=="gfx1100"?1100:0;
}
inline bool locality_descriptor_valid(const rns8_locality_desc& d) {
  return d.struct_size==sizeof(d) && d.abi_version==RNS8_ABI_VERSION &&
    d.order>=RNS8_LOCALITY_ORDINARY && d.order<=RNS8_LOCALITY_SAME_OUTPUT &&
    (d.target_id==0 || d.target_id==942 || d.target_id==950 || d.target_id==1100) &&
    d.topology_version==1 && d.cache_domains>=1 && d.cache_domains<=128 &&
    d.concurrent_groups_per_domain>=1 && d.concurrent_groups_per_domain<=65536;
}
// The source is the already-admitted complete active schedule. Sorting is a
// permutation: it never creates/removes work or reassigns a basis index. Each
// safe-K phase replays this ordering on the same stream, preserving all writers.
inline rns8_status lower_locality_schedule(std::vector<rns8_plan_tile_schedule_entry>& entries,
    uint64_t k,const rns8_locality_desc& d,const std::string& actual_arch,rns8_locality_info& out) {
  if(!k || !locality_descriptor_valid(d))return RNS8_INVALID_ARGUMENT;
  rns8_locality_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  info.requested_order=d.order;info.target_id=locality_target_id(actual_arch);info.topology_version=1;
  info.applied=info.target_id && (!d.target_id || d.target_id==info.target_id) && d.order!=RNS8_LOCALITY_ORDINARY;
  info.applied_order=info.applied?d.order:RNS8_LOCALITY_ORDINARY;
  info.cache_domains=d.cache_domains;info.concurrent_groups_per_domain=d.concurrent_groups_per_domain;
  info.cache_bytes_per_domain=d.cache_bytes_per_domain;info.active_entries=entries.size();
  if(entries.size()>UINT64_MAX/sizeof(rns8_plan_tile_schedule_entry))return RNS8_RANGE_ERROR;
  info.directory_bytes=entries.size()*sizeof(rns8_plan_tile_schedule_entry);
  info.k_phases=(k-1)/65536+1;
  auto add_product=[](uint64_t& total,uint64_t a,uint64_t b)->bool {
    if(b && a>(UINT64_MAX-total)/b)return false;total+=a*b;return true;
  };
  std::set<std::tuple<uint32_t,int64_t,int64_t>> seen_a,seen_b;
  for(const auto& e:entries) {
    if(e.row_extent<=0 || e.col_extent<=0 || e.row_offset<0 || e.col_offset<0 ||
       e.group_index>=e.selected_prefix || e.selected_prefix>20 || (e.flags&RNS8_TILE_SCHEDULE_ZERO_OUTPUT))return RNS8_INVALID_ARGUMENT;
    const uint64_t rows=static_cast<uint64_t>(e.row_extent),cols=static_cast<uint64_t>(e.col_extent);
    if(!add_product(info.requested_source_bytes,rows,k) || !add_product(info.requested_source_bytes,cols,k))return RNS8_RANGE_ERROR;
    if(seen_a.emplace(e.group_index,e.row_offset,e.row_extent).second && !add_product(info.unique_source_bytes,rows,k))return RNS8_RANGE_ERROR;
    if(seen_b.emplace(e.group_index,e.col_offset,e.col_extent).second && !add_product(info.unique_source_bytes,cols,k))return RNS8_RANGE_ERROR;
    if(rows>UINT64_MAX/cols || info.k_phases>(UINT64_MAX/2))return RNS8_RANGE_ERROR;
    if(!add_product(info.output_read_write_bytes,rows*cols,2*info.k_phases-1))return RNS8_RANGE_ERROR;
  }
  if(info.applied)std::sort(entries.begin(),entries.end(),[&](const auto& a,const auto& b) {
    // Group size is one residue plane in this consumer. Tail extents are
    // compatible cost buckets; native/raw representation reuse is not assumed.
    const auto bucket_a=std::make_pair(a.row_extent,a.col_extent),bucket_b=std::make_pair(b.row_extent,b.col_extent);
    if(bucket_a!=bucket_b)return bucket_a<bucket_b;
    auto key=[&](const auto& e) {
      const uint64_t row=e.tile_row,col=e.tile_col,plane=e.group_index;
      return d.order==RNS8_LOCALITY_OUTPUT_NEIGHBOR?std::make_tuple(row,plane,col):
          d.order==RNS8_LOCALITY_MODULUS?std::make_tuple(plane,row,col):std::make_tuple(row,col,plane);
    };
    return key(a)<key(b);
  });
  uint64_t fingerprint=1469598103934665603ULL;
  auto mix=[&](uint64_t value){fingerprint^=value;fingerprint*=1099511628211ULL;};
  mix(info.target_id);mix(info.applied_order);mix(info.topology_version);
  for(const auto& e:entries){mix(e.tile_row);mix(e.tile_col);mix(e.group_index);mix(e.row_extent);mix(e.col_extent);}
  info.permutation_fingerprint=fingerprint?fingerprint:1;out=info;return RNS8_SUCCESS;
}
} // namespace rns8::detail
