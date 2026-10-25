#pragma once
#include "locality_schedule.hpp"
#include "native_wmma_queue.hpp"
#include "encoded_b_layout.hpp"
namespace rns8::detail {
// The immutable queue already owns (task/fragment, output tile, modulus group).
// Reorder only these complete identities, never the fragment's partial address.
inline rns8_status lower_native_locality(const std::vector<NativeWmmaTask>& tasks,
    std::vector<NativeWmmaWork>& work,const rns8_locality_desc& d,const std::string& actual_arch,
    rns8_locality_info& out,uint64_t& transient_bytes) {
  if(!locality_descriptor_valid(d) || tasks.empty() || work.empty())return RNS8_INVALID_ARGUMENT;
  rns8_locality_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  info.requested_order=d.order;info.target_id=locality_target_id(actual_arch);info.topology_version=1;
  info.applied=info.target_id && (!d.target_id || d.target_id==info.target_id) && d.order!=RNS8_LOCALITY_ORDINARY;
  info.applied_order=info.applied?d.order:RNS8_LOCALITY_ORDINARY;info.cache_domains=d.cache_domains;
  info.concurrent_groups_per_domain=d.concurrent_groups_per_domain;info.cache_bytes_per_domain=d.cache_bytes_per_domain;
  info.active_entries=work.size();info.directory_bytes=work.size()*sizeof(NativeWmmaWork);info.k_phases=1;
  const auto add=[](uint64_t& dst,uint64_t value){if(value>UINT64_MAX-dst)return false;dst+=value;return true;};
  // Exact union of consumed source byte intervals, not padded allocation spans.
  // Native group reuse does not multiply raw storage by the residue count.
  std::vector<std::pair<uintptr_t,uintptr_t>> intervals;
  const auto span=[&](const void* base,uint64_t offset,uint64_t bytes){const auto p=reinterpret_cast<uintptr_t>(base);
    if(!base || !bytes || offset>UINTPTR_MAX-p || bytes>UINTPTR_MAX-p-offset)return false;
    intervals.emplace_back(p+offset,p+offset+bytes);return true;};
  for(const auto& t:tasks){
    for(int64_t row=0;row<t.m;++row)if(!span(t.a,uint64_t(row)*t.lda*8,uint64_t(t.k)*8))return RNS8_RANGE_ERROR;
    if(t.encoded_b){const auto layout=encoded_b_layout(t.k,t.n,t.ldb,t.prefix,UINT64_MAX);
      if(layout.status!=RNS8_SUCCESS || !span(t.encoded_b,0,layout.payload_bytes))return RNS8_RANGE_ERROR;
    }else for(int64_t q=0;q<t.k;++q)if(!span(t.b,uint64_t(q)*t.ldb*8,uint64_t(t.n)*8))return RNS8_RANGE_ERROR;
  }
  std::sort(intervals.begin(),intervals.end());uintptr_t first=0,end=0;
  for(auto interval:intervals){if(interval.first>end){if(!add(info.unique_source_bytes,end-first))return RNS8_RANGE_ERROR;first=interval.first;end=interval.second;}else end=std::max(end,interval.second);}
  if(!add(info.unique_source_bytes,end-first))return RNS8_RANGE_ERROR;
  transient_bytes=intervals.capacity()*sizeof(intervals.front());
  const auto bucket=[&](const NativeWmmaWork& w){const auto& t=tasks[w.task];
    return std::make_tuple(bool(t.encoded_b),t.unsigned_inputs,t.group,std::min<uint32_t>(t.group,t.prefix-w.first_plane),
      std::min<int64_t>(16,t.m-int64_t(w.tile_row)*16),std::min<int64_t>(16,t.n-int64_t(w.tile_col)*16),t.k);};
  for(const auto& w:work){if(w.task>=tasks.size())return RNS8_INVALID_ARGUMENT;const auto& t=tasks[w.task];
    const int64_t rows=std::min<int64_t>(16,t.m-int64_t(w.tile_row)*16),cols=std::min<int64_t>(16,t.n-int64_t(w.tile_col)*16);
    if(rows<=0 || cols<=0 || w.first_plane>=t.prefix || w.first_plane%t.group)return RNS8_INVALID_ARGUMENT;
    const uint64_t planes=std::min(t.group,t.prefix-w.first_plane);
    const uint64_t b_bytes=t.encoded_b?uint64_t(cols)*t.encoded_k_words*4*planes:uint64_t(cols)*t.k*8;
    if(!add(info.requested_source_bytes,uint64_t(rows)*t.k*8) || !add(info.requested_source_bytes,b_bytes) ||
       !add(info.output_read_write_bytes,uint64_t(rows)*cols*planes))return RNS8_RANGE_ERROR;
  }
  if(info.applied)std::sort(work.begin(),work.end(),[&](const auto& a,const auto& b){
    const auto ba=bucket(a),bb=bucket(b);if(ba!=bb)return ba<bb;
    const auto key=[&](const NativeWmmaWork& w){return d.order==RNS8_LOCALITY_OUTPUT_NEIGHBOR?std::make_tuple(w.tile_row,w.first_plane,w.tile_col,w.task):
      d.order==RNS8_LOCALITY_MODULUS?std::make_tuple(w.first_plane,w.tile_row,w.tile_col,w.task):std::make_tuple(w.tile_row,w.tile_col,w.first_plane,w.task);};
    return key(a)<key(b);
  });
  uint64_t hash=1469598103934665603ULL;const auto mix=[&](uint64_t x){hash^=x;hash*=1099511628211ULL;};
  mix(info.target_id);mix(info.applied_order);mix(info.topology_version);
  for(const auto& w:work){mix(w.task);mix(w.tile_row);mix(w.tile_col);mix(w.first_plane);}
  info.permutation_fingerprint=hash?hash:1;out=info;return RNS8_SUCCESS;
}
}
