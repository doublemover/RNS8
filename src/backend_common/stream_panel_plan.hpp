#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
#include <algorithm>
#include <array>
namespace rns8::detail {
struct StreamPanelProblem {
  uint32_t version=1;
  rns8_semantics semantics=RNS8_BOUNDED_I64;
  int64_t m=0,n=0,k=0,lda=0,ldb=0,ldc=0;
  uint32_t panel_m=0,panel_n=0,panel_k=0,prefix=0;
  uint64_t device_budget=0,host_budget=0;
  uint64_t retained_b_budget=0;
  rns8_stream_gemm_schedule schedule=RNS8_STREAM_GEMM_SERIAL;
};
// Direct mapping keeps lookup and eviction O(1), with a fixed metadata payload.
// The complete source binding is retained: a generation alone is not an identity.
struct StreamPanelCacheEntry {
  const void* source=nullptr;
  uint64_t instance=0,generation=0,panel=0;
  bool matches(const void* p,uint64_t i,uint64_t g,uint64_t key) const noexcept {
    return source==p && instance==i && generation==g && panel==key;
  }
};
static_assert(sizeof(StreamPanelCacheEntry)==32,"stream cache receipt requires a 64-bit host ABI");
enum StreamPanelSlot : size_t { stream_a,stream_b,stream_c0,stream_c1,
    stream_native_a,stream_native_b,stream_export,stream_status,
    stream_a1,stream_b1,stream_native_a1,stream_native_b1,stream_export1,stream_status1,stream_slot_count };
struct StreamPanelResources {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t device_bytes=0,host_bytes=0,minimum_device_bytes=0,minimum_host_bytes=0;
  uint64_t a_span=0,b_span=0,c_span=0,tile_count=0,panels_per_tile=0,total_steps=0;
  uint64_t b_cache_offset=0,b_cache_stride=0,b_cache_entries=0,b_cache_bytes=0;
  uint64_t b_panels=0,b_cold_loads=0,b_warm_loads=0;
  uint64_t pinned_bytes=0,ordinary_bytes=0,control_bytes=0;
  uint32_t streams=0,events=0;
  std::array<uint64_t,stream_slot_count> host_offsets{};
  std::array<uint64_t,stream_slot_count> offsets{},bytes{};
};
inline bool stream_panel_storage(uint64_t m,uint64_t n,uint64_t k,uint32_t p,StreamPanelResources& out,
    rns8_stream_gemm_schedule schedule=RNS8_STREAM_GEMM_SERIAL) {
  AddressDomain a,b,c;
  if(!checked_host_plane_address(m,k,k,p,1,a) || !checked_host_plane_address(k,n,n,p,1,b) ||
     !checked_host_plane_address(m,n,n,p,1,c)) return false;
  AddressDomain na,nb,nc;
  if(!checked_host_plane_address(m,k,k,1,8,na) || !checked_host_plane_address(k,n,n,1,8,nb) ||
     !checked_host_plane_address(m,n,n,1,8,nc)) return false;
  out.bytes={a.byte_extent,b.byte_extent,c.byte_extent,c.byte_extent,
             na.byte_extent,nb.byte_extent,nc.byte_extent,4};
  if(schedule==RNS8_STREAM_GEMM_OVERLAPPED) {
    out.bytes[stream_a1]=a.byte_extent;out.bytes[stream_b1]=b.byte_extent;
    out.bytes[stream_native_a1]=na.byte_extent;out.bytes[stream_native_b1]=nb.byte_extent;
    out.bytes[stream_export1]=nc.byte_extent;out.bytes[stream_status1]=4;
  }
  constexpr uint64_t limit=uint64_t(PTRDIFF_MAX);
  uint64_t total=0;
  for(size_t i=0;i<stream_slot_count;++i) {
    if(total>limit-255 || out.bytes[i]>limit-255) return false;
    total=(total+255)&~UINT64_C(255);out.offsets[i]=total;
    const auto padded=(out.bytes[i]+255)&~UINT64_C(255);
    if(padded>limit-total) return false;
    total+=padded;
  }
  if(na.byte_extent>limit-nb.byte_extent) return false;
  const uint64_t inputs=na.byte_extent+nb.byte_extent;
  if(nc.byte_extent>limit-inputs) return false;
  out.device_bytes=total;out.host_bytes=inputs+nc.byte_extent;out.ordinary_bytes=out.host_bytes;
  if(schedule==RNS8_STREAM_GEMM_OVERLAPPED) {
    uint64_t pinned=0;
    for(auto slot:{stream_native_a,stream_native_b,stream_export,stream_status,
        stream_native_a1,stream_native_b1,stream_export1,stream_status1}) {
      if(pinned>limit-255 || out.bytes[slot]>limit-255)return false;
      out.host_offsets[slot]=pinned;
      const auto span=(out.bytes[slot]+255)&~UINT64_C(255);
      if(span>limit-pinned)return false;pinned+=span;
    }
    // Payload allowance for eleven opaque handle slots, two export records and
    // ring flags. Driver-internal stream/event allocation overhead is opaque.
    out.control_bytes=256;out.pinned_bytes=pinned;out.ordinary_bytes=out.control_bytes;
    if(pinned>limit-out.control_bytes)return false;
    out.host_bytes=pinned+out.control_bytes;out.streams=3;out.events=8;
  }
  return true;
}
inline StreamPanelResources plan_stream_panels(const StreamPanelProblem& p) {
  StreamPanelResources out;
  if(p.version!=1 || p.m<=0 || p.n<=0 || p.k<=0 || !p.panel_m || !p.panel_n || !p.panel_k ||
     !p.prefix || p.prefix>20 || p.lda<p.k || p.ldb<p.n || p.ldc<p.n ||
     (p.schedule!=RNS8_STREAM_GEMM_SERIAL && p.schedule!=RNS8_STREAM_GEMM_OVERLAPPED)) return out;
  if(p.semantics!=RNS8_BOUNDED_I64 && p.semantics!=RNS8_BOUNDED_U64) {
    out.status=RNS8_UNSUPPORTED_BACKEND;return out;
  }
  AddressDomain a,b,c;
  if(!checked_host_plane_address(p.m,p.k,p.lda,1,8,a) ||
     !checked_host_plane_address(p.k,p.n,p.ldb,1,8,b) ||
     !checked_host_plane_address(p.m,p.n,p.ldc,1,8,c)) {out.status=RNS8_RANGE_ERROR;return out;}
  out.a_span=a.byte_extent;out.b_span=b.byte_extent;out.c_span=c.byte_extent;
  const auto m=std::min<uint64_t>(p.m,p.panel_m),n=std::min<uint64_t>(p.n,p.panel_n),
             k=std::min<uint64_t>(p.k,p.panel_k);
  if(!qualified_i32_gemm_address(m,n,k,k,n,n,p.prefix) ||
      !qualified_i32_matrix_address(m,k,k,1,8) || !qualified_i32_matrix_address(k,n,n,1,8) ||
      !qualified_i32_matrix_address(m,n,n,1,8)) {
    out.status=RNS8_UNSUPPORTED_BACKEND;return out;
  }
  if(!stream_panel_storage(m,n,k,p.prefix,out,p.schedule)) {out.status=RNS8_RANGE_ERROR;return out;}
  StreamPanelResources minimum;
  if(!stream_panel_storage(1,1,1,p.prefix,minimum,p.schedule)) {out.status=RNS8_RANGE_ERROR;return out;}
  out.minimum_device_bytes=minimum.device_bytes;out.minimum_host_bytes=minimum.host_bytes;
  const uint64_t mt=(uint64_t(p.m)-1)/m+1,nt=(uint64_t(p.n)-1)/n+1;
  out.panels_per_tile=(uint64_t(p.k)-1)/k+1;
  if(mt>UINT64_MAX/nt || mt*nt>UINT64_MAX/out.panels_per_tile) {out.status=RNS8_RANGE_ERROR;return out;}
  out.tile_count=mt*nt;out.total_steps=out.tile_count*out.panels_per_tile;
  const auto upload=out.bytes[stream_native_a]+out.bytes[stream_native_b];
  if(out.total_steps>UINT64_MAX/upload || out.tile_count>UINT64_MAX/(out.bytes[stream_export]+4)) {
    out.status=RNS8_RANGE_ERROR;return out;
  }
  out.b_panels=nt*out.panels_per_tile;
  out.b_cold_loads=out.b_warm_loads=out.total_steps;
  out.b_cache_offset=out.device_bytes;
  out.b_cache_stride=(out.bytes[stream_b]+255)&~UINT64_C(255);
  if(out.device_bytes>p.device_budget || out.host_bytes>p.host_budget) {
    out.status=RNS8_WORKSPACE_TOO_SMALL;return out;
  }
  const auto device_limit=std::min<uint64_t>(p.device_budget,uint64_t(PTRDIFF_MAX));
  const auto host_limit=std::min<uint64_t>(p.host_budget,uint64_t(PTRDIFF_MAX));
  out.b_cache_entries=std::min({out.b_panels,p.retained_b_budget/out.b_cache_stride,
      (device_limit-out.device_bytes)/out.b_cache_stride,
      (host_limit-out.host_bytes)/sizeof(StreamPanelCacheEntry)});
  // Do not reserve payload for a cache whose every slot conflicts on each scan.
  if(out.b_cache_entries<=out.b_panels/2) out.b_cache_entries=0;
  out.b_cache_bytes=out.b_cache_entries*out.b_cache_stride;
  out.device_bytes+=out.b_cache_bytes;
  out.host_bytes+=out.b_cache_entries*sizeof(StreamPanelCacheEntry);
  out.ordinary_bytes+=out.b_cache_entries*sizeof(StreamPanelCacheEntry);
  if(out.b_cache_entries) {
    // One pass visits B panels in ordinal order. Slots with two or more keys
    // miss on every pass; singleton slots hit after the first row tile.
    const auto q=out.b_panels,s=out.b_cache_entries;
    const auto singleton=q-s<s?s-(q-s):0;
    const auto collisions=q-singleton;
    out.b_cold_loads=q+(mt-1)*collisions;
    out.b_warm_loads=mt*collisions;
  }
  out.status=RNS8_SUCCESS;return out;
}
struct StreamPanelStep {
  uint64_t index=0,row=0,col=0,k_begin=0,rows=0,cols=0,depth=0;
  bool first=false,last=false;
};
// Serial epoch-local cursor. No completion can skip/duplicate a K interval or
// advance a different source binding. Caller publishes only after complete().
class StreamPanelCoverage {
 public:
  StreamPanelCoverage(const StreamPanelProblem& problem,const StreamPanelResources& resources,
                      uint64_t epoch,uint64_t a_instance,uint64_t a_generation,
                      uint64_t b_instance,uint64_t b_generation)
    : p_(problem),r_(resources),epoch_(epoch),ai_(a_instance),ag_(a_generation),
      bi_(b_instance),bg_(b_generation) {}
  bool next(StreamPanelStep& step) const {
    if(r_.status!=RNS8_SUCCESS || !epoch_ || !ai_ || !bi_ || complete()) return false;
    const uint64_t m=std::min<uint64_t>(p_.m,p_.panel_m),n=std::min<uint64_t>(p_.n,p_.panel_n),
                   k=std::min<uint64_t>(p_.k,p_.panel_k),nt=(uint64_t(p_.n)-1)/n+1;
    const uint64_t tile=done_/r_.panels_per_tile,kp=done_%r_.panels_per_tile;
    step={done_,(tile/nt)*m,(tile%nt)*n,kp*k,0,0,0,kp==0,kp+1==r_.panels_per_tile};
    step.rows=std::min<uint64_t>(m,p_.m-step.row);
    step.cols=std::min<uint64_t>(n,p_.n-step.col);
    step.depth=std::min<uint64_t>(k,p_.k-step.k_begin);
    return true;
  }
  bool acknowledge(uint64_t index,uint64_t epoch,uint64_t ai,uint64_t ag,uint64_t bi,uint64_t bg) {
    if(r_.status!=RNS8_SUCCESS || !epoch_ || !ai_ || !bi_ || complete() || index!=done_ ||
       epoch!=epoch_ || ai!=ai_ || ag!=ag_ || bi!=bi_ || bg!=bg_) return false;
    ++done_;return true;
  }
  bool complete() const {return r_.status==RNS8_SUCCESS && done_==r_.total_steps;}
 private:
  StreamPanelProblem p_;StreamPanelResources r_;
  uint64_t epoch_,ai_,ag_,bi_,bg_,done_=0;
};
}
