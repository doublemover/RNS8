#include "core/api_internal.hpp"
#include "backend_common/stream_panel_plan.hpp"
#include "backend_common/stream_panel_cost.hpp"
#include "core/hip_resources.hpp"
#include <mutex>
#include <cstring>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct StreamPanelOverlap;
struct rns8_stream_gemm_plan {
  rns8_context* ctx=nullptr;
  rns8_stream_gemm_desc desc{};
  StreamPanelProblem problem;
  StreamPanelResources resources;
  std::vector<uint64_t> host_a,host_b,host_c;
  std::unique_ptr<StreamPanelCacheEntry[]> b_cache;
  void* arena=nullptr;
  uint64_t bound=0,epoch=0;
  bool quarantined=false;
  bool retiring=false;
  std::unique_ptr<StreamPanelOverlap> overlap;
  std::mutex mutex;
};
#include "api_stream_gemm_overlap.inc"
namespace {
rns8_status stream_description(const rns8_stream_gemm_desc* d,StreamPanelProblem& p,
                               StreamPanelResources& r,uint64_t& bound) {
  if(!d || !valid_abi(d->struct_size,d->abi_version,sizeof(*d)) || d->reserved) return RNS8_INVALID_ARGUMENT;
  p={1,d->semantics,d->m,d->n,d->k,d->lda,d->ldb,d->ldc,d->panel_m,d->panel_n,d->panel_k,
     d->prefix,d->device_budget_bytes,d->host_budget_bytes,d->retained_b_budget_bytes,d->schedule};
  r=plan_stream_panels(p);
  if(r.status!=RNS8_SUCCESS && r.status!=RNS8_WORKSPACE_TOO_SMALL) return r.status;
  if(d->semantics==RNS8_BOUNDED_I64 && (d->max_abs_a>(UINT64_C(1)<<63) || d->max_abs_b>(UINT64_C(1)<<63)))
    return RNS8_RANGE_ERROR;
  const boost::multiprecision::cpp_int exact=boost::multiprecision::cpp_int(d->k)*d->max_abs_a*d->max_abs_b;
  if(exact>(d->semantics==RNS8_BOUNDED_I64?boost::multiprecision::cpp_int(INT64_MAX):
                                           boost::multiprecision::cpp_int(UINT64_MAX)) ||
     !exact_range_fits(d->semantics==RNS8_BOUNDED_I64?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED,
                       exact,modulus_product(d->prefix))) return RNS8_RANGE_ERROR;
  bound=exact.convert_to<uint64_t>();return r.status;
}
bool stream_input_bound(const void* source,int64_t rows,int64_t cols,int64_t ld,bool signed_values,uint64_t bound) {
  for(int64_t row=0;row<rows;++row) for(int64_t col=0;col<cols;++col) {
    uint64_t magnitude;
    if(signed_values) {
      const auto value=static_cast<const int64_t*>(source)[row*ld+col];
      magnitude=value<0?uint64_t(0)-uint64_t(value):uint64_t(value);
    } else magnitude=static_cast<const uint64_t*>(source)[row*ld+col];
    if(magnitude>bound) return false;
  }
  return true;
}
}
rns8_status rns8_query_stream_gemm(const rns8_stream_gemm_desc* desc,rns8_stream_gemm_resources* out) {
  return guard_api([&]() -> rns8_status {
    if(!out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out))) return RNS8_INVALID_ARGUMENT;
    StreamPanelProblem p;StreamPanelResources r;uint64_t bound;
    const auto status=stream_description(desc,p,r,bound);
    if(status!=RNS8_SUCCESS && status!=RNS8_WORKSPACE_TOO_SMALL) return status;
    rns8_stream_gemm_resources info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    info.device_bytes=r.device_bytes;info.host_bytes=r.host_bytes;
    info.minimum_device_bytes=r.minimum_device_bytes;info.minimum_host_bytes=r.minimum_host_bytes;
    info.source_a_span_bytes=r.a_span;info.source_b_span_bytes=r.b_span;info.destination_span_bytes=r.c_span;
    info.output_tiles=r.tile_count;info.k_panels_per_tile=r.panels_per_tile;
    info.retained_b_entries=r.b_cache_entries;info.retained_b_device_bytes=r.b_cache_bytes;
    info.cold_b_panel_loads=r.b_cold_loads;info.warm_b_panel_loads=r.b_warm_loads;
    info.pinned_host_bytes=r.pinned_bytes;info.ordinary_host_bytes=r.ordinary_bytes;info.control_reservation_bytes=r.control_bytes;
    info.stream_count=r.streams;info.event_count=r.events;
    *out=info;return status;
  });
}
rns8_status rns8_create_stream_gemm(rns8_context* ctx,const rns8_stream_gemm_desc* desc,rns8_stream_gemm_plan** out) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !out) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !hip_direct_compiled()) return RNS8_UNSUPPORTED_BACKEND;
    auto plan=std::make_unique<rns8_stream_gemm_plan>();
    auto status=stream_description(desc,plan->problem,plan->resources,plan->bound);
    if(status!=RNS8_SUCCESS) return status;
    if(ctx->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    plan->ctx=ctx;plan->desc=*desc;
    if(desc->schedule==RNS8_STREAM_GEMM_OVERLAPPED) {
      status=create_stream_overlap(*plan);if(status!=RNS8_SUCCESS)return status;
    } else {
      plan->host_a.resize(plan->resources.bytes[stream_native_a]/8);
      plan->host_b.resize(plan->resources.bytes[stream_native_b]/8);
      plan->host_c.resize(plan->resources.bytes[stream_export]/8);
    }
    const auto capacity=uint64_t(plan->host_a.capacity())+plan->host_b.capacity()+plan->host_c.capacity();
    const auto metadata=plan->resources.b_cache_entries*sizeof(StreamPanelCacheEntry);
    if(capacity>(desc->host_budget_bytes-metadata)/8) return RNS8_WORKSPACE_TOO_SMALL;
    if(plan->resources.b_cache_entries)
      plan->b_cache=std::make_unique<StreamPanelCacheEntry[]>(plan->resources.b_cache_entries);
    status=hip_direct_allocate(ctx->device_id,plan->resources.device_bytes,&plan->arena);
    if(status!=RNS8_SUCCESS) return status;
    ++ctx->prepared_references;*out=plan.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_stream_gemm(rns8_stream_gemm_plan* plan,const void* a,const void* b,void* c,
    uint64_t ai,uint64_t ag,uint64_t bi,uint64_t bg,rns8_stream_gemm_result* out) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)) ||
       !a || !b || !c || !ai || !bi || (reinterpret_cast<uintptr_t>(a)%8) ||
       (reinterpret_cast<uintptr_t>(b)%8) || (reinterpret_cast<uintptr_t>(c)%8)) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(plan->retiring)return RNS8_INVALID_ARGUMENT;
    if(plan->quarantined || plan->ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto& d=plan->desc;const auto& r=plan->resources;
    if(!nonoverlapping_byte_ranges(c,r.c_span,a,r.a_span) || !nonoverlapping_byte_ranges(c,r.c_span,b,r.b_span))
      return RNS8_INVALID_ARGUMENT;
    if(plan->epoch==UINT64_MAX) return RNS8_RANGE_ERROR;
    const bool signed_values=d.semantics==RNS8_BOUNDED_I64;
    if(!stream_input_bound(a,d.m,d.k,d.lda,signed_values,d.max_abs_a) ||
       !stream_input_bound(b,d.k,d.n,d.ldb,signed_values,d.max_abs_b)) return RNS8_RANGE_ERROR;
    rns8_stream_gemm_result result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
    result.epoch=++plan->epoch;
    if(plan->overlap) {
      const auto status=execute_stream_overlap(*plan,a,b,c,ai,ag,bi,bg,result);*out=result;return status;
    }
    const auto pointer=[&](StreamPanelSlot slot){return static_cast<uint8_t*>(plan->arena)+r.offsets[slot];};
    void* current=pointer(stream_c0);void* next=pointer(stream_c1);
    StreamPanelCoverage coverage(plan->problem,r,result.epoch,ai,ag,bi,bg);StreamPanelStep step;
    const auto fail=[&](rns8_status status) {
      // A failed conversion must never leave a cache tag for partially written
      // residues. Drop all tags before draining, including earlier valid panels.
      for(uint64_t i=0;i<r.b_cache_entries;++i) plan->b_cache[i]={};
      // A failed operation return alone is not proof that DMA/kernels stopped.
      if(hip_direct_synchronize(plan->ctx->device_id)!=RNS8_SUCCESS) {
        plan->quarantined=true;plan->ctx->hip_failure_poisoned=true;result.quarantined=1;
      }
      result.status=status;*out=result;return status;
    };
    while(coverage.next(step)) {
      const auto copy_panel=[](std::vector<uint64_t>& destination,const void* source,uint64_t row,uint64_t col,
                                uint64_t rows,uint64_t cols,uint64_t ld) {
        for(uint64_t r=0;r<rows;++r)
          std::memcpy(destination.data()+r*cols,static_cast<const uint8_t*>(source)+((row+r)*ld+col)*8,cols*8);
      };
      copy_panel(plan->host_a,a,step.row,step.k_begin,step.rows,step.depth,d.lda);
      const auto b_key=step.index%r.b_panels;
      StreamPanelCacheEntry* entry=r.b_cache_entries?&plan->b_cache[b_key%r.b_cache_entries]:nullptr;
      void* b_residues=entry?static_cast<uint8_t*>(plan->arena)+r.b_cache_offset+
          (b_key%r.b_cache_entries)*r.b_cache_stride:pointer(stream_b);
      const bool b_hit=entry && entry->matches(b,bi,bg,b_key);
      auto status=hip_direct_copy_host_to_device(plan->ctx->device_id,pointer(stream_native_a),
                                                plan->host_a.data(),step.rows*step.depth*8);
      if(status!=RNS8_SUCCESS) return fail(status);
      result.h2d_bytes+=step.rows*step.depth*8;++result.h2d_transfers;
      const auto convert=signed_values?hip_direct_native_i64_to_rns_device:hip_direct_native_u64_to_rns_device;
      status=convert(plan->ctx->device_id,pointer(stream_native_a),pointer(stream_a),step.rows,step.depth,d.prefix);
      if(status!=RNS8_SUCCESS) return fail(status);
      if(b_hit) ++result.b_panel_cache_hits;
      else {
        if(entry) *entry={};
        copy_panel(plan->host_b,b,step.k_begin,step.col,step.depth,step.cols,d.ldb);
        status=hip_direct_copy_host_to_device(plan->ctx->device_id,pointer(stream_native_b),
                                             plan->host_b.data(),step.depth*step.cols*8);
        if(status!=RNS8_SUCCESS) return fail(status);
        result.h2d_bytes+=step.depth*step.cols*8;++result.h2d_transfers;
        status=convert(plan->ctx->device_id,pointer(stream_native_b),b_residues,step.depth,step.cols,d.prefix);
        if(status!=RNS8_SUCCESS) return fail(status);
        if(entry) *entry={b,bi,bg,b_key};
        ++result.b_panel_loads;
      }
      if(step.first) {
        status=hip_direct_zero(plan->ctx->device_id,current,r.bytes[stream_c0]);
        if(status!=RNS8_SUCCESS) return fail(status);
      }
      status=hip_direct_gemm_rns_axpby_device(plan->ctx->device_id,pointer(stream_a),b_residues,current,next,
          step.rows,step.cols,step.depth,step.depth,step.cols,step.cols,step.cols,d.prefix,1,1);
      if(status!=RNS8_SUCCESS) return fail(status);
      std::swap(current,next);
      if(!coverage.acknowledge(step.index,result.epoch,ai,ag,bi,bg)) return fail(RNS8_INTERNAL_ERROR);
      ++result.completed_panels;
      if(step.last) {
        void* export_buffer=pointer(stream_export);void* status_buffer=pointer(stream_status);
        std::size_t export_bytes=r.bytes[stream_export],status_bytes=r.bytes[stream_status];
        if(signed_values)
          status=hip_direct_export_i64_device(plan->ctx->device_id,current,&export_buffer,&export_bytes,
              &status_buffer,&status_bytes,step.rows,step.cols,d.prefix,plan->bound,
              reinterpret_cast<int64_t*>(plan->host_c.data()),step.cols,false);
        else status=hip_direct_export_u64_device(plan->ctx->device_id,current,&export_buffer,&export_bytes,
              &status_buffer,&status_bytes,step.rows,step.cols,d.prefix,plan->bound,
              plan->host_c.data(),step.cols,false);
        if(status!=RNS8_SUCCESS) return fail(status);
        result.output_started=1;
        result.d2h_bytes+=step.rows*step.cols*8+4;result.d2h_transfers+=2;
        for(uint64_t row=0;row<step.rows;++row)
          std::memcpy(static_cast<uint8_t*>(c)+((step.row+row)*d.ldc+step.col)*8,
                      plan->host_c.data()+row*step.cols,step.cols*8);
        ++result.completed_tiles;
      }
    }
    if(!coverage.complete()) return fail(RNS8_INTERNAL_ERROR);
    result.status=RNS8_SUCCESS;result.output_valid=1;*out=result;return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_stream_gemm(rns8_stream_gemm_plan* plan) {
  if(!plan) return RNS8_SUCCESS;
  std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
  plan->retiring=true;
  if(plan->quarantined && hip_direct_synchronize(plan->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
  if(plan->overlap) {
    const auto status=close_stream_overlap(*plan);if(status!=RNS8_SUCCESS)return status;
  }
  const auto status=hip_direct_free(plan->ctx->device_id,plan->arena);
  if(status!=RNS8_SUCCESS) return status;
  plan->arena=nullptr;--plan->ctx->prepared_references;
  lock.unlock();delete plan;return RNS8_SUCCESS;
}
#include "api_stream_gemm_planner.inc"
