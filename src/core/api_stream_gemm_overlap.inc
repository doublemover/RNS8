#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
struct StreamPanelOverlap {
  hip_pinned_host_buffer pinned;
  hip_unique_stream upload,compute,download;
  std::array<hip_unique_event,2> uploaded,consumed,export_ready,downloaded;
};
static rns8_status create_stream_overlap(rns8_stream_gemm_plan& plan) {
  hip_scoped_device selected(plan.ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  auto state=std::make_unique<StreamPanelOverlap>();
  if(state->pinned.ensure(plan.ctx->device_id,plan.resources.pinned_bytes)!=hipSuccess ||
      state->upload.create_non_blocking()!=hipSuccess || state->compute.create_non_blocking()!=hipSuccess ||
      state->download.create_non_blocking()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  for(auto* events:{&state->uploaded,&state->consumed,&state->export_ready,&state->downloaded})for(auto& event:*events) {
    hipEvent_t handle=nullptr;
    if(hipEventCreateWithFlags(&handle,hipEventDisableTiming)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    event.reset(handle);
  }
  plan.overlap=std::move(state);return RNS8_SUCCESS;
}
static bool drain_stream_overlap(StreamPanelOverlap& state) {
  // Attempt every drain even if an earlier stream fails. A return code never
  // authorizes reuse of another still-running stream's pinned/device slots.
  bool drained=true;
  for(auto* stream:{&state.upload,&state.compute,&state.download})
    if(*stream && hipStreamSynchronize(stream->get())!=hipSuccess)drained=false;
  if(drained)state.pinned.mark_completed();return drained;
}
static rns8_status close_stream_overlap(rns8_stream_gemm_plan& plan) {
  hip_scoped_device selected(plan.ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  auto& state=*plan.overlap;
  if(!drain_stream_overlap(state))return RNS8_BACKEND_FAILURE;
  for(auto* events:{&state.uploaded,&state.consumed,&state.export_ready,&state.downloaded})for(auto& event:*events)
    if(event) {
      if(hipEventDestroy(event.get())!=hipSuccess)return RNS8_BACKEND_FAILURE;
      (void)event.release();
    }
  for(auto* stream:{&state.upload,&state.compute,&state.download})if(*stream) {
    if(hipStreamDestroy(stream->get())!=hipSuccess)return RNS8_BACKEND_FAILURE;
    (void)stream->release();
  }
  if(state.pinned.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  plan.overlap.reset();return RNS8_SUCCESS;
}
static rns8_status execute_stream_overlap(rns8_stream_gemm_plan& plan,const void* a,const void* b,void* c,
    uint64_t ai,uint64_t ag,uint64_t bi,uint64_t bg,rns8_stream_gemm_result& result) {
  hip_scoped_device selected(plan.ctx->device_id);
  if(selected.status()!=hipSuccess){result.status=RNS8_BACKEND_FAILURE;return result.status;}
  auto& state=*plan.overlap;const auto& r=plan.resources;const auto& d=plan.desc;
  const bool sign=d.semantics==RNS8_BOUNDED_I64;
  const auto device=[&](StreamPanelSlot slot){return static_cast<uint8_t*>(plan.arena)+r.offsets[slot];};
  const auto host=[&](StreamPanelSlot slot){return static_cast<uint8_t*>(state.pinned.get())+r.host_offsets[slot];};
  const StreamPanelSlot as[]{stream_a,stream_a1},bs[]{stream_b,stream_b1};
  const StreamPanelSlot nas[]{stream_native_a,stream_native_a1},nbs[]{stream_native_b,stream_native_b1};
  const StreamPanelSlot outs[]{stream_export,stream_export1},flags[]{stream_status,stream_status1};
  bool input_used[2]{},output_pending[2]{};
  std::array<StreamPanelStep,2> exports;
  state.pinned.mark_pending();
  const auto fail=[&](rns8_status status) {
    for(uint64_t i=0;i<r.b_cache_entries;++i)plan.b_cache[i]={};
    if(!drain_stream_overlap(state)) {
      plan.quarantined=true;plan.ctx->hip_failure_poisoned=true;result.quarantined=1;
    }
    result.status=status;return status;
  };
  const auto scatter=[&](unsigned slot)->rns8_status {
    if(!output_pending[slot])return RNS8_SUCCESS;
    if(hipEventSynchronize(state.downloaded[slot].get())!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(*reinterpret_cast<const int*>(host(flags[slot])))return RNS8_RANGE_ERROR;
    const auto& step=exports[slot];result.output_started=1;
    for(uint64_t row=0;row<step.rows;++row)
      std::memcpy(static_cast<uint8_t*>(c)+((step.row+row)*d.ldc+step.col)*8,
          host(outs[slot])+row*step.cols*8,step.cols*8);
    output_pending[slot]=false;++result.completed_tiles;
    result.completed_panels=std::max(result.completed_panels,step.index+1);
    return RNS8_SUCCESS;
  };
  const auto copy_panel=[](void* dst,const void* src,uint64_t row,uint64_t col,uint64_t rows,uint64_t cols,uint64_t ld) {
    for(uint64_t r=0;r<rows;++r)std::memcpy(static_cast<uint8_t*>(dst)+r*cols*8,
        static_cast<const uint8_t*>(src)+((row+r)*ld+col)*8,cols*8);
  };
  // This cursor certifies submitted coverage only. Whole-result publication
  // additionally requires every status download and all three terminal drains.
  StreamPanelCoverage submitted(plan.problem,r,result.epoch,ai,ag,bi,bg);StreamPanelStep step;
  void* current=device(stream_c0);void* next=device(stream_c1);uint64_t tile_index=0;
  while(submitted.next(step)) {
    const unsigned slot=unsigned(step.index&1);
    if(input_used[slot]) {
      // CPU may refill pinned memory after its DMA read, while the old device
      // panel remains readable until the compute-consumed event below.
      if(hipEventSynchronize(state.uploaded[slot].get())!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
      if(hipStreamWaitEvent(state.upload.get(),state.consumed[slot].get(),0)!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
    }
    copy_panel(host(nas[slot]),a,step.row,step.k_begin,step.rows,step.depth,d.lda);
    const auto key=step.index%r.b_panels;
    auto* entry=r.b_cache_entries?&plan.b_cache[key%r.b_cache_entries]:nullptr;
    void* b_residues=entry?static_cast<uint8_t*>(plan.arena)+r.b_cache_offset+(key%r.b_cache_entries)*r.b_cache_stride:device(bs[slot]);
    const bool hit=entry && entry->matches(b,bi,bg,key);
    if(!hit) {
      if(entry)*entry={};
      copy_panel(host(nbs[slot]),b,step.k_begin,step.col,step.depth,step.cols,d.ldb);
    }
    if(hipMemcpyAsync(device(nas[slot]),host(nas[slot]),step.rows*step.depth*8,hipMemcpyHostToDevice,state.upload.get())!=hipSuccess)
      return fail(RNS8_BACKEND_FAILURE);
    result.h2d_bytes+=step.rows*step.depth*8;++result.h2d_transfers;
    if(!hit) {
      if(hipMemcpyAsync(device(nbs[slot]),host(nbs[slot]),step.depth*step.cols*8,hipMemcpyHostToDevice,state.upload.get())!=hipSuccess)
        return fail(RNS8_BACKEND_FAILURE);
      result.h2d_bytes+=step.depth*step.cols*8;++result.h2d_transfers;
    }
    if(hipEventRecord(state.uploaded[slot].get(),state.upload.get())!=hipSuccess ||
        hipStreamWaitEvent(state.compute.get(),state.uploaded[slot].get(),0)!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
    auto status=hip_direct_enqueue_panel_pack(plan.ctx->device_id,device(nas[slot]),device(as[slot]),step.rows,step.depth,d.prefix,sign,state.compute.get());
    if(status!=RNS8_SUCCESS)return fail(status);
    if(hit)++result.b_panel_cache_hits;
    else {
      status=hip_direct_enqueue_panel_pack(plan.ctx->device_id,device(nbs[slot]),b_residues,step.depth,step.cols,d.prefix,sign,state.compute.get());
      if(status!=RNS8_SUCCESS)return fail(status);
      if(entry)*entry={b,bi,bg,key};++result.b_panel_loads;
    }
    if(step.first && hipMemsetAsync(current,0,r.bytes[stream_c0],state.compute.get())!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
    status=hip_direct_enqueue_panel_gemm(plan.ctx->device_id,device(as[slot]),b_residues,current,next,
        step.rows,step.cols,step.depth,d.prefix,state.compute.get());
    if(status!=RNS8_SUCCESS)return fail(status);std::swap(current,next);
    if(hipEventRecord(state.consumed[slot].get(),state.compute.get())!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
    input_used[slot]=true;
    if(!submitted.acknowledge(step.index,result.epoch,ai,ag,bi,bg))return fail(RNS8_INTERNAL_ERROR);
    if(step.last) {
      const unsigned out_slot=unsigned(tile_index++&1);
      if((status=scatter(out_slot))!=RNS8_SUCCESS)return fail(status);
      status=hip_direct_enqueue_panel_export(plan.ctx->device_id,current,device(outs[out_slot]),device(flags[out_slot]),
          step.rows,step.cols,d.prefix,plan.bound,sign,state.compute.get());
      if(status!=RNS8_SUCCESS)return fail(status);
      if(hipEventRecord(state.export_ready[out_slot].get(),state.compute.get())!=hipSuccess ||
          hipStreamWaitEvent(state.download.get(),state.export_ready[out_slot].get(),0)!=hipSuccess ||
          hipMemcpyAsync(host(outs[out_slot]),device(outs[out_slot]),step.rows*step.cols*8,hipMemcpyDeviceToHost,state.download.get())!=hipSuccess ||
          hipMemcpyAsync(host(flags[out_slot]),device(flags[out_slot]),4,hipMemcpyDeviceToHost,state.download.get())!=hipSuccess ||
          hipEventRecord(state.downloaded[out_slot].get(),state.download.get())!=hipSuccess)return fail(RNS8_BACKEND_FAILURE);
      result.d2h_bytes+=step.rows*step.cols*8+4;result.d2h_transfers+=2;
      exports[out_slot]=step;output_pending[out_slot]=true;
    }
  }
  // Flush in logical order; each host scatter observes its own current event.
  unsigned first=output_pending[0] && output_pending[1] && exports[1].index<exports[0].index?1:0;
  for(unsigned i=0;i<2;++i) {const auto status=scatter(first^i);if(status!=RNS8_SUCCESS)return fail(status);}
  if(!submitted.complete() || !drain_stream_overlap(state))return fail(RNS8_BACKEND_FAILURE);
  if(result.completed_tiles!=r.tile_count || result.completed_panels!=r.total_steps)return fail(RNS8_INTERNAL_ERROR);
  result.status=RNS8_SUCCESS;result.output_valid=1;return RNS8_SUCCESS;
}
#else
struct StreamPanelOverlap {};
static rns8_status create_stream_overlap(rns8_stream_gemm_plan&){return RNS8_UNSUPPORTED_BACKEND;}
static rns8_status close_stream_overlap(rns8_stream_gemm_plan&){return RNS8_UNSUPPORTED_BACKEND;}
static rns8_status execute_stream_overlap(rns8_stream_gemm_plan&,const void*,const void*,void*,uint64_t,uint64_t,uint64_t,uint64_t,rns8_stream_gemm_result& out) {
  out.status=RNS8_UNSUPPORTED_BACKEND;return out.status;
}
#endif
