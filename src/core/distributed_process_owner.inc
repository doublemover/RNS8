Admission initialize_device_transport_runtime() {
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  return rns8_rocshmem_runtime_init()==0?Admission{}:reject(RNS8_BACKEND_FAILURE,"rocshmem_runtime_initialize");
#else
  return reject(RNS8_UNSUPPORTED_BACKEND,"rocshmem_not_compiled");
#endif
}
Admission allocate_device_transport_arena(uint64_t bytes,void*& output) {
  if(output || !bytes || bytes>SIZE_MAX)return reject(RNS8_INVALID_ARGUMENT,"empty_symmetric_arena_output_required");
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  auto* arena=rns8_rocshmem_arena_allocate(bytes);if(!arena)return reject(RNS8_WORKSPACE_TOO_SMALL,"symmetric_arena_allocation");output=arena;return {};
#else
  return reject(RNS8_UNSUPPORTED_BACKEND,"rocshmem_not_compiled");
#endif
}
Admission release_device_transport_arena(void* arena) {
  if(!arena)return {};
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  const auto code=rns8_rocshmem_arena_free(arena);
  return code==0?Admission{}:reject(code==2?RNS8_BUSY:RNS8_INVALID_ARGUMENT,"symmetric_arena_release");
#else
  return reject(RNS8_UNSUPPORTED_BACKEND,"rocshmem_not_compiled");
#endif
}
Admission finalize_device_transport_runtime() {
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  return rns8_rocshmem_runtime_finalize()==0?Admission{}:reject(RNS8_BUSY,"symmetric_arena_still_retained");
#else
  return reject(RNS8_UNSUPPORTED_BACKEND,"rocshmem_not_compiled");
#endif
}
ProcessSession::ProcessSession(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
ProcessSession::~ProcessSession(){
  if(impl_ && !impl_->closed){auto* state=impl_.release();auto* head=Impl::quarantine.load();
    do{state->quarantine_next=head;}while(!Impl::quarantine.compare_exchange_weak(head,state));}
}
const HipProgress& ProcessSession::progress()const{return impl_->progress;}
Admission ProcessSession::discover(void* communicator,uint32_t ordinal,Rank& output) {
#if RNS8_PROCESS_ENABLED
  if(!communicator)return reject(RNS8_INVALID_ARGUMENT,"mpi_communicator_required");
  int initialized=0;if(MPI_Initialized(&initialized)!=MPI_SUCCESS || !initialized)return reject(RNS8_INVALID_ARGUMENT,"live_mpi_required");
  char node[MPI_MAX_PROCESSOR_NAME]{};int length=0;
  if(MPI_Get_processor_name(node,&length)!=MPI_SUCCESS || length<=0 || length>128)return reject(RNS8_BACKEND_FAILURE,"mpi_node_identity");
  std::vector<Rank> devices;std::vector<Edge> edges;auto admitted=HipExecution::discover({ordinal},devices,edges);if(!admitted)return admitted;
  auto rank=std::move(devices.front());rank.node_id=std::string(node,size_t(length));output=std::move(rank);return {};
#else
  (void)communicator;(void)ordinal;(void)output;return reject(RNS8_UNSUPPORTED_BACKEND,"process_mpi_hip_not_compiled");
#endif
}
Admission ProcessSession::create(std::shared_ptr<const Plan> plan,std::shared_ptr<const Inputs> inputs,
    const ProcessBinding& binding,std::unique_ptr<ProcessSession>& output) {
  if(output)return reject(RNS8_BUSY,"process_owner_already_present");
  if(!plan || !inputs || !binding.mpi_communicator || !binding.setup_timeout_ns)return reject(RNS8_INVALID_ARGUMENT,"process_plan_inputs_binding_required");
  if(!compiled(plan->request().transport))return reject(RNS8_UNSUPPORTED_BACKEND,"selected_process_transport_not_compiled");
  try {
    auto admitted=validate_inputs(*plan,*inputs);if(!admitted)return admitted;
    auto impl=std::make_unique<Impl>();impl->plan=std::move(plan);impl->inputs=std::move(inputs);impl->binding=binding;
    auto owner=std::unique_ptr<ProcessSession>(new ProcessSession(std::move(impl)));
    // Publish a cleanup owner before the first asynchronous duplication/copy.
    output=std::move(owner);
#if RNS8_PROCESS_ENABLED
    return output->impl_->initialize();
#else
    return reject(RNS8_UNSUPPORTED_BACKEND,"process_not_compiled");
#endif
  }catch(const std::bad_alloc&){return reject(RNS8_INTERNAL_ERROR,"process_preparation_allocation");}
}
Admission ProcessSession::execute(uint64_t epoch,uint64_t timeout_ns) {
  auto& p=*impl_;const auto& r=p.plan->request();
  if(p.closed || p.failed || !p.usable || !epoch || epoch==UINT64_MAX || epoch<=p.progress.epoch || !timeout_ns)return reject(RNS8_INVALID_ARGUMENT,"fresh_live_process_epoch_required");
  if(p.readers || p.progress.state==State::pending)return reject(RNS8_BUSY,"process_epoch_retained");
  if(epoch>(UINT64_MAX>>2) || epoch>(UINT64_MAX-p.plan->work().size())/(p.plan->work().size()+1))return reject(RNS8_RANGE_ERROR,"process_signal_epoch_overflow");
#if RNS8_PROCESS_ENABLED
  try {
    // All host allocation and binding proofs precede replacing the prior epoch.
    auto coverage=std::make_unique<CoverageLedger>(*p.plan);auto hash=process_binding(*p.plan,*p.inputs,epoch);
#ifdef RNS8_BUILD_MANIFEST_JSON
    hash=sha256(hash+RNS8_BUILD_MANIFEST_JSON);
#else
    return reject(RNS8_UNSUPPORTED_BACKEND,"compiled_process_build_manifest_required");
#endif
    hip_scoped_device selected(p.device);if(selected.status()!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_submit_device");
    p.coverage=std::move(coverage);p.binding_hash=std::move(hash);p.deadline=process_deadline(timeout_ns);
    p.progress={};p.progress.epoch=epoch;p.progress.expected=p.plan->work().size();p.progress.state=State::pending;p.progress.integrity_requested=r.native_integrity;
    p.active.store((epoch<<2)|1,std::memory_order_release);
    Admission admitted;
    if(!(admitted=p.hip(hipMemsetAsync(p.status.get(),0,4,p.stream.get()),"process_status_reset")))return admitted;
    if(p.rank==r.owner && !(admitted=p.hip(hipMemsetAsync(p.gather.get(),0,size_t(r.m*r.n*r.prefix),p.stream.get()),"process_gather_reset")))return admitted;
    if(!(admitted=p.wait_device()))return admitted;
    p.manifest={};std::memcpy(p.manifest.data(),p.binding_hash.data(),64);wire_u64(p.manifest.data()+64,epoch);wire_u64(p.manifest.data()+72,p.rank);
    if(!(admitted=p.mpi_started(MPI_Iallgather(p.manifest.data(),128,MPI_BYTE,p.manifests.data(),128,MPI_BYTE,p.comm,&p.request),"process_epoch_agreement")) || !(admitted=p.wait_mpi()))return admitted;
    for(uint32_t rank=0;rank<p.world;++rank) {
      const auto* manifest=p.manifests.data()+rank*128;
      if(std::memcmp(manifest,p.binding_hash.data(),64)!=0 || wire_u64(manifest+64)!=epoch || wire_u64(manifest+72)!=rank)return p.fail(RNS8_INVALID_ARGUMENT,"process_plan_input_epoch_disagreement");
    }
    uint64_t local_completed=0;
    for(const auto& work:p.plan->work()) {
      if((p.active.load(std::memory_order_acquire)&3)==2)return p.fail(RNS8_CANCELED,"process_owner_cancel",0,work.id);
      if(process_now()>=p.deadline)return p.fail(RNS8_TIMEOUT,"process_operation_deadline",0,work.id);
      if(work.rank==p.rank){if(!(admitted=p.shard(work)))return admitted;++local_completed;p.progress.local_completed=local_completed;}
      if(work.rank==r.owner) {
        if(p.rank==r.owner && !(admitted=p.assemble(work,static_cast<const int8_t*>(p.local_output.get())+work.byte_offset,hipMemcpyDeviceToDevice)))return admitted;
      }else if(p.rank==work.rank || p.rank==r.owner){if(!(admitted=p.transfer(work)))return admitted;}
    }
    Admission proof;
    if(p.rank==r.owner){if(p.coverage->completed()!=p.plan->work().size())return p.fail(RNS8_INVALID_ARGUMENT,"process_global_coverage_missing");proof=p.reconstruct();if(p.failed)return proof;}
    // Status-only collective: no residue addition, narrowing, or floating sum.
    p.manifest={};std::memcpy(p.manifest.data(),p.binding_hash.data(),64);wire_u64(p.manifest.data()+64,epoch);wire_u64(p.manifest.data()+72,p.rank);
    wire_u64(p.manifest.data()+80,local_completed);wire_u64(p.manifest.data()+88,uint64_t(proof.status));
    if(!(admitted=p.mpi_started(MPI_Iallgather(p.manifest.data(),128,MPI_BYTE,p.manifests.data(),128,MPI_BYTE,p.comm,&p.request),"process_final_rank_acknowledgments")) || !(admitted=p.wait_mpi()))return admitted;
    for(uint32_t rank=0;rank<p.world;++rank) {
      const auto* manifest=p.manifests.data()+rank*128;uint64_t expected=0;for(const auto& work:p.plan->work())if(work.rank==rank)++expected;
      if(std::memcmp(manifest,p.binding_hash.data(),64)!=0 || wire_u64(manifest+64)!=epoch || wire_u64(manifest+72)!=rank || wire_u64(manifest+80)!=expected)return p.fail(RNS8_INVALID_ARGUMENT,"process_incomplete_rank_ack",0,rank);
      const auto operation=wire_u64(manifest+88);if(operation)return p.fail(rns8_status(operation),"process_remote_proof_failure",0,rank);
      if(p.rank==r.owner){admitted=p.coverage->acknowledge(rank);if(!admitted)return p.fail(admitted.status,admitted.reason);}
    }
    if(p.rank==r.owner && !p.coverage->complete())return p.fail(RNS8_INVALID_ARGUMENT,"process_membership_publication_guard");
    uint64_t pending=(epoch<<2)|1;
    if(!p.active.compare_exchange_strong(pending,(epoch<<2)|3,std::memory_order_acq_rel))return p.fail(RNS8_CANCELED,"process_owner_cancel_before_publish");
    p.progress.published_epoch=epoch;p.progress.output_current=p.rank==r.owner;p.progress.state=State::published;return {};
  }catch(const std::bad_alloc&){if(p.progress.epoch==epoch)return p.fail(RNS8_INTERNAL_ERROR,"process_epoch_allocation");return reject(RNS8_INTERNAL_ERROR,"process_admission_allocation");}
#else
  (void)r;return reject(RNS8_UNSUPPORTED_BACKEND,"process_not_compiled");
#endif
}
Admission ProcessSession::cancel(uint64_t epoch)noexcept {
  auto& p=*impl_;if(p.rank!=p.plan->request().owner)return reject(RNS8_UNSUPPORTED_BACKEND,"cancel_on_process_output_owner_only");
  if(!epoch || epoch>(UINT64_MAX>>2))return reject(RNS8_INVALID_ARGUMENT,"live_process_epoch_required");
  uint64_t expected=(epoch<<2)|1;
  if(p.active.compare_exchange_strong(expected,(epoch<<2)|2,std::memory_order_acq_rel) || expected==((epoch<<2)|2))return {};
  return reject(RNS8_INVALID_ARGUMENT,"live_process_epoch_required");
}
Admission ProcessSession::retain(uint64_t epoch){auto& p=*impl_;if(p.closed || !p.progress.output_current || epoch!=p.progress.published_epoch)return reject(RNS8_INVALID_ARGUMENT,"published_process_epoch_required");if(p.readers==UINT64_MAX)return reject(RNS8_RANGE_ERROR,"process_reader_overflow");++p.readers;return {};}
Admission ProcessSession::release(uint64_t epoch){auto& p=*impl_;if(!p.readers || epoch!=p.progress.published_epoch)return reject(RNS8_INVALID_ARGUMENT,"retained_process_epoch_required");--p.readers;return {};}
Admission ProcessSession::resident_limbs(uint64_t epoch,int& device,const void*& pointer)const {
  const auto& p=*impl_;if(p.closed || !p.readers || !p.progress.output_current || epoch!=p.progress.published_epoch)return reject(RNS8_INVALID_ARGUMENT,"retained_process_result_required");
#if RNS8_PROCESS_ENABLED
  device=p.device;pointer=p.result.get();return {};
#else
  (void)device;(void)pointer;return reject(RNS8_UNSUPPORTED_BACKEND,"process_not_compiled");
#endif
}
Admission ProcessSession::export_limbs(uint64_t epoch,uint64_t* destination,uint64_t ld) {
  auto& p=*impl_;const auto& r=p.plan->request();AddressDomain span;
  if(!destination || p.closed || !p.progress.output_current || epoch!=p.progress.published_epoch ||
      !checked_host_plane_address(r.m,r.n,ld,1,24,span))return reject(RNS8_INVALID_ARGUMENT,"published_process_export_required");
#if RNS8_PROCESS_ENABLED
  hip_scoped_device selected(p.device);if(selected.status()!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_export_device");
  p.deadline=process_deadline(p.binding.setup_timeout_ns);
  auto admitted=p.hip(hipMemcpyAsync(p.host_export.data(),p.result.get(),p.host_export.size()*8,hipMemcpyDeviceToHost,p.stream.get()),"process_host_export");
  if(!admitted || !(admitted=p.wait_device()))return admitted;
  for(uint64_t row=0;row<r.m;++row)std::copy_n(p.host_export.data()+row*r.n*3,size_t(r.n*3),destination+row*ld*3);
  return {};
#else
  return reject(RNS8_UNSUPPORTED_BACKEND,"process_not_compiled");
#endif
}
Admission ProcessSession::close() {
  auto& p=*impl_;if(p.closed)return {};if(p.readers)return reject(RNS8_BUSY,"process_consumers_retained");
  p.usable=false;p.progress.output_current=false;
#if RNS8_PROCESS_ENABLED
  if(p.request_active){int complete=0;const auto code=MPI_Test(&p.request,&complete,MPI_STATUS_IGNORE);if(code!=MPI_SUCCESS || !complete)return reject(RNS8_BUSY,"mpi_request_quarantined");p.request_active=false;}
  hip_scoped_device selected(p.device>=0?p.device:0);
  if(p.device>=0 && selected.status()!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_cleanup_device");
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  if(p.rccl && p.failed && !p.rccl_aborted){if(ncclCommAbort(p.rccl)!=ncclSuccess)return reject(RNS8_BACKEND_FAILURE,"rccl_abort_failed");p.rccl=nullptr;p.rccl_aborted=true;}
#endif
  if(p.stream){const auto code=hipStreamQuery(p.stream.get());if(code==hipErrorNotReady)return reject(RNS8_BUSY,"process_device_work_quarantined");if(code!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_failed_drain_retained");}
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  if(p.rccl){if(ncclCommDestroy(p.rccl)!=ncclSuccess)return reject(RNS8_BACKEND_FAILURE,"rccl_destroy_failed");p.rccl=nullptr;}
#endif
  // Invalidate before the first free; retryable partial cleanup cannot expose a
  // formerly published pointer whose earlier storage has already been released.
  p.progress.output_current=false;
  for(auto* buffer:{&p.a,&p.b,&p.packed,&p.local_output,&p.gather,&p.partials,&p.result,&p.directory,&p.status,&p.receive}){buffer->mark_completed();if(buffer->reset()!=RNS8_SUCCESS)return reject(RNS8_BACKEND_FAILURE,"process_failed_free_retained");}
  if(p.payload.reset()!=hipSuccess || p.host_status.reset()!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_pinned_free_retained");
  if(p.ready){if(hipEventDestroy(p.ready.get())!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_event_free_retained");(void)p.ready.release();}
  if(p.stream){if(hipStreamDestroy(p.stream.get())!=hipSuccess)return reject(RNS8_BACKEND_FAILURE,"process_stream_free_retained");(void)p.stream.release();}
  if(p.comm!=MPI_COMM_NULL && MPI_Comm_free(&p.comm)!=MPI_SUCCESS)return reject(RNS8_BACKEND_FAILURE,"process_collective_comm_free");
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  if(p.arena_retained){if(rns8_rocshmem_arena_release_lease(p.binding.symmetric_arena)!=0)return reject(RNS8_BACKEND_FAILURE,"rocshmem_arena_lease_release");p.arena_retained=false;}
#endif
#endif
  p.closed=true;p.progress.output_current=false;return {};
}
