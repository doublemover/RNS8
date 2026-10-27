Admission initialize() {
  const auto& r=plan->request();int initialized=0,finalized=0;
  if(MPI_Initialized(&initialized)!=MPI_SUCCESS || MPI_Finalized(&finalized)!=MPI_SUCCESS || !initialized || finalized)return fail(RNS8_INVALID_ARGUMENT,"live_mpi_required");
  deadline=process_deadline(binding.setup_timeout_ns);
  const auto borrowed=*static_cast<MPI_Comm*>(binding.mpi_communicator);
  auto admitted=mpi_started(MPI_Comm_idup(borrowed,&comm,&request),"mpi_communicator_duplicate");if(!admitted || !(admitted=wait_mpi()))return admitted;
  if(MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN)!=MPI_SUCCESS)return fail(RNS8_BACKEND_FAILURE,"mpi_error_handler");
  int current_rank=0,count=0;
  if(MPI_Comm_rank(comm,&current_rank)!=MPI_SUCCESS || MPI_Comm_size(comm,&count)!=MPI_SUCCESS || count!=int(r.ranks.size()))return fail(RNS8_INVALID_ARGUMENT,"mpi_plan_rank_count");
  rank=uint32_t(current_rank);world=uint32_t(count);device=int(r.ranks[rank].ordinal);
  if(!(admitted=ProcessSession::resources(*plan,rank,binding,resources)))return admitted;
  if(r.m>INT32_MAX || r.n>INT32_MAX || r.k>INT32_MAX || r.lda>INT32_MAX || r.ldb>INT32_MAX || r.m*r.n>INT32_MAX)return fail(RNS8_UNSUPPORTED_BACKEND,"process_kernel_address_width");
  Rank actual;if(!(admitted=ProcessSession::discover(&comm,uint32_t(device),actual)))return admitted;
  if(actual.node_id!=r.ranks[rank].node_id || actual.physical_id!=r.ranks[rank].physical_id || actual.target!=r.ranks[rank].target)return fail(RNS8_INVALID_ARGUMENT,"process_physical_rank_binding");
  hip_scoped_device selected(device);if(selected.status()!=hipSuccess)return hip(selected.status(),"process_selected_device");
  size_t free=0,total=0;if(!(admitted=hip(hipMemGetInfo(&free,&total),"process_free_memory")))return admitted;
  if(resources.device_payload_bytes-resources.symmetric_arena_bytes>free)return fail(RNS8_WORKSPACE_TOO_SMALL,"process_available_memory");
  maximum=maximum_fragment(*plan);manifests.resize(size_t(world)*manifest.size());
  if(rank==r.owner)host_export.resize(size_t(r.m*r.n*3));
  if(!(admitted=hip(stream.create_non_blocking(),"process_stream")) || !(admitted=hip(ready.create(),"process_event")))return admitted;
  const bool needs_native=plan->resources().ranks[rank].native_inputs!=0;
  if((needs_native && (!(admitted=allocate(a,inputs->lhs().size()*8)) || !(admitted=allocate(b,inputs->rhs().size()*8)))) ||
      !(admitted=allocate(packed,plan->resources().ranks[rank].packed_inputs)) ||
      !(admitted=allocate(local_output,plan->resources().ranks[rank].local_outputs)) || !(admitted=allocate(status,4)))return admitted;
  if(!(admitted=hip(host_status.ensure(device,4),"process_status_host")))return admitted;
  if(r.transport==Transport::mpi_host && maximum && !(admitted=hip(payload.ensure(device,size_t(maximum)),"process_pinned_payload")))return admitted;
  if(needs_native) {
    if(!(admitted=hip(hipMemcpyAsync(a.get(),inputs->lhs().data(),inputs->lhs().size()*8,hipMemcpyHostToDevice,stream.get()),"process_A_upload")) ||
        !(admitted=hip(hipMemcpyAsync(b.get(),inputs->rhs().data(),inputs->rhs().size()*8,hipMemcpyHostToDevice,stream.get()),"process_B_upload")) || !(admitted=wait_device()))return admitted;
  }
  if(rank==r.owner) {
    uint64_t offset=0;tiles.reserve(plan->tiles().size());
    for(const auto& tile:plan->tiles()) {
      DeviceTile record;record.row=tile.row;record.col=tile.col;record.rows=tile.rows;record.cols=tile.cols;record.prefix=tile.prefix;
      record.partial_offset=offset;record.fragments=plan->k_fragments();offset+=tile.rows*tile.cols*tile.prefix*record.fragments;
      Integer magnitude=tile.magnitude;for(auto& word:record.bound){word=static_cast<uint64_t>(magnitude&UINT64_MAX);magnitude>>=64;}tiles.push_back(record);
    }
    if(!(admitted=allocate(gather,r.m*r.n*r.prefix)) || !(admitted=allocate(partials,plan->resources().ranks[rank].partial_gather)) ||
        !(admitted=allocate(result,r.m*r.n*24)) || !(admitted=allocate(directory,tiles.size()*sizeof(DeviceTile))) ||
        (r.transport==Transport::rccl && !(admitted=allocate(receive,maximum))))return admitted;
    if(!(admitted=hip(hipMemcpyAsync(directory.get(),tiles.data(),tiles.size()*sizeof(DeviceTile),hipMemcpyHostToDevice,stream.get()),"process_tile_directory")) || !(admitted=wait_device()))return admitted;
  }
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  if(r.transport==Transport::rccl) {
    if(rank==r.owner && ncclGetUniqueId(&rccl_id)!=ncclSuccess)return fail(RNS8_BACKEND_FAILURE,"rccl_unique_id");
    if(!(admitted=mpi_started(MPI_Ibcast(&rccl_id,int(sizeof(rccl_id)),MPI_BYTE,int(r.owner),comm,&request),"rccl_bootstrap_broadcast")) || !(admitted=wait_mpi()))return admitted;
    ncclConfig_t config=NCCL_CONFIG_INITIALIZER;config.blocking=0;
    const auto code=ncclCommInitRankConfig(&rccl,int(world),rccl_id,int(rank),&config);
    if(code!=ncclSuccess && code!=ncclInProgress)return fail(RNS8_BACKEND_FAILURE,"rccl_nonblocking_init",int(code));
    if(!(admitted=wait_rccl()))return admitted;
  }
#endif
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  if(r.transport==Transport::rocshmem) {
    int pe=-1,pes=0;
    if(!binding.symmetric_arena || binding.symmetric_arena_bytes<resources.symmetric_arena_bytes ||
        rns8_rocshmem_world(&pe,&pes)!=0 || pe!=int(rank) || pes!=int(world))return fail(RNS8_INVALID_ARGUMENT,"rocshmem_world_or_arena");
    hipPointerAttribute_t attr{};
    if(hipPointerGetAttributes(&attr,binding.symmetric_arena)!=hipSuccess || attr.device!=device)return fail(RNS8_INVALID_ARGUMENT,"rocshmem_arena_device");
    const auto lease=rns8_rocshmem_arena_retain(binding.symmetric_arena,resources.symmetric_arena_bytes);
    if(lease!=0)return fail(lease==2?RNS8_BUSY:RNS8_INVALID_ARGUMENT,"exclusive_rocshmem_helper_arena_required");
    arena_retained=true;
  }
#endif
  usable=true;return {};
}
