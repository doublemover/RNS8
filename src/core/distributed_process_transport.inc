Admission shard(const Work& work) {
  const auto& r=plan->request();const auto& tile=plan->tiles()[work.tile];const auto k=work.k_end-work.k_begin;
  auto* ap=static_cast<int8_t*>(packed.get());auto* bp=ap+tile.rows*k;
  auto* output=static_cast<int8_t*>(local_output.get())+work.byte_offset;
  auto code=hipError_t(rns8_hip_distributed_pack(static_cast<const uint64_t*>(a.get()),static_cast<const uint64_t*>(b.get()),ap,bp,
      tile.row,tile.col,tile.rows,tile.cols,k,r.lda,r.ldb,plan->signed_value(),work.modulus,work.k_begin,stream.get()));
  if(code==hipSuccess)code=hipError_t(rns8_hip_direct_ring_gemm_i8_device_on_stream(ap,bp,output,int(tile.rows),int(tile.cols),int(k),int(k),int(tile.cols),int(tile.cols),int(work.modulus),uint32_t((uint64_t{1}<<32)/work.modulus),int(work.logical_plane),int(r.prefix),65536,stream.get()));
  const auto admitted=hip(code,"process_local_shard");if(!admitted)return admitted;return wait_device();
}
Admission message(bool sending,void* buffer,int bytes,int peer,int tag) {
  expected_receive_bytes=sending?-1:bytes;
  auto admitted=mpi_started(sending?MPI_Isend(buffer,bytes,MPI_BYTE,peer,tag,comm,&request):MPI_Irecv(buffer,bytes,MPI_BYTE,peer,tag,comm,&request),"mpi_fragment_enqueue");
  if(!admitted)return admitted;
  return wait_mpi();
}
Admission transfer(const Work& work) {
  const auto& r=plan->request();const bool sending=rank==work.rank;
  auto* source=sending?static_cast<int8_t*>(local_output.get())+work.byte_offset:nullptr;
  Admission admitted;
  if(r.transport==Transport::mpi_host && sending) {
    if(!(admitted=hip(hipMemcpyAsync(payload.get(),source,size_t(work.bytes),hipMemcpyDeviceToHost,stream.get()),"mpi_source_stage")) || !(admitted=wait_device()))return admitted;
  }
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  if(r.transport==Transport::rocshmem && sending) {
    auto* send=static_cast<uint8_t*>(binding.symmetric_arena)+maximum*world;
    if(!(admitted=hip(hipMemcpyAsync(send,source,size_t(work.bytes),hipMemcpyDeviceToDevice,stream.get()),"rocshmem_source_stage")))return admitted;
    const uint64_t signal=progress.epoch*(plan->work().size()+1)+work.id+1;
    if(!(admitted=hip(hipError_t(rns8_rocshmem_send(binding.symmetric_arena,maximum,work.bytes,signal,int(rank),int(r.owner),stream.get())),"rocshmem_device_put_signal")) || !(admitted=wait_device()))return admitted;
  }
#endif
  if(sending)envelope=fragment_envelope(*plan,binding_hash,progress.epoch,work);
  if(!(admitted=message(sending,envelope.data(),int(envelope.size()),int(sending?r.owner:work.rank),1)))return admitted;
  if(!sending){uint64_t id=UINT64_MAX;admitted=validate_envelope(*plan,binding_hash,progress.epoch,work.rank,envelope,id);
    if(!admitted || id!=work.id)return fail(RNS8_INVALID_ARGUMENT,"network_fragment_order",0,work.id);}
  if(r.transport==Transport::mpi_host) {
    if(!(admitted=message(sending,payload.get(),int(work.bytes),int(sending?r.owner:work.rank),2)))return admitted;
  }
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  if(r.transport==Transport::rccl) {
    auto code=ncclGroupStart();if(code!=ncclSuccess)return fail(RNS8_BACKEND_FAILURE,"rccl_group_start",int(code));
    const auto queued=sending?ncclSend(source,size_t(work.bytes),ncclInt8,int(r.owner),rccl,stream.get()):
        ncclRecv(receive.get(),size_t(work.bytes),ncclInt8,int(work.rank),rccl,stream.get());
    code=ncclGroupEnd();
    if((queued!=ncclSuccess && queued!=ncclInProgress) || (code!=ncclSuccess && code!=ncclInProgress))return fail(RNS8_BACKEND_FAILURE,"rccl_matching_byte_transfer",int(queued!=ncclSuccess && queued!=ncclInProgress?queued:code));
    if(!(admitted=wait_rccl()) || !(admitted=wait_device()))return admitted;
  }
#endif
  if(!sending) {
    const void* received=r.transport==Transport::mpi_host?payload.get():receive.get();
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
    if(r.transport==Transport::rocshmem) {
      const uint64_t signal=progress.epoch*(plan->work().size()+1)+work.id+1;
      if(!(admitted=hip(hipError_t(rns8_rocshmem_acquire(binding.symmetric_arena,maximum,signal,int(work.rank),stream.get())),"rocshmem_signal_acquire")) || !(admitted=wait_device()))return admitted;
      received=static_cast<const uint8_t*>(binding.symmetric_arena)+maximum*work.rank;
    }
#endif
    if(!(admitted=assemble(work,received,r.transport==Transport::mpi_host?hipMemcpyHostToDevice:hipMemcpyDeviceToDevice)))return admitted;
  }
  // A receiver acknowledgment prevents symmetric-inbox or staging reuse before
  // the destination stream has consumed the complete fragment. Buffer is owned.
  wire_u64(manifest.data(),work.id);
  admitted=message(!sending,manifest.data(),8,int(sending?r.owner:work.rank),3);
  if(!admitted)return admitted;
  if(sending && wire_u64(manifest.data())!=work.id)return fail(RNS8_INVALID_ARGUMENT,"network_ack_work_mismatch",0,work.id);
  return {};
}
Admission assemble(const Work& work,const void* source,hipMemcpyKind kind) {
  const auto& r=plan->request();const auto& tile=plan->tiles()[work.tile];
  auto* destination=partials?static_cast<int8_t*>(partials.get())+work.gather_offset:
      static_cast<int8_t*>(gather.get())+uint64_t(work.logical_plane)*r.m*r.n+tile.row*r.n+tile.col;
  auto admitted=hip(hipMemcpy2DAsync(destination,size_t(partials?tile.cols:r.n),source,size_t(tile.cols),size_t(tile.cols),size_t(tile.rows),kind,stream.get()),"process_owner_assembly");
  if(!admitted || !(admitted=wait_device()))return admitted;
  admitted=coverage->accept(work.id,work.rank);if(!admitted)return fail(admitted.status,admitted.reason,0,work.id);
  ++progress.transfers_queued;return {};
}
Admission reconstruct() {
  const auto& r=plan->request();hipError_t code=hipSuccess;
  if(partials)code=hipError_t(rns8_hip_distributed_reduce_partials(static_cast<const int8_t*>(partials.get()),static_cast<int8_t*>(gather.get()),static_cast<const DeviceTile*>(directory.get()),r.m,r.n,uint32_t(tiles.size()),r.prefix,plan->basis_packet(),stream.get()));
  if(code==hipSuccess)code=hipError_t(rns8_hip_distributed_export(static_cast<const int8_t*>(gather.get()),static_cast<uint64_t*>(result.get()),int(r.m*r.n),int(r.n),r.tile_rows,r.tile_cols,static_cast<const DeviceTile*>(directory.get()),plan->signed_value(),plan->bounded(),static_cast<int*>(status.get()),plan->basis_packet(),stream.get()));
  if(code==hipSuccess && r.native_integrity)code=hipError_t(rns8_hip_distributed_integrity(static_cast<const uint64_t*>(a.get()),static_cast<const uint64_t*>(b.get()),static_cast<const uint64_t*>(result.get()),r.m,r.n,r.k,r.lda,r.ldb,plan->signed_value(),static_cast<int*>(status.get()),stream.get()));
  if(code==hipSuccess)code=hipMemcpyAsync(host_status.get(),status.get(),4,hipMemcpyDeviceToHost,stream.get());
  auto admitted=hip(code,"process_global_crt");if(!admitted || !(admitted=wait_device()))return admitted;
  const auto flags=*static_cast<const uint32_t*>(host_status.get());progress.integrity_failure_mask=flags&~1u;
  progress.integrity_completed=r.native_integrity && !(flags&1u);
  return {distributed_commit_status(flags),flags?"process_global_proof_failed":"admitted",rank,UINT64_MAX};
}
