#include "distributed_process.hpp"
#include "backend_common/distributed_protocol.hpp"
#include "backend_common/distributed_integrity.hpp"
#include "core/hip_resources.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#if defined(RNS8_ENABLE_DISTRIBUTED_MPI) && RNS8_ENABLE_DISTRIBUTED_MPI
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#endif
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
#include <rccl/rccl.h>
#endif
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
extern "C" int rns8_rocshmem_send(void*,uint64_t,uint64_t,uint64_t,int,int,void*);
extern "C" int rns8_rocshmem_acquire(void*,uint64_t,uint64_t,int,void*);
extern "C" int rns8_rocshmem_world(int*,int*);
extern "C" int rns8_rocshmem_runtime_init();
extern "C" void* rns8_rocshmem_arena_allocate(uint64_t);
extern "C" int rns8_rocshmem_arena_free(void*);
extern "C" int rns8_rocshmem_arena_retain(void*,uint64_t);
extern "C" int rns8_rocshmem_arena_release_lease(void*);
extern "C" int rns8_rocshmem_runtime_finalize();
#endif
#if defined(RNS8_ENABLE_DISTRIBUTED_MPI) && RNS8_ENABLE_DISTRIBUTED_MPI && defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#define RNS8_PROCESS_ENABLED 1
extern "C" int rns8_hip_distributed_pack(const uint64_t*,const uint64_t*,int8_t*,int8_t*,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,bool,uint32_t,uint64_t,void*);
extern "C" int rns8_hip_basis_gemm_stream(const int8_t*,const int8_t*,int8_t*,int,int,int,const uint16_t*,uint32_t,void*);
extern "C" int rns8_hip_distributed_reduce_partials(const int8_t*,int8_t*,const rns8::detail::distributed::DeviceTile*,uint64_t,uint64_t,uint32_t,uint32_t,rns8::detail::BasisPacket,void*);
extern "C" int rns8_hip_distributed_export(const int8_t*,uint64_t*,int,int,uint32_t,uint32_t,const rns8::detail::distributed::DeviceTile*,bool,bool,int*,rns8::detail::BasisPacket,void*);
extern "C" int rns8_hip_distributed_integrity(const uint64_t*,const uint64_t*,const uint64_t*,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,bool,int*,void*);
#else
#define RNS8_PROCESS_ENABLED 0
#endif
namespace rns8::detail::distributed {
namespace {
#if RNS8_PROCESS_ENABLED
uint64_t process_now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
uint64_t process_deadline(uint64_t duration){const auto now=process_now();return duration>UINT64_MAX-now?UINT64_MAX:now+duration;}
#endif
uint64_t maximum_fragment(const Plan& plan){uint64_t bytes=0;for(const auto& work:plan.work())bytes=std::max(bytes,work.bytes);return bytes;}
}
bool ProcessSession::compiled(Transport transport)noexcept {
#if RNS8_PROCESS_ENABLED
  if(transport==Transport::mpi_host)return true;
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  if(transport==Transport::rccl)return true;
#endif
#if defined(RNS8_ENABLE_ROCSHMEM) && RNS8_ENABLE_ROCSHMEM
  if(transport==Transport::rocshmem)return true;
#endif
#endif
  (void)transport;return false;
}
Admission ProcessSession::resources(const Plan& plan,uint32_t rank,const ProcessBinding& binding,ProcessResources& output) {
  const auto& r=plan.request();if(rank>=r.ranks.size() || !process_transport(r.transport))return reject(RNS8_INVALID_ARGUMENT,"process_rank_or_transport");
  ProcessResources result;result.device_payload_bytes=plan.resources().ranks[rank].peak;
  result.pinned_bytes=4;const auto maximum=maximum_fragment(plan);
  if(r.transport==Transport::mpi_host && !add_bytes(result.pinned_bytes,maximum))return reject(RNS8_RANGE_ERROR,"process_pinned_extent");
  if(r.transport==Transport::rccl && rank==r.owner && !add_bytes(result.device_payload_bytes,maximum))return reject(RNS8_RANGE_ERROR,"process_receive_extent");
  if(r.transport==Transport::rocshmem) {
    if(!multiply(maximum,r.ranks.size()+1,result.symmetric_arena_bytes) ||
        !aligned_bytes(result.symmetric_arena_bytes,result.symmetric_arena_bytes) ||
        !add_bytes(result.symmetric_arena_bytes,r.ranks.size()*8) ||
        !aligned_bytes(result.symmetric_arena_bytes,result.symmetric_arena_bytes) ||
        !add_bytes(result.device_payload_bytes,result.symmetric_arena_bytes))return reject(RNS8_RANGE_ERROR,"symmetric_arena_extent");
  }
  result.transport_overhead_known=r.transport==Transport::mpi_host;
  if(!result.transport_overhead_known && (!binding.acknowledge_unknown_transport_overhead || !binding.transport_headroom_bytes))
    return reject(RNS8_UNSUPPORTED_BACKEND,"explicit_opaque_transport_headroom_required",rank);
  result.transport_headroom_bytes=binding.transport_headroom_bytes;
  if(!add_bytes(result.device_payload_bytes,binding.transport_headroom_bytes))return reject(RNS8_RANGE_ERROR,"transport_headroom_overflow",rank);
  result.host_metadata_bytes=plan.resources().contract_metadata;
  if(!add_bytes(result.host_metadata_bytes,r.ranks.size()*128+4096))return reject(RNS8_RANGE_ERROR,"process_metadata_extent",rank);
  uint64_t host_peak=plan.resources().host_model;
  if(!add_bytes(host_peak,r.ranks.size()*128+4096))return reject(RNS8_RANGE_ERROR,"process_host_peak_overflow",rank);
  if(result.device_payload_bytes>r.ranks[rank].memory_budget || result.pinned_bytes>r.pinned_budget ||
      host_peak>r.host_budget)return reject(RNS8_WORKSPACE_TOO_SMALL,"process_resource_budget",rank);
  output=result;return {};
}
struct ProcessSession::Impl {
  std::shared_ptr<const Plan> plan;std::shared_ptr<const Inputs> inputs;ProcessBinding binding;
  HipProgress progress;ProcessResources resources;uint32_t rank=0,world=0;int device=-1;
  uint64_t deadline=0,readers=0,maximum=0;bool closed=false,failed=false,usable=false;
  std::atomic<uint64_t> active{0}; // (epoch<<2)|pending1/cancel2/terminal3
  Impl* quarantine_next=nullptr;inline static std::atomic<Impl*> quarantine{nullptr};
#if RNS8_PROCESS_ENABLED
  MPI_Comm comm=MPI_COMM_NULL;MPI_Request request=MPI_REQUEST_NULL;
  bool request_active=false,arena_retained=false;int expected_receive_bytes=-1;
  hip_unique_stream stream;hip_unique_event ready;
  hip_direct_device_buffer a,b,packed,local_output,gather,partials,result,directory,status,receive;
  hip_pinned_host_buffer payload,host_status;
  std::vector<DeviceTile> tiles;std::vector<uint64_t> host_export;
  std::vector<uint8_t> manifests;std::array<uint8_t,128> manifest{};Envelope envelope{};
  std::string binding_hash;std::unique_ptr<CoverageLedger> coverage;
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  ncclComm_t rccl=nullptr;ncclUniqueId rccl_id{};bool rccl_aborted=false;
#endif
  Admission fail(rns8_status code,const char* reason,int runtime=0,uint64_t work=UINT64_MAX) {
    if(!failed){progress.operation=reject(code,reason,rank,work);progress.runtime_error=runtime;}
    failed=true;active.store((progress.epoch<<2)|3,std::memory_order_release);progress.output_current=false;progress.state=State::quarantined;return progress.operation;
  }
  Admission mpi_started(int code,const char* phase) {
    request_active=request!=MPI_REQUEST_NULL;
    if(code!=MPI_SUCCESS)return fail(RNS8_BACKEND_FAILURE,phase,code);
    request_active=true;return {};
  }
  Admission wait_mpi() {
    while(request_active) {
      int complete=0;MPI_Status observed{};const auto code=MPI_Test(&request,&complete,&observed);
      if(code!=MPI_SUCCESS)return fail(RNS8_BACKEND_FAILURE,"mpi_request_progress",code);
      if(complete){request_active=false;
        if(expected_receive_bytes>=0){int received=-1;const auto counted=MPI_Get_count(&observed,MPI_BYTE,&received);
          if(counted!=MPI_SUCCESS || received!=expected_receive_bytes)return fail(RNS8_INVALID_ARGUMENT,"mpi_truncated_fragment",counted);}
        expected_receive_bytes=-1;break;}
      if(process_now()>=deadline)return fail(RNS8_TIMEOUT,"mpi_progress_deadline");
      if((active.load(std::memory_order_acquire)&3)==2)return fail(RNS8_CANCELED,"process_owner_cancel");
      std::this_thread::yield();
    }
    return {};
  }
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
  Admission wait_rccl(){
    for(;;){ncclResult_t async=ncclSuccess;const auto code=ncclCommGetAsyncError(rccl,&async);
      if(code!=ncclSuccess || (async!=ncclSuccess && async!=ncclInProgress))return fail(RNS8_BACKEND_FAILURE,"rccl_async_error",int(code!=ncclSuccess?code:async));
      if(async==ncclSuccess)return {};
      if((active.load(std::memory_order_acquire)&3)==2)return fail(RNS8_CANCELED,"process_owner_cancel");
      if(process_now()>=deadline)return fail(RNS8_TIMEOUT,"rccl_progress_deadline");
      std::this_thread::yield();}
  }
#endif
  Admission wait_device(){
    const auto recorded=hipEventRecord(ready.get(),stream.get());if(recorded!=hipSuccess)return fail(RNS8_BACKEND_FAILURE,"process_event_record",int(recorded));
    for(;;){const auto code=hipEventQuery(ready.get());if(code==hipSuccess)return {};
      if(code!=hipErrorNotReady)return fail(RNS8_BACKEND_FAILURE,"process_device_observe",int(code));
#if defined(RNS8_ENABLE_RCCL) && RNS8_ENABLE_RCCL
      if(rccl){ncclResult_t async=ncclSuccess;const auto rc=ncclCommGetAsyncError(rccl,&async);
        if(rc!=ncclSuccess || (async!=ncclSuccess && async!=ncclInProgress))return fail(RNS8_BACKEND_FAILURE,"rccl_device_progress",int(rc!=ncclSuccess?rc:async));}
#endif
      if((active.load(std::memory_order_acquire)&3)==2)return fail(RNS8_CANCELED,"process_owner_cancel");
      if(process_now()>=deadline)return fail(RNS8_TIMEOUT,"process_device_deadline");
      std::this_thread::yield();}
  }
  Admission hip(hipError_t code,const char* phase){return code==hipSuccess?Admission{}:fail(RNS8_BACKEND_FAILURE,phase,int(code));}
  Admission allocate(hip_direct_device_buffer& value,uint64_t bytes){if(!bytes)return {};const auto code=value.allocate(device,size_t(bytes));return code==RNS8_SUCCESS?Admission{}:fail(code,"process_device_allocation");}
#include "distributed_process_initialize.inc"
#include "distributed_process_transport.inc"
#endif
};
#include "distributed_process_owner.inc"
}
#undef RNS8_PROCESS_ENABLED
