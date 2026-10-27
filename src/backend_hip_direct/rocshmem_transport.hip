#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#include <cstdint>

namespace {bool runtime_ready=false;void* owned_arena=nullptr;uint64_t arena_bytes=0,arena_leases=0;}
extern "C" int rns8_rocshmem_world(int* rank,int* world) {
  if(!runtime_ready)return 1;*rank=rocshmem::rocshmem_my_pe();*world=rocshmem::rocshmem_n_pes();return 0;
}
extern "C" int rns8_rocshmem_runtime_init(){if(runtime_ready)return 1;rocshmem::rocshmem_init();runtime_ready=true;return 0;}
extern "C" void* rns8_rocshmem_arena_allocate(uint64_t bytes){
  if(!runtime_ready || owned_arena || !bytes)return nullptr;owned_arena=rocshmem::rocshmem_malloc(size_t(bytes));
  if(owned_arena)arena_bytes=bytes;return owned_arena;
}
extern "C" int rns8_rocshmem_arena_free(void* pointer){
  if(!runtime_ready || pointer!=owned_arena)return 1;if(arena_leases)return 2;
  rocshmem::rocshmem_free(pointer);owned_arena=nullptr;arena_bytes=0;return 0;
}
extern "C" int rns8_rocshmem_arena_retain(void* pointer,uint64_t bytes){
  if(pointer!=owned_arena || !owned_arena || bytes>arena_bytes)return 1;if(arena_leases)return 2;++arena_leases;return 0;
}
extern "C" int rns8_rocshmem_arena_release_lease(void* pointer){if(pointer!=owned_arena || !arena_leases)return 1;--arena_leases;return 0;}
extern "C" int rns8_rocshmem_runtime_finalize(){if(owned_arena || !runtime_ready)return 1;rocshmem::rocshmem_finalize();runtime_ready=false;return 0;}

// One calling thread performs the one-sided transfer. The source is a symmetric
// arena slice; signal ordering and quiet establish remote completion before the
// host emits the epoch-bound control envelope. No oversubscribed spin barrier.
__global__ void rns8_rocshmem_put_signal_kernel(uint8_t* arena,uint64_t maximum,
    uint64_t bytes,uint64_t sequence,int sender,int owner,int world) {
  const uint64_t signals=(maximum*(uint64_t(world)+1)+255)&~uint64_t(255);
  rocshmem::rocshmem_putmem_signal(arena+maximum*sender,arena+maximum*world,size_t(bytes),
      reinterpret_cast<uint64_t*>(arena+signals)+sender,sequence,rocshmem::ROCSHMEM_SIGNAL_SET,owner);
  rocshmem::rocshmem_quiet();
}
__global__ void rns8_rocshmem_acquire_kernel(uint8_t* arena,uint64_t maximum,
    uint64_t sequence,int sender,int world) {
  const uint64_t signals=(maximum*(uint64_t(world)+1)+255)&~uint64_t(255);
  // The sender's quiet/event precedes its MPI envelope, so the matching signal
  // is already due. If a device/transport contradicts that, the host deadline
  // quarantines this kernel and the borrowed arena instead of freeing them.
  (void)rocshmem::rocshmem_signal_wait_until(reinterpret_cast<uint64_t*>(arena+signals)+sender,
      rocshmem::ROCSHMEM_CMP_EQ,sequence);
}
extern "C" int rns8_rocshmem_send(void* arena,uint64_t maximum,uint64_t bytes,
    uint64_t sequence,int sender,int owner,void* stream) {
  const int world=rocshmem::rocshmem_n_pes();
  hipLaunchKernelGGL(rns8_rocshmem_put_signal_kernel,dim3(1),dim3(1),0,static_cast<hipStream_t>(stream),
      static_cast<uint8_t*>(arena),maximum,bytes,sequence,sender,owner,world);
  return int(hipGetLastError());
}
extern "C" int rns8_rocshmem_acquire(void* arena,uint64_t maximum,uint64_t sequence,int sender,void* stream) {
  const int world=rocshmem::rocshmem_n_pes();
  hipLaunchKernelGGL(rns8_rocshmem_acquire_kernel,dim3(1),dim3(1),0,static_cast<hipStream_t>(stream),
      static_cast<uint8_t*>(arena),maximum,sequence,sender,world);
  return int(hipGetLastError());
}
