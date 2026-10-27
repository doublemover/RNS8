#pragma once
#include "distributed_hip.hpp"
namespace rns8::detail::distributed {
// Explicit collective setup/teardown outside deadline-polled execution. Use the
// same helper runtime that owns device-linked rocSHMEM symbols; independently
// static-linking a second runtime would create unrelated contexts/heaps.
Admission initialize_device_transport_runtime();
Admission allocate_device_transport_arena(uint64_t bytes,void*& output);
Admission release_device_transport_arena(void*);
Admission finalize_device_transport_runtime();
struct ProcessBinding {
  // Pointer to a live MPI_Comm. MPI is initialized externally; create duplicates
  // this communicator without changing its error handler or finalizing MPI.
  void* mpi_communicator=nullptr;
  uint64_t setup_timeout_ns=0;
  // RCCL opaque allocations are not portably measurable. The caller supplies
  // explicit per-rank headroom and acknowledges this unknown overhead.
  uint64_t transport_headroom_bytes=0;
  bool acknowledge_unknown_transport_overhead=false;
  // rocSHMEM only: externally initialized PE world must match MPI rank order.
  // Borrow an identically sized symmetric arena on every PE. No collective heap
  // allocation or runtime initialization occurs inside bounded execution.
  void* symmetric_arena=nullptr;
  uint64_t symmetric_arena_bytes=0;
};
struct ProcessResources {
  uint64_t symmetric_arena_bytes=0,pinned_bytes=0,transport_headroom_bytes=0;
  uint64_t device_payload_bytes=0,host_metadata_bytes=0;
  bool transport_overhead_known=true;
};
// One process per selected rank; local arithmetic and owner reconstruction use
// the same real HIP shard/export kernels as HipExecution. MPI transports control
// and explicit staged bytes; RCCL and rocSHMEM move unchanged residue bytes.
// The first process implementation is serialized and explicit; no overlap or
// graph support is inferred. Inputs are immutable snapshots on every rank.
class ProcessSession {
 public:
  static bool compiled(Transport) noexcept;
  static Admission resources(const Plan&,uint32_t rank,const ProcessBinding&,ProcessResources&);
  static Admission discover(void* mpi_communicator,uint32_t local_ordinal,Rank&);
  static Admission create(std::shared_ptr<const Plan>,std::shared_ptr<const Inputs>,
      const ProcessBinding&,std::unique_ptr<ProcessSession>&);
  ~ProcessSession();
  ProcessSession(const ProcessSession&)=delete;ProcessSession& operator=(const ProcessSession&)=delete;
  // A synchronous, deadline-polled operation. Fresh epoch accepted before any
  // payload issue. Errors retain unsafe HIP/MPI/RCCL/arena lifetimes. Every rank
  // participates, including zero-work ranks; only owner has a borrowable result.
  Admission execute(uint64_t epoch,uint64_t timeout_ns);
  const HipProgress& progress()const;
  // The output owner's sole concurrent-host exception: atomically request
  // discard before publication. Does not preempt HIP/MPI; peers stop by their
  // bounded progress deadline. Nonowners return unsupported.
  Admission cancel(uint64_t epoch)noexcept;
  Admission retain(uint64_t epoch);Admission release(uint64_t epoch);
  Admission resident_limbs(uint64_t epoch,int& device,const void*& pointer)const;
  Admission export_limbs(uint64_t epoch,uint64_t* destination,uint64_t ld);
  // Call collectively after all operations/consumers are terminal, before MPI
  // or rocSHMEM finalization or releasing the borrowed arena. Destruction never
  // starts a collective: an unclosed owner is retained in process quarantine.
  Admission close();
 private:
  struct Impl;explicit ProcessSession(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
}
