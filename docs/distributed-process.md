# Explicit process and network transports

`ProcessSession` in `src/core/distributed_process.hpp` implements the separately
admitted M3 process path. One MPI process owns each selected GPU. It executes
the same real native-to-residue shard kernels, modular K-fragment reduction and
checked owner CRT kernels as the [single-process owner](distributed-hip.md).
Modulus, M, N, 2-D and K partitions use the retained `Plan` membership; arbitrary
ordered bases and ragged tile prefixes keep their existing uniqueness proofs.
This source extension is experimental. No physical cluster execution, overlap,
communication speedup or production qualification is asserted.

| Transport | Payload | Control and completion |
|---|---|---|
| `mpi_host` | Explicit bounded pinned D2H -> MPI bytes -> H2D staging | Nonblocking MPI requests, HIP events, receiver acknowledgments |
| `rccl` | Matching RCCL INT8 send/receive of unchanged centered bytes | MPI envelopes, nonblocking RCCL progress/errors, HIP events |
| `rocshmem` | Device-initiated symmetric byte put plus ordered signal and quiet | MPI envelopes after sender completion, device signal acquisition, receiver acknowledgment |

No transport adds unrelated residue planes. K fragments enter the existing
modulus-specific integer reduction before CRT; no byte allreduce or floating
reduction performs exact reconstruction. The current process scheduler handles
one fragment at a time per participating rank. It reports that serialization
instead of claiming communication overlap. Ordinary `HipExecution` rejects
network plans explicitly and directs callers to the process owner.

## Build and dependencies

Use the native dependency setup in [platform-linux](platform-linux.md).
The ordinary CPU/direct-HIP build has no MPI, RCCL or rocSHMEM dependency. Enable
only the selected real providers on a native Linux ROCm qualification host:

```sh
cmake -S . -B build/process -G Ninja \
  -DRNS8_ENABLE_HIP=ON -DRNS8_AMDGPU_TARGETS=gfx942 \
  -DRNS8_ENABLE_DISTRIBUTED_MPI=ON \
  -DRNS8_ENABLE_RCCL=ON -DRNS8_ENABLE_ROCSHMEM=ON
cmake --build build/process
```

MPI3 nonblocking communicator duplication/collectives are required. RCCL
configuration compiles and links its nonblocking communicator, asynchronous
error, matching P2P, grouping and abort APIs. Native Windows does not enable
RCCL/rocSHMEM. There is no substitute implementation when a selected provider
is missing. The build manifest records MPI headers/libraries, transport options,
RCCL/rocSHMEM header trees and selected transport-library hashes.

rocSHMEM compiles into a separate device-linked shared helper with `hipcc
-fgpu-rdc`; archiving unresolved device code into a host-linked library would
not suffice. Explicit runtime/arena setup uses that helper's own rocSHMEM
instance. Static-linking a second unrelated runtime into the application is not
an initialization mechanism for this helper. The installed package retains the
helper dependency and its device code.

## Binding, resources and ownership

Each `Rank` has a node identity, rank-local HIP ordinal, physical GPU identity
and target. `ProcessSession::discover` obtains the MPI processor name and real
HIP inventory. Ordinals may repeat on different named nodes; duplicate devices
on one node reject. No ordinal or MPI rank is described as a physical-card count.
The complete node/ordinal/target map joins the immutable plan key.

Pass a pointer to a live `MPI_Comm` in `ProcessBinding`. Creation duplicates it
without changing the borrowed communicator's error handler or finalizing MPI.
Every process supplies immutable input snapshots and the same complete plan.
`resources` prices the actual rank payload, pinned staging, receive scratch,
symmetric arena and bounded protocol metadata before allocation. It includes
all existing plan reserves. RCCL/rocSHMEM require explicit caller-supplied
headroom and acknowledgment that opaque transport allocations are not portably
measured. This reserve is an admission input, not a measured library peak.

rocSHMEM initialization and symmetric allocation are explicit collective setup
outside deadline-polled execution. Call `initialize_device_transport_runtime`,
query resources, and collectively allocate the identical required arena size
with `allocate_device_transport_arena`. The arena contains one bounded receive
slot per rank, one send slot and an aligned signal array. The PE world must match
the MPI rank order and selected GPU. The session retains an exclusive helper arena lease;
release and a second session binding are BUSY while that lease survives. No symmetric allocation,
initialization, global spin barrier or hidden pageable staging occurs inside
`execute`.

Creation can return a nonnull cleanup owner after a failed setup. Keep it and
all borrowed arena/runtime lifetimes until `close` succeeds. Once close begins,
execution and old-result borrowing remain disabled even if a later free fails.
Failed frees retain their original handles for retry.

## Epochs, visibility and failures

Before the first payload, all ranks agree on a digest of the complete plan,
native snapshots encoded in stable little-endian chunks, input identities,
epoch and actual compiled library build manifest. A fixed112-byte envelope
encodes its protocol tag, epoch, work ID, source, owner, exact extent and full
binding digest. Native C++ structs and host-endian integers are never wire
records. Receives verify actual byte counts, exact expected membership/order,
and duplicate/stale/foreign fragments. An explicit coverage ledger tracks IDs;
a count alone cannot certify membership.

The receiver acknowledges each fragment only after its HIP assembly event.
The sender cannot overwrite staging or a symmetric inbox before that
acknowledgment. rocSHMEM uses ordered put/signal/quiet and signal acquisition;
the nonwrapping signal sequence includes the epoch. Every rank, including idle
ranks, supplies a final exact work count and proof status. Only complete
membership plus successful owner CRT/status can publish the owner result.
Signed/unsigned exports are always three little-endian limbs per cell, with
two's complement for signed values. `retain`/`release` cover the last actual
resident consumer; tokens do not own buffers implicitly.

`execute` polls MPI, RCCL and HIP completion against one host monotonic
deadline. Timeout/error preserves outstanding requests, source/destination
storage and the symmetric lease. Root-only `cancel(epoch)` is the one concurrent
host exception: it atomically prevents the requested root result from being
published, without promising kernel/MPI preemption. Peers stop by their progress
deadlines. Other calls sharing a session remain externally serialized.
Failed contexts are not reused under a later epoch. Cleanup aborts a failed
RCCL communicator, then still requires an observed safe HIP drain; abort alone
does not authorize freeing buffers. Outstanding MPI requests remain retained.

Call `close` collectively after every request, device user and result consumer
is terminal, before releasing the arena or finalizing MPI/rocSHMEM. Destruction
never starts a collective or frees potentially live work: an unclosed owner is
retained in process quarantine. Collective setup/teardown also needs an external
job watchdog; an operation deadline cannot preempt an upstream blocking setup
routine or failed external process.

## Authored qualification path

```sh
scripts/linux_process_transport_smoke.sh build/process/rns8-process-smoke \
  mpi-host modulus 0,1
scripts/linux_process_transport_smoke.sh build/process/rns8-process-smoke \
  rccl contraction 0,1
scripts/linux_process_transport_smoke.sh build/process/rns8-process-smoke \
  rocshmem hybrid 0,0 /path/to/two-node-hostfile
```

The runner bounds the complete process job, including setup and teardown. The
executable runs two epochs of padded full-width signed and unsigned products,
checks complete Boost outputs and retained-reader admission, and emits an
explicit non-promoting hardware receipt. It is not registered as an automatic
multi-GPU test. CPU fixtures cover node-local identity, envelope corruption,
stale epochs, complete/duplicate membership, rank acknowledgments, byte budgets
and opaque-headroom rejection. The source pass has CPU/MSVC, Linux
OpenMPI/RCCL-header and rocSHMEM-header syntax checks; real linked Linux ROCm
execution and memory-ordering/fault litmus qualification remain outstanding.

Upstream contracts: [RCCL P2P](https://rocm.docs.amd.com/projects/rccl/en/latest/userguide/source/usage/p2p.html),
[RCCL nonblocking communicator APIs](https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/api-library.html),
and [rocSHMEM ordering](https://rocm.docs.amd.com/projects/rocSHMEM/en/docs-7.14.0/api/memory_ordering.html).
