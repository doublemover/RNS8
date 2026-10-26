# Experimental modulus-sharded HIP execution

`src/core/distributed_hip.hpp` implements one exact product across explicitly
selected runtime devices. It extends the [shared plan and CPU mock](distributed-mock.md)
with real allocations, streams, events, source replication, per-modulus GEMM,
peer or pinned-host transfers, owner-device CRT and publication. It is a private
experimental C++ owner, not a stable public C ABI or a qualified production route.
The source is written for native Windows HIP and Linux ROCm; their runtime and
hardware qualification remain separate. No RCCL dependency is introduced.
The ordinary operation inventory now reports that source/compiled availability,
but returns an explicit topology/binding rejection: a single-context query lacks
the selected-rank transport and complete budgets needed to admit this owner.
It never reports an unbound distributed route as successful zero scratch.

## Preparation and arithmetic

`HipExecution::discover` reports explicit ordinal, UUID/BDF/visible-device
identity and directional peer capability. Package and partition identities
remain unknown where the runtime cannot establish them. Visible device count
is never reported as physical card count. Peer mode requires positive runtime
access in both directions; pinned mode explicitly allocates portable pinned
memory. Failed capability/allocation checks reject without changing transport
or silently reducing the selected device inventory.

`create` binds a shared immutable `Plan` and generation-bound `Inputs` snapshot.
It independently computes native row/column magnitude and L1 bounds and requires
each actual tile basis to prove uniqueness for that conservative range. Checked
GPU reconstruction then enforces the claimed output bound. Large cancellation
can use a wider basis without a full CPU GEMM or trusting a small claimed bound
to conceal a CRT alias. Caller certificate references
are supplemented by the existing exact arithmetic checker for every actual rank
target/build and the maintained signed INT8, INT32, 65,536-term slab with centered
carry-in. Its returned hashes are available through `arithmetic_certificates()`;
layout and target-execution verdicts are not inferred from those host proofs.

Preparation synchronously uploads each active rank's native snapshot once and
finishes its default-stream copies before nonblocking work can read them.
The packing kernel applies the existing exact unsigned reduction and explicit
signed `2^64 mod m` correction. Per-rank slots reuse packed A/B storage while
the maintained direct-HIP tiled GEMM reduces each safe slab to centered bytes.
Every tile/plane owns an explicit output byte range; arbitrary local subsets
are never relabeled as a short global prefix.

## Progress, publication and lifetime

Calls sharing an owner require external host serialization. Epoch arguments
are scoped to that owner; they are not independently retaining public tokens.
Only one epoch is in flight. `submit` accepts a fresh nonwrapping epoch and a
positive operation deadline. After partial enqueue failure the accepted epoch
remains observable through `query`, including original runtime code and work/rank.
Validation before the first enqueue preserves the preceding published epoch.

Host progress observes source-ready events before gathering. Peer copies use
the explicit source/destination devices. Pinned transport orders source D2H
completion, destination H2D and destination completion before reusing its single
bounded staging slot. There are no device-wide spin barriers, cross-device
event-time subtraction, pageable copies during submit, or host callbacks.
Packing/compute may advance while earlier shard copies remain pending.

An owner-stream event covers all expected copies, per-tile exact CRT, range
status and status readback. Only that observed event can publish the complete
result. Adaptive tile prefixes, including proved zero tiles, reconstruct with
their own complete bases. No partial export API exists.

`export_limbs` stages and then publishes padded host output as three little-endian
64-bit limbs per cell; signed results use 192-bit two's complement. It rejects
aliasing with immutable native inputs. `retain`/`release` protect resident
borrows through the last actual consumer. `materialize` can create a sufficient
uniform default basis directly from the complete GPU limb result; it never
extends an insufficient isolated rank. It rejects active borrows and charges
both old and replacement materializations before allocation.

Cancellation stops new work and discards the epoch after outstanding streams
drain. An observation timeout retains all reservations; the operation deadline
is a separate host-monotonic deadline. Neither promises kernel preemption.
Failed drain or cleanup retains the entire owner, events and both DMA endpoints
in an allocation-free quarantine. `reap_quarantined` retries progress and cleanup;
unsafe resources are never declared reusable. Explicit `close` returns BUSY for
pending work or result readers. Callers must release borrows and obtain successful
close before destroying the handle; destructor quarantine preserves safety but
cannot discharge a consumer whose retention was abandoned.

## Resource contract

The shared plan prices native replication, one to three reusable packing slots,
all retained local result fragments, the complete owner gather, CRT/output,
tile/status descriptors, pinned staging and explicit overhead reserves. Packing
capacity follows the largest live tile window, not the total number of output
tiles. Checked rank budgets are also compared to runtime free memory.

`HostStorage::exact_reference` includes the simulator's full integer oracle and
host fragments. `HostStorage::limbs_only` prices the real runtime's retained
three-limb host export staging without those nonexistent mirrors. The CPU mock
rejects the latter model. Input/proof/descriptor storage and explicit host
overhead remain charged in both. These are requested reservations; allocator,
driver, cache and physical HBM/RSS peaks still require observation.

## Future qualification commands

Build the ordinary CMake HIP configuration for every selected target, for example
`-DRNS8_AMDGPU_TARGETS=gfx942` on a native Linux qualification host. A local gfx1100
binary does not establish that a gfx942 code object exists. Enable the normal
test and benchmark targets. The registered `real HIP modulus shards gather and
publish one exact generation` test uses one selected device by default. To test
real transport, set `RNS8_DISTRIBUTED_DEVICES=0,1` and
`RNS8_DISTRIBUTED_TRANSPORT=pinned` or `peer`. Unsupported peer mode is explicit.
The fixture covers signed/unsigned bounded/exact-wide results, P1/P20, padded
inputs/outputs, zero and nonuniform tile prefixes, suffix materialization,
retained readers and cancellation. Its Boost oracle is independent of device CRT.

`rns8-distributed-calibrate CASE.json unique-run-id` records complete operations.
The [single-device case](../benchmarks/distributed/single-smoke.json) and
[two-device staged case](../benchmarks/distributed/two-staged-smoke.json) share
one exact separable signed workload. The 8192-square controls have explicit
8 GiB complete host and 16 GiB per-rank device budgets; edit ordinal inventories
for the actual host. Complete-output audits use a closed form independently
checked by Boost dots, not an infeasible cubic host oracle. Three sustained
bursts require at least 16 calls and target 250 ms, with an explicit 4096-call cap.
Independent checks and JSON stay outside timing; required status/publication stay
inside. One host monotonic clock measures cross-device elapsed time.

`tools/collect_distributed_calibration.py` uses the existing artifact/request
capture wrapper and counterbalanced independent-process planner. `--plan-only`
does not run GPUs. Collection validates immutable cases, exact expected inventory,
runtime identities, budget/traffic accounting, completion and matched process
sets. Ratios remain descriptive; no automatic default or release promotion occurs.
No collector or timing campaign has been launched for this source pass.

Native Linux gfx942 2/4/8-device tests, native Windows supported transport tests,
real failed-drain fault injection, topology/NUMA/counter traces and the existing
1.55x 8192-square Linux release gate remain external qualification. M/N/K splits,
RCCL, multi-process/multi-node and device-initiated transport remain explicitly
separate experiments, as scoped by issue #50's M3 milestone.

The runtime sequencing follows AMD's versioned
[HIP multi-device reference](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/how-to/hip_runtime_api/multi_device.html).
API documentation does not establish performance or execution on another target.
