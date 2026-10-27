# Experimental distributed HIP execution

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

Cancellation records terminal events on each allocated work stream and the
owner gather stream before polling their completion. This covers queued work
without requiring further shard scheduling, and gives the Windows runtime an
explicit completion point for batched commands. Reservations stay pending until
every marker completes; recording/query failures retain the quarantined owner.

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

Preparation uploads use the same conservative ownership rule. Native snapshots
and the lowered tile directory remain owned if an initial copy or default-stream
completion fails. Quarantine observes those initialization streams before freeing
either host or device storage; no local descriptor vector becomes a dangling DMA
source when preparation unwinds.

## Optional independent integrity before commit

Set `Request::native_integrity` to compare the complete reconstructed result
against independent original-native products modulo 257 and 263. The owner
retains native inputs even when its rank owns no computational plane. Neither
checking channel contributes to the computational CRT range. The explicit
policy is part of plan/capture identity and the native replication is budgeted.

The scalar checker runs on the owner stream after CRT and before status readback
and the final completion event. Query reports requested/completed/mismatch
separately. Range errors and `RNS8_VERIFICATION_FAILED` both prevent publication;
cancellation or backend failure cannot fabricate a completed successful check.
An error divisible by 257*263 is invisible. This is a declared integrity model,
not arbitrary-error correction or a replacement for deterministic exactness.
CPU corruption/invisibility fixtures and actual HIP signed/unsigned check
fixtures pass applicable local CPU/gfx1100 checks in the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json). Actual multi-device behavior and cost remain unqualified.

The collector accepts an explicit boolean `native_integrity` in its case and
requires matching completion/zero-mask reports. Checked/unchecked modes remain
different comparison cohorts; check work stays inside timed complete calls.

## Resource accounting

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
The [single-device case](../benchmarks/cases/distributed.json) and
[two-device staged case](../benchmarks/cases/distributed.json) share
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
1.55x 8192-square Linux release gate remain external qualification. RCCL,
multi-process/multi-node and device-initiated transport remain separately gated
M3 transport experiments; issue #50 does not require them before a correct M1.

The runtime sequencing follows AMD's versioned
[HIP multi-device reference](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/how-to/hip_runtime_api/multi_device.html).
API documentation does not establish performance or execution on another target.

## M/N, 2-D output and K-fragment source paths

The real owner consumes the same explicit axes as the CPU plan. Output-region
axes reuse its existing replicated native snapshots, bounded packing slots,
peer/pinned transfer ordering and final complete CRT owner. A tile region owns
its whole active basis. The2-D option is block-cyclic output ownership with
replicated inputs; it is not a panel-broadcast implementation.

K-fragment packing reads only the descriptor's nonempty `[k_begin,k_end)` interval.
Each shard runs the maintained65536-term centered INT8/INT32 slab primitive.
All fragment bytes have separate retained storage and unique owner gather spans.
After every expected transfer has been enqueued on the ordered owner stream, a
separate kernel sums at most64 centered bytes per cell in INT32, reduces modulo
the actual channel, and writes the complete residue matrix before CRT. It never
uses ordinary byte allreduce or a small final bound to justify partial sums.

The80-byte device tile record freezes partial offsets and counts alongside the
range proof. Host/device resources, pending flags and retryable cleanup include
the entire partial gather. Existing cancellation, missing-coverage, failed-drain
and retained-reader rules still apply. Authored CPU/Boost and HIP fixtures cover
all axes, odd tiles, adaptive zero-prefix regions and cancellation across huge
partial integers. These CPU/mock and one-device HIP execution fixtures pass the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json). One-device execution does not test physical
peer transport, rank failures or multi-GPU scaling.

The calibrator accepts the explicit version2 case fields `axis`, `k_fragments`,
`rank_grid_rows` and `rank_grid_cols`. The compact catalogue contains single-device
and two-device staged controls. Capture validation checks actual axis/count,
all fragment work and partial-gather bytes. These cases have not been executed.
