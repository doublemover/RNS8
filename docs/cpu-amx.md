# Optional CPU AMX-INT8 provider

Tracking: [#54](https://github.com/doublemover/RNS8/issues/54).
Source implementation, not hardware or performance qualification. The scalar/
OpenMP CPU reference and Boost oracle remain independent; AMX is never selected
by AUTO merely because its code was compiled.

## Admission and public contract

`RNS8_ENABLE_CPU_AMX` is opt-in. The current implementation builds on Linux
x86-64 with GCC >=11 or Clang >=14 and isolates tile instructions in a
`target("amx-tile,amx-int8")` function. Other platforms/builds retain the same
query API and reject explicit AMX execution. This is not an Arm implementation
or an assertion that all compiler versions/hardware have been qualified.

Select `RNS8_BACKEND_CPU_AMX` when creating a context and plan. Host packing,
bounded signed/unsigned and exact-wide signed/unsigned GEMM, finite byte rings
and fields, ordinary CRT exports, and synchronous prepared executions/groups
share the existing representation, range, generation, and lifetime contracts.
The backend is named `cpu-amx-int8`, never `cpu-reference`. No wraparound,
GPU-stream/graph, sparse, specialized Gram, exact-lift, or resident-DAG support
is inferred from dense GEMM support. Ordinary host CRT is intentionally reused;
its arithmetic is not a matrix-engine self-comparison.

`rns8_query_cpu_amx(0, ...)` observes without requesting permission and never
executes AMX instructions. Fields independently report compilation, CPUID tile
and INT8, OSXSAVE, palette/TMUL dimensions, supported xstate, dynamic permission,
and XCR0. `request_permission=1`, explicit AMX context creation, and each
participating execution worker may request XTILEDATA using `arch_prctl`.
Failure (including an insufficient alternate signal stack) returns unsupported
before tile execution. There are no persistent configuration/permission changes,
signal-handler probes, speculative SIGILL recovery, or silent scalar substitution.
A scalar fallback is an explicit independently created CPU-reference context.

Linux grants dynamic xstate permission to a process/thread group; nevertheless
all executing workers recheck their own usable xstate before a team barrier.
Only if **every** worker is admitted does any worker enter the tile function.
A context created on one thread does not authorize an unchecked worker.

## Arithmetic and storage

Each job computes one 16x16 output tile for one modulus. A is packed to 16x64
centered bytes in worker-local scratch. B is packed into 16 rows of 64 bytes:
`Btile[(k/4)*64 + 4*n + k%4]`; `TDPBSSD` consumes four adjacent K bytes per
INT32 cell. Tails are zero padded, never loaded out of bounds. All 2..256 moduli,
including even/composite rings, use the same signed-centered representation.

An INT32 tile accumulates at most 65,536 products before local modular reduction.
The next slab starts from the previous **centered** residue. The exact absolute
bound is `65536*128*128 + 128 = 1073741952 < INT32_MAX`; the plan's arithmetic
certificate explicitly includes the signed prior and centered reduction. The
kernel never writes a full-matrix INT32 temporary or runs a separate global
reduction pass. Tile-local stores/reloads and compiler-generated traffic still
exist; no bandwidth/ISA throughput claim follows from this source count.

Immutable adaptive plan tiles are subdivided into 16x16 jobs; active-prefix and
proven-zero output rules remain authoritative. A complete private byte-output
stage is published only after the whole admitted worker team finishes. Normal
public GEMM failure-currentness rules still apply: a failed call need not leave
the old logical C exportable even though admission failure does not copy bytes.

Workspace construction prepares all worker tile buffers, a B layout cache, and
the complete output stage outside parallel/hot regions. Workspace-owned payload
is queried with `rns8_get_cpu_amx_workspace_info`, including actual capacities,
prepared/last team sizes, B identity/generation, pack/reuse counters, layout
version, and safe K. B reuse requires matching stable matrix instance, current
content generation, finite modulus, and logical layout/stride. This does not
trust caller source-version labels or pointer reuse. Different plans/workspaces
do not accidentally share transformed B storage.

The checked lower envelope is
`1024*prefix*ceil(N/16)*ceil(K/64) + prefix*M*N + workers*sizeof(Worker) + sizeof(owner)`.
Finite operations use one physical plane even though their public prefix is zero.
Vector capacities, allocator overhead, OpenMP runtime, CRT `cpp_int` temporaries,
and ordinary matrix allocations are distinct; the query reports owned capacities,
not total process RSS. Exact input/range bookkeeping may allocate outside the
microkernel. There is no claim of an allocation-free complete public call.

The prepared worker ceiling is frozen with the plan/workspace identity. Each
call also respects the current configured CPU thread ceiling and work threshold.
Inside an existing OpenMP parallel region execution is serial: no nested worker
pool. Affinity/NUMA policy is inherited; the library does not pin threads or
move pages. Use the process/OpenMP controls appropriate to a measured cohort.

## Deferred validation

Authored regressions cover pure capability admission, independent packed-layout
interpretation, every byte pair, prime/composite/even moduli, unaligned padded
native inputs, 16-cell tails, 65,536/131,072 K boundaries, signed/unsigned full
64-bit extrema, all twenty residue planes and three-limb CRT, adaptive tiles,
thread ceilings, repeated-B mutation, and prepared-group refresh.
They have **not** been executed. AMX source/object compilation is not proof of
OS admission, emitted instructions executing correctly, or performance/energy.
The optional second architecture remains evidence-gated by the issue; none is
claimed from a portable build. Target execution, independent exact comparison,
end-to-end timing, affinity/NUMA sensitivity, and hardware adoption remain open.

## Instruction and OS references

- [Linux dynamic xstate/AMX permissions](https://www.kernel.org/doc/html/latest/arch/x86/xstate.html)
- [Intel AMX INT8 intrinsics example and tile configuration](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)
- [Intel AMX capability discovery](https://github.com/intel/Intel-AMX-Detection/blob/main/amx_detection.c)

These define the platform contract; they are not RNS8 qualification receipts.

## Matched calibration implementation (not executed)

`rns8-cpu-amx-calibrate` uses the public context/plan/matrix/workspace APIs for
both the independent CPU reference and AMX. The 560 checked-in input cases in
`benchmarks/cpu-amx/` cover tiny, 16-cell tails, long-K skinny, and square
products; bounded and exact-wide signed/unsigned and finite mod255/mod256;
prefixes 5/9/20 as legal for those contracts; one/four requested threads; cold,
resident, A-only/B-only/both repacking. Cold calls own construction, pack, GEMM,
CRT export and destruction; steady calls retain exactly the declared inputs.
Packing unchanged values still advances host generation, so the B-cache tests
measure the real generation contract rather than assuming source-version reuse.

The harness emits setup, native packing/setup, public GEMM (including internal
AMX B packing when needed), CRT/export, and cleanup phases. Sample totals sum
operation and cleanup intervals, **excluding the intervening independent full
output/residue oracle**; they are not falsely labeled one continuous wall
interval. Sustained bursts include complete calls and harness clock/JSON
bookkeeping, with no oracle inside. Full Boost comparison, every residue plane,
and padded output checks run on separate audited invocations. Three final audit
calls and all warmups/repeats/burst calls are included in the exact cache and
packing-count receipt.

`tools/collect_cpu_amx.py` records a deterministic counterbalanced independent
process plan before launching anything, binds the executable/build manifest,
requires both scalar and AMX modes in each matched cohort, validates OS/ISA
admission, actual owned payload and all generation/cache counts, and preserves
AMX-slower outcomes. It never marks a capture promotion-eligible. Environment
records CPU model, visible physical cores/hardware threads, OS/compiler,
requested/max/observed team sizes, inherited CPU affinity and allowed NUMA nodes,
queried memory policy or its explicit error, and selected OpenMP placement
variables. Allowed nodes are **not** asserted to be actual page placement.
Optional readable package-0 RAPL counters are captured as shared-system modular
energy deltas; process attribution and unknown multiple wraps are not invented.

Collector rejection fixtures cover missing OS permission, forged ISA/currentness,
incomplete pairs, reused process receipts, incorrect packing/cache counts,
underreported storage, altered safe K, false NUMA observation, and energy scope.
All corpus and regression execution remains deferred by request.
