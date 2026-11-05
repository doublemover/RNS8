# Persistent native WMMA queue

The private HIP research entry point `hip_direct_gemm_native_wmma_queue`
accepts up to 4096 independent native 64-bit matrix products. Each descriptor
selects signed or unsigned inputs, shape, pitches, the default RNS prefix,
and one, two, or four channels per matrix-engine group. It produces centered
residue planes; the caller remains responsible for the semantic range proof.
This entry point does not implement strict modulo-2^64 arithmetic.

The host builder proves the qualified 32-bit address domain and rejects every
output/output or output/input span overlap across the batch. Shared readers
are legal. It enumerates tile/channel groups round-robin across active tasks,
with linear enumeration cost after sorting the memory spans. Queue storage is
budgeted before allocation, and failed admission preserves the caller's queue.

Each 32-thread CTA atomically claims a descriptor and owns its complete K
reduction. The existing RDNA3 native grouped WMMA tile implementation is shared
with the ordinary grid launch. Each modulus has an independent accumulator;
65536-element K slabs prevent signed INT32 overflow. Output tails and partial
channel groups retain their original masks. A bounded terminal-claim count
prevents the 32-bit queue cursor from wrapping.

After admission, queues whose descriptors all share signedness and modulus group
select a kernel specialized for that pair. Shapes, pitches and prefixes may still
differ. Mixed queues retain the general dispatch kernel. This avoids reserving
registers for larger modulus groups when a uniform group-one queue executes;
the selection does not change arithmetic or the round-robin work order.

Native conversion uses a per-channel reciprocal and the remainder of `2^32`,
initialized before the K loop. Each 64-bit input is split into two unsigned
32-bit words. Reducing each word and then their radix-weighted sum needs only
32-bit reciprocal reductions: the recombination is at most 65280 for a byte
modulus. Signed inputs subtract `2^64 mod m` when negative, including `INT64_MIN`.
This avoids a large constant-division switch inside each encoded product.
The CPU reducer fixture checks every modulus from 2 through 256 against Boost;
GPU fixtures include word carries, full-width boundaries and safe K-slab splits.

The synchronous backend probes the compiled device capability before touching
outputs. Descriptor upload, capability probing, and persistent execution have
separate timing labels. `outputs_may_be_written` becomes true immediately before
the compute launch; callers must treat outputs as invalid after a subsequent
failure. The one-shot entry point allocates device descriptors per invocation
and frees them on return.

The private `NativeWmmaPreparedQueue` additionally retains admitted descriptors
and device storage across synchronous calls. Preparation performs admission,
allocation, capability probing and upload before replacing any prior prepared
state. Its preparation budget charges both old and replacement descriptor
allocations at the transactional peak. Execution resets the cursor and launches
against the retained descriptors.
Concurrent preparation/execution returns `RNS8_BUSY`; a device execution failure
poisons the object until successful re-preparation. Referenced allocations must
remain alive at the original addresses and layouts. Contents can change between
completed calls because this object caches no operand values or transformed
panels. Callers own range proofs and exclusive buffer access. This private owner
does not supply public matrix generations, reader/writer reservations or graph
publication semantics. Its differential fixture passed at `51ae8d8`, including
changed input values, transactional preparation failure and all six uniform
signedness/group combinations. All 14 native calibration modes passed their
smoke collection at that revision; sustained prepared-path measurements are
being collected separately.

The same owner now offers `execute_ordinary` as a matched scheduling control:
one ordinary grid per admitted task followed by one final synchronization,
without a queue cursor. It retains the identical preparation and allocation
footprint, including device descriptors unused by this control, plus one host
metadata descriptor per task. The scratch budget covers device allocations;
host task/work vectors are separately retained metadata. Preparation replaces
the host descriptors only after device admission/upload succeeds. Both launch
modes share exclusion and failure poisoning. This control isolates repeated
execution from per-call allocation/capability checks; it is not a claim that an
ordinary-only implementation needs queue metadata. Its added GPU coverage and
measurements await the next coordinated build.

CPU mapping, admission, collision, and budget fixtures and a Boost-based GPU
differential fixture cover heterogeneous shapes, full-width signed/unsigned
values, prefix/group tails, padding, K=65537, and varying worker counts. These
fixtures passed the local all-feature campaign at `c7f7e05`; the uniform-kernel
extension passed all six signedness/group combinations at `6d8ab25` (469 passes,
three target-inapplicable skips across the full configuration). The reciprocal
conversion passed the fresh `436d235` campaign: 472 passes and three expected
skips across 475 registered tests, with all 13 calibration modes also passing.
Public prepared
groups still execute sequentially. Public integration, asynchronous ownership,
matched prepared-path performance qualification remain open under issue #38.
The [worker sweep and resource inspection](native-queue-performance-2026-10-23.md)
record the completed scheduling experiment without claiming a win over the
best ordinary launch.

## Procedural schedules and measured completion contract

The new private prepared queue has a versioned, immutable schedule header and
three explicit identities. The original expanded round-robin list remains the
default and still uses its original singleton-claim kernel. `task_major` retains
one exclusive work range per task and binary-searches the range containing a
claimed ordinal. `compact_round_robin` additionally stores distinct work-count
breakpoints. A breakpoint records the cumulative work before its first round,
that round's active-task count and its ending round. The decoder locates the
round by binary search and selects its active task by a bounded original-order
scan. This exactly preserves the reference round-robin assignment sequence,
not a claim about asynchronous completion order. Task-major deliberately does
**not** claim that fairness equivalence.

For T tasks and W assignments, explicit metadata is O(T+W); both procedural
modes retain O(T) records. Task-major lookup is O(log T); compressed fair lookup
is O(log T+T), capped at 4096 tasks. Construction sorts spans and breakpoints,
O(T log T), with no enumeration of W assignments. The CPU admission decoder
checks canonical ranges, phases, work counts, ABI, reserved fields and budget
before publication. Kernel directory data is private immutable prepared state,
not an untrusted device-produced schedule. Irregular/adaptive coverage retains
its separate explicit representation. The whole K reduction and existing
plane/tile tail masks are unchanged.

The explicit `ranked_round_robin` candidate (public schedule 3, private
`rank-select`) uses header ABI 2. Its first, all-active phase selects the task
directly from the round rank. Later sparse phases binary-search prefix counts
to select a bitmap word, then select its bit in five bounded subdivisions. It
preserves the same reference fair sequence; mode 2 retains the original scan.
For T tasks and P distinct count phases, the extra directory is
`4*(P-1)*(2*ceil(T/32)+1)` bytes. Equal-count queues allocate no rank directory.
Construction and retained/peak payload accounting charge these actual tables;
the worst-case representation is O(P*T), rather than O(T).

Authored exact-sequence and malformed-directory fixtures cover 1/64/512/4096
tasks, late survivors, many count breakpoints and chunk crossings. Private
targeted cases add 512/4096 equal-count tasks and one late survivor among 4096 tasks;
the public shared-B case exercises schedule 3 at 64 tasks. Compact catalog record
repetition emits ignored expanded fixtures without storing repeated task lists.
Collectors check the selected decoder identity and independently price physical
directory bytes. Host/device syntax checks are separate from the deferred exact
execution, ISA/resource, preparation-cost and sustained-latency qualification.

Homogeneous encoded-B specialization is an explicit dispatch-policy experiment.
`rns8_prepare_native_group_dispatch` accepts policy 0 (current native-only
specialization), 1 (native or encoded uniform specialization), or 2 (forced
generic task dispatch). All tasks must agree on group, signedness and operand
representation; dimensions and prefixes may differ. Group-zero hints retain
per-task dispatch, including mixed representations. Both explicit and procedural
queues and their owned graphs use the selected body. Ordinary-grid controls
reject nondefault policies. Refresh preserves the policy; failed preparation
preserves the previous queue. The additive dispatch-info query reports the
actual group and representation without changing existing descriptor layouts.

Private/public cases accept optional integer `dispatch_policy`. Collectors bind
it to the actual launch hint, independently derived from task controls. Private
nondefault-policy collections compare queue schedules; default collections also
retain ordinary grids. Separate diagnostic completion submissions remain generic
and cannot qualify specialization latency. Authored differential sources cover
all groups and signedness, encoded/native and mixed representations, partial
prefixes, padded tails, K=65537, direct/graph submission and refresh. Device
compilation alone supplies no ISA/register, correctness or speedup evidence.

The public calibrator's `host_diagnostics: true` requires a separate
`-DRNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS=ON` build. The default compiles these
counter hooks away. An observer is borrowed on the calling thread for cold
setup and one complete warm observation **after all timed intervals**, then
removed even on exception. It reports attempt counts at prepared submission,
proof-cache and managed runtime-handle sites: stream-device/capture queries,
device selection, event creation/record/query/wait/destruction, proof builds
and reuse, and native graph launch. These are scoped source call counts, not
process-wide profiler events or generated GPU work. Existing successful HIP
allocation/transfer API-payload counters are reported separately; host heap
allocations, physical traffic and other threads are outside this diagnostic.

Diagnostic configuration participates in the compiled manifest. Its timings
are identified as ineligible for release-build latency claims even while the
observer is absent. Normal sustained callbacks return typed observations;
JSON construction and exact CPU checks happen after the measured intervals.
Optional `integrity: true` enables the public native compute sentinel and
requires consistent generation-bound per-child reports in addition to the
independent full Boost result comparison. Native sentinel checks do not replace
the exact oracle. The 96 opt-in controls cover 1/8/32/64 tail tasks, shared or
distinct B, persistent/ordinary/owned-graph modes, stable or mutated B, and
integrity off/on. They are source-ready fixtures, not executed qualification.

A one-task 8192-square, prefix-20/group-one schedule has 5,242,880 assignments.
The explicit assignment payload alone is 80 MiB. The fair procedural payload is
`sizeof(NativeWmmaTask) + 16 + 16 + 4` bytes; its 32-byte header is a kernel
argument rather than another device allocation. This is a capacity calculation,
not a speedup or advice to run a large regular GEMM through a persistent queue.
Actual vector capacities, builder payload high-water marks, retained host
metadata, descriptor upload payload and transactional old-plus-new device
payload peaks are exposed separately. Allocator bookkeeping/object headers,
RSS and driver residency are not represented as measured heap/device usage.

Claims support chunks 1–32 and switch to singleton claims near the observed
worker tail. Admission reserves `workers * maximum_chunk` counter headroom.
A stale tail observation may retain a larger claim; out-of-range ordinals are
masked, and no worker waits for another workgroup. The new claim loop has a
counter-read atomic as well as the claim atomic. Lookup/atomic overhead and
long-K fairness can therefore outweigh metadata savings. No automatic selector
or public cache promotion is implied. Failed replacement retains the old
schedule; all execute variants retain BUSY, drain and poison behavior. Warm
production calls allocate/upload no immutable descriptors.

`rns8-queue-calibrate` and `tools/collect_queue_calibration.py` provide matched
explicit/task-major/fair/ordinary controls with identical native tile arithmetic,
representation, reuse and synchronized resident-output boundaries. The cases in
`benchmarks/native-queue/` cover many tiny jobs, strongly skewed K/counts, short
batches, larger jobs, tails, encoded B and explicit B mutation/re-encoding.
CPU construction, prepared setup, first use, hook-disabled warm batches,
rebuild costs, upload/retained/peak payloads and useful tile/channel throughput
are separate fields. Every observation audits every residue and padding/extra
plane against Boost outside its production timer; warm execution allocation and
host-transfer counters must remain zero. Independent process order is rotated
and reversed, captures retain executable/case hashes and raw observations, and
cohorts never choose the luckiest process as their representative.

Per-task completion is a **separate diagnostic submission**. A preceding kernel
establishes one device wall-clock epoch; each completed tile/channel assignment
publishes its stores before incrementing the task's completion counter. The last
assignment records its task's elapsed ticks. There are no inter-workgroup spin
barriers. Readback verifies exact required counts and reports p50/p95/max task
completion with raw ticks and the queried wall-clock rate. This diagnostic uses
the general dispatch kernel and adds atomics, fences and a launch; its fairness
measurements must not be mislabeled production latency. A separate GPU-event
companion likewise supplies diagnostic, not production, timing.
[HIP's timer documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.1/how-to/hip_cpp_language_extensions.html#timer-functions)
specifies the fixed-frequency wall clock/rate query and warns against using
`clock64()` for RDNA3. Ordinary grids have no synthetic per-task queue telemetry,
and this private synchronous owner has no supported graph-replay control.

The new CPU property/mutation/budget/4096-task/counter-bound fixtures and GPU
full-reference/telemetry/replacement fixtures are **authored, not executed**.
Host-only syntax checking does not compile the HIP kernels. Compilation on the
supported target, differential execution, independent latency/fairness collection
and ISA/resource qualification remain deferred. Public generation/read-write
reservation integration remains distinct work: the private harness explicitly
owns every borrowed buffer and synchronously rebuilds changed encoded B.

## Public generation-owned queue and graph opt-in

`rns8_prepare_native_group` now attaches the native queue to an existing public
`rns8_prepared_group`. Every child remains an ordinary prepared exact-wide GEMM:
its context, immutable plan, matrices and workspace retain their existing owners.
The new binding adds a modulus group (1/2/4), optional generation-bound encoded B,
and explicit schedule/claim/worker policy. Raw and encoded tasks may coexist;
shared B is read-only and independently reserved once per child. This is an
experimental direct-HIP WMMA route, not AUTO dispatch or a CPU matrix-engine claim.

Preparation stages all task contracts, checked raw spans, input generations,
exact range proofs, cache references, unique workspaces, descriptor allocation,
events and a nonblocking stream before replacing the queue. The device metadata
budget includes old plus replacement descriptors. Failed admission does not
change output generations. A retained encoded B cannot be destroyed; changing B
makes both its materialization and the group ineligible. Refresh rebuilds raw
bindings and rejects a stale encoded cache rather than blessing it. Supply a new
cache through preparation to replace it transactionally. Individual child input
refresh cannot bypass the group's independently retained generation binding.

`rns8_submit_native_group_hip` borrows one nondefault same-device stream, reserves
**all** output writers/workspaces and input readers before the first enqueue, and
returns one existing completion token per child. Each child's event is recorded
after the full queue; this proves complete task coverage without pretending that
host observation order measures device task finish order. Existing query, wait,
timeout, cancellation and downstream prepared dependencies apply unchanged.
Cancellation suppresses only that child's publication. Reverse-order observation
releases only the observed child's readers. Every child must be observed before
cursor reuse, refresh or destruction. Launch/event-record failures invalidate
all children, and a failed drain retains reservations and poisons the context.

`rns8_capture_native_group_hip_graph` captures the same cursor reset and queue
body on an owned stream. Replay uses `rns8_submit_native_group_hip_graph` with the
same host proof/generation/reservation checks; completion events remain outside
capture. Replacement invalidates the old graph. Ordinary prepared-child graphs
remain ordinary residue GEMM: they are not silently changed to native graphs.
`rns8_execute_prepared_group` uses the configured native route synchronously and
drains every issued token even when an earlier child fails.

The `ordinary_grid_control` uses the **same** raw-native/encoded-B representations
and prepared contracts, but launches one native grid per task. The metadata query
reports actual retained device/host descriptor bytes, preparation peak and graph
readiness. Warm submission performs no descriptor upload or HIP allocation;
host proof evaluation remains inside the public call. Resource cleanup is checked:
failed descriptor frees preserve the owner, and encoded-B destruction no longer
releases source/context references when its device free fails.

The current source pass also retains a proof-array shell and a compact table of
unique input readers/multiplicities at native preparation or refresh. Warm
submission does not rebuild that topology map. It checks the actual borrowed
stream's device and capture status once per batch, then performs each child's
mutable generation, proof, reservation, epoch and graph-envelope checks. Stream
admission authority exists only for that submission. Child completion events and
independent tokens remain unchanged. The metadata query charges the retained
proof and reader-table objects; multiprecision/range payloads remain dynamic.
These host-control changes have authored mock coverage and await the deferred
combined runtime validation; no new latency or allocation claim is qualified.

Written regressions cover signed/unsigned tails, shared B, both representations,
all three schedules, ordinary/persistent controls, owned graph replay, cancellation,
reverse completion, no warm HIP allocation, stale cache/rebind, budget failure and
padded three-limb exact outputs. **These regressions have not been executed.**
HIP compilation, fault injection, public complete-route measurements, task fairness,
and target-specific qualification remain unverified; source ownership and launch
counts are not measured speedups.

### Complete public controls (authored, not run)

`rns8-public-queue-calibrate` and `tools/collect_public_queue_calibration.py`
now measure the actual public owner rather than treating private raw-pointer queue
latency as public-call latency. Cases in `temp/benchmark-cases/current/public-native-queue/` pair
persistent, identical-representation ordinary native grids, native graph replay
and the ordinary resident-RNS control. The latter does not consume or charge a
fictional encoded cache. Heterogeneous signed/unsigned tasks include shared B,
skewed rows, prefix/group tails and K=65537; explicit/task-major/compact schedules
are separate identities. Reuse 1/4 executes real complete groups, never divides a
single operation's time by a nominal batch count.

Each timed group includes host proof/currentness/reservation, actual execution and
completion, and every padded three-limb public export. Mutated-B cases also include
packing, new generation-bound encoded payloads, descriptor replacement and graph
recapture inside each operation. Initial setup is separately retained; replacement
cache peaks count old plus new unique B payloads, not duplicates per shared reader.
`rns8_get_encoded_b_info` exposes layout ABI, source instance/generation/currentness,
prefix/device and actual retained allocation bytes without making a cache current.
The native metadata query reports descriptor bytes, not undocumented HIP graph,
event or driver allocations. The collector independently checks schedule bytes,
logical useful work, cache payloads, output bytes and real mutation/rebuild counts.

Independent child processes are counterbalanced and artifact-bound. Reports retain
raw observations and per-process medians/ranges, including setup plus an actually
observed reuse batch. Three sustained intervals count actual completed groups and
mutation generations. Full Boost/padding comparisons occur after each observation
and interval, not inside production timing; intermediate overwritten outputs in a
sustained interval are not claimed independently audited. Device per-task fairness
is **not** inferred from post-batch host events: the separate private completion
telemetry is still a diagnostic companion, not public latency evidence. No report
promotes a route automatically, supplies measured HBM traffic, or claims a speedup.
