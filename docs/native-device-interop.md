# Native HIP buffer leases and prepared transfers

The experimental public C boundary provides `rns8_query_native_device`,
`rns8_create_native_device_lease`, `rns8_import_native_device`,
`rns8_import_native_device_verified`, `rns8_export_native_device`, and `rns8_destroy_native_device_lease`.
Ordinary import/export remain explicitly synchronous direct HIP; the prepared
exact-wide adapter below adds stream, owned-graph and dependency execution. CPU builds
support descriptor/resource queries and reject lease creation. No host pointer
API accepts a device pointer implicitly.

A versioned descriptor names native i64/u64 dtype, read OR write access,
allocation base/capacity, byte offset, rows/columns/leading dimension, device,
owner identity, nonzero binding generation, explicit BORROW or COPY_OWNED policy, producer
completion and acquire/release callbacks. Signedness does not select arithmetic
semantics: import/export independently admit bounded or exact-wide matching
matrix semantics. Finite and wrap semantics are not admitted. Explicit exact-wide limb output is
supported as described below.
A matrix created with `RNS8_MATRIX_RETAIN_NATIVE_INPUT` receives an owned compact
device copy on import. Conversion reads that copy; native and residue currentness
publish together after completion. Destroying the lease or recycling its source
cannot change the owned snapshot. Reimport refreshes the same allocation. There
is no borrowed external pointer stored as the matrix's native representation.

The pure query validates the complete accessible view, pointer arithmetic,
alignment, stride and qualified index domain, and returns compact output plus
four-byte status scratch requirements. It does not query HIP or call owners.
Creation first acquires an owner reference, then checks HIP pointer attributes,
same device, unmanaged device memory and actual allocation base/capacity using
`hipMemGetAddressRange`. Interior views use byte_offset; passing an interior
pointer as an allocation base rejects. Unsupported pointer-query/runtime
capabilities are explicit failures. A failed creation after acquisition releases
that reference; an unsuccessful acquire transfers no reference.

The caller must complete initial producer work before creation, truthfully set
producer_complete=1, and exclude external writes/recycling outside the explicit
mutation window described below. A write lease excludes other external readers/writers until each
synchronous export returns success. Binding generations and owner IDs describe
the immutable view; they cannot detect a caller violating this lifetime
contract. Changed address/layout/owner requires destroy/recreate; a stable view can use the
explicit generation refresh below. Recycled allocations need
new owner IDs. No cache hit is inferred from pointer equality or external
source generation, and external generations never enter host-pack's version
cache. Overlapping immutable READ leases may coexist. A WRITE lease rejects any
overlapping live lease in the same context before acquisition. Separate contexts
and out-of-band framework writes remain the caller's synchronization obligation.

Import reuses existing strided device pack kernels into ordinary owned RNS
storage. It certifies the complete native type envelope (2^63 signed magnitude
or UINT64_MAX unsigned), checks unique CRT coverage before writes, and publishes
that conservative scalar proof. It never treats a caller's small bound as a
verified fact and never downloads input payload for a host scan. Every import
reads the leased view, including a newly leased recycled address. Retained-native
imports add a separately timed `interop_native_retain_d2d` logical-row copy and
retain rows*cols*8 bytes in the matrix; no host native mirror or duplicate upload
buffer is allocated. The normal matrix allocation report includes those bytes.
Input/output leases overlapping either the matrix's residue or native allocation
reject before writes, preserving both representations. A failed issued import
invalidates both, and uncertain completion retains the existing matrix/lease pins.

The opt-in verified import scans actual device magnitudes with the same block
reduction used by narrow native admission, extended to unsigned values. It
charges eight retained device scratch bytes and reads back only the eight-byte
maximum. The scan handles INT64_MIN without signed negation, excludes padding,
and completes before the matrix write transaction. Its measured magnitude then
passes the same exact uniqueness check and replaces the conservative type
bound. This supports tighter exact-GEMM/native-output admission without trusting
caller assertions. A failed scan or insufficient CRT range preserves the old
matrix value; failed drain still pins the matrix and owner. The input remains
immutable across scan and conversion under the lease contract.

Export requires current device residues and a current unique scalar proof
fitting the native type. Ordinary bounded results without that proof reject;
use a path that publishes such a proof, rather than inventing one at export.
The proof must cover every residue plane consumed by native export, currently
the matrix's complete basis. A narrowed exact-wide computation in a larger
allocation is rejected until those suffix planes are explicitly materialized;
capacity alone is not evidence. Ordinary plan-specific host export can still
read its actually written prefix.
Existing CRT kernels reconstruct into lease-owned compact scratch, then read
only a four-byte status. Only successful range/status verification permits the
D2D logical-row copy to the external destination. Padding is preserved. The
scratch budget includes all retained export payload/status allocation. Static
rejections and reconstruction range failure preserve destination bytes; an
issued commit failure makes the destination invalid. Consumers require a
successful return, not merely an observed device event.

Each operation pins its matrix and lease until real completion. An error is
followed by an explicit drain. Failed drain retains the matrix reservation,
external owner and scratch and poisons the context. Destruction retries the
drain/free and preserves the lease on failure; successful destruction invokes
release exactly once. Callbacks run on the caller's host thread, must not throw
or reenter this lease, and are never HIP runtime callbacks. Context destruction
is BUSY while leases exist. This uses existing matrix reservation/context poison
rules, not a second async completion type.

This is zero host **payload** round trip, not zero-copy conversion: import writes
RNS storage, export stages native data and performs a D2D copy, and compact
status travels to the host. Conversion/reconstruction timings remain separate
from status and commit traffic. Authored CPU descriptor and HIP full-width
Boost/residue/output/padding/ownership fixtures await the consolidated campaign.
Prepared async/graph binding is implemented below. DLPack remains optional;
new execution/failed-drain qualification and matched performance receipts remain
unverified. No measured speedup or
broader target support is claimed.

The installed `examples/device-native` consumer supplies custom HIP producer and
consumer kernels around public exact GEMM. It independently checks all output
and padding and separately labels validation-only host payload readback. Its
standalone CMake recipe uses explicit hipcc object compilation and exported
static/shared targets; build/runtime qualification is pending the final batch.

## Copy-owned read leases

`RNS8_NATIVE_DEVICE_COPY_OWNED` is an explicit synchronous read-only alternative
to borrowing. Set `snapshot_budget_bytes` to at least the queried compact native
size. Creation acquires the external owner, validates its actual device allocation,
and copies only logical rows into a new compact D2D allocation. It releases the
external owner exactly once after successful copy completion. The caller may then
mutate or free the original source; repeated ordinary or verified imports read
the immutable snapshot. Creating a fresh lease is the refresh operation. Export
requires a borrowed WRITE lease; copied destinations are rejected.

Query reports `snapshot_bytes` separately from import/export scratch. BORROW
requires a zero snapshot budget and retains the original lifetime contract.
Snapshot traffic is exactly rows*cols*8 logical D2D bytes, not zero-copy. It adds
one retained compact allocation; operation scratch budgets remain separate.
An unpublished snapshot whose copy/drain/free fails remains owned by the poisoned
context, including the external owner reference when it may still be in use.
Context destruction retries cleanup and returns failure until safe release.
Public handles remain BUSY lifetime references to the context. Host callbacks
must not throw or reenter the lease.

CPU descriptor tests and GPU signed/unsigned padded-view differential cases are
authored for budget rejection, early external release, source overwrite/free,
repeated verified import and exactly-once callback release. Execution remains
pending the final consolidated campaign; no copy-mode speedup is claimed.

## Explicit multiword device output

The same lease/export API accepts `RNS8_NATIVE_DEVICE_SIGNED_LIMBS` and
`RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS` for exact-wide matrices. Set `limb_count`
to 1..32, access to WRITE, and policy to BORROW. Scalar I64/U64 descriptors
require a zero limb count. Limb import and copy-owned limb leases reject.
Each logical integer occupies that many contiguous native-byte-order uint64
words, least-significant word first. Signed output is two's complement at
exactly 64*limb_count bits; unsigned output is a magnitude. Rows, columns and
leading dimension count integers. Byte offsets still count bytes and require
eight-byte alignment. This is an explicit word tensor, not a native int128 type.

The resource query checks the complete padded byte span and charges compact
rows*cols*limb_count*8 staging plus four status bytes. Export requires the full
materialized basis and a unique scalar magnitude proof fitting the requested
width. The symmetric signed admission conservatively excludes the signed
minimum when its sign is not separately proven. Too-small widths and budgets
reject before external writes. Existing signed/unsigned CRT kernels write the
staging allocation; four-byte status observation precedes logical-row D2D
commit. Padding stays untouched. Ownership, failed-drain retention and external
consumer readiness follow the scalar lease contract.

Authored descriptor and GPU/Boost fixtures cover widths 1, 2, 3 and 32,
full-width input products, signed negative results, range rejection, padding,
owner release and exact payload/status transfer counters. Execution is deferred
to the consolidated campaign.


## Prepared import/export and producer readiness

`rns8_prepare_native_transfer_hip` returns the existing prepared execution owner,
not a second completion type. It admits exact-wide i64/u64 import and scalar or
1..32-word export. The shared operation provider exposes the same domains,
complete native spans, proof prerequisites, algorithms and known memory budget.
Export freezes an actually proven/materialized prefix rather than requiring
unused allocation capacity. Stream exports now share the ordinary generic and
fixed-prefix CRT kernels for every admitted prefix/word width.

Import either uses the complete native type envelope or verifies an explicit
magnitude ceiling on device. The latter does **not** trust a caller assertion:
a device maximum scan and status comparison precede status-gated commits of
staged RNS and optional compact retained-native data. Every replay resets status.
Export similarly stages reconstruction and uses a status-gated logical-row copy.
Four pinned status bytes are copied on the same stream before the completion
event. No allocation, host payload copy or whole-device synchronization occurs
inside prepared submission/replay. Preparation charges staged payload, status/
scan, pinned status and the retained native matrix allocation; copied lease
snapshots and opaque runtime objects are separately owned costs.

`rns8_begin_native_device_write` opens a mutation window only when the lease has
no device user and no overlapping live lease. This prevents silently changing
another immutable view's generation. `rns8_end_native_device_write` asserts that
producer work has already completed; its HIP variant records an owned readiness
event on a verified same-device borrowed nondefault/noncapturing producer stream
without waiting. Both require a strictly newer generation. Refresh prepared
inputs after ending the window. Captured bodies retain stable addresses and
proof ceilings; current-generation readiness waits are enqueued **outside** the
captured body before each replay. Changed view/owner/device/width requires a new
lease/owner, not a pointer-equality cache hit.

One lease has one in-flight transfer. Distinct immutable read leases can feed
independent consumers and complete in reverse order. Active operations, pending
producer readiness and mutation windows remain in the context's alias registry;
failed drains preserve owner references. Groups reject duplicate lease
reservations and conflicting external/internal byte spans before any enqueue.
A failed external producer event cannot release its allocation before a proven
drain. Destruction waits for explicitly recorded readiness and preserves failed
event/free identities for retry.

Native import → GEMM → native export can be submitted as token-qualified stream
or owned-graph pipelines. A producer's device work may run before host proof
observation, but its failure suppresses all downstream publication. An external
consumer must wait for successful operation status and current-result reporting;
physical writes alone are not a valid-result signal. `rns8_get_prepared_transfer_info`
reports the actual external owner/generation, retained payload/status bytes,
verified-bound mode, prefix and width. Cancellation never authorizes consumption.

The installed device-native example's `--prepared` mode demonstrates this path
using five borrowed streams, custom device producer/consumer kernels, current
readiness events, captured import/GEMM/export bodies and one final host observation.
Its setup and validation traffic are labeled separately. New all-prefix/width,
small-bound rejection, graph-generation, independent reader, alias, padding,
lease lifetime and failure-propagation fixtures are authored, **not executed**.
No overlap, latency or throughput benefit is claimed from this source work.


## Matched application cost experiments

`rns8-native-interop-calibrate` and `tools/collect_native_interop.py` compare
complete custom HIP producer → public RNS8 product → custom HIP consumer
transactions. Stream and owned-graph candidates use the same three-limb,
padded signed/unsigned output as the explicit device→host→host-pack and
host-export→device control. Inputs include small and full-width boundaries.
The host composition has both default eager and explicit scalar-first proof
controls. Device candidates and the resident floor also select scalar-first.
Compare against the strongest same-work host control; an eager-only comparison
can charge host metadata propagation that conservative device import does not
produce. Normal host pack still constructs its documented input facts, so a
whole-transaction benefit includes conversion/proof work as well as payload copies.
Every timed call advances participating external owner generations, including
unchanged values; import independently verifies its envelope. The consumer digests every
output limb. An independent Boost product and padding/digest audit run outside
every measured sample in **all** modes, not just the control.

A separate resident control repeats the same external application work but
retains already-established, unchanged internal inputs. It intentionally
excludes input binding/conversion and is a labeled cost floor, **not** a matched
speedup claim. It is never included in candidate/control ratios. Initial
allocation and route-specific binding are timed as cold setup, separately from
first-call, p50/p95, and three sustained bursts. Device candidates establish
input facts through verified device imports; the host control packs host input
without creating unused native leases or transfer owners. Preparing a native
export requires one initial product to establish its output proof. The host
control prepares only its product and pays for its first product in first-call
timing. Only participating leases receive acquire/release callbacks (three for
device candidates, one for the resident floor, zero for the host control).
The two true device candidates issue imports, product, and export across
nondefault streams and observe only the final dependency completion.

Version-three captures name the proof policy, payload transfers, compact status bytes (12 per
device candidate, 4 for the resident export), validation-only readback,
conservative all-route preflight payload estimates, and separate plan workspace.
Four same-epoch public allocation snapshots observe RNS8-owned requested device
and pinned-host payload before setup, after setup, after calls, and after cleanup.
Live payload must return to baseline; peaks and successful allocation/free
counters remain available after retirement. These observations exclude the
application's four explicit HIP allocations, whose requested bytes are reported
separately. The actual capacity of the four ordinary native host vectors is
also reported. Boost objects/dynamic limbs, other library heap allocations,
opaque runtime/driver objects, physical VRAM/RSS and profiler launch counts
remain unmeasured. The preflight estimate is not an observed retained-byte count
or a physical memory limit. Wall subphases
include overlapping enqueues/observations; they are **not** per-kernel GPU
durations. Hardware launch counts and physical memory peaks still require a
named-target profiler. The first version-two collection stopped after two passing
captures when the stream candidate omitted explicit GEMM submission preparation.
Graph capture supplies this preparation internally. The stream setup now calls
`rns8_prepare_hip_submission` for its four operations. The complete fresh retry
passes **80/80 captures / 37,771 transactions**, including exact outputs, callbacks
and requested-payload retirement, at compiled source **0f26d96** with the runtime
three-warmup profiles from **29de830**. Both focused builds and eight collector
checks pass. The seven native-device GPU fixtures also pass in the unchanged
production source's full **4e21e9c** qualification.

That retry exposed the eager-proof control confound above: its large apparent
ratios do not qualify an interoperability gain. Its input-binding-excluded
resident control also retained unnecessary eager product proof propagation.
The [reviewed v2 evidence](../metadata/native-interop-eager-controls-20261025.json)
retains all observations, the initial failure, setup/first-call costs, payload
peaks and this limit. The distinct v3 experiment adds the scalar-first host
control, repairs the resident floor, and awaits measurement. No threshold was
widened and the v2 captures will not be pooled into it.

The collector counterbalances independent processes, refuses reused process
identities, checks immutable executable/build/environment provenance, requires
complete phase/sustained observations, observed-payload retirement, and exact validation, and reports only
same-cohort device/host ratios, with the strongest control selected by sustained
cost. The eager comparison remains visible as a diagnostic. Captures remain nonpromoting.
Each sustained process mean divides total elapsed burst time by total calls;
reported cohort means are medians of independent process means.
