# Prepared execution and completion ownership

`rns8_prepare_gemm` creates a reusable binding for one ordinary dense GEMM.
It validates the plan, workspace, operands and output ownership, records input
instance/generation pairs, and retains the context, plan, workspace and matrices.
Destroying any retained handle returns `RNS8_BUSY` without changing the handle.
Destroy prepared executions first, then retry resource destruction. Shared A/B
is retained once. Multiple prepared executions may retain the same resource.
Users of C++ RAII handles must likewise destroy the prepared execution before
the wrappers that own its resources.

The ordinary `rns8_execute_prepared` API is synchronous. It invokes the existing
GEMM implementation and returns after its ordinary completion/status boundary.
It may allocate and synchronize; it is not a HIP-stream or graph-capture API.
Calls involving shared handles require external host-thread serialization.
No concurrency, stream-overlap or graph-replay qualification follows from this
interface.

The lifecycle is prepared → executing → completed or failed → executing.
Preflight failure leaves the prior completion epoch unchanged and does not
modify the output token. Once execution begins, a new epoch is assigned and
the operation status is recorded even if the operation fails. Output currentness
and value publication use the existing GEMM write guard and generation owner;
there is no second matrix-validity state. A preflight range failure can preserve
an old value while the new operation's completion still reports failure.

Each token contains a unique prepared-instance identity and a nonwrapping epoch.
`rns8_get_prepared_completion` accepts only the latest epoch of that instance.
It reports the exact input generations, output generation and operation status.
`output_is_current` is true only for a successful operation whose output still
has that generation and a current storage domain. A later repack or another
writer cannot make an old completion certify its value. Completion queries
require the prepared handle to remain alive; tokens do not retain it.

Inputs may change between executions. Such a change makes execution reject the
stale binding until `rns8_refresh_prepared_inputs` validates and binds the new
generations. Refresh does not execute work or alter previous completion history.
Generation exhaustion rejects preparation/refresh/execution rather than reusing
an ambiguous identity.

The explicit HIP submission subset below adds stream/event completion and a
fixed-envelope graph replay path. Independent groups, resident DAG submission,
prepared transfers, dependencies and explicit caller-graph bindings are
implemented below. Their supported envelopes remain operation-specific.

## Explicit HIP submission

Call `rns8_prepare_hip_submission` on a prepared dense direct-HIP bounded i64/u64, exact-wide, strict wrap64, or
finite-ring/field execution to allocate its completion event. The subset has fixed
shape/basis/prefix or one explicit finite modulus, and no adaptive tile schedule.
The dense stream-provider section below adds individually admitted accelerators;
other semantics/backends return unsupported. Finite operations use their existing
enqueue-only finite kernel wrapper, including dynamic composite moduli and ordered
safe K slabs. They require current operands with the plan's modulus and do not
invent an exact-integer range proof. Strict wrap64 uses eight unsigned byte-limb
planes and the existing 36-product kernels, with the same accumulator-width
selection as synchronous execution. It never uses odd-modulus CRT. Completion
publishes only byte-limb currentness, and graph replay binds byte-limb allocation
addresses. These paths share the cancellation, reader reservation, generation,
and grouped-submission contracts.
`rns8_submit_prepared_hip` takes a caller-owned non-default HIP stream and checks
its device and capture state. Keep that stream alive until completion is observed.
Ordinary submit rejects an already capturing stream. Use the explicit caller-graph
binding below to embed a body; no synchronous fallback is selected.

Submission checks input generations and computes the complete exact range proof
before changing the output or epoch. Host proof work can allocate; device event
storage is prepared ahead of submission. It then reserves input readers, the
output writer and workspace, invalidates C through the existing lifecycle owner,
enqueues the existing GEMM kernel path without publication, and records an event.
Returning success means accepted work. No global synchronization occurs on the
successful enqueue path. Inputs shared by independent submissions may have
multiple readers; mutations, conflicting output writers and workspace reuse
are rejected until their reservations are released.

`rns8_query_prepared_hip` returns BUSY while the event is pending;
`rns8_wait_prepared_hip` waits for that event. Both observe completion and only
then publish C's logical generation and exact proof on success. The output stays
unavailable even if the GPU finishes before the host observes completion. Query/
wait success means the completion record was retrieved: inspect its
`operation_status` and `output_is_current`. A device failure leaves C invalid.
An enqueue/event-record failure drains the stream before releasing reservations
and returns the failure with the issued token. A failed preflight does not issue
an epoch or alter the caller's token. Old epochs never certify later writes.

The host must serialize calls sharing handles; device work may overlap across
independent streams. Ordinary consumers observe producer completion before using
the new value; explicit token dependencies below retain pending producers and
propagate their eventual status. Destruction remains BUSY while handles are retained or executions are
pending. Event errors require an explicit blocking wait to drain before reservations are released;
fatal device-context errors remain backend failures and are not repaired here.

CPU reservation fixtures and real HIP two-stream fixtures are authored for
shared readers completed in reverse order, write/workspace collisions, pending
destruction, rebinding, stale epochs, range rejection, tails and K-slab crossings.
Registered applicable execution is covered by the [dated validation](local-all-features-20261023.md);
overlap and enqueue-time gains still require measurements.
The lifetime/capture distinction follows the active
[HIP 7.1 graph contract](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_runtime_api/hipgraph.html).

## Owned fixed-envelope HIP graph

`rns8_capture_prepared_hip_graph` captures the supported GEMM launches on an
internally owned stream and instantiates an owned graph. Capture leaves C's
currentness and generation unchanged; it records the enqueue-only backend path.
No device allocation, synchronous transfer or host callback occurs in the
captured body. Event and graph construction happen outside replay. A failed
replacement capture preserves the previous executable graph.

`rns8_submit_prepared_hip_graph` shares ordinary submission's range/generation
preflight, reservations, epoch, event and query/wait publication. It verifies
the captured A/B/C device addresses before enqueue. The immutable envelope is
one fixed-shape dense exact-wide, strict wrap64, or finite operation, its fixed basis/prefix or
finite modulus, and all
ordered K-slab launches. Shape, topology, layout, basis or address changes need
a new prepared binding/capture. Input values can change at stable addresses
between completed submissions with explicit input refresh. No adaptive growth
outside this envelope is silently accommodated.

Graph replay is an explicit API choice; ordinary stream submission does not
silently change paths. Calling submission while the caller's stream is already
capturing remains unsupported. The host preflight may allocate proof metadata,
and runtime-internal HIP allocations are not asserted absent. Authored tests
check capture without publication, repeated replay, stable-address rebinding,
stale epochs and zero new allocations through RNS8's device-allocation helper
during replay. Registered applicable tests are covered by the
[dated validation](local-all-features-20261023.md); capture cost, overlap,
leaks and replay break-even have not been measured.

## Heterogeneous prepared groups

`rns8_create_prepared_group` retains an immutable list of up to 65536 prepared
executions. Tasks may have different shapes, semantics and prefixes. Shared
readers are legal; duplicate writers and cross-task output/input hazards are
rejected in expected linear time. The group retains execution handles, which
in turn retain all resources. Destroy groups before executions.

Group refresh validates all tasks before changing any generation binding.
Group execution likewise preflights every binding before starting the first
task. It executes ordinary calls sequentially, stops on the first failure, and
returns the count of issued completion records. Earlier successful results
remain available. Later token slots are untouched. This is an ordinary-launch
baseline for #38. It does not upload a tile schedule or use a persistent
matrix-engine queue.

`rns8_submit_prepared_group_hip` submits an independent same-device batch to
caller-owned streams. Prepare each member's HIP submission resources first.
`rns8_submit_prepared_group_hip_graph` instead replays each member's previously
captured graph. Streams may repeat; workspaces must be distinct. All members
must satisfy the fixed-prefix exact-wide submission contract. Different
contexts may be used only when they address the same physical HIP device.

Before the first enqueue, the batch checks every stream, binding, address,
range proof, workspace, epoch and aggregate shared-input reader count. Proofs
are constructed once and moved into the individual completion owners. A
preflight failure leaves all output generations and token slots untouched and
reports zero submissions. After enqueue starts, a backend failure stops the
batch: earlier submissions remain owned by their execution handles, the failed
submission also receives a token, and later slots remain untouched. The returned
count identifies issued tokens, rather than successful or completed operations.

Query or wait each issued token on its execution handle. Only observation of
that member's completion publishes its output and releases its reservations.
Group destruction and refresh return busy while any member remains pending.
Host calls sharing handles must be serialized. This API batches host admission
and enqueue; overlap, throughput and graph replay break-even remain unmeasured.
Authored coverage includes shared A/B inputs, mixed GEMM/epilogues, reverse-order
completion observation, shared-workspace rejection, reader-count overflow and
a later member's range failure before any output mutation.

## Prepared exact epilogues

`rns8_prepare_exact_axpby` binds the exact-wide integer expression
`alpha*A*B + beta*D` to the same prepared execution lifecycle. Alpha and beta are
immutable original integer coefficients; D's instance/generation participates
in binding and refresh. Completion records expose the expression flag,
coefficients and D identity explicitly. The unreleased completion record grew;
callers must rebuild and supply its current struct size.

The CPU/HIP synchronous routes call the ordinary range-admitted epilogue API.
Explicit HIP submission and owned capture enqueue the fused epilogue on the
requested stream without global synchronization or early publication. Replay
checks D's captured device address as well as A/B/C. Each distinct A/B/D handle
is retained and reserved exactly once; D may share A/B when descriptors match,
but never C. Groups include D in cross-task read/write hazard detection.

Original-coefficient whole-expression bounds are recomputed before submission.
D mutation requires refresh even when beta is zero, preserving the bound
workload identity. A captured expression cannot silently become ordinary GEMM
or acquire different coefficients. Change the expression by preparing a new
execution. Source-generated completion metadata does not qualify timing or
cache equivalence by itself.

Authored CPU/HIP differential cases cover extreme signed/unsigned coefficients,
shared A/B/D lifetime counts, distinct addend reservations, stale D binding,
stream and capture execution, and completion identities. Registered applicable
execution is covered by the [dated validation](local-all-features-20261023.md). The ordinary async range-exhaustion fixture now uses a legal
full-width plan and creates a wider-than-native A with an exact outer product
before replay, rather than requesting an inadmissibly small initial basis.


## Failed stream cleanup

`rns8_get_prepared_async_failure` reports the original failure stage, operation status,
backend error when available, first stream-cleanup error, device, epoch and
whether resources remain reserved. A zero backend error on an enqueue failure
means the backend wrapper returned only an RNS8 status. Diagnostics require the
exact issued token and do not synchronize or release resources.

A failed stream drain does not prove that GPU work stopped. Input readers,
output writers and workspaces therefore remain reserved; destruction stays
busy and new arithmetic on that context is rejected. Query reports the failure.
Wait retries stream draining and releases reservations only after a successful
drain, without publishing failed outputs or consulting a stale completion event.
The context remains poisoned after cleanup: destroy its handles and recreate it
before further arithmetic. No implicit device reset is performed. Submission,
observation and owned graph/event cleanup select the context device and restore
the caller's previous device.

Applicable host-side failure fixtures pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).
Real hardware fault recovery and cross-device behavior remain unqualified.


## Cooperative result discard

`rns8_cancel_prepared_hip` accepts the latest issued token and marks pending work for
discard. It does not preempt kernels, synchronize, release reservations or
revoke a terminal published result. Continue query/wait until terminal
observation. A successful device operation then reports `RNS8_CANCELED`, leaves
its attempted outputs unavailable and releases reservations; a backend failure
takes precedence over cancellation. Repeated cancellation is idempotent. A new
submission clears the request and issues a fresh epoch. External host
serialization is required for calls sharing handles.

Submission rejects null, legacy-default and per-thread-default stream handles.
The caller owns a nondefault stream and must keep it alive through terminal
observation, including any failed-drain recovery. The cancellation and special
stream fixtures pass the applicable lanes of the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).


## Bounded completion observation

`rns8_wait_prepared_hip_for` polls using a steady-clock deadline in microseconds.
Zero performs one poll. `RNS8_TIMEOUT` leaves the completion output unchanged,
does not cancel, and preserves all reservations and the issued epoch. The
caller can poll, cancel or wait again under the same external host-serialization
rule. This is a bounded observation attempt, not a kernel execution deadline;
OS scheduling and runtime query overhead can exceed the requested interval.
Queries and timed waits never drain a failed stream. They report the backend
failure, quarantine the context, and require ordinary blocking wait for cleanup.

| State | Value publication | Resources | Next operation |
|---|---|---|---|
| Prepared/idle | Previous value remains current | Handles retained | Submit or destroy |
| Admission rejected | Previous value and token unchanged | No execution reservation | Correct inputs and retry |
| Submitted/pending | Attempted output unavailable | Readers, writers and workspace reserved | Query, timed wait, cancel or wait |
| Cancel requested | Attempted output remains unavailable | Reservations unchanged | Observe terminal completion |
| Successful observation | Value and proof published together | Execution reservations released | Query idempotently or submit fresh epoch |
| Canceled completion | No value published | Execution reservations released | Submit fresh epoch or destroy |
| Failed/quarantined | No attempted value published | Retained until successful drain | Blocking wait cleanup, then recreate poisoned context |

Only one epoch can be pending on each owner. Tokens identify observations and
do not own handles. Data-ready events alone do not certify public currentness;
publication happens during successful host observation after proof admission.


`rns8_get_prepared_hip_support` reports the implementation's shape/backend/semantics
support without allocating events, performing range proofs or submitting work.
Its typed record exposes the single-epoch limit, borrowed nondefault-stream
requirement, possible host proof allocation, owned-graph, cancellation and timed
observation support. Unsupported records contain zero feature flags. This query
does not admit current values or a caller stream and is not hardware qualification.
Preparation reuses the same support predicate, then performs current admission.

Synchronous dense, finite, wrap and prepacked-B preflight now reports `RNS8_BUSY`
consistently for pending input writers or output readers/writers, before backend
admission or output invalidation. Shared input readers remain legal. Sparse
preflight follows the same rule. This is early admission/status parity; the
existing write guard also rejects conflicting output reservations at mutation.
The all-semantics lifecycle fixtures pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json);
fixture coverage does not broaden the advertised asynchronous semantic subset.


## Token-qualified producer binding

`rns8_bind_prepared_dependency(consumer, producer, token)` retains one exact
producer epoch whose output occupies one or more consumer input slots.
`rns8_bind_prepared_dependencies` atomically replaces up to three distinct
producer-output bindings for A, B and epilogue D. All tokens, output owners and
complete dependency branches are checked before any reference change. Duplicate
producers/outputs and cycles reject without dropping the old binding set.
Bindings require the same context. Bound producers cannot start another
epoch or be destroyed. Consumer execution/submission performs nonblocking producer
observation; `RNS8_BUSY` leaves the consumer token, epoch and output untouched.
Retry explicitly after progress. No background worker or hidden host wait is used.

The producer's pending proof stays owned by its prepared execution until ordinary
completion publishes it. Only successful, current publication refreshes those
consumer input generations. Unrelated input mutations reject, as do stale output
receipts. Failed/canceled producers propagate their operation status without a
consumer launch. This provides a pending-value binding with host-driven progress;
it does not promise an event-only cross-stream pipeline before status is known.

After consumer acceptance, its normal matrix-reader reservations retain the value
and the producer binding is released. Before acceptance, explicit unbind with a null
producer/zero token or consumer destruction releases it. Ordinary refresh is BUSY
while bound. Prepared groups reject dependent members. A multi-producer consumer
requires every successful current publication; BUSY, failure or stale publication
from any branch prevents launch. A zero-count plural call unbinds all producers.
Prepare HIP event resources before binding pending producers.
Shared handles still require external host serialization, and borrowed streams must
survive completion. A completion token alone never retains an owner.

CPU fixtures cover epoch retention, stale receipts and unrelated mutations. HIP
fixtures cover successful publication and canceled-producer rejection. Additional
CPU/HIP joins bind three producers, observe HIP completion in reverse order,
execute an exact epilogue, reject an invalid rebind without dropping holds, and
reject a cycle reachable through the second dependency branch. Applicable
CPU/HIP cases pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).


C++ owners expose `close()` returning the C destruction status. `RNS8_BUSY`
retains the handle so callers can observe completion or destroy retained prepared
owners and retry. Move assignment now throws `rns8::Error` on failed close and
preserves both source and destination handles; it is no longer `noexcept`.
Move construction remains nonthrowing. Declare context before plans, matrices and
workspace, then destroy prepared/graph/dependency owners before closing those
resources. Destructors do not wait for asynchronous work and cannot report BUSY;
use explicit successful `close()` when lifetime order is not guaranteed. Do not
let a busy owner leave scope and lose the handle needed for later cleanup.


Finite prepared submission also uses the existing independent-group, cancellation
and owned-graph paths. Successful observation publishes the plan modulus and
clears exact-range/axis metadata; cancellation leaves the output unavailable.
Authored HIP fixtures cover moduli 2/250/251/256, padded inputs/outputs, K=17 and
65537, two-stream shared readers observed in reverse order, grouped graph replay,
stale source generations, refresh, pending ownership and cancellation. This
extension passes applicable gfx1100 fixtures in the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).


Matrix destruction now also preserves its C handle after a failed backend release,
matching the C++ `Matrix::close()` ownership rule. Successfully released addresses
are cleared individually; a retry visits only remaining allocations. Values are
invalidated during cleanup, and a failed destruction handle is for cleanup retry
only. Active matrix readers or writers reject destruction with BUSY before any
release. If creation fails and device cleanup also fails, the context retains the
unpublished matrix in an allocation-free intrusive list, rejects further arithmetic,
and retries those releases during context destruction. Context destruction also
retains an unreleased hipBLASLt handle on backend failure. Creation still returns a
null matrix output. These bookkeeping CPU fixtures pass the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json); no real device-fault campaign was run.

### Strict wrap64 resident edges

Resident MatMul DAGs also accept `RNS8_WRAP_U64_MOD_2_64` on CPU reference and
direct HIP. Preflight checks byte-limb storage and external input generations;
intermediate edges carry the explicit wrap contract without an integer range
proof. Synchronous execution, caller-stream submission and captured replay all
use byte-limb products. Async completion publishes every output only after the
whole DAG completes successfully; cancellation leaves outputs unavailable.
Captured replay binds the byte-limb allocation addresses. Exact epilogues,
base extension and wide-limb export nodes retain their exact-wide restrictions.

The source regression cases cover unsigned full-width boundaries, CPU Boost
references, non-square padded host buffers, K=4096/4097 accumulator selection,
shared-reader groups observed in reverse order, stale generations, cancellation,
and resident fan-out with unpublished intermediate values. These applicable
cases pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).

### Bounded prepared submissions

Prepared direct-HIP streams and captured replay also accept dense bounded i64/u64
plans with a fixed prefix and no adaptive tile schedule. They use the same
resident residue validation and enqueue kernel as synchronous bounded GEMM.
The plan's declared output bound remains the caller's contract; submission does
not reinterpret bounded storage as exact-wide or invent a dynamic range proof.
Export still uses the checked signed or unsigned bounded API. The prepared object
requires already-current RNS inputs; lazy native conversion must happen before
submission. Bounded resident MatMul DAGs use the separate resident-graph APIs.
The stream-provider section below describes individually admitted accelerators.
Fixtures cover narrow and full-width declared
bounds, signed extrema, unsigned maxima, pitched export, replay, refresh
and cancellation; applicable execution passes the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json).

Bounded RNS currentness also records the number of materialized planes. A small
declared output bound can select fewer planes than a matrix's allocated capacity.
Prepared admission, refresh and execution reject a later wider-prefix consumer
with `RNS8_RANGE_ERROR`; current storage alone does not authorize unwritten
planes. Pending producers use their actual selected write prefix. Native import
explicitly encodes the full destination basis again. This is plane availability,
not an exact-wide magnitude proof or automatic base extension.

## Event-dependent prepared submission

`rns8_submit_prepared_hip_pipeline` explicitly consumes the existing token
bindings without waiting for host publication. Prepare each owner's HIP event
before producers start, bind up to three producer tokens, then submit the consumer
on a non-default stream. Pending producer events become `hipStreamWaitEvent`
dependencies. Successful submission performs no host wait or device-wide drain.
The current envelope is fixed-prefix bounded i64/u64 or exact-wide GEMM on the
prepared direct-HIP/builtins/CK/rocWMMA/hipBLASLt providers, and direct-HIP exact
axpby. Finite/wrap pending-token consumers and caller-graph pipeline submission
are not admitted by this particular entrypoint.

Before issuing work, preflight checks each producer's identity, latest epoch,
semantic contract, written prefix, failure state and pending proof. Only inputs
bound to those producers may have an active writer. Unbound inputs must still
match their original generations and be current. Exact admission uses copies of
producer proof metadata without copying device payloads or publishing values.

Consumer observation also observes its producers and records their actual
published generations. Producer cancellation or failure invalidates the consumer
result even if its kernel ran successfully. Producer handles, epochs and input
readers stay reserved until consumer completion or safe failure cleanup. A failed
accepted submission issues a token and still requires blocking observation;
failed drains retain ownership. Up to 64 nested pending event-dependency levels
are admitted to bound host observation recursion. Independent siblings do not
consume that depth limit. This path makes no measured overlap or speedup claim.

The authored join fixture submits three independent producers and an axpby
consumer before observing any producer, verifies unpublished storage and retained
owners, checks the exact output, and repeats with late producer cancellation.
These applicable fixtures pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).

Private direct-HIP device and pinned-host temporary buffer owners retain their
pointer and capacity if an explicit reset/free fails. A replacement allocation then reports the failure
without replacing that owner. Destruction or move replacement hands a failed
free to the existing device-drain retry queue using a cleanup record allocated
before device allocation. The cleanup record distinguishes device and pinned-host allocations; the retry
queue uses the corresponding release API. Successful device drain/free is required to retire
that record; another device's records remain queued. The small host ownership
record is metadata, separate from requested device payload capacity. Invalid
zero-size or negative-device replacement preserves the current allocation.
This cleanup bookkeeping does not authorize freeing resources with live public
reservations or establish a new asynchronous execution path.

### Cleanup and retry ownership

Workspace, prepack-cache, result-cache and sparse-matrix destruction consume the
handle only on success. A failed release retains the pointer and byte accounting;
retry destruction after recovering the backend. Do not execute with a partially
retired handle. Construction owns device buffers through preallocated RAII cleanup
records from the first allocation: an exception or failed release transfers
unpublished buffers to the existing device-drain cleanup queue, not a lost raw
pointer. No failure-injection or hardware qualification is implied by this rule.


## Prepared exact conversion variant

Exact HIP base extension now uses the same prepared execution and completion
token, rather than an unrelated transfer completion system. See
[mixed-radix lift](mixed-radix-lift.md) for its frozen range/suffix envelope,
retained pinned status and status-gated device commit. It composes with existing
pending-value dependencies and independent groups; no whole-device wait is
hidden in submission. Counted token-ordered writers keep a producer→lift chain
reserved until its last actual writer finishes. Ordinary arithmetic defaults
are unchanged. The explicit caller-graph binding can capture one prepared lift
body under the same retained owner.

Destroy first retires graph/event identities with checked HIP status, then
releases transfer scratch. A failed destroy preserves the handle for cleanup
retry, marks it closing and rejects further execution; it never erases a failed
HIP object's only identity. Applicable local execution passes the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json); real cleanup-failure qualification remains deferred.

## Owned pinned-host transfer boundaries

`rns8_create_pinned_host_buffer_hip` owns a pinned native rectangle on one
explicit direct-HIP context. `rns8_prepare_host_transfer_hip` binds it to an
exact-wide matrix using the **existing** prepared execution, completion token,
stream, dependency, cancellation and owned-graph APIs. This is not a second
completion system or an implicit change to synchronous pack/export. Signed and
unsigned 64-bit imports and scalar or 1–32-limb exports are supported; bounded,
finite, wrap, borrowed pageable pointers and compiled resident-graph transfer
nodes are not inferred from this adapter.

The buffer descriptor uses element strides: for limb output, one element is
`limb_count * 8` bytes. Allocation includes `rows * ld` elements, including the
last row's padding; DMA copies only logical columns. A checked size query is
available without HIP. Creation rejects an insufficient explicit pinned budget
before allocation. Buffer storage is zeroed but initially **not current**.

### CPU and DMA ownership

1. Open an exclusive `rns8_begin_pinned_host_write` window, write logical input
   (and optional padding), then publish a strictly newer, nonzero generation
   with `rns8_end_pinned_host_write`. All CPU writes must have finished first.
2. Prepare the import and optionally capture it outside execution. An existing
   prepared owner must be refreshed after the buffer generation changes.
3. Submit on a borrowed nondefault stream, or replay an owned graph. Multiple
   independent imports can read the same input buffer. CPU reads may share this
   immutable input; CPU writes remain BUSY until **all** DMA readers finish.
4. For output, prepare export from a current proved matrix; bind a pending
   producer token before pipeline submission. An accepted output write makes
   the buffer unreadable, including its previous value. Wait/query completion
   success only retrieves a record: check `operation_status` and
   `output_is_current` before opening `rns8_begin_pinned_host_read`.
5. End every CPU access window. Destroy prepared dependents before their
   producers, then buffers, matrices/workspaces/plans, then context. Destruction
   is BUSY while referenced or in use. `rns8::PinnedHostBuffer::close` preserves
   its handle on failure; its destructor neither waits nor repairs a wrong
   destruction order. Explicit close/retry is necessary for fallible cleanup.

CPU pointers are valid **only inside their access window**. Keeping an escaped
pointer and touching it during DMA violates the contract even if its address
has not changed. Calls sharing handles require external host serialization.
A successful export increments the buffer's generation and permits replay at
stable addresses. A failed or canceled attempted export remains unreadable;
there is no rollback promise for its private contents. Device drain failure
keeps reservations and storage alive and poisons the context rather than
assuming DMA stopped.

### Preparation and publication costs

Preparation retains a compact native device bounce buffer, a conversion stage,
a four-byte pinned status, and a private stream. Imports optionally retain the
matrix's requested native representation. The total-scratch limit includes the
already-owned full padded host rectangle, status, both device stages, and that
optional native copy. Provider queries report the same known byte boundary;
HIP's opaque stream/event/graph allocations are additional, not claimed zero.

The submitted path performs pitched `hipMemcpy2DAsync` plus existing guarded
conversion/CRT kernels: no allocation, host callback, pageable staging or host
synchronization is inserted into enqueue. Imports publish RNS/native currentness
only after bound status and completion observation. Exports hold their host
writer reservation through CRT status and DMA completion, including upstream
failure/cancellation observation. An old host export remains a valid independent
snapshot after the source matrix changes, but its original prepared token no
longer certifies a current source generation.

The upstream [HIP asynchronous execution contract](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html)
and [host-memory guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html)
explain why pinned storage and explicit lifetime are needed. They do not qualify
this implementation or guarantee useful copy/compute overlap. The authored
regressions cover padded import→GEMM→limb-export stream/graph pipelines, failed
bound publication, stale generations, budget arithmetic and two shared H2D
readers completed in reverse order. Applicable CPU-ASan and gfx1100 HIP execution
passes the [October27 implementation validation](../metadata/implementation-closeout-20261027.json). Physical overlap traces, performance
and broader platform qualification remain deferred.

Base-extension discovery is a value-preserving transfer contract, not a new
native GEMM: a short source/target basis is admitted using the current source
uniqueness proof and concrete owner binding. It does not relax ordinary public
GEMM plan creation's full-native range floor. The composed prepared-lift fixture
therefore uses legal prefix17→prefix20 public plans; a separate pure provider
fixture covers prefix3→prefix9 discovery, missing proof and reversed prefixes.
Both lift device scratch and pinned status are explicitly pending until checked
completion/drain. These applicable fixtures pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).
# Matched pinned-host transfer calibration

`rns8-host-transfer-calibrate` compares ordinary pageable synchronous calls,
pinned synchronous calls, pinned borrowed-stream pipelines and pinned owned
graph pipelines at one readable padded three-limb host-result boundary. Every
mode imports both native64 operands on each call and performs the same product
and final export. Pinned imports use the full native64 envelope without a
redundant input-bound scan; ordinary packing keeps its engine-owned host scan.
Both use prefix20 and explicit signed/unsigned exact-wide semantics. The ordinary
export control explicitly disables optional nested pinned export staging.

The driver reports construction/proof bootstrap and graph capture separately,
submission and terminal observation, two source generations, full outside-timer
Boost/padding audits and bounded sustained calls. Synchronous submission already
includes completion. Queried matrix/transfer payloads, caller capacities and a
bounded reference reservation are charged; graph/driver setup remains opaque.
No overlap is inferred from stream count. The 64 matched cases include skinny,
tail, 128³ and K65537 workloads with small and full-width inputs.

```powershell
python tools/benchmark_cases.py --output temp/benchmark-cases/current
$hostTransferCases = (Get-ChildItem -LiteralPath temp/benchmark-cases/current/host-transfers -Filter '*.json').FullName
python tools/collect_host_transfers.py --binary build/issues-all/rns8-host-transfer-calibrate.exe --cases $hostTransferCases --output temp/host-transfers --runs 5
```

On Linux, pass the case paths through the shell's normal glob handling. The
collector counterbalances independent processes, binds artifact/build/environment
identity, validates both generations and complete payload/audit/call receipts,
and preserves p50/p95/sustained regressions. Its gains table contains only modes
that improve all three observed metrics over the matched ordinary control.
Applicable collector and correctness checks pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).
No new timing campaign or speedup is claimed.

## Stream providers and explicit caller graphs

Prepared dense GEMM now has enqueue-only builtins, CK, rocWMMA and hipBLASLt
adapters alongside direct HIP. Each keeps its ordinary semantic, modulus, shape,
address and target admission. Builtins retain their65536 K ceiling. CK and
rocWMMA pass the supplied stream through packing, every matrix product and
modular carry. Their stream path bypasses timing-event setup and synchronization;
CK explicitly disables invoker timing. Existing synchronous controls still drain.
The rocWMMA prepared stream route uses its named baseline packing policy.

hipBLASLt preparation retains descriptors and selected algorithms for the full
and tail K geometries, and warms both with zero private pack buffers outside
capture. Replays pack current operands, issue INT8→INT32 matmul on the supplied
stream, then reduce with integer carries. No heuristic, transient descriptor,
prepack-cache rebuild or timing event occurs in enqueue. Retained workspace and
context references last through completion; an undrained preparation keeps its
cleanup owner and workspace reservation and poisons the context. Opaque library
setup remains additional to the known payload lower bound.

The [October27 implementation validation](../metadata/implementation-closeout-20261027.json) separately qualifies applicable adapter
correctness on Windows/gfx1100; older direct-HIP receipts do not qualify the new
adapters or Linux/CDNA. The fixtures exercise signed
and unsigned full-width outputs, bounded exports, finite composite/prime rings,
K tails, independent groups, shared readers and repeated graphs against the
independent exact references, including actual stream, owned-graph and
caller-graph execution. These checks do not establish overlap or elapsed gains.

Caller embedding is a separate four-step interface:

1. Call `rns8_prepare_hip_submission` before capture to create completion and
   backend resources.
2. Begin capture on a borrowed nondefault same-device stream, then call
   `rns8_capture_prepared_in_caller_hip_graph`. This records one RNS8 body and
   retains its prepared owner; it does not execute, publish, or allocate an epoch.
3. End capture yourself and pass that exact graph to
   `rns8_finalize_caller_hip_graph`. RNS8 instantiates and owns a separate
   executable. The original graph can then be destroyed by the caller.
4. Launch through `rns8_submit_caller_hip_graph`, then observe its ordinary
   prepared token. Destroy the binding after completion and before destroying
   the prepared execution.

Only one RNS8 body per caller capture is admitted. A capture-ID registry rejects
additional bodies, because independently reserving one owner cannot protect the
other bodies. Multi-operation RNS8 composition continues to use resident DAGs.
Other caller nodes must preserve RNS8's input/lease contracts and may not read
unpublished results: a later node is not a substitute for host-observed exactness
status. Their own external pointers must remain alive through execution. Do not
mutate/remove captured RNS8 nodes, independently instantiate/launch the original
graph, or recycle its addresses behind the binding.

Finalization preserves an existing executable on replacement failure. Submission
rechecks stable addresses, generations, ranges and stream state before reserving
storage and issuing its epoch. Cancellation, deadlines, failure diagnostics and
quarantine use the existing prepared state machine. A partial capture failure
can return a cleanup-only binding; discard the caller capture/graph before
releasing it. The caller stream and graph are never destroyed by RNS8.
