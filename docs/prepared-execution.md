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
fixed-envelope graph replay path. Grouped and DAG submission remain separate
implementation and qualification work. No speedup is claimed.

## Explicit HIP submission

Call `rns8_prepare_hip_submission` on a prepared dense direct-HIP exact-wide, strict wrap64, or
finite-ring/field execution to allocate its completion event. The subset has fixed
shape/basis/prefix or one explicit finite modulus, and no adaptive tile schedule.
Other semantics/backends return unsupported. Finite operations use their existing
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
Stream capture is explicitly unsupported; there is no synchronous fallback.

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
independent streams. Consumers must observe producer completion before using the
new value. Destruction remains BUSY while handles are retained or executions are
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

These recovery changes and host-side failure fixtures are authored for the next
consolidated validation pass. Hardware fault recovery and cross-device behavior
are not newly qualified by this change.


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
stream fixtures are included in the pending consolidated validation batch.


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
The new all-semantics lifecycle fixtures are authored for the final batch and
do not extend the supported asynchronous semantic subset.


## Token-qualified producer binding (authored, validation pending)

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
reject a cycle reachable through the second dependency branch. All newly authored cases remain queued for the consolidated campaign.


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
extension is source-only pending the final consolidated validation campaign.


Matrix destruction now also preserves its C handle after a failed backend release,
matching the C++ `Matrix::close()` ownership rule. Successfully released addresses
are cleared individually; a retry visits only remaining allocations. Values are
invalidated during cleanup, and a failed destruction handle is for cleanup retry
only. Active matrix readers or writers reject destruction with BUSY before any
release. If creation fails and device cleanup also fails, the context retains the
unpublished matrix in an allocation-free intrusive list, rejects further arithmetic,
and retries those releases during context destruction. Context destruction also
retains an unreleased hipBLASLt handle on backend failure. Creation still returns a
null matrix output. These bookkeeping changes and CPU fixtures are authored for
the consolidated batch; no device-fault or allocation-fault campaign was run.

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
and resident fan-out with unpublished intermediate values. Execution of these
cases is pending the consolidated validation campaign.
