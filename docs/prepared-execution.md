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

Call `rns8_prepare_hip_submission` on a prepared dense direct-HIP exact-wide
execution to allocate its completion event. The subset has fixed shape/basis/
prefix and no adaptive tile schedule. Other semantics/backends return unsupported.
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
pending. Event errors trigger a stream drain before reservations are released;
fatal device-context errors remain backend failures and are not repaired here.

CPU reservation fixtures and real HIP two-stream fixtures are authored for
shared readers completed in reverse order, write/workspace collisions, pending
destruction, rebinding, stale epochs, range rejection, tails and K-slab crossings.
Compilation and execution are pending; no overlap or enqueue-time gain is claimed.
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
one fixed-shape dense exact-wide operation, its fixed basis/prefix, and all
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
during replay. Compilation and execution are pending; capture cost, overlap,
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
stream and capture execution, and completion identities. Runtime qualification
is pending. The ordinary async range-exhaustion fixture now uses a legal
full-width plan and creates a wider-than-native A with an exact outer product
before replay, rather than requesting an inadmissibly small initial basis.
