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

The explicit HIP submission subset below adds stream/event completion. Grouped
and DAG submission, fixed graph envelopes and graph replay remain separate
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
baseline for #38, not a persistent matrix-engine queue, uploaded tile schedule
or async submission.
