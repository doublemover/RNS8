# Resident DAG execution

`rns8_create_resident_graph` accepts an ordered list of 1-4096 typed
`rns8_resident_node` records on CPU or direct HIP. MatMul supports exact-wide
and explicit finite ring/field semantics; other node kinds remain exact-wide.
MatMul nodes reference an existing plan,
workspace, input matrices and distinct output matrix. ExactAxpby nodes also
reference an addend in `d` and signed integer `alpha`/`beta` coefficients and
compute `alpha*a*b + beta*d`. Unsigned semantics reject negative coefficients.
The addend can be external or produced earlier in the DAG, and is retained and
bound like every other input. Other kinds require these epilogue fields zero.
BaseExtend nodes reference
the matrix in `c`, an explicit target prefix and a staging budget; their other
resource fields must be null. The graph retains these
handles and the context; destroy returns BUSY while a graph holds a resource.
One output has one producer. Reads of a produced value must follow its producer,
so cycles, forward references and overwriting a logical value are rejected.
Shared read-only values and diamonds are supported. BaseExtend is an in-place
materialization of the same logical value, so it does not create a second writer
or change content generation. Its position must follow the value's producer and
precede any consumer needing the additional planes.

Creation and execution propagate exact-wide bounds through the existing
axis/moment/scalar proof implementation and finite moduli through each finite
edge before arithmetic begins. Proof metadata
is copied once per external input; matrix payloads are never copied for planning.
This catches downstream range exhaustion and insufficient extension capacity or
staging budget before earlier nodes overwrite outputs. Proof propagation updates
available prefixes only after an explicit BaseExtend node; omitted lifts fail
preflight. It also checks extension busy state, storage/device identity and the
source value's unique reconstruction range.
ExactAxpby uses the same complete-expression bound and operand validation as
the standalone API: `abs(alpha)*bound(AB) + abs(beta)*bound(D)` must fit the root
basis. The unpublished AB need not fit independently (for example, alpha zero).
All three inputs still require materialized planes even with zero coefficients.
Its output carries a conservative scalar bound into downstream consumers.
Execution uses existing synchronous GEMM, exact epilogue and CPU/HIP lift publication guards.
Each intermediate remains resident and can feed multiple later nodes.

`EXPORT_LIMBS` nodes explicitly read the matrix in `c` through `plan` into a
borrowed host `export_limbs` buffer. `export_ld` is measured in matrix cells and
`export_limb_count` selects 1-32 little-endian 64-bit limbs per cell; signedness
comes from the exact-wide plan. Other resource, epilogue and extension fields
must be zero. Keep the buffer alive for the graph lifetime and disjoint from
library-owned storage. Export destinations in the same graph cannot overlap,
including their inter-row padding spans. Shape, stride, byte-span overflow and
alignment are checked before execution.

The complete graph proof includes the export's available prefix and destination
width. A symmetric magnitude bound must fit the signed/unsigned destination;
it conservatively rejects a signed-minimum boundary when the proof cannot
distinguish that sign. A later narrow export therefore rejects before earlier
GEMMs write. Wider exports remain available. Export nodes retain their plan and
source handles, preserve the logical value's generation and call the ordinary
staged CPU/HIP export implementation. An explicit export node can also read an
external resident value. Additional exports outside the graph remain supported.

External input instances and generations are bound at creation. Repacking one
requires `rns8_refresh_resident_graph_inputs`, which rechecks the complete proof
chain before changing bindings. Stale bindings reject execution before writes.
Execution reports its epoch, completed-node count, status and failing operation
index; an external-input binding error has no operation index. Runtime failure
retains completed earlier nodes, applies ordinary failure semantics to the
failing node, and leaves later nodes untouched. Calls sharing resources require
external host-thread serialization.

This first #42 executor requires existing uniform storage compatible with every
edge. It does not allocate intermediates, silently fill missing planes, increase
prefixes implicitly, reassociate arithmetic or infer wraparound semantics. All referenced
intermediates remain allocated until callers release them after graph destruction.
The former MatMul-only node record is replaced by the typed record in this
unreleased API; callers must rebuild and set its current struct size. Unknown
kinds and mixed node fields fail closed. Authored CPU/HIP chain fixtures cover
the full-native planner prefix -> fixed prefix 20, signed negative values, failed staging admission, missing
or forward lifts, identity preservation and stale external bindings. Registered applicable execution evidence is recorded in the
[dated validation](local-all-features-20261023.md).

Liveness-based allocation reuse, native conversion IR nodes,
ragged inputs, backwards materialization demand, cost-based representation
selection and broader asynchronous node support remain open. The explicit
fixed-prefix direct-HIP asynchronous subset is described below. Preflight and ordinary GEMM
both perform admission; this baseline exposes that setup cost rather than
claiming a measured dispatch or proof-work reduction.

Authored epilogue DAG fixtures cover signed/unsigned CPU/HIP chains with a
produced addend, repeated execution, extreme coefficients, Boost differential
exports, forward/alias rejection, and downstream range rejection before writes.
Registered applicable execution evidence is recorded in the
[dated validation](local-all-features-20261023.md).

Authored export-node fixtures cover CPU/HIP signed and unsigned GEMM/export,
repeated execution, padded host destinations, an export-only reader preserving
generation, forward and overlapping destinations, malformed fields, overflow
and later-node width rejection before earlier output mutation. Registered
applicable execution is covered by the [dated validation](local-all-features-20261023.md).

## Explicit asynchronous direct-HIP DAGs

`rns8_prepare_resident_graph_hip` prepares a completion event for a dense graph
of fixed-prefix exact-wide MatMul/ExactAxpby or finite ring/field MatMul nodes. `rns8_submit_resident_graph_hip` preflights every
binding and propagates every intermediate proof, then enqueues the ordered DAG
on one caller-owned nondefault stream. Internal dependencies follow stream
order. BaseExtend/Export nodes and adaptive tile schedules reject as unsupported
before launch. Owned capture supports this same fixed resource envelope.

The graph reserves each distinct external input once, each output as a writer,
and each distinct workspace once. A workspace shared by serial nodes is legal.
All graph handles remain retained. Repacking, refresh, synchronous execution,
resubmission and destruction return busy while pending. The completion token
binds graph identity and epoch; another graph's or an earlier submission's token
cannot observe this execution.

No intermediate is publicly current while the DAG is pending. Query/wait
observes the final event, publishes every output with its precomputed bound or finite modulus and
new generation, and releases all reservations. Publication swaps already
computed proofs rather than allocating after GPU completion. An enqueue or
event failure requires a successful stream drain before releasing reservations without publishing
any attempted output; nodes never attempted retain their earlier values. This
whole-DAG asynchronous publication rule differs from synchronous execution,
which can retain successfully completed earlier nodes. Completion reports zero
published nodes on failure and the launch-failure index when available.

The API rejects externally capturing streams and does not allocate device
storage during submission. Proof construction can allocate host metadata.
Authored signed/unsigned mixed MatMul/epilogue tests cover produced addends,
diamonds, dependency chains, shared workspaces, reader overflow, stale tokens,
busy lifetime rules and no early publication. CPU paths explicitly reject HIP
preparation. Registered applicable runtime/lifetime cases are covered by the
[dated validation](local-all-features-20261023.md). Broader failure/leak campaigns,
overlap and submission-latency measurements remain separate gates.

`rns8_capture_resident_graph_hip` owns a HIP graph containing the ordered kernels
and `rns8_replay_resident_graph_hip` replays it with the same whole-DAG range,
reservation, token and observed-publication rules. Capture does not publish or
change any matrix value. It retains every A/B/C/D address, including internal
edges; a changed address rejects replay before reservation or output writes.
Stable-address input changes require ordinary explicit input refresh. Shape,
coefficients, topology or storage changes require a new graph/capture envelope.
Replacement capture preserves the previous executable until instantiation
succeeds. All outputs count as attempted on graph launch and remain invalid on
a failed replay. Event recording stays outside the captured kernel graph.

Authored replay tests cover diamonds, produced addends, signed/unsigned extreme
coefficients, replacement capture, address substitution rejection, no capture
publication and zero new allocations through the RNS8 device-allocation helper
during submission/wait. HIP-internal allocations are not covered by that
counter. Registered applicable replay tests are covered by the
[dated validation](local-all-features-20261023.md). Broader capture-leak/instruction
qualification and replay break-even measurements remain separate gates.

Fresh validation at `167aff0` exposed a reservation/invalidation ordering bug:
submission reserved output writers, then generic invalidation rejected those
same reservations with BUSY. Owned graph invalidation now clears currentness
without releasing or rechecking its own writer reservation. Each ordinary
submission invalidates only attempted outputs; replay invalidates all outputs
at launch. Completion retains the existing publication and reservation-release
rules. The bounded retry includes both diamond and produced-addend DAG cases.


## Failed stream cleanup

`rns8_get_resident_graph_async_failure` reports the original failure stage, operation status,
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

`rns8_cancel_resident_graph_hip` accepts the latest issued token and marks pending work for
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

`rns8_wait_resident_graph_hip_for` polls using a steady-clock deadline in microseconds.
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


`rns8_get_resident_graph_hip_support` reports the implementation's shape/backend/semantics
support without allocating events, performing range proofs or submitting work.
Its typed record exposes the single-epoch limit, borrowed nondefault-stream
requirement, possible host proof allocation, owned-graph, cancellation and timed
observation support. Unsupported records contain zero feature flags. This query
does not admit current values or a caller stream and is not hardware qualification.
Preparation reuses the same support predicate, then performs current admission.


## Paired complex nodes

`RNS8_RESIDENT_COMPLEX_MATMUL` composes the experimental exact-wide signed
complex API into synchronous CPU/direct-HIP graphs. It pairs `a/a_imag`,
`b/b_imag` and `c/c_imag`, chooses explicit 3M or 4M, and uses
`max_staged_residue_bytes` as the scratch budget. Each output has one distinct
writer and can feed later ordinary or complex nodes. Input/output cross-aliases,
forward references, mixed fields and duplicate writers are rejected.

Whole-graph preflight propagates both integer magnitude proofs using the same
complex arithmetic/resource admission routine as standalone execution. It
retains all four input and both output handles, binds every external generation,
and checks all downstream ranges before the first write. A successful node
publishes both components together. Synchronous graph failure can retain earlier
successful nodes, exactly as for ordinary nodes. Async/capture support queries
reject graphs containing complex nodes; no async paired-publication claim is made.

Authored CPU/Boost and direct-HIP fixtures cover both methods, downstream real
GEMM consuming both components, stale input refresh, retained output ownership,
aliased output and scratch rejection. New execution remains pending the final
consolidated validation pass.


## Finite ring and field products

MATMUL nodes also accept explicit finite-ring or finite-field plans. The graph
propagates each produced value's modulus through its metadata-only preflight;
every consumer must match that modulus and semantic contract. Unpublished
intermediates need no invented integer range proof. External inputs still require
current storage and bound generations. Modulus changes reject refresh and all
execution before writes. Exact axpby, lift, complex and limb-export node contracts
remain exact-wide; finite support does not reinterpret those operations.

CPU and ordinary HIP execution use the public finite GEMM. HIP asynchronous and
owned-capture execution use the existing finite enqueue-only kernel, including
its ordered K slabs. Final event observation publishes all node generations and
explicit moduli; cancellation/failure retains the existing graph invalidation and
reservation rules. Authored CPU/HIP diamond fixtures compare every intermediate
and output under moduli 2/250/251/256, replay captured graphs, check unpublished
intermediates and reject changed-modulus bindings. Execution remains pending.
