# Runtime product arena

`rns8_create_product_recipe` owns a synchronous CPU/direct-HIP executor for a
fixed topologically ordered exact-wide product DAG. It accepts current public
matrix inputs and publishes a final public matrix through `rns8_execute_product_recipe`,
or an explicitly ordered set of node results through the multi-output API. Inputs must share explicit
signedness, the default basis, a common input/output prefix and modulus-major storage. It
retains context/input lifetimes and binds their engine instance/generation;
repacking requires explicit input rebinding or creating a new recipe. Changed or busy inputs reject execution.

Preparation derives every intermediate shape, proves every complete product with
a conservative magnitude envelope, and checks the common CRT range. It computes
all last-reader positions before assigning storage. Deterministic best-fit slots
are 256-byte aligned; a slot can be reused only when its prior value's last reader
is strictly earlier than the new producer. An input still read by the current
node can never share that node's destination. Every requested observable value remains live until
publication. Cycles/forward IDs, incompatible dimensions, insufficient exact range
and arena budget violations reject before execution.

The actual executor allocates one CPU vector or HIP device buffer and dispatches
each product into its assigned region. HIP products use the existing direct RNS
kernel; CPU products independently reduce every scalar accumulation. Intermediate
values stay private until every requested result is ready for publication.
Kernel failure before publication preserves the old outputs. Copy failure invalidates
the entire requested output batch. Successful publication installs the complete prefix and exact magnitude proof.
Replay overwrites each logical value before reading it and rechecks input bindings.

The explicit budget covers aligned intermediate payload storage plus the maximum
simultaneous HIP suffix-lift scratch, not retained public input/output matrices,
host proof/descriptor metadata or runtime allocator overhead. `arena_bytes` and the sum of unreused logical output bytes are exposed
separately; alignment can make tiny cases larger than that unaligned sum. Lifetime
holds are released on destruction; execution/destruction require ordinary external
host exclusion. Concurrent execute attempts return BUSY.

Authored CPU cases cover linear and diamond reuse, all overlapping live intervals,
budget boundaries, lifetime/currentness and replay. A HIP diamond uses independent
Boost full-output comparison. They have not been executed in this frontload.
No speedup or global allocation optimality is claimed.

## Consumer-driven exact prefix materialization

The optional `minimize_compute_prefix` policy selects the smallest default-basis
prefix certified by each node's conservative complete-product bound. It reserves
each value's maximum required consumer prefix, rather than the common ceiling.
Before a consumer runs, any missing input suffix is reconstructed from the already
unique exact value and appended. A later consumer can extend the same value again.
Every requested private result is extended to the full public output prefix before any copy.
No input prefix is truncated in place and no external matrix is mutated for lifting.

CPU lifting checks canonical centered residues, reconstructs with exact arithmetic,
checks the known magnitude, and writes the private suffix. HIP uses the existing
staged exact lift with the same bound; its maximum suffix-plus-status scratch is
reserved separately from the arena. `lift_scratch_bytes()` exposes that peak.
Replay resets materialization state and never treats leftover suffix bytes as valid.
Authored linear/diamond CPU and HIP fixtures exercise both policies, replay, exact
exports and the combined arena/lift budget boundary; execution is still pending.

This policy minimizes compute prefix under a conservative bound, not total measured
cost. Small prefixes can lose after lift overhead; the full-prefix route is the
control. Layout conversion/ragged inputs, public graph arena integration,
multi-stream execution and an exhaustive prefix/layout cost optimizer remain
separate work.

## Public recipe lifecycle

Pass an ABI-tagged `rns8_product_recipe_desc` with input handles, ordered product
nodes, explicit prefix policy and total arena/lift budget. Input IDs precede node
output IDs. Creation retains inputs/context; failures preserve the output handle.
`rns8_get_product_recipe_resources` reports aligned arena bytes, peak lift scratch,
node count and the unaligned sum of unreused payloads. The latter is diagnostic,
not a guaranteed speedup or allocator peak. `rns8_get_product_recipe_prefixes`
returns each prepared compute prefix and preserves the array when capacity is too
small. Resources/prefix choices are immutable after creation.

The final output must match the prepared shape, explicit exact-wide semantics,
common external prefix and backend/device, and must differ from every input.
The caller can reuse the recipe with distinct compatible outputs. Intermediate
values are private unless selected as observable outputs at creation; they cannot be rebound. Repacking inputs requires
explicitly rebinding compatible inputs or creating another recipe; successful destruction releases retained lifetimes. There is no
public async submission. Destruction may drain quarantined failed work before
releasing its storage. External host exclusion
covers use/destruction, as with other synchronous owning APIs.

The older resident graph API still accepts explicit intermediate handles and its
own supported operations. This specialized product API provides owned scratch and
consumer-driven precision for GEMM-only DAGs; it does not replace that graph API.


## Rebinding and failure ownership

`rns8_rebind_product_recipe_inputs` replaces all input handles and generation
snapshots in one transaction. Each input must retain the compiled shape, semantics,
prefix, backend and device, with a current exact range proof no larger than its
original input envelope. This permits repeated execution after repacking without
allocating another arena or changing selected compute prefixes. A larger bound
requires a new recipe. Failed validation preserves every old binding and hold;
duplicate handles retain one lifetime reference. Output proofs remain the original
conservative compiled bounds.

Execution reserves each unique input reader and the output writer before work.
Intermediate failures preserve the old public output unless publication has begun.
An uncertain device drain retains the arena, input readers and output ownership,
and poisons the context. Destruction returns a failure while retaining the recipe
when draining or freeing fails, so callers can retry destruction safely. Private
RAII destruction cannot report errors and retains a failed owner instead of freeing
possibly active storage. This is source-level ownership behavior awaiting the
consolidated validation campaign, not a completed device-failure qualification.


## Multiple observable results

`rns8_create_product_recipe_outputs` takes the ordinary recipe descriptor and an
ordered list of distinct produced value IDs. `rns8_execute_product_recipe_outputs`
binds an equally sized array of distinct compatible destination matrices. External
input IDs cannot be requested or used as destinations. Requested intermediate
values remain available to later consumers and are retained until publication;
the arena planner includes those extended lifetimes and full-prefix storage in
its budget. The final node need not be requested. All listed nodes still execute;
this API does not silently remove unobserved work from the supplied recipe.

All destination shapes, semantics, devices, reservations and aliases are checked
before execution. Every private result is extended before any public copy starts.
All copies must succeed before any output is marked current. A publication failure
invalidates the complete batch, and an uncertain drain retains every output hold
and input reader reservation. This is synchronous batch publication under external
host exclusion, not an atomic snapshot for concurrent readers. Output order follows
the requested IDs, including a non-topological order.

The single-output API keeps its final-node default. A recipe explicitly requesting
one intermediate can also use single-output execution. The internal calibrated
cost selector still models one final result and rejects other observable sets;
full-prefix and minimum-compute-prefix policies support multiple results without
claiming global cost optimality. Authored CPU/HIP fixtures compare signed and
unsigned diamond intermediates and final outputs to Boost, exercise both prefix
policies, and cover retained live ranges, exact budget rejection, duplicate/null/
input aliases and all-output precommit preservation. Execution remains pending.

## Physically retained lift workspace

HIP recipes now allocate the aligned product arena **and** their maximum
status-plus-suffix workspace together during preparation. The arena size remains
separately queryable; `lift_scratch_bytes` is physically retained capacity, not
merely a budget reservation. The four-byte status begins at the aligned arena
end and the byte suffix follows it, so odd cell/suffix sizes cannot misalign the
status. Linear, diamond and multi-output materializations reuse that one region
serially; compatible input rebinding does not move it.

The workspace-taking lift still clears status, validates the source residues and
magnitude, synchronizes/checks the result, and copies only a completely accepted
suffix. It preserves the existing commit-start and public all-output publication
transactions. Borrowed scratch is checked for capacity, alignment and overlap
with all source/destination planes. No lift allocation/free is performed during
recipe replay. A caller without a persistent owner retains the one-shot helper,
now using one combined status/suffix allocation rather than two separate buffers.
Failed drains/cleanup retain the existing whole recipe owner and all reservations.

For the32×32 all-ones three-node minimum-prefix chain, compute prefixes remain
1/2/3, arena capacity25,600 bytes and retained lift capacity17,412 bytes. Creation
physically reserves43,012 bytes once. The three lifts no longer allocate/free
workspace during replay; this is a source allocation count, not a hardware
measurement. Regression coverage checks positive replay and compatible-rebind
allocation/free deltas, every prefix pair, status reuse after rejected proofs,
full output residues and exact budget boundaries. These regressions are unrun.

Internal cost profiles are now version2: HIP lift entries must price the retained
workspace implementation, including reset/check and staged copy, while CPU
entries price inline host extension. Version1 profiles are rejected instead of
silently relabeling old one-shot preparation costs. Unknown estimates remain
unknown; retaining memory does not make the lift free. Full hardware compilation,
complete-call/sustained measurements and failure qualification remain deferred.
