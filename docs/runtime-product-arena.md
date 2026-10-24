# Runtime product arena

`rns8_create_product_recipe` owns a synchronous CPU/direct-HIP executor for a
fixed topologically ordered exact-wide product DAG. It accepts current public
matrix inputs and publishes one final public matrix through `rns8_execute_product_recipe`. Inputs must share explicit
signedness, the default basis, a common input/output prefix and modulus-major storage. It
retains context/input lifetimes and binds their engine instance/generation;
repacking requires creating a new recipe. Changed or busy inputs reject execution.

Preparation derives every intermediate shape, proves every complete product with
a conservative magnitude envelope, and checks the common CRT range. It computes
all last-reader positions before assigning storage. Deterministic best-fit slots
are 256-byte aligned; a slot can be reused only when its prior value's last reader
is strictly earlier than the new producer. An input still read by the current
node can never share that node's destination. The final value remains live until
publication. Cycles/forward IDs, incompatible dimensions, insufficient exact range
and arena budget violations reject before execution.

The actual executor allocates one CPU vector or HIP device buffer and dispatches
each product into its assigned region. HIP products use the existing direct RNS
kernel; CPU products independently reduce every scalar accumulation. Intermediate
values stay private: only the final staged result is copied to the public output.
Kernel failure before that copy preserves the old output. Copy failure invalidates
it. Successful publication installs the complete prefix and exact magnitude proof.
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
The final private result is extended to the full public output prefix before copy.
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
values are private and cannot be exported or rebound. Repacking inputs requires
creating another recipe; destruction releases retained lifetimes. There is no
public async submission or hidden wait in destruction. External host exclusion
covers use/destruction, as with other synchronous owning APIs.

The older resident graph API still accepts explicit intermediate handles and its
own supported operations. This specialized product API provides owned scratch and
consumer-driven precision for GEMM-only DAGs; it does not replace that graph API.
