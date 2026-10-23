# Exact resident DAG execution

`rns8_create_resident_graph` accepts an ordered list of 1-4096 typed exact-wide
`rns8_resident_node` records on CPU or direct HIP. MatMul nodes reference an existing plan,
workspace, input matrices and distinct output matrix. BaseExtend nodes reference
the matrix in `c`, an explicit target prefix and a staging budget; their other
resource fields must be null. The graph retains these
handles and the context; destroy returns BUSY while a graph holds a resource.
One output has one producer. Reads of a produced value must follow its producer,
so cycles, forward references and overwriting a logical value are rejected.
Shared read-only values and diamonds are supported. BaseExtend is an in-place
materialization of the same logical value, so it does not create a second writer
or change content generation. Its position must follow the value's producer and
precede any consumer needing the additional planes.

Creation and execution propagate every node's exact bound through the existing
axis/moment/scalar proof implementation before arithmetic begins. Proof metadata
is copied once per external input; matrix payloads are never copied for planning.
This catches downstream range exhaustion and insufficient extension capacity or
staging budget before earlier nodes overwrite outputs. Proof propagation updates
available prefixes only after an explicit BaseExtend node; omitted lifts fail
preflight. It also checks extension busy state, storage/device identity and the
source value's unique reconstruction range.
Execution uses existing synchronous GEMM and CPU/HIP lift publication guards.
Each intermediate remains resident and can feed multiple later nodes. Export
only the requested outputs through their ordinary plan/export APIs.

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
prefix 3 -> 9 -> 20, signed negative values, failed staging admission, missing
or forward lifts, identity preservation and stale external bindings. Execution
of these fixtures remains pending.

Liveness-based allocation reuse, native conversion/export IR nodes,
ragged inputs, backwards materialization demand, cost-based representation
selection and asynchronous completion remain open. Preflight and ordinary GEMM
both perform admission; this baseline exposes that setup cost rather than
claiming a measured dispatch or proof-work reduction.
