# Exact resident DAG execution

`rns8_create_resident_graph` accepts an ordered list of 1-4096 exact-wide
MatMul nodes on CPU or direct HIP. Each node references an existing plan,
workspace, input matrices and distinct output matrix. The graph retains these
handles and the context; destroy returns BUSY while a graph holds a resource.
One output has one producer. Reads of a produced value must follow its producer,
so cycles, forward references and overwriting a logical value are rejected.
Shared read-only values and diamonds are supported.

Creation and execution propagate every node's exact bound through the existing
axis/moment/scalar proof implementation before arithmetic begins. Proof metadata
is copied once per external input; matrix payloads are never copied for planning.
This catches downstream range exhaustion before earlier nodes overwrite outputs.
Execution uses existing synchronous GEMM and its publication/identity guard.
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
prefixes, reassociate arithmetic or infer wraparound semantics. All referenced
intermediates remain allocated until callers release them after graph destruction.
Liveness-based allocation reuse, explicit conversion/base-extension IR nodes,
ragged inputs, backwards materialization demand, cost-based representation
selection and asynchronous completion remain open. Preflight and ordinary GEMM
both perform admission; this baseline exposes that setup cost rather than
claiming a measured dispatch or proof-work reduction.
