# Runtime product arena

`RuntimeProductRecipe` is an internal synchronous CPU/direct-HIP executor for a
fixed topologically ordered exact-wide product DAG. It accepts current public
matrix inputs and publishes one final public matrix. Inputs must share explicit
signedness, the default basis, a common prefix and modulus-major storage. It
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

The explicit budget covers aligned intermediate payload storage, not retained
public input/output matrices, host proof/descriptor metadata or runtime allocator
overhead. `arena_bytes` and the sum of unreused logical output bytes are exposed
separately; alignment can make tiny cases larger than that unaligned sum. Lifetime
holds are released on destruction; execution/destruction require ordinary external
host exclusion. Concurrent execute attempts return BUSY.

Authored CPU cases cover linear and diamond reuse, all overlapping live intervals,
budget boundaries, lifetime/currentness and replay. A HIP diamond uses independent
Boost full-output comparison. They have not been executed in this frontload.
No speedup or global allocation optimality is claimed.

This closes the previous absence of an executable shared-arena adapter for this
bounded subset. It does not implement automatic precision/layout selection,
conversion/lift nodes, ragged inputs, public graph arena integration, multi-stream
execution or an exhaustive prefix/layout optimizer. Those remain separate work.
