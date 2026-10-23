# Prepared execution and completion ownership

`rns8_prepare_gemm` creates a reusable binding for one ordinary dense GEMM.
It validates the plan, workspace, operands and output ownership, records input
instance/generation pairs, and retains the context, plan, workspace and matrices.
Destroying any retained handle returns `RNS8_BUSY` without changing the handle.
Destroy prepared executions first, then retry resource destruction. Shared A/B
is retained once. Multiple prepared executions may retain the same resource.
Users of C++ RAII handles must likewise destroy the prepared execution before
the wrappers that own its resources.

The current API is synchronous. `rns8_execute_prepared` invokes the existing
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

This is the first synchronous slice of issue #43. Explicit streams, retained
in-flight reservations, dependency tokens, fixed graph resource envelopes,
data-ready versus status-ready events, graph replay and asynchronous wait/query
remain separate implementation and qualification work. No speedup is claimed.
