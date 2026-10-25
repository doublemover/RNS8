# Offline typed recipe admission

`tools/planning_recipe.py` adds a typed provider boundary around the existing
arithmetic verifier and resource-constrained predictor. It does not launch
kernels, alter public planning, create reviewed cache entries or qualify a
target. Providers enumerate concrete stage descriptions; the central gate
admits recipes before consulting costs.

The records distinguish problem shape/strides, full expression and output
boundary; generation-bound input range/materialization facts; supported
kernel/certificate/layout combinations; calibrated rates; complete tile/plane/
K-slab coverage; and aligned temporary storage. Initial expressions are `A*B`
under finite-ring/field or bounded signed/unsigned semantics. Other expressions
reject explicitly until their composed proof and recipe are supplied. An
epilogue cannot reuse an ordinary GEMM identity.

Every basis plane needs its ordered algebraic certificate. The actual current
operand envelope must fit each bounded-integer certificate, and the chosen K
slab must respect every instruction bound. Kernel capability records bind
target/compiler/artifact, certificate, opcode, layout and instruction-evidence
digest. These records consume evidence identities; a supplied digest is not
itself verified hardware evidence. Cost environments must match the target
artifact, and every stage symbol and work term needs calibration. Use
`profile_from_calibrations` to apply the existing independent-run, ISA and
reference admission checks when importing measurements. CPU mock fixtures
construct explicitly fabricated profiles instead.

Capabilities also declare the actual kernel output boundary. Every terminal
stage must produce the requested boundary, so renaming a residue-only recipe's
result to native integers cannot omit reconstruction. Duplicate symbol/proof/
layout capability keys reject as ambiguous rather than allowing ordering to
select an output policy. Providers still supply complete conversion dependencies;
the boundary check does not synthesize missing stages or hardware evidence.

Coverage is checked as disjoint half-open ranges over the full flattened
output-tile/plane/K-slab domain. Duplicate arrivals cannot hide missing work.
Address admission conservatively checks 64-bit element spans. Providers must
account for conversions, imports/exports and the contract's declared reuse in
their work/stage descriptions; the boundary does not invent amortization or
infer omitted kernels.

Workspace queries return success, unsupported, overflow or unknown. Failure
never becomes zero scratch. Successful queries assign aligned nonoverlapping
arena regions with checked 64-bit arithmetic; padding counts against the
budget. The default arena is conservative and does not reuse storage across lifetimes.
Each region names its producer and direct consumers, while prediction retains
the producer allocation through all dependent consumers. A retained region
survives to the end of the graph. The model's lifetime peak and the arena's
actual required bytes are reported separately.

Shortlists are deterministic for the same typed inputs, with ties resolved by
workspace then recipe name. Cost changes can reorder mathematically admissible
recipes but cannot waive failed proofs. Binding rechecks instance, generation,
range and materializations before a caller could submit. This is an offline
integration boundary, not a C++ runtime provider or asynchronous submitter;
those adapters and qualified calibration cohorts remain pending under #30.

Authored mock-provider tests cover deterministic serialization, cost-only rank
changes, missing instructions/calibration, stale bindings, insufficient CRT
range, incomplete/duplicate coverage, workspace fallback, alignment overflow,
unknown query results and malformed numerical stage fields. Execution remains
pending during frontloaded development.


## Dependency-proven arena reuse

A provider can explicitly set `ResourcePlan.reuse_ordered_lifetimes=True` to
reuse temporary bytes. The allocator proves a strict dependency path from the
old producer and every declared reader to the new producer. Unordered branches,
a consumer writing during its own read, and retained values cannot share those
bytes. It does not rely on predicted completion times, equal timestamps, stage
list order or incidental serialization on a resource.

Allocation uses deterministic aligned first-fit placement against all conflicting
regions. Counts and aligned endpoints remain checked against uint64 capacity;
large mutually exclusive regions need not fail because their unshared sum would
overflow. The result exposes each region's offset and the physical high-water
size. The predictor charges one retained arena of that size, rather than counting
aliased regions as independent allocations. No runtime allocator or route is
changed by this offline option.

Authored fixtures cover ordered reuse, forks, retained results, producer/consumer
overlap, reversed declarations, uint64 limits, cyclic graphs and predictor
accounting. They await the consolidated validation pass. Runtime adapters and
measured planner promotion remain separate work.

## Runtime discovery and shortlist (implementation, not qualification)

`rns8_query_operation_support` is a read-only, versioned provider query. It
separates source inventory, compiled availability, context observation and the
always-independent correctness/performance qualification fields. It does not
probe a GPU, allocate device storage, read a reviewed cache or authorize caller
proof assertions. Concrete methods, representations, coefficient signedness,
full padded spans, execution modes and checked scratch are explicit. Ordinary
plan creation, prepared admission and narrow/bilinear owners consume the same
provider; their existing actual-value, generation, pointer, alias and target
binding remains mandatory. Opaque runtime/graph setup is unknown, not zero.

`rns8-inspect --operations [--json]` renders the same records for a documented
sample problem; it is **not** a universal support matrix or an on-device receipt.
The C++ `operation_support` wrapper preserves all rejection records.

`rns8_plan_pipeline` returns a non-promoting shortlist of executable ordinary
plan descriptors and stage DAGs. For uniform exact-wide input it compares the
minimum unique-integer prefix with a possibly stronger prefix satisfying
`4 B <= M` for the real quarter-CRT owner. A resident output requests no CRT.
It charges native import, packing, per-plane multiply-add, reconstruction and
readback, with retained-A/B and setup-reuse contracts. Calibrations bind backend,
exact target and this build identity; missing/stale rates produce unknown cost,
not a zero-cost winner. Positive finite lower/upper rates schedule dependencies
and shared host/copy/compute resources explicitly. Costs cannot alter arithmetic
or resource admissibility or create an AUTO reviewed-cache record.

Memory admission uses a conservative **simultaneous live payload sum**, not the
peak of optimistic/slow endpoint schedules. This remains sound for interior
rates and deliberately reports opaque setup separately. The budget is explicit
payload storage, not a claim about process RSS, allocator overhead, library
caches or graph internals. A shortlist still requires ordinary plan/owner
creation and current-value binding; hypothetical range flags never certify an
actual matrix. CPU table-driven parity, budget, stronger-prefix, stale-cost and
resident-boundary regressions are authored; execution and hardware/model
qualification are deferred.
