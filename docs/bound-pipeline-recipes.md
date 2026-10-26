# Bound whole-route recipes

`rns8/pipeline_recipe.h` supplies explicit C enumeration, selection, preparation
and execution over current matrix-owned values. This closes the execution gap
between the [offline planner](planning-recipes.md) and its existing operation
providers. It does not change AUTO or install reviewed cache entries.

## Routes and admission

The initial envelope is a uniform exact-wide signed/unsigned plan with current
inputs. Candidates select an ordinary prepared RNS product, generation-bound
encoded B with its ordinary native-grid consumer, a windowed split-K owner,
one of four narrow native methods, or a persistent native queue with optional
explicit-work locality order. Each uses the shared operation provider and the
existing implementation's static predicates, then checks concrete native
materialization, proof, prefix, stride and resource requirements.

The query launches nothing, allocates no device memory and returns caller-owned
records with required-count/truncation semantics. Each record has typed status,
reason, provider/build/target identity, payload and cost fields. An unsupported
route stays visible with its rejection. No fallback changes its identity.

`resource_status=1` describes the known logical payload floor: workspace,
encoded representation, schedule buffers and checked export storage. Opaque
HIP event/graph allocations, C++ control objects and allocator bookkeeping remain
unknown. `max_extra_payload_bytes` limits that declared payload envelope, rather
than promising an allocator-level process peak. The query bounds schedule and locality interval preparation before building
large host directories. Locality source intervals reserve their checked count
once, avoiding repeated vector growth. No elapsed gain is inferred from this
source optimization. External input/output matrices
and caller host destinations are outside this *extra* payload accounting.

## Pricing and selection

First enumerate without prices. A supplied price must match the candidate's
`cost_identity_sha256`, build identity and target. Preparation, complete producer
and final export have independent lower/upper intervals. Preparation is amortized
over the explicit reuse count. Prices declare modeled or measured origin and a
calibration identity; neither origin establishes qualification.

Only complete known costs enter selection/Pareto comparison. Unknown costs
remain unknown, rather than zero. Admission is independent of prices. The
separate binding identity includes input instances/generations and the output
instance; cost identity excludes process-specific handles but retains logical,
proof and materialization facts. No neighboring-shape interpolation occurs.

## Execution and lifetime

Preparation re-enumerates and validates the selected binding, ignoring forged
cost/status/resource fields. It acquires the real workspace, representation,
split-K, prepared-execution/group and graph owners. It retains context, plan and
all borrowed matrices. A changed input generation requires a new recipe.

Synchronous recipes can retain resident output or perform a final checked CRT
export using algorithm0..4 and explicit signed, unsigned or canonical limb
semantics. The first successful producer permits lazy checked-export preparation;
subsequent calls reuse/rebind it. Charge this first export setup to preparation
when collecting prices. Host row stride is bound in the recipe. Export stages
before caller writes. If export fails after successful compute, the successful
resident C remains current: publication is sequential, not atomic across both
boundaries.

Borrowed-stream and owned-graph recipes accept resident output only, for routes
whose actual prepared owners support that mode. Submission returns the real
child token, including an issued token on partial enqueue failure. Query, wait
and cancel use that child's completion machinery. Calls sharing handles must
be serialized; tokens do not retain the recipe. Ordinary unsupported graph,
narrow or split-K combinations reject before execution.

Destroy drains an issued epoch and closes children in dependency order. A failed
drain/free preserves the cleanup-only handle for retry. Preparation exceptions
retain complete owners on an allocation-free cleanup queue, retried by
`rns8_reap_pipeline_recipe_cleanup`; that retry launches no arithmetic.

Fixtures cover CPU/direct-HIP exports across all five CRT algorithms, resource
and cost identity, stale/forged binding, retained lifetimes, and real stream/graph
replay. This source pass is compile checked; these new fixtures have not run.
Measured routing, held-out regret, opaque peak accounting and target qualification
remain [issue30](https://github.com/doublemover/RNS8/issues/30) evidence gates.
