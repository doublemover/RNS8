# Offline typed recipe admission

## Layout-aware materialization primitives

The private CPU/direct-HIP residue consumer accepts explicit uniform or tiled
A/B/output descriptors with a complete ordered basis and declared current
prefix. `convert_residue_planes_host` and the corresponding HIP operation copy
only `[first_plane, end_plane)`, allowing a retained packed view to grow without
reading unmaterialized higher planes or changing its physical capacity.
Basis/shape/range/alias validation completes before any write.

The HIP consumer stages 16x32 and 32x16 byte tiles and reduces its INT32 carry
every 32 K terms. Its explicit packed-dot path uses the shared signed-byte dot
implementation; the scalar-source build selects scalar updates in the source.
LLVM may still combine them into dot instructions: the `6ce489b` consumer emits
eight dot4 instructions in all three inspected configurations. It is therefore
not a physical scalar-versus-dot4 control. This is a
vector/LDS consumer, with no matrix-instruction or performance qualification
implied. It does not change ordinary GEMM routing.

Boost differential fixtures cover every default prefix, ordered-basis changes,
signed native extrema, partial packing, shape/pitch tails, padding poison,
K=65535/65536/65537 and rejected aliases. The CPU/ASan and gfx1100 differentials passed at `6ce489b`, including packed-dot
and scalar-control configurations. The [dated validation](local-validation-20261025.md#subsequent-materialization-integration)
records exact artifact and coverage boundaries.

## Cost-planned product materializations

The opt-in [experimental product API](../include/rns8/experimental_product.h)
adds a synchronous CPU/direct-HIP owner that chooses both compute prefixes and
uniform versus packed tiled B operands. It composes with the existing exact
ordered-basis and multiple-output product recipe contracts. Ordinary recipe
creation and default GEMM routing remain unchanged.

First query `rns8_query_product_materialization_identity` with the actual inputs,
graph and requested output IDs. Its successful query reports `admission_status`
separately: a range-exhausted graph names its first `blocking_node` and stops
bound propagation there. All admitted nodes expose their independent minimum
prefix. A price cannot waive that proof. The identity binds full ordered basis,
semantics, shapes, proof envelopes, topology, observations, prefix policy,
target and compiled build using SHA-256. A missing build manifest is unsupported.

Supply `rns8_product_materialization_profile` with caller-estimated integer
nanoseconds for uniform and packed GEMM, exact prefix lifting, and actual
incremental packing. Set `cost_schema_version` to
`RNS8_PRODUCT_MATERIALIZATION_COST_SCHEMA_VERSION` (2) and `node_costs_size` to
`sizeof(rns8_product_materialization_node_costs)`. Creation rejects an older
outer profile size, any other schema, or any other node stride before reading
node tables. This is a hard cutover; schema 1 tables are unsupported.

Each node has `uniform_gemm_ns[21]` indexed by compute prefix and `lift_ns[441]`
indexed by `old_prefix*21+new_prefix`. The capacity-aware tables are
`packed_gemm_ns[441]`, indexed by `capacity*21+compute_prefix`, and
`pack_ns[9261]`, indexed by `(capacity*21+old_prefix)*21+new_prefix`. Valid packed
GEMM cells satisfy `1 <= compute_prefix <= capacity <= 20`; valid packing cells
satisfy `0 <= old_prefix < new_prefix <= capacity <= 20`. Capacity means the
physical packed B allocation used by all selected readers, including readers
that compute at a smaller prefix or run before higher planes are packed. It is
not the current materialized prefix. Fill all cells with `UINT64_MAX` first,
then populate the valid cells actually priced; zero is a known zero estimate.
Unpriced and unused cells remain unknown. A node table is 81,312 bytes, so
multiple-node callers should allocate their tables on the heap. Tables and
labels are copied during creation. These prices are model inputs and do not
establish hardware qualification or update AUTO caches.
`FULL_PREFIX` preserves full compute prefixes; `MINIMUM_COMPUTE_PREFIX` allows
the planner to choose any independently admitted prefix.

One packed view belongs to each logical B value within a replay. Its physical
capacity covers all selected consumers. Conversion copies only newly demanded
planes, and all readers share the retained prefix. Each replay starts with no
packed planes and charges its conversions; cross-execution cache reuse is not
assumed. Raw B lifetime ends after its last actual read or pack growth, while
packed lifetime continues through its last packed reader. A reader and writer
in the same stage cannot alias scratch. Requested intermediate outputs remain
live through final publication.

Uniform and packed regions share one aligned arena. Admission includes its
physical high-water size and the retained lift/status/packet region; padding,
fragmentation and conversion storage count against the descriptor's hard
budget. External matrices and host proof/descriptor memory are separate. The
executor reserves the whole arena before execution and performs no new HIP
allocations/frees for packing or lifting during replay.

Small choice sets are enumerated exhaustively. Strict chains additionally use
cost dynamic programming keyed by the preceding prefix, shared external B
materialized prefixes and their final physical capacities. The frontier is
explicitly bounded; truncation or unknown prices remove its global lower-bound claim. Larger DAGs use deterministic feasible
seeds and bounded coordinate improvement. Resource admission always evaluates
the actual liveness arena. A bounded unsuccessful search reports unsupported,
not proof that every possible schedule exceeds memory.

`rns8_get_product_materialization_choices` reports each selected layout/prefix,
capacity, lifetime and arena region. `rns8_get_product_materialization_info`
reports estimated cost, search scope, retained bytes and completed pack calls.
Unknown prices never become free work or a claim of global optimality. Existing
execute, multi-output, rebind and destroy APIs retain their generation/range,
transactional publication and failed-cleanup rules. This owner adds no async or
capture support.

The `_ragged` query projects current ragged-source proofs into the conversion
destinations without changing their contents or generations. Its matching
creation API validates prices before gathering. A 3x5 ragged producer can feed
the 16x16 packed B consumer, and refresh can replace it with a 5x3 producer.
The shared ragged budget also charges retained source owners, conversion
destinations and peak gather scratch. Source mutation invalidates replay until
an explicit successful refresh; the recipe retains every borrowed owner.
The projected bound is the ragged owner's certified gathered magnitude, which
can be tighter than its source matrix's conservative envelope. Query and
post-gather identities therefore agree even when a narrower destination would
not fit the original envelope.

Authored fixtures cover chain and diamond selection versus exhaustive tiny
domains, hard budgets, growth/reuse, arena interference, unknown/overflowing
costs, rejected schema/stride/outer-size headers, capacity-sensitive shared B
prices, stale build identity, range-blocking squaring, multiple observed outputs,
default/experimental bases, both signedness contracts, mutation/rebind,
differently tiled ragged producer refresh, and HIP
allocation counters. The installed
[materialized product example](../examples/downstream-cmake/materialized_product_recipe.cpp)
checks the exported header/functions with static and shared linkage. Functional
execution of the earlier cost contract passed in the consolidated `6ce489b`
batch; that evidence does not validate the schema 2 cutover. Schema 2 source
and fixtures require integrated validation. Independent measured qualification
remains pending; passing correctness does not establish a gain.

### Matched materialization controls

The existing `rns8-lift-recipe-calibrate` driver adds `recipe-packed-full`,
`recipe-packed-minimum` and `recipe-layout-model`. They use the resident recipe
executor with the same final full-prefix three-limb output and independent
integer audit as `manual-full`, `recipe-full` and `recipe-minimum`. Full packed
preserves every compute prefix; minimum packed forces the packed consumer while
allowing prefix choices; layout model also allows uniform consumers.

`manual-scalar-first` uses the public GEMM API with
`RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_EXACT_VALUE_BOUND |
RNS8_PLAN_EXACT_SCALAR_FIRST`. It preserves full compute prefixes and tries the
scalar range envelope before axis refinement. The collector uses this public
control as its baseline when present, then `manual-full` otherwise.

`native-roundtrip` executes the same DAG through public native64 imports and
exports. At each node it imports both host operands with fresh source versions
(once when A and B are the same value), executes full-prefix GEMM, and exports
each nonfinal result to a retained one-limb host buffer. The final result uses
the common three-limb export and may exceed native64. Exact preflight rejects
any nonfinal value outside the chosen signed/unsigned native64 contract before
HIP execution. Matrices, workspaces and native host buffers persist across
replays; independent intermediate and final audits run outside sample timing.
These internal transfers contribute to the same final-output cohort. The
`eager-observe` mode requests additional externally observable outputs and
remains a separate cohort.

The model's integer prices are explicit analytical work estimates, not measured
stage calibration. Captures identify layout model v2, cost schema 2 and physical
shared B capacity indexing. The v3 capture requires logical native64 H2D/D2H
bytes per replay, retained native host payload bytes, and actual HIP transfer
bytes/calls summed over the declared warm sample count. Logical native bytes
are separate from device copies of residue/staging data; retained host bytes
count buffer payload rather than process memory. Captures also require a
materialization receipt (null for other modes): selected packed
operands, unique view capacities/lifetimes/arena offsets, packing calls, search
scope and estimated cost. The collector independently replays packing growth,
prefix lifting and the cost trace, checks tile padding, interference and the hard
retained budget, and rejects omitted or inconsistent receipts. Earlier capture
schemas are rejected. The native roundtrip and scalar-first sources and receipt
fixtures await integrated execution; earlier measurements do not validate them.

Small signed chain and unsigned diamond CTest cases exercise all three routes;
the 128x128 chain files support a larger sustained comparison. Existing
collection records setup, first call, production p50/p95, sustained completed
calls, export, exact audit and replay allocation counters. Compare matching
manual/compiled cases and charge setup amortization; these controls alone do
not supply held-out measured stage prices or a qualified speedup.

Explicit ordered experimental bases are supported by the CPU/direct-HIP product
recipe executor, including minimum-prefix lifting and multiple outputs. Every
input, output and rebind must carry the same complete ordered basis. The basis
and provider are part of private cost-profile admission, and HIP budgets include
the generic full-target lift staging and immutable packet. See
[experimental bases](experimental-bases.md) for the opt-in API and limitations.

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
Native-device and pinned-host import queries require `output_limbs=0` for
scalar native64 storage; limb views are export-only. Both import families reject
a nonzero width before compiled inventory and resource pricing, matching their
concrete transfer descriptors. Authored provider regressions also compare static
rejections with CPU synchronous/prepared epilogue and narrow/bilinear admission,
including preservation of a previously published output. Those additions await
integrated execution; they do not qualify unsupported devices or complete the
remaining representation, distributed and asynchronous provider gates.

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

### Explicit current-value fixed-basis plans

`RNS8_PLAN_EXACT_VALUE_BOUND | RNS8_PLAN_FORCE_FIXED_PREFIX` admits an exact-wide
plan with a 1–20-plane capacity chosen for current values, rather than for every
possible native64 product. Ordinary plan creation without this explicit opt-in
retains its full-native admission floor. The new flag is invalid for bounded,
finite or wrap semantics and without fixed-prefix execution.

This is **not** a trusted caller range assertion. The static arithmetic
certificate proves the physical modular lowering; its integer output bound is
unknown. Every ordinary/prepared/group/resident expression still binds current
matrix generations, materialized planes and engine-owned range proofs before
writes. Insufficient range rejects without replacing the previous output. An
exact epilogue proves the complete expression, not an unobserved product whose
coefficient may be zero. A changed input can invalidate an otherwise reusable
plan or prepared execution.

Before binding, schedule range bits are zero (unknown), and required/selected
prefix fields name the fixed capacity, not a proved output range. The policy is
part of the autotune key, arithmetic identity and immutable plan fingerprint.
The public pipeline shortlist emits these explicit descriptors; its supplied
magnitude is a planning estimate only and cannot bypass binding.

Public narrow, bilinear, native-queue/integrity/interop and lift-producer
calibrators now request this policy where their declared small-value workload
uses a short fixed basis. Their full-width and automatically selected controls
remain separately identified. This repairs the prior mismatch between authored
small-prefix cases and ordinary full-native admission; no measured result or
previous execution receipt is claimed. Regression source covers every capacity,
default rejection, illegal flag combinations, runtime range failure, unchanged
old outputs, stale prepared values and whole-expression epilogues. Execution is
deferred.


The [focused matched observations](focused-closeout-measurements-20261025.md#packed-storage-and-the-analytical-model)
now compare four shapes through 256³ across five processes per mode. Full-prefix
compiled graphs win against the public resident sequence, while packed B and
the analytical layout model lose to the best compiled control. The prices remain
unqualified. The new stage measurements/importer and native per-operation
roundtrip source below require their own validation and measurements. Physical
peak/capacity evidence and broader held-out selection remain open under #42.

## Measured resident stage prices

`rns8-product-stage-calibrate` measures completed production backend calls for
uniform GEMM, packed-layout GEMM, exact suffix lift, partial-plane packing and
final publication. The case explicitly lists candidate compute prefixes and
physical packed capacities. Packing includes every listed old/new pair; an
unmeasured cell stays unknown. The all-one DAG has an independent Boost integer
reference for every logical output, and each stage checks the complete physical
buffer, including untouched suffix planes and tile padding, outside timing.
This fixture qualifies no other dataset or shape.

Samples use `steady_clock` around synchronous wrappers with diagnostic hooks
disabled. Their boundary includes launch/status/synchronization and staged
suffix publication, but excludes fixture upload and oracle readback. Reported
transfer counters count wrapper copies; actual HBM/cache traffic remains
unavailable. The nominal clock tick is recorded separately from empirically
qualified timer resolution. Calibration wall time charges context setup,
warmups, samples, fixture transfers and audits; it excludes process startup and
JSON output serialization. Training is a cost when evaluating setup amortization.
The frozen file carries an integer `training_charge_ns` for all five training
processes. Matched summaries include it in setup and break-even projections;
conditional reuse of an already-trained profile is reported separately.

For example, after a clean build:

```powershell
python tools/collect_product_stages.py --binary build/rns8-product-stage-calibrate.exe --case benchmarks/lift-recipe/17-stage-training.json --runs 5 --output temp/product-training
python tools/collect_lift_recipe.py --binary build/rns8-lift-recipe-calibrate.exe --cases benchmarks/lift-recipe/17-chain-3-i-recipe-measured.json --stage-prices temp/product-training/prices.json --runs 5 --output temp/product-held-out
python tools/collect_product_stages.py --review-held-out-plan temp/product-held-out/plan.json --prices temp/product-training/prices.json --relative-tolerance 0.25 --output temp/product-held-out/discrepancy.json
```

The tolerance must be chosen before collection. Five independent training
processes produce integer nanosecond estimates and conservative Student-t
intervals over process means, with explicit independence/stationarity assumptions.
An interval reaching zero leaves that cell unknown. The importer binds the
graph, independently computed minimum prefixes, build, target, device name,
runtime, schema/axes, unique process IDs and capture/artifact digests. Parsing
is transactional. Malformed, stale or partial files cannot leave partially
applied prices, lower the exact minimum or install an AUTO route.

`recipe-measured` uses only these supplied cells and reports their immutable
content digest. Interval endpoints normalize to exact binary64 bit words before
sorted UTF-8 JSON hashing, avoiding C++/Python float-format differences. Capture
also emits the actual queried graph/build/minimum-prefix identity. Collection
independently checks that identity, replays the selected stage cost trace and
rejects pooling different price files into one measured-mode cohort.
Held-out processes must be disjoint from training and share one graph, build,
dataset and selected plan. The discrepancy report compares the entire measured
process interval against the expanded prediction, including final publication;
final CRT/export remains a separate common boundary. A failed prediction is
retained as a failed report. These profiles remain **unqualified estimates**:
no ISA/counter qualification, controlled clocks, new-shape generalization,
selection-regret success or performance promotion is inferred from collection.
