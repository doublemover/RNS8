# Mixed-radix exact suffix candidate

The HIP lift wrapper accepts an explicit mixed-radix candidate. Ordinary
synchronous CPU/HIP entrypoints retain their existing default; the prepared HIP
owner selects wide Garner or shared digits explicitly. Both routes
reconstruct the canonical source value once, verify its centered/unsigned
magnitude against the caller's bound, stage all missing residues and commit
only after every cell passes.

The candidate retains the canonical Garner digits produced during that same
reconstruction. Each new modulus evaluates those digits with byte-base Horner
arithmetic instead of reducing a three-limb magnitude. Its unsigned result is
`X mod q`. A negative signed representative is `X-P`, where P is the complete
source product, so the candidate subtracts `P mod q` before centering. This
correction is essential: for source basis [256], canonical X=255 represents
-1, whose residue modulo 255 is 254, not zero.

Every Horner intermediate is below `255*256+255`; the product-modulus chain is
also bounded within 16 bits. Wide reconstruction remains necessary for the
unique-value/magnitude check. Digits are temporary registers/local storage for
one lift launch and are not a cross-generation cache. Twenty per-thread digits
can increase register pressure or spill; no speedup or storage reduction is
claimed without actual ISA/resource and timing evidence.

Authored independent Boost tests cover every source prefix 1-19, all suffix
moduli through prefix 20, canonical boundaries and negative correction. HIP
fixtures compare both candidates across every source/target prefix pair and
signed/unsigned semantics, preserve untouched suffix canaries and verify that a
false magnitude proof leaves all residues unchanged. Registered applicable tests are covered by the [dated validation](local-all-features-20261023.md);
performance and broader target qualification remain open.


## Prepared stream and graph ownership

`rns8_prepare_exact_lift_hip` returns the **existing** `rns8_prepared_execution`:
submit/query/wait/cancel/deadline/group/capture APIs and latest-epoch tokens are
shared with GEMM. Preparation freezes the source prefix, target suffix, signed
interpretation, maximum independent bound and allocation address. It allocates
one retained suffix/status region plus four pinned status bytes; the total known
scratch budget includes both, while opaque stream/event/graph memory remains
additional. No-op lift still has an explicit completion owner.

Submission stages the suffix, then launches a separate status-guarded commit
kernel and a four-byte status readback on the same stream. The commit cannot
write any suffix cell if any source cell failed its residue/range proof. Neither
allocation nor a device-wide synchronization occurs during replay. Logical
identity/generation does not change; source planes are untouched. Success only
publishes the larger materialized/proven prefix after observed completion.
Cancellation or failure leaves the earlier source prefix valid (unless an
upstream producer itself failed). Partial newly written suffix bytes are not
published as a larger representation.

Token-qualified chains can submit GEMM → lift → GEMM without host observation
between stages. Consecutive writers hold counted, event-ordered reservations;
observing the first producer cannot release the lift's writer. Actual range
metadata is staged separately from public currentness. Downstream kernels may
execute before host status observation, but a failed proof/integrity/canceled
producer prevents downstream publication. Independent readers and unrelated
writers cannot bypass these reservations. Refresh at stable addresses is allowed
only inside the frozen source-uniqueness/bound envelope; growth requires reprepare.

Authored new regressions cover both algorithms at every source/target pair,
untouched planes/canaries, no allocation on replay, cancellation, generation
preservation, and a three-stream producer/integrity/lift/consumer chain. These
new cases have **not been executed**; older dated receipts do not qualify them.


## Direct-HIP automatic prefix transaction

`rns8_gemm_exact_wide_hip_auto` is an explicit synchronous selector for uniform
exact-wide default-basis matrices. It computes the same conservative/axis bound
as ordinary GEMM, chooses the smallest sufficient prefix at or above the plan
floor, and honors a forced fixed prefix. It validates both source uniqueness
proofs even for a zero product. The caller's immutable plan is never replaced.

If an input needs extension, its existing planes are copied into a private
full-selected-prefix region; the selected wide-Garner or shared-digit lift
fills only that private suffix. GEMM also targets private staging. Only after
both input checks and the product succeed are resident suffixes and output
copied; all three metadata publications follow the all-copy success point.
Input integer identity/source generation is unchanged. A failed suffix copy
cannot damage original source planes, and its uncommitted region remains hidden.
A failed output copy invalidates C. An undrainable device error retains exclusive
A/B/C and workspace reservations until an explicit successful drain, without
unpoisoning the context or publishing results.

Workspace-owned staging is retained across replay. The explicit budget charges
full temporary inputs only when they need lifting, full temporary output,
4-byte alignment between regions, and one maximum reusable suffix/status region.
The four-byte host status and opaque runtime cost are additional; no native
input/output payload is downloaded. This route does not silently select a
vendor/matrix-engine backend or claim measured performance. Authored shared-input,
second-input-failure, no-allocation replay, padded export and ownership regressions
have not been executed.


## Shared lift/recipe cost harness (authored, not executed)

`rns8-lift-recipe-calibrate` and `tools/collect_lift_recipe.py` bind identical
integer/output boundaries across full-prefix manual execution, transactional
minimum-prefix execution, and retained full/minimum/cost-model recipe arenas.
Chains and diamonds exercise consumer-demand extension and shared operands.
The public `manual-scalar-first` control combines fixed-prefix, exact-value and
scalar-first proof flags. The `native-roundtrip` control reimports both native64
operands at every node with fresh source versions and exports each nonfinal
result to one limb; its common final export can be wide. Exact preflight rejects
an intermediate outside signed/unsigned native64 before HIP execution. Native
host buffers, matrices and workspaces are retained; audits run outside timing.
Internal roundtrip materializations belong to the same final-output cohort,
while eager extra observations remain separate. The collector prefers the
scalar-first public baseline when supplied.

Capture v3 separates per-replay logical native H2D/D2H payload and retained
native host buffer payload from actual HIP copy bytes/calls summed across the
warm samples. It records the fixed-plan proof policy and rejects older capture
schemas. Layout analytical prices use cost schema 2 and physical shared B
capacity indexing. These source additions await integrated validation.
The explicit analytical cost model is unqualified; it is recorded in each
capture and cannot be mistaken for a calibrated or AUTO-promoted profile.

Separate one-product cohorts compare wide-Garner, shared-digit, and captured
prepared lifts with a private CPU-lift control that transfers source residue
planes down and only the new suffix up, and with producing the wider basis from
the start. No-op lifts are included. This private residue-transfer control is
labeled; it is not a public native-host import API. All routes finish with the
same padded three-limb export and independent full-integer audit. Extra eager
intermediate observations live in a different cohort and never establish a
like-for-like speedup.

Captures retain cold setup, first use, complete phase timings, p50/p95 and three
sustained bursts, actual HIP allocation/free counters around warm samples,
selected prefixes, arena/retained-lift/automatic-staging bytes, and explicit
payload boundaries. The collector rejects recipe replay allocation and
independently checks every selected prefix against propagated exact bounds.
Immutable artifact/build/target/environment cohorts require independent
processes. Break-even reuse is computed only when a matched measured replay
saving is positive; otherwise it remains unknown. These are enabling tools,
not results: no tests, benchmarks or GPU qualification were run.
