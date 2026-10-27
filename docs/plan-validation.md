# Immutable plan validation

Plan construction now performs exhaustive structural/lowering validation before
publishing the opaque handle. It creates a shared const contract containing the
workspace fingerprint, canonical structural key and active tile counts for each
prefix. Public plans have no mutation API: their owned bounds, masks, schedule
and lowering metadata remain immutable for the handle's lifetime. Input arrays
are copied; changing those caller arrays does not change an existing plan.

Ordinary validation, export and prepack queries use the construction validation
and cached fingerprint. HIP workspace checks compare precomputed active counts
instead of revisiting every tile for every prefix. Device/context, matrix input
generations and currentness, output ownership, workspace fields/capacities and
value-dependent range admission remain per-call obligations. No hash serves as
an arithmetic proof.

A workspace retains its original immutable contract and one most recently
admitted equivalent contract. A distinct plan first passes fingerprint rejection
and full canonical-key equality. Its retained identity then permits constant-time
compatibility checks on repeat use. This preserves equivalent-plan reuse and
survives destruction of the original plan without retaining a dangling pointer.
Switching among more than two equivalent plan objects may repeat the structural
comparison; no unbounded global interning cache is introduced. Calls sharing a
workspace continue to require external host serialization.

`revalidate_plan_schedule_contract` is the exhaustive internal diagnostic path.
Internal corruption fixtures explicitly discard the seal before changing private
plan fields and retain their rejection coverage. Internal builders must discard
the seal before modification, then fully validate and publish a new contract;
schedule configuration does this invalidation. A changed contract never updates
an existing workspace's uploaded descriptors in place. Public schedule replacement
requires a new plan and compatible workspace. Copying a private plan to modify it
also requires discarding its inherited seal.

Construction uses RAII so exceptions during array, string or contract allocation
do not leak a partially initialized plan. The structural key duplicates encoded
metadata once per plan; this trades preparation memory for avoiding repeated
traversals. Fingerprint values retain their previous algorithm and encoding.

Thread-local diagnostic counters distinguish exhaustive validation, fingerprint
traversal and distinct-plan structural comparison. Authored tests use a 16,384
square, thin-K plan with 16,384 tiles and full all-false masks, repeated original/
equivalent workspace checks, forced hash collisions, caller-array changes, stale
workspace fingerprints and corrupted coverage/prefix/mask fields. Registered
applicable execution is covered by the [dated validation](local-all-features-20261023.md). Zero traversal counts are a work-count assertion;
first-preparation, steady-state host and end-to-end GPU timings remain unmeasured.

## Typed arithmetic guard (implementation; execution deferred)

Plan construction now checks an exact-integer, topologically ordered proof DAG
before sealing. The declared ordinary lowering binds the ordered basis, signed
input alphabet, K slab, prior residue, physical intermediate widths, explicit
modular reductions, semantics, output-range obligation, compiler manifest and
lowering identity. A uint64 wrap operation is a distinct modular node, not a
fictional exact 65-bit temporary. Native 192-bit software accumulation and the
small-K uint32 wrap diagonals have separate contracts. Full-integer recovery
still requires the existing current input/output-range proof; an algebraic guard
never substitutes for that proof.

`rns8_get_plan_arithmetic_certificate` exposes five independent verdicts and the
SHA-256 identities. Algebraic/finite exactness is conditional on the declared
input contract. Layout and target execution remain **not run**; floating rewrite
is **unsupported** for these integer guards. Evidence strings cannot promote a
verdict. Unsupported node kinds, overflow, noncoprime bases, nonintegral division,
modular-to-integer reinterpretation and noninjective layout maps fail closed.

The certificate digest is in the version-5 autotune key; its complete canonical
encoding is also in the workspace structural key, so a digest is not the sole
equivalence test. Native prepared groups recheck the shared slab invariant under
a distinct native-encoding/WMMA lowering and bind each child's source instance,
generation and encoded-B layout plus the group schedule into their own digest.
Submission consumes immutable guards without symbolic checking per output cell.
Existing reviewed version-4 cache entries do not silently qualify these builds.

Regression sources cover standard SHA-256 vectors, unsafe fused dots, safe
reduce-before-combine, domain/division/layout rejection, K slab boundaries and
public immutable identity queries. They were **written, not executed** in this
implementation pass; host C++ syntax checks are not HIP compilation, device
correctness, ISA evidence or scientific qualification.

The existing offline `ArithmeticCertificate` also has a typed floating-contract
extension, rather than a second integer checker. It decodes OCP E2M1, E2M3,
E3M2, E4M3 and E5M2 plus E8M0 scales from the
[OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
OCP FP8 is not treated as AMD FNUZ. Code tables must equal the declared logical
integer tables on an explicit binary grid. The guard covers stored operands,
all product/reduction partials, per-block exponent rebasing, the initial carry,
every outer accumulation and integer destination recovery before reduction.

The precision/exponent/rounding/denormal model, fold/balanced tree, scale-placement
operation, full block schedule and compiler/target identity are serialized.
Generated lowering/model artifact digests are recomputed; changed assumptions
without regenerated artifacts fail closed. Only the **new block** can be rebased
before addition: applying its new scale to an old accumulator is unsupported.
The conservative rational grid proof does not infer exactness from cancellation
or identical exceptional floating results. The extended concrete two-evaluator
rewrite oracle separately models a complete multi-block outer sequence.

These are offline, explicitly declared binary models, not a hardware instruction
qualification or default floating route. Authored codebook, stale-artifact,
scale/precision/FTZ, overflow, destination and outer-carry fixtures remain unrun.

## Actual AUTO decisions and one-shot plan ownership

`rns8_get_plan_auto_decisions` returns the bounded trace captured during the
actual reviewed-cache selection pass. Each considered provider records a typed
reason, disposition, key digest, workspace requirement and an admitted median
only when one exists. Semantic/modulus/tile exclusions, absent compilation,
probe failure, absent/unvalidated/runtime-mismatched entries and workspace
mismatch are distinct. Losing eligible hits report slower or tied—not an
invented absence. The builtin backend is included in the actual candidate set.
A CPU baseline context records that accelerator candidates were not evaluated.
Explicit-backend plans say AUTO was not evaluated. Querying never probes or
rereads a later cache snapshot.

Benchmark metadata captures that immutable trace from its queried plan and
projects rejected candidates from those records. Missing traces say unknown;
they do not manufacture a backend list. The schema checks reason/disposition,
observed-hit consistency, candidate-set completeness and measured ordering.
This is an explanation of existing reviewed-cache authority, not an analytical
shortlist granting itself promotion.

The bounded signed/unsigned one-shot shortcut now hands its already admitted
plan back to the resident owner when prefix nine is inapplicable. A selected
minimal prefix stays minimal; the fallback no longer destroys/seals the same
plan twice. Success/error paths retain ordinary single ownership. Authored CPU
ownership and HIP low-prefix/fixed-nine controls check sealing counts and exact
outputs. Their execution and one-shot latency measurement remain deferred.
