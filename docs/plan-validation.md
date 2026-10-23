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
workspace fingerprints and corrupted coverage/prefix/mask fields. Compilation
and execution are pending. Zero traversal counts are a work-count assertion;
first-preparation, steady-state host and end-to-end GPU timings remain unmeasured.
