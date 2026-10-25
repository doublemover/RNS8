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
