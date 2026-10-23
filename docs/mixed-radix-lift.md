# Mixed-radix exact suffix candidate

The private HIP lift wrapper now accepts an explicit mixed-radix candidate.
Public CPU/HIP lift and graph behavior retain the existing default. Both routes
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
false magnitude proof leaves all residues unchanged. Execution and performance
qualification remain pending during frontloaded development.
