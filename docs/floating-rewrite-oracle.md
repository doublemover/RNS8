# Offline floating rewrite oracle

`tools/floating_rewrite_oracle.py` implements the issue #45 host-only increment.
It decodes all sixteen E2M1 bit patterns and E8M0 codes 0..252, widens scale-code
addition, and evaluates separate binary32 operations with gradual underflow.
Fold and balanced trees are explicit; every leaf must occur exactly once.
An integer/rational round-to-nearest-even state machine is compared with a
binary64 host evaluator that rounds each operation through binary32 storage.
The hoisted form applies the combined exact power of two after the unscaled tree.
No FMA, FTZ, cross-block accumulation or native matrix opcode is modeled.

Reports bind the decoder/checker version, inputs, scales, topology, zero/NaN
policy, denominator and signed destination width. They separately report algebraic
equality, floating rewrite equivalence, finite exact recovery, physical layout and
target execution. Verdicts concern the supplied input block; they are not an
exhaustive floating instruction certificate. Matching infinities can establish
rewrite equality for that block but fail finite recovery. NaN equivalence remains
unknown. Recovery requires the finite exact numerator, integral denominator
cancellation and destination range before any modular reduction.

Static guard fields report the integer-product cap, topology-specific sufficient
rewrite band and conservative finite-root bound requested in the issue. They do
not authorize runtime dispatch or prove the behavior of AMD instructions. A failed
guard does not select a different semantic program. The original computation must
remain the fallback when this model is eventually connected to a runtime owner.

Authored fixtures cover the supplied overflow/topology and underflow counterexamples,
all decoded element pairs, scale boundaries, malformed trees, k=1/2/31/32/33/64,
RNE ties and signed-zero normalization. Codes 253..255 are rejected. Execution is
pending the consolidated campaign. Existing integer certificate/composition reports
now carry distinct verdicts and versioned identities; assumed layout permutations
remain unknown, and a hardware evidence reference alone never qualifies execution.
