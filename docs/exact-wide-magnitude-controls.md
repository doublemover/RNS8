# Public exact-wide magnitude controls

The public persistent and host-batch benchmark keeps `uniform-small` unchanged
and adds `wide-two-limb-constant`, `wide-three-limb-constant`,
`wide-two-limb-mixed`, and `wide-three-limb-mixed`. These are explicit dense
single-GEMM exact-wide profiles, not changes to production arithmetic or the
modulus basis. Constant native operands are respectively `2^40` and `2^62`.
The mixed controls add seeded ten-bit magnitude jitter and, for signed inputs,
independent operand signs. At K=512 the constants produce exactly `2^89` and
`2^133`: magnitude bit lengths 90 and 134, requiring two and three magnitude
limbs. The profile name does not assert that every shape/mixed-sign result uses
that many limbs; captures report actual minima and maxima.

Before allocation/timed repetitions, an arbitrary-precision scan of the actual
inputs checks `B=K*max_abs(A)*max_abs(B)`, strict `P>2B` (signed) or `P>B`
(unsigned), and the requested signed/unsigned native limb width. This is an
additional fixture gate, not a bypass of the public pack/GEMM proof machinery.
After timed repetitions, independent Boost integer GEMM checks every output
limb, including high words, and every allocated output padding cell. Batch
receipts aggregate every task, not a selected sample. The input scan and oracle
have separate setup/postflight times and are outside repeated public-call
samples. Dependent chains do not receive a misleading single-product receipt.

`exact_wide_magnitude` records decimal exact bounds/product, input magnitude
maxima, actual output magnitude bits/limbs, storage width, negative outputs,
complete oracle coverage and timing scope. Schema checks reject incomplete or
arithmetically inconsistent receipts; comparison identity includes the profile
and the existing seed/distribution/shape/prefix/residency contract. Older
captures retain the historical `uniform-small` identity when the field is absent.

The `exact-wide-magnitude` scenario family contains a deliberately small paired
surface: 8x7x512, prefix20, three exported limbs, padded output, signed/unsigned,
small/constant/mixed profiles, public CPU/direct-HIP and cold/reused inputs.
These are collection recipes, not collected results or promotion evidence.
Independent process qualification and hook-disabled/sustained public timing
remain governed by the normal evidence policy and issue #31.

## Offline named range profiles

The legacy modulus-search report schema3 no longer substitutes product bit
length for semantic coverage. It publishes exact products and named strict
thresholds. Full unsigned64 requires `P>2^64-1`; the default ladder's eight-plane
product is only 13,057,518,384,392,951,040, so the first sufficient prefix is nine.
Full signed64 conservatively covers magnitude `2^63`; full signed native-wide
GEMM uses `B=K*2^126`, and unsigned uses `B=K*(2^64-1)^2`. Noncoprime prefixes
cannot certify a range. User-requested minimum product bits remain a separate
search diagnostic. No runtime basis or selector is changed by this report.

## Validation status

Regression sources and collection fixtures are authored. Only syntax/static
checks were performed during implementation; no tests, benchmarks, GPU work,
performance outcome or scientific qualification has been executed or claimed.
