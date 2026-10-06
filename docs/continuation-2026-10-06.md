# Cloud correctness continuation — 2026-10-06

Base: `3855bb5b3e4dcb8d1111f0d4e478ebd19e2bb8b1`.
The earlier audit and its hardware limitations remain in effect.

## Tiled zero-output inference

Removed an active host-mirror scan from direct HIP tiled GEMM. It treated a
zero residue in modulus 256 as an integer-zero proof, treated any zero row or
column as proof of a whole zero tile, and initialized an unscanned operand's
mask to all zero. It also modified host schedule flags without rebuilding the
associated device schedule. The immutable, caller-proven plan and its existing
zero-row/column contracts are retained. No new inferred skipping is introduced.

Regression source covers mixed zero/nonzero tiles, nonzero multiples of 256,
and neither/one/both coherent host mirrors against CPU reference results.
Focused CPU checks: 14 cases, 30,331 assertions passed. Differential translation
unit receives a host syntax check; actual HIP compilation/execution remains
unrun. The implementation cache revision is 3 to invalidate earlier timing keys.
