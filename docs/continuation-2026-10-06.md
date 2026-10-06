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

## Exact-wide chain admission

The public API now records actual host-packed maximum magnitudes and initialized
prefixes. Multiprecision K*max(A)*max(B) admission prevents arbitrary-depth CRT
aliasing before computation. Signed admission is strict at twice the bound.
Known output proofs propagate through dense, grouped, incremental, prepacked B
and sparse operations; raw sparse planes receive their canonical/centered input
interpretation at packing. Larger allocated prefixes no longer imply that an
operation wrote those planes. Output aliases are rejected for exact-wide routes.
Malformed resident sparse repacks now stage before commit, preserving the old
source version, bytes and range proof on a late invalid group.
Public enum and descriptor layouts are unchanged; metadata is private to opaque
handles. No new user flags or runtime tuning policy were added.

Focused CPU exact-wide tests: 32 cases, 1,520 assertions passed before adding
the transactional malformed-sparse-repack regression. The original
focused failure was a regression-fixture assumption: max_prefix is a ceiling,
not an instruction to disable adaptive selection. The fixed-prefix test now
requests RNS8_PLAN_FORCE_FIXED_PREFIX explicitly; the failed log is retained.
Full consolidated CPU/sanitizer/tooling validation follows before checkpointing.

Remaining: conservative rejection can require a wider fixed prefix or a different
algorithm even when cancellation would make the answer small. Automatically
lifting residues to a wider ladder and tighter per-row/per-column range analysis
are not implemented. HIP compilation, hardware correctness and performance,
Windows ABI/runtime and LeakSanitizer remain unverified.

## Consolidated acceptance

- CPU Debug: 216/216 CTest entries passed.
- ASan+UBSan: 216/216 passed with enum sanitization retained and
  `ASAN_OPTIONS=detect_leaks=0`. The initial LeakSanitizer discovery failure is
  retained; this environment does not support its runtime inspection.
- Golden tooling: 49/49 passed.
- Updated direct-HIP differential translation unit: host C++17 syntax passed.
  No HIP compiler or GPU was used.
- Independent same-source public-API reproduction linked against the preserved
  audit baseline: `(UINT64_MAX²) × UINT64_MAX` returned success with an aliased
  residue representative. The repaired library returns `RNS8_RANGE_ERROR` and
  leaves the destination value 7 intact.
- The initial build-preset name error and initial regression-fixture failure are
  retained in the recovery logs. They are not represented as passing commands.

No timing campaign, hardware qualification, remote push, merge or deployment was
performed. Bounds are conservative; hardware/backend-specific validation is
still required before publishing optimized-path or production-readiness claims.
