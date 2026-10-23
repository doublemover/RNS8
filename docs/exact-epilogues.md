# Exact-wide linear GEMM epilogue

`rns8_gemm_exact_axpby(ctx, plan, A, B, D, C, alpha, beta, workspace)` computes
`C = alpha*A*B + beta*D` with explicit exact-wide signed or unsigned plan
semantics. CPU-reference and direct-HIP backends are supported. Unsigned plans
require nonnegative coefficients; signed plans accept the full int64 coefficient
domain, including INT64_MIN. C must be distinct from every input. Read-only
inputs may alias when their descriptors are compatible.

All inputs must already have current RNS storage and exact proofs covering the
plan's prefix. D has the output shape and the same basis, semantics and storage
contract. The API validates ownership, descriptors, workspace and device address
domains before writes. It computes the existing exact GEMM magnitude bound once,
then admits `abs(alpha)*bound(AB) + abs(beta)*bound(D)` against strict signed or
unsigned basis uniqueness. The bound uses multiprecision coefficient magnitude
and never negates INT64_MIN in a native signed type. It deliberately does not
rely on cancellation between AB and D.

Only the complete expression must fit the basis: intermediate AB is computed
modularly and is never published as an integer matrix. CPU arithmetic reduces
each dot-product step and both coefficients before composition; the HIP path
uses the existing fused modular output kernel with no AB global temporary.
Success publishes one new logical value and a conservative complete-expression
proof. An execution failure leaves C invalid; admission rejection preserves its
previous value. The normal matrix write guard owns that state transition.

The output currently retains a scalar magnitude proof, without inferred exact
axis sums or covariance. Prepared/grouped/graph execution of this expression,
bounded-i64/u64 and finite-ring public forms, additional algebraic operators and
matrix-engine epilogue integration remain open under #47. The API does not
automatically select this candidate for ordinary GEMM.

CPU and HIP API fixtures cover positive/negative and extreme coefficients, zero
terms, signed/unsigned exact exports, aliases and over-range rejection with
preserved output generation. Compilation and execution are pending; no fused
performance advantage has been measured.

## Explicit rocWMMA candidate

The private `rocwmma_exact_axpby_device` candidate consumes resident planar
signed-byte residues and uses one rocWMMA wave per 16x16 output tile/modulus.
Cooperative LDS staging handles arbitrary M/N/K tails. Independent INT32
fragments accumulate at most 65536 terms before a local centered reduction;
the carry is bounded by 128. The final shared reducer combines original integer
coefficients reduced modulo each channel with D, then stores each C byte once.
The kernel uses 512 bytes of input LDS and 1024 bytes of INT32 fragment LDS per
CTA, with no global INT32, AB-residue or packed-input temporary.

This is a feature-detected, opt-in rocWMMA baseline. It is not selected by the
public exact epilogue API. Its private residue-domain contract requires callers
to supply integer output range admission; modular equality alone does not
certify a unique reconstructed integer. Invalid shapes, address domains and
C/A/B/D overlaps reject before launch. Coefficients remain full-width integers;
negative coefficients and signed-byte inputs do not imply a field inverse.

Authored CPU/Boost differential fixtures cover prefix 1/3/20, complete and
masked tiles, K 1/17/65535/65536/65537, zero/unit/extreme coefficients, negative
D, aliases and output padding. The candidate and full test executable compiled
with VS2026/HIP 7.1 for gfx1100. Instruction/resource inspection,
runtime qualification and comparison with the direct scalar and unfused
resident baselines remain required. LDS staging/barriers and D reads are real
costs; no dispatch promotion or performance improvement is claimed.

VS2026/MSVC 14.51 marks standard type traits as non-specializable. The current
rocWMMA dependency specializes them for its floating wrapper types even when
only integer fragments are used. `rocwmma_headers.hpp` scopes Clang's
`invalid-specialization` diagnostic suppression to that vendor include on
MSVC 19.50+, leaving RNS8 diagnostics intact. Dependency probes include the
same wrapper after `<chrono>` so their STL/include environment matches the
actual translation unit. This does not qualify floating-point backends.
