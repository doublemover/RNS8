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
