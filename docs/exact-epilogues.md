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
axis sums or covariance. Public [prepared/grouped execution](prepared-execution.md) and
[resident DAGs](resident-graphs.md) support this expression, including the
explicit fixed-prefix direct-HIP asynchronous subset. Bounded-i64/u64 and
finite-ring public forms, additional operators and public matrix-engine dispatch
integration remain open under #47. The API does not
automatically select this candidate for ordinary GEMM.

CPU and HIP API fixtures cover positive/negative and extreme coefficients, zero
terms, signed/unsigned exact exports, aliases and over-range rejection with
preserved output generation. Registered applicable tests are covered by the [dated validation](local-all-features-20261023.md).
The private resident-residue comparison has measured results below; the public
exact-wide API retains its separate output/range boundary.

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
with VS2026/HIP 7.1 for gfx1100. The [512³ matched residue comparison](7900xtx-large-throughput-2026-10-23.md#resident-residue-epilogue)
records a scoped improvement over the unfused direct baseline; applicable
correctness/ISA checks are in the [dated validation](local-all-features-20261023.md).
LDS staging/barriers and D reads remain real costs. Native exact-wide export,
other targets and automatic dispatch promotion require their own evidence.

VS2026/MSVC 14.51 marks standard type traits as non-specializable. The current
rocWMMA dependency specializes them for its floating wrapper types even when
only integer fragments are used. `rocwmma_headers.hpp` scopes Clang's
`invalid-specialization` diagnostic suppression to that vendor include on
MSVC 19.50+, leaving RNS8 diagnostics intact. Dependency probes include the
same wrapper after `<chrono>` so their STL/include environment matches the
actual translation unit. This does not qualify floating-point backends.

The fresh validation at `167aff0` passed the rocWMMA epilogue differential but
failed the no-divide ISA gate. Coefficients and modulus reciprocals now reduce
once per plane on the host and travel in a small by-value kernel argument.
The device epilogue uses only bounded INT32 products and reciprocal reduction;
it no longer computes full-width coefficient remainders per output cell. The
expanded signed-minimum fixtures passed on retry. The four remaining reciprocal
instructions were traced to signed/unsigned native-B packing, whose fallback
still used a variable modulus. Fixed reductions now cover all 28 backend ladder
entries. The subsequent full gfx1100 rocWMMA suite passed 430 tests, including
the unchanged no-divide ISA gate and a new Boost packing differential over all
28 planes, full-width inputs and padded tails. No timing gain is inferred from
the ISA repair alone.

## Zero-channel lowering and production-call controls

The CPU, direct-HIP synchronous/enqueue-only (prepared/graph) and explicit
rocWMMA candidate now share a canonical coefficient lowering. Original signed
64-bit integers are reduced once per modulus into a320-byte by-value table of
modulus, reciprocal and two nonnegative coefficients. There is no separate
coefficient cache/allocation/upload kernel. Runtime output cells consume that
lowering rather than performing full-width coefficient remainder operations.
Any emitted instruction savings still require ISA inspection.

A channel with `alpha mod p == 0` skips its complete dot product; a channel with
`beta mod p == 0` skips its D load. The branch is uniform across each plane's
wave/block, including rocWMMA's barriers. Both-zero channels fill zero. The
CPU path follows the same algebra. The reduced expression fits INT32 because
its magnitude is at most `2*255*128=65280`, even for arbitrary signed-byte
input residues. Existing generic/unfused controls remain available. No default
backend selection or probabilistic status policy changes.

Only **original integer alpha==0** skips the expensive AB range-refinement
traversal. A nonzero coefficient divisible by one modulus still contributes
its original magnitude to the final CRT bound. All original operands must
remain valid/current, have sufficient range certificates and obey ownership,
alias and completion contracts, even when their arithmetic contribution is
zero. Prepared and graph publication continue through their existing owners.

Native calibration accepts named `extreme`, `zero`, `copy`, `unit`, `general`
and `channel-zero` coefficients. The extreme full-width-input control is
preserved, and all three direct/rocWMMA/unfused routes retain the same original
integer expression identity. `temp/benchmark-cases/current/native-calibration/epilogue-coefficients.json`
supplies tail, medium and large controls; some large full-width expressions
have only a residue contract, not a claim that prefix20 uniquely reconstructs
the unreduced integer result.

`rns8-epilogue-calibrate` separately measures the **actual public exact API plus
padded signed-limb export**, on bounded small signed inputs permitting every
named expression within prefix20. Its resident/repack fixtures live in
`benchmarks/public-epilogue/`. It records context/plan/allocation/initial pack
setup, first call, disjoint pack/expression/export host phases and three bounded
sustained intervals. The expression phase includes the real proof and public
ownership work. A separate duplicate range-check diagnostic is never subtracted
from it. Device timing hooks are disabled. Full Boost integer output, every
residue channel and padding are checked outside production intervals; JSON
assembly and reference checks are also outside sustained intervals.

`tools/collect_epilogue_calibration.py` rotates/reverses case order between
independent processes, binds raw receipts to executable/build/device identity,
and reports process medians/ranges without pooling distinct coefficient or
residency workloads. Native resident-only and complete-public-output timings
have different boundaries and cannot be substituted for one another. Tests,
GPU compilation, campaigns, emitted ISA and profitability are all unverified
for this change; source term counts are not measured device instructions.
