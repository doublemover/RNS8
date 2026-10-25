# Word-size modular research baseline

This opt-in private path accepts canonical uint64 values and an explicit modulus
from 2 through UINT64_MAX. Ring and prime-field domains are separate; the field
request checks primality using deterministic 64-bit Miller-Rabin. A prime power
is a ring here, never an implicit extension field. The public finite-u8 API and
its centered byte storage are unchanged.

The CPU and direct HIP baselines compute each product by binary double/add with
modular reduction at each step. `a >= p-b ? a-(p-b) : a+b` avoids overflow even
when p is UINT64_MAX. No signed overflow, floating conversion, residue inverse,
or assumption of a wider GPU integer type is needed. This is a correctness
control, not a throughput optimization. Input matrices use explicit padded
strides. C cannot overlap either input, and padding is not written. The private
HIP boundary requires same-device valid allocations, as other private helpers
do; it is not an external-buffer ownership API.

Host descriptor validation proves spans and GPU index limits. A device scan
rejects noncanonical logical input cells before output starts. A four-byte flag
is the baseline scratch allocation; concatenated MFMA products add an explicit
coefficient budget described below. Each issued phase is drained, and the output
started flag distinguishes preflight rejection from an issued output kernel.
The timing hook measures the arithmetic kernel separately from the canonical
scan; any eventual production benchmark must also report the complete call.

The host decomposition planner enumerates (1,1), (1,2), (1,3), (1,4), (2,2),
(2,3), plus swapped asymmetric pairs. Integer binary search with Boost integers
computes ceil(p^(1/u)) and ceil(p^(1/v)); no floating roots are accepted. The
reported sufficient bound is lambda*alpha*beta+p-1 <= 2^t, evaluated exactly.
This boolean is not a backend certificate: it proves that inequality only and
does not certify a compiler, reduction schedule, rounding mode or instruction.
An independent digit splitting/recombination reference uses exact products and
modular scaling without division by residues, including composite moduli.

Authored fixtures cover full-width boundary arithmetic, prime and composite
moduli, rejection and output preservation, padding, swapped decompositions,
and complete Boost GEMM comparisons at bit widths 8, 16, 23, 26, 27, 31, 35,
39, 42, 48, 50, 52 and 64. They remain unexecuted during development frontload.

Still required before promotion: compiled scalar/MFMA qualification, executed
matched cold/reused/resident calibration, resource/ISA receipts and target-specific
qualification. The matrix-engine and concatenation source routes are described below.
No decomposition cache or public ABI is introduced. CDNA FP64 qualification
requires the deferred hardware; that does not substitute for implementing the
authored host and GPU candidates.

## Explicit packed-byte dot-product candidate

An opt-in gfx1100-family candidate lowers each canonical word into eight
unsigned bytes in the retained full-width control, forms all 64 byte-pair dot products, and groups them into 15
radix-256 diagonals. UDOT4 consumes four K terms per instruction with zero tail
lanes. Each fixed 4096-term slab bounds the largest diagonal by
`4096*8*255*255 = 2130739200 < INT32_MAX`; the hardware accumulator therefore
cannot wrap or saturate. Every slab is reduced modulo the actual target before
weighted recombination using overflow-safe modular multiply/add. Precomputed
15 radix weights are kernel arguments, not a persistent decomposition cache.
The target guard and capability probe reject unsupported hardware before writes.

The method works for composite rings and primes without inverses and retains
the canonical scan, alias checks and four-byte scratch of the baseline. It is
a packed-dot control, not a claimed WMMA implementation of the asymmetric
floating decomposition. Independent host digit-product fixtures and GPU
4095/4096/4097/8193 slab-boundary/full-width fixtures are authored. Its launch,
register traffic and modular recombination costs require measurement; no
speedup or AUTO selection is claimed.

The [word-modular calibration driver](word-modular-calibration.md) supplies
explicit baseline/packed-dot4/byte-WMMA and admitted scalar binary64 schedules, ring/prime-field
and resident/cold controls, full independent exact references, swapped digit/slab
and bit-width cases, and strict provenance collection. These are authored
interfaces and fixtures, with execution evidence pending.

## Unsigned byte-WMMA integer candidate

The private opt-in byte-WMMA candidate uses one wave per 16x16 output tile.
Canonical words use `ceil(bit_length(p-1)/8)` unsigned bytes, all byte pairs and
`2*bytes-1` radix-256 diagonals. A 4096-term slab bounds every diagonal by
`4096*bytes*65025`, at most 2130739200. Zero-filled K and spatial tails preserve
the complete product. Overflow-safe integer modular weights recombine diagonals
and slabs; prime and composite rings use the same exact arithmetic without
inverses. Both intrinsic operand signedness flags are false.

Only one diagonal's eight accumulators stay live at once. The implementation
rereads native operands for successive diagonals rather than retaining a packed
tile cache. Its target/builtin probe, canonical scan, integer weight generation,
status transfers and required drains remain part of each checked invocation.
Matched calibration records this full boundary alongside baseline and packed
dot4 controls. Tail, slab, full-width Boost, preservation and metadata fixtures
are authored, with no execution or performance qualification yet. This integer
candidate does not implement asymmetric floating matrix-engine decomposition.


## Explicit scalar binary64 asymmetric schedule

A separate private candidate takes an explicit left digit count, right digit
count and K slab (1-65536). The host recomputes integer ceil roots alpha and beta
and admits only `lambda * alpha * beta + p - 1 <= 2^53`. The request cannot supply
a trusted proof flag or substitute a floating approximation for this test. Failed
admission preserves output and reports RANGE_ERROR. Full uint64 moduli generally
reject this binary64 schedule; the integer baseline remains available.

For each output cell and slab, the kernel decomposes canonical uint64 inputs into
unsigned alpha/beta digits. It accumulates each digit-pair coefficient with scalar
binary64 FMA. Digits, products and every nonnegative partial sum are exact integers
within the representable range, so every FMA result is exact. There is no relaxed
error tolerance, sign cancellation assumption or division by ring elements.
Precomputed integer radix weights combine the coefficients with the existing
overflow-safe modular multiply/add, then reduce between slabs. Output stays in
the same canonical uint64 ring or prime-field domain.

This is a scalar binary64 executable schedule, not a WMMA or tensor-core claim.
It introduces no automatic selection or decomposition cache. The coefficient
arrays may spill; register pressure, emitted instructions and useful throughput
need actual compiler/device evidence. CPU bound/weight fixtures and complete
Boost HIP comparisons cover asymmetric pairs and slab tails with padding and
rejected-output preservation. All remain unexecuted in the development frontload.


The opt-in canonical-width dot4 sibling now specializes B=1..8 from the checked
modulus. It omits provably zero high-byte terms without changing integer product
semantics; the full-width dot4 and byte-WMMA controls remain distinct. See the
[matched calibration boundary](word-modular-calibration.md#canonical-width-packed-dot-control).

## CDNA FP64 matrix-engine and concatenated schedules

`asymmetric_fp64_mfma` is a separate compiled gfx90a/gfx942 candidate using
`__builtin_amdgcn_mfma_f64_16x16x4f64`, never a relabelled scalar FMA loop.
The [official builtin signature](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
and [CDNA3 instruction contract](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cdna/cdna3-dense-mfma-builtins.html)
specify one double per input lane and four double accumulators. Modifiers are
all zero. An actual device capability probe checks the compiled target before
output. No CDNA4, RDNA or unsupported compiler fallback is called MFMA.

A is `(lane%16,lane/16)` and B is its transposed coordinate map. Crucially,
FP64 C is `(4*reg+lane/16,lane%16)`, **not** the INT32 map with four consecutive
rows in each lane. `MfmaF64Layout` checks bijection at compile time; these
coordinates were independently derived from the same pinned AMD calculator
source cited in the [wrap64 mapping guide](wrap64-signed-radix.md).

Each pair starts at zero for each admitted slab. Canonical digits and all
nonnegative partial sums satisfy the exact integer inequality already checked
by the host; every result is at most 2^53. Rounding association therefore cannot
lose an integer bit under the declared FP64 contract. A final instruction with
fewer than four active K positions receives exact zeros. The certificate bounds
active terms, and separately reports padded instruction terms. The integer
conversion, modular coefficient weighting and slab merge use the same exact
non-inverting ring arithmetic as the scalar control. No floating quotient or
approximate modular reduction is introduced.

Two additional explicit choices batch the pair products into an expanded
`(M*u) x (N*v)` MFMA problem. `asymmetric_fp64_mfma_rows` uses word-major axes
`(i*M+row,j*N+col)`; `asymmetric_fp64_mfma_columns` uses cell-major axes
`(row*u+i,col*v+j)`. Each coefficient dot is reduced and weighted exactly per
slab before it is stored. A separate integer kernel merges the u*v canonical
weighted coefficients for each output. These are actual different physical
batching layouts, not inferred performance winners; swapped u/v fixtures price
both asymmetric orientations.

The concatenated methods budget `8*M*N*u*v + 4` scratch bytes, including the
live status flag. Allocation, capability, input-domain and coefficient-stage
failure leave output untouched. The output-started bit changes only before the
final merge. If completion is uncertain, coefficient storage is retained rather
than freed while potentially live. The fused method retains its four-byte
scratch boundary. Caller-owned input/output spans are outside this explicit
scratch budget. All methods remain private research APIs with no persistent
word cache or implicit public finite-u8 promotion.

Full Boost/tail/padding/domain/budget/layout regressions and cold, repeated-B,
resident, swapped-pair, slab±1 and sustained comparison fixtures are authored.
The selected ISA adapter binds the exact arithmetic/layout recipe and captured
artifact to explicit symbols, without asserting observed dynamic execution.
HIP compilation, all regression execution, physical FP64 mapping/exactness and
profitable-cohort or negative performance reports remain deferred.
