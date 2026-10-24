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
is the only scratch allocation. Each issued phase is drained, and the output
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

Still required before promotion: compiled scalar binary64 qualification, concatenation
layouts and matrix-engine lambda derivation, asymmetric matrix-engine decomposition, matched cold
and warm calibration, resource/ISA receipts, and target-specific qualification.
No decomposition cache or public ABI is introduced. CDNA FP64 qualification
requires the deferred hardware; that does not substitute for implementing the
remaining host and GPU candidates.

## Explicit packed-byte dot-product candidate

An opt-in gfx1100-family candidate lowers each canonical word into eight
unsigned bytes, forms all 64 byte-pair dot products, and groups them into 15
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
explicit baseline/packed-dot4 and admitted scalar binary64 schedules, ring/prime-field
and resident/cold controls, full independent exact references, swapped digit/slab
and bit-width cases, and strict provenance collection. These are authored
interfaces and fixtures, not execution evidence or a matrix-engine schedule.


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
