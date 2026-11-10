# Finite-ring Freivalds verification

`rns8_verify_finite_product` is an opt-in, synchronous, read-only verifier for
`A(M,K) B(K,N) = C(M,N)` in the explicitly selected finite ring or field.
CPU reference and direct HIP use the same SplitMix64-v1 binary challenge stream.
The implementation and fixtures are authored; consolidated qualification is
pending. No speedup or hardware qualification is claimed here.

## Contract and probability

The caller packs all three matrices before invoking the verifier and excludes
concurrent mutation/destruction until it returns. The verifier admits current
storage, matching semantics/moduli/backend/device, shapes, writer reservations
and optional expected generations before generating any challenge. Its report
binds the result to all three instance IDs and content generations. An existing
reader reservation is allowed: the operation writes none of the matrices.

For each trial it generates a binary vector r, computes B*r, then compares
A*(B*r) with C*r modulo the ring modulus. The arithmetic reduces every term;
its largest unsigned intermediate is 65280 even for modulus 256. All stored
bytes are checked against the unique centered representation, including entries
multiplied by zero. Malformed storage takes precedence over a mismatch.

For a fixed nonzero error matrix over **any ring**, choose a nonzero coefficient
in an error row and condition on all other challenge bits. At most one setting
of the remaining binary bit can make that row zero: otherwise subtraction would
make the chosen nonzero coefficient zero. Thus independent uniform binary
challenges give an ideal false-acceptance bound of 2^-t after t trials. Composite
moduli and zero divisors do not invalidate this argument. This is not the
q^-t theorem for uniform field-valued challenges.

Version 1 uses a user-supplied **replay seed**, not unpredictable fresh entropy.
SplitMix64 is not a cryptographic PRNG. The report's ideal probability describes
the independent-bit theorem; it is not an information-theoretic guarantee for
the generated deterministic sequence or for an adversary who knows the seed.
The fixture explicitly constructs a nonzero error invisible to a known challenge.
No probability multiplication across residue planes is performed.

Passing certifies only the tested congruence under this sampling model. An
integer error equal to the modulus is invisible. Exact integer equality needs
an independently established error bound and sufficient coprime modulus product,
or an exact integer projection. Neither is implemented by this finite-only API.
Bounded, exact-wide and wraparound matrices are rejected, and default GEMM,
export and AUTO selection never enable probabilistic verification implicitly.

## Options and outcomes

Initialize both options and report `struct_size`/`abi_version`. Set sampler to
`RNS8_VERIFY_SPLITMIX64_BINARY_V1`, repetitions to 1..1048576, and the explicit
workspace budget. Set `RNS8_VERIFY_CHECK_GENERATIONS` to compare all three
`expected_generations`; otherwise they are captured in the result. The context
and matrix lifetime rules of the ordinary synchronous API apply.

`RNS8_SUCCESS` publishes `passed=1`; `RNS8_VERIFICATION_FAILED` publishes
`passed=0` and the number of trials completed before the first mismatch. Invalid
arguments/storage, insufficient workspace, busy writers, unsupported contracts
and backend errors preserve the report. They never publish a partial pass.
The report's exponent represents the ideal bound exactly; the floating value
is clamped upward to the smallest positive normal double below 2^-1022, never
to zero. A mismatch is a detected error, not a false-acceptance event.

The legacy `rns8_freivalds_verify` adapter uses this same finite-only replay
contract with unlimited logical workspace. Its scalar probability has the same
ideal-model meaning. Null legacy operands and other semantic domains retain
unsupported status. Prefer the versioned API for generation and sampler metadata.

## Memory and execution

CPU scratch is N challenge bytes plus 4*K projection bytes. Direct HIP uses the
host challenge, a device challenge, device projection and host/device status
words: 2*N + 4*K + 8 bytes. Budgets cover these logical array payloads, excluding
allocator bookkeeping and fixed stack metadata. Checked matrix storage and
ptrdiff-sized products precede allocation and launch. Scratch is allocated once
per verification and reused across trials.

HIP launches a B*r kernel followed by an A*(B*r)/C*r comparison on the same
ordered stream, then reads one status word. Matrices stay resident. The host
copies a new challenge each trial. This first implementation assigns one thread
per row with a grid-stride loop; it is a correctness-first projection path, with
no claim of bandwidth-optimal reduction or large-shape performance.

## Consolidated qualification coverage

Authored fixtures compare complete products to an independent integer reference
for moduli 2, 4, 6, 251 and 256, with K=65 and the K=65537 reduction boundary.
They cover replay, deliberate corruption, preservation, budget boundaries, busy
writers, stale generations and malformed centered bytes on CPU and HIP. CPU
fixtures exhaust every two-column error and binary vector for several prime and
composite rings, all-ones cancellation, known-seed nullspaces, modular blindness,
unsupported domains, invalid counts/samplers and high-repetition bound underflow.

The authored [public verifier calibration](finite-verifier-calibration.md)
provides verification-alone, compute-plus-verify and complete modular reference
controls, including allocation, random-vector generation, transfer, projection,
synchronization and report construction. Timings from the kernel projection hook
alone do not represent the public verifier. Check-plane controls, execution of
the new benchmark, measured results and production promotion remain pending.
