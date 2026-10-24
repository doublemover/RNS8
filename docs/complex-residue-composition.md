# Exact complex residue composition

Issue #59 starts with the private `hip_direct_gemm_complex_residues` experiment.
It accepts split real/imaginary residue views with a common explicit basis and
domain, independent component pitches, and either three or four real products.
It is not yet a public complex-integer API or an automatically selected route.

## Arithmetic contract

For three products, `P0=Ar*Br`, `P1=Ai*Bi` and
`P2=(Ar+Ai)*(Br+Bi)`. The result is `Cr=P0-P1` and `Ci=P2-P0-P1`.
Each sum is reduced and recentered in its ring before signed INT8 GEMM. Each
GEMM uses the existing direct backend's safe K slabs and produces centered
residues. Final combinations therefore add at most three byte residues; they
never add unrestricted near-limit INT32 accumulators. The identity needs no
division and supports every finite modulus from 2 through 256, including even
and composite rings.

The four-product control computes `Ar*Br`, `Ai*Bi`, `Ar*Bi` and `Ai*Br`, with
the same backend, staged dual output and final commit boundary. Neither route
exports intermediate integers. P2 is a ring value with no claimed standalone
integer uniqueness proof.

Signed bounded and exact-wide default-basis domains require original-component
magnitude certificates. With K terms, admission constructs both bounds in Boost:

```
real_bound = K * (amax_real*bmax_real + amax_imag*bmax_imag)
imag_bound = K * (amax_real*bmax_imag + amax_imag*bmax_real)
```

The common basis product must exceed twice each bound independently. Bounded
signed output additionally requires both bounds at most `INT64_MAX`. The caller
must bind the supplied certificates to the current input values. This private
primitive does not scan original inputs or create generation-bound proofs.
Finite rings require no integer uniqueness certificate. Unsigned integer complex
semantics, strict wraparound, alternate CRT admission, transposition, conjugation
and complex alpha/beta are not offered by this entry point.

## Memory and publication

Admission checks all six views, their common basis/domain, dimensions, strides,
qualified address ranges and every output/input or output/output overlap before
allocation or output writes. Shared read-only components are permitted. The
three-product scratch envelope is one compact sum panel per operand, three
compact product planes and two staged outputs. The four-product envelope is
four product planes and two staged outputs. Budget rejection occurs before
device work; failed admission is not a valid zero-byte plan.

Both outputs remain untouched until the final dual-store kernel. A shared
`commit_started` flag marks that boundary. Device stores are not atomic across
the pair: if the final launch or completion fails, the caller must invalidate
**both** components. Input lifetime/exclusion and atomic logical publication
remain the caller's responsibility. No transformed panel cache, public matrix
reader/writer reservation, prepared-group integration or asynchronous ownership
is claimed. Those are required before a public paired-value operation can ship.

## Validation and remaining work

The implementation passed its post-commit all-feature Windows build and full
validation at `51ae8d8`: 489 passes and three expected target skips across 492
registered tests. CPU tests cover independently derived real/imaginary bounds, strict uniqueness
boundaries, scratch limits, cross-component aliases, invalid bases and the 3M
identity over every byte ring with native extrema and values beyond 64 bits.
Incorrect input-sum and recombination signs must be detected by the reference.
The GPU differential compares both algorithms to independent four-product Boost
arithmetic, covering zero/pure-real/pure-imaginary products, strong cancellation,
full-width signed extrema, 70-bit resident components, odd tails, distinct strides,
padding/canaries, moduli 2/4/6/127/251/256 and K=65537. Default-basis outputs also
passed through the actual signed wide CRT export and were compared as full integers
against both independently computed complex components.

No performance benefit is asserted from three products alone. Matched fresh and
reused-input measurements must charge transformed panels, all launches, scratch,
joint commit and any final dual export. Public ownership/generation integration,
fault-path qualification, explicit operation metadata and those measurements
remain open under #59.

## Internal matrix ownership integration

The new internal `complex_exact_signed_gemm` connects the raw composition to
existing exact-wide signed matrix handles and a compatible workspace. It is
restricted to CPU reference and direct HIP. It obtains original-component
magnitude certificates from each current matrix's own exact-range proof;
callers cannot supply replacement bounds. Both output bounds are admitted
independently against the common basis before device work. Unsigned, bounded,
finite and wrap public-handle contracts are not inferred from this entry point.
The raw private primitive above retains its separate bounded/finite support.

All cross-component output aliases, currentness, proof prefixes, outstanding
writer/reader conflicts, generation exhaustion and scratch limits are checked
before mutation. Shared read-only handles are reserved once. The synchronous
operation reserves both output writers and the workspace, retains old values
through staging, and publishes both engine generations and range proofs only
after successful completion. A failure after the dual-store boundary clears
currentness and proofs on both outputs. Reservation cleanup also runs when
host allocation/proof work throws. Like other synchronous APIs, host callers
must externally exclude concurrent access to the same handles; this is logical
paired publication under that contract, not atomic GPU stores or a lock-free API.

The CPU path uses bounded modular intermediates and stages two output arrays;
the admission budget conservatively charges the same larger GPU plan envelope.
No transformed values are cached. Changes to either component are reread and
re-admitted on each call. Authored CPU/HIP tests cover independent Boost output,
changed imaginary components, legal shared readers, aliases, busy reservations,
stale proofs, exhausted outputs, resident values above 64 bits and over-range
composition. This new owner awaits the next coordinated build and full pass.
It remains internal while failure-path qualification, measured execution,
public API metadata and graph/prepared-group integration are completed.
