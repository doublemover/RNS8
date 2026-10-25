# Exact complex residue composition

Issue #59 provides an explicit experimental
`rns8_gemm_complex_exact_signed` API for CPU-reference and direct HIP. It uses
signed exact-wide matrix handles, explicit three/four-product selection and a
scratch budget. The new public binding awaits the final coordinated validation;
its underlying matrix owner passed the earlier `1bb5e33` pass.

The private `hip_direct_gemm_complex_residues` primitive accepts split
real/imaginary residue views with a common explicit basis/domain, independent
component pitches, and either three or four real products. It also supports raw
bounded/finite experiments outside the public signed exact-wide subset. Neither
interface changes automatic routing.

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
compact product planes. The staged components reuse dead internal p0/p1 slots
only after the owning cell has loaded every product. The four-product envelope
is four product planes with the same p0/p1 reuse. No public input/output alias is
introduced; final dual publication remains a separate stage. Budget rejection occurs before
device work; failed admission is not a valid zero-byte plan.

Both outputs remain untouched until the final dual-store kernel. A shared
`commit_started` flag marks that boundary. Device stores are not atomic across
the pair: if the final launch or completion fails, the caller must invalidate
**both** components. Input lifetime/exclusion and atomic logical publication
remain the raw caller's responsibility. The public matrix owner below supplies
reader/writer reservations and paired publication for synchronous exact-wide
signed execution. No transformed panel cache, prepared-group integration or
asynchronous ownership is claimed.

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

The matched resident measurements below charge transformed panels, all launches,
scratch and joint commit. Final dual integer export is outside their contract.
The new public ownership binding and synchronous graph node await their own
validation. Fault-path qualification, broader prepared/async integration and
complete exported application measurements remain open under #59.

## Internal matrix ownership integration

The internal `complex_exact_signed_gemm`, exposed through the experimental
public entry point above, connects the raw composition to
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
composition. The owner passed the coordinated all-feature validation at
`1bb5e33`: 491 passes, three expected sparse-target skips and no failures.
The public entry point exposes only this synchronous signed contract. Invalid
method values fail before mutation. Transpose/conjugation and unsigned, bounded,
finite and wrap public-handle contracts remain unsupported; callers export the
two exact-wide components through existing signed limb exports. Synchronous graph
integration is described below; prepared-group and async integration remain separate work.

## Matched resident measurements at 1bb5e33

RX 7900 XTX / gfx1100, VS2026 and HIP 7.1; full signed 64-bit components,
prefix twenty, seed 3135, three warmups, nine measured pairs and five independent
processes per case. Each timed sustained burst runs for at least 250 ms without
reference checks between calls. All 540 paired calls and all thirty final burst
outputs passed independent full-output Boost comparisons; bursts total 8,336 calls.

| Shape M×N×K | 3M median µs [process range] | 4M median µs [process range] |
|---|---:|---:|
| 32×32×32 | 601.43 [518.31, 737.60] | 532.03 [441.95, 961.93] |
| 1×512×512 | 696.08 [676.56, 873.24] | 675.73 [565.85, 1070.10] |
| 512×512×512 | 4300.42 [4248.64, 4520.65] | 5380.03 [5182.36, 5410.74] |

The large case improves by **1.25×**, with separated process ranges. Its useful
complex-integer throughput is **0.250 TOPS versus 0.200 TOPS**, counting eight
real scalar operations per complex multiply-accumulate, once per logical output.
Small cases show no benefit. Scratch is allocated and freed inside every call;
no transformed-panel cache is used. Native input upload, four initial packs and
common allocations are outside these resident timings, as is final integer
export. These results do not establish an end-to-end exported application gain.
For the historical layout measured below, explicit scratch was 35 MiB for 3M and 30 MiB for 4M (not the current layout-v2 capacity). Separate cold input
preparation medians are 431.91 ms [351.10, 488.99] and 351.74 ms [328.76, 428.06]
respectively, including the common allocation/upload/packing boundary. Cold
process effects are substantial and must not be hidden by the steady-state gain.

Reproduce using `benchmarks/native-calibration/complex-residues.json` with
`--runs 5 --repeats 9 --warmups 3 --timeout 900`. Local captures are
`temp/1bb5e33-complex/`, with sibling `.summary.json` and `.sha256.txt` files.


The paired owner is also available as `RNS8_RESIDENT_COMPLEX_MATMUL` in the
[synchronous resident graph](resident-graphs.md#paired-complex-nodes). Graph
preflight propagates both proofs and tracks both output writers before any
node executes. Both components can feed later arithmetic. This integration is
authored with CPU/Boost and HIP fixtures; async complex nodes remain unsupported.

## Layout-v2 dead-product storage reuse (unqualified)

Ordinary and graph-enqueued HIP routes now derive their staged-component offsets
from the same checked `ComplexResidueScratch` layout. Recombination loads p0/p1/
p2 (and p3 for4M) into one cell owner's scalar values before either store. The
common host/device algebra helper returns both components; only then can that
cell overwrite p0/p1. No other cell reads those locations, and p2/p3 remain
separate. This removes two full component buffers from both methods without
removing the final paired commit or relaxing public aliases, currentness,
reader/writer reservations, or generation publication. Graph preparation retains
the reduced maximum across nodes and uses those same offsets on replay.

For512³/P20, the queried scratch capacities are now25MiB (3M) and20MiB (4M),
10MiB below each historical layout. These are explicit capacity calculations,
not measured allocation peaks, memory traffic, launch reductions or speedups.
Calibration receipts identify layout2, reused stage bytes and accounting scope;
collectors reject the previous layout's capacity as a matched new capture.
The CPU alias-schedule regressions, exact-budget GPU cases and existing graph
replay differentials are authored but have not been executed for this change.
Historical measurements above do not validate the new storage layout.
