# Backend implementation and selection

Select a backend explicitly in `rns8_context_options.requested_backend`, or use
`AUTO` with the policy below. The benchmark accepts the names in this table;
the [public enum](../include/rns8/semantics.h) defines C API identities.
[Readiness](platform-readiness.md) separates availability, compiled declarations
and actual device validation.

## Public paths

| Backend / benchmark name | State and enablement | Arithmetic / constraints | Evidence or implementation owner |
|---|---|---|---|
| `cpu-reference` | Available public path | Bounded, exact-wide, finite; independent Boost reference, optional OpenMP | [Correctness](correctness.md), [developer checks](../CONTRIBUTING.md) |
| `hip-direct` | Available public path with `RNS8_ENABLE_HIP=ON` | Bounded, wide, finite and separate byte-limb wrap64; target/address restrictions | [Local device validation](local-all-features-20261023.md) |
| `hip-vector-alu-int64` | Available public HIP path | Bounded i64/u64 only; native storage and exact wide accumulation | [Matched measurements](7900xtx-large-throughput-2026-10-23.md#public-bounded-i64-pipelines-and-reuse) |
| `hipblaslt` | Opt-in public path; `RNS8_ENABLE_HIPBLASLT=ON` | Fixed-prefix bounded, wide, finite; library availability; separate INT32 scratch/reduction; no adaptive tile bounds | [Implementation](../src/backend_hipblaslt/README.md) |
| `ck` | Opt-in public path; `RNS8_ENABLE_CK=ON` | Bounded including adaptive schedules, wide, finite; headers, shapes and byte-modulus contracts constrain admission | [Implementation](../src/backend_ck/README.md) |
| `rocwmma` | Opt-in public path; `RNS8_ENABLE_ROCWMMA=ON` | Bounded including adaptive schedules, wide, finite; target matrix instructions and header compatibility | [Implementation](../src/backend_rocwmma/README.md) |
| `amdgpu-builtins` | Opt-in public path; `RNS8_ENABLE_AMDGPU_BUILTINS=ON` | Target-specific dense RNS/finite and explicit Sparse-A; accumulation/modulus/device restrictions | [Source](../src/backend_amdgpu_builtins/amdgpu_builtins_backend.cpp), [validation](local-all-features-20261023.md) |
| `wrap64-byte-limb` | Available public CPU context; direct HIP provides the GPU route | Strict modulo 2⁶⁴ only, distinct byte-limb storage | [Implementation](../src/backend_wrap64/README.md) |

All optional GPU paths require HIP. CMake feature detection checks components;
`RNS8_PROBE_ACCELERATORS` is discovery only. The direct-HIP and CPU paths do not
require optional matrix libraries. Windows correctness and measurements are
scoped in the linked dated records; Linux and other GPU families need their own
execution evidence.

RDNA3 builtin kernels use actual WMMA instructions. CDNA3 `gfx942` MFMA and
RDNA4 matrix/sparse variants also exist in source. Source coverage does not
establish execution for an entire family. Sparse-A is an explicit public
structured-sparsity contract with target restrictions; dense GEMM does not infer
sparsity. Private block-sparse research is a separate path.

Sparse-A source preflight is shared with raw-device dispatch: current kernels
admit `gfx942`, `gfx1200` and `gfx1201`, K divisible by4 up to65536, supported
static byte moduli or default-basis prefix1..20, and complete packed/input/output
address spans within their index domain. This is source admissibility, not a
GPU/OS qualification statement. In particular, dense `gfx1100` availability
does not enable sparse instructions. An unsupported sparse request now rejects
before output invalidation, preserving its old bytes, generation and range
proof. CPU predicate and gfx1100 negative-preservation fixtures pass the
[current local batch](../metadata/remaining-code-validation-20261027.json);
supported sparse-target arithmetic remains unqualified.

## AUTO and reviewed caches

An AUTO context probes direct HIP and can use CPU when HIP is unavailable.
Within a HIP AUTO context, an eligible plan may select a compiled backend from
a reviewed exact-match cache. Admission binds semantics, target/runtime, kernel,
workspace and the full plan key. Missing/rejected entries retain the baseline
context backend. There is no runtime timing search or shape interpolation.

`implementation_revision=2` participates in that key. Earlier cache entries do
not carry validation across implementation repairs. Reviewed caches are routing
inputs, not cryptographic attestations; retain the original capture, artifact
hash and build provenance. [Performance review](performance.md) owns promotion.

## Metadata, ownership and execution

`rns8_get_backend_capability_info` reports compiled/source metadata;
`rns8_get_plan_backend_info` adds selected kernel, accumulation and workspace;
`rns8_get_plan_packing_info` reports domains/currentness and continuations.
These APIs describe the selected implementation without running a campaign.
The context-free capability accessor therefore returns
`exact_differential_validated=0` and `performance_validated=0`: no run receipt is
bound to that inventory. Zero means unknown in this scope, not a failed test.
Correctness-backend roles and compiled availability remain separate fields.
An AUTO plan admitted through the existing reviewed exact-match cache can
expose that scoped exact/performance evidence; ordinary plans do not infer it
from compile switches. The named dated validation/measurement reports above
retain their actual target, source and toolchain qualification.

New raw benchmark diagnostics may consequently carry false in the metadata
validation field while retaining their independently required per-capture
reference/checksum checks. The schema accepts this unknown inventory state;
it still requires a Boolean, and rejects performance validation without bound
exact-differential evidence. Historic captures declaring true remain readable.
This does not relax reviewed comparison/provenance/variance or routing gates.

Ordinary GEMM/grouped calls are synchronous and operate on compatible current
resident resources. Public [prepared execution](prepared-execution.md) retains
resources and binds generations. Explicit fixed-prefix direct-HIP stream
submission, owned HIP capture/replay, same-device prepared groups and
[resident DAG submission](resident-graphs.md#explicit-asynchronous-direct-hip-dags)
have separate completion-token and reservation rules. Benchmark graph/streaming
flags are separate experiments; their larger combinations do not extend the
public API envelope.

## Experimental paths and reserved APIs

| Family | Current boundary | Owner |
|---|---|---|
| Native-byte WMMA, grouped native conversion, queue and split-K | Private register/LDS controls plus explicit prepared native groups and windowed split-K owners; ordinary dispatch unchanged | [Native narrow](native-narrow.md), [native tiled](native-tiled-wmma.md), [queue](persistent-native-queue.md), [split-K](native-split-k.md) |
| [Bound whole-route planner](bound-pipeline-recipes.md) | Explicit preparation/execution of representation, split-K, narrow, queue, stream/graph and final checked CRT alternatives; unpriced routes remain unknown | [Issue30](https://github.com/doublemover/RNS8/issues/30) |
| [CDNA5 transfer research](cdna5-tdm-research.md) | Private ordinary/TDM/two-stage cluster-multicast raw-native controls; LLVM23 gfx1250 syntax only, no target run | [Issue52](https://github.com/doublemover/RNS8/issues/52) |
| Matrix-owned narrow native | Opt-in synchronous public exact-wide scalar/byte products; fresh domain/partial-sum scans and complete RNS output; applicable local checks pass | [Narrow owner](narrow-native-owner.md) |
| Encoded B | Explicit generation-bound public cache of actual K-contiguous B residues with grouped WMMA consumers; applicable local checks pass | [Encoded-B owner](encoded-b-native.md) |
| Runtime product arena | CPU/direct-HIP exact-wide DAG, strict last-reader reuse and opt-in public packed-materialization planning; capacity-aware price schema and measured-stage ingestion | [Runtime arena](runtime-product-arena.md), [materializations](planning-recipes.md#cost-planned-product-materializations) |
| Matrix-owned adaptive native | Opt-in public exact-wide direct-HIP call with retained native inputs; synchronous; applicable local checks pass | [Adaptive ownership](adaptive-native-product.md#explicit-matrix-owned-native-inputs) |
| Paired complex 3M/4M | Opt-in public signed exact-wide CPU/direct-HIP and synchronous graph composition; local binding checks pass | [Complex owner](complex-residue-composition.md) |
| Wide-index residue control | Private scalar64-bit offsets plus public panel streaming with a completed >32GiB native-input capacity study; each path keeps its own indexing/output contract | [Wide-index control](wide-index-gemm.md), [capacity study](out-of-core-panels.md#real-32-gib-input-on-the-24-gib-rx-7900-xtx) |
| Strassen / rank-23 / rank48 | Opt-in public one-level direct-HIP exact-wide Strassen; private rank-23 and canonical retained/recomputed rank48; applicable local checks pass; performance qualification pending | [Strassen](strassen-device.md), [rank-23](rank23-device.md), [#48](https://github.com/doublemover/RNS8/issues/48) |
| Packed INT4/IU4 | Explicit public finite signed-nibble owner and versioned packed B; gfx1100/1101/1102 implementation; target/performance qualification deferred | [Specification](RNS8_RESEARCH_SPEC.md#171-int4), [#46](https://github.com/doublemover/RNS8/issues/46) |
| Floating encodings / integer splitting | Explicit finite gfx942/gfx1201 FP8 and gfx950 fixed/dynamic/mixed FP4/FP6/FP8 kernels with conditional instruction models; separate caller-owned host radix128 split/merge; floating target execution/performance unqualified | [FP8 and split contracts](fp8-modular-research.md), [rewrite oracle](floating-rewrite-oracle.md), [#45](https://github.com/doublemover/RNS8/issues/45) |
| Redundant checks / Freivalds | Native-input independent residue sentinel, producer-attached public integrity and distributed commit check; public finite Freivalds verifier with explicit randomness metadata | [Integrity](device-integrity.md), [#49](https://github.com/doublemover/RNS8/issues/49) |
| Sparse/hybrid | Public structured Sparse-A and generation-bound block-sparse owners with explicit scalar/WMMA/hybrid methods; provider restrictions and measured cohorts remain scoped | [Block sparse](block-sparse.md), [#51](https://github.com/doublemover/RNS8/issues/51) |
| Multi-GPU | Private HIP all-axis shards, staged/peer and MPI/RCCL/rocSHMEM process transports, bounded ownership and independent commit integrity; actual linked cluster qualification pending | [Real HIP owner](distributed-hip.md), [Linux helpers](platform-linux.md#existing-bounded-host-scripts), [#50](https://github.com/doublemover/RNS8/issues/50) |

The [integer split API](../include/rns8/integer_split.h) exposes exact signed
radix128 host components with explicit strides, capacity and transactional merge.
The seeded Freivalds entrypoint delegates to the public finite verifier; other arithmetic domains
remain unsupported. Strassen accepts the explicitly documented subset. The [research index](research/README.md) groups focused technical notes.

Sparse entry points also return `RNS8_BUSY` for a dense B with a pending writer,
or an output with pending readers or a writer, before target admission and output
mutation. Concurrent read-only reservations on B remain valid. State fixtures
cover both sparse RNS and finite-ring calls and pass the current local batch.


The internal epilogue and exact-suffix-lift support providers now share static
shape/prefix/address/resource admission between public owners and raw wrappers.
Records distinguish unknown resources from zero scratch and never mark a device
qualified. Lift reports suffix payload and the additional device status word
separately: the public staged-residue budget still limits payload, preserving its
documented meaning. Pointer aliases, generations, value proofs and stream ownership
remain dynamic admission steps. Provider parity/budget fixtures pass applicable local validation;
the [bound recipe layer](bound-pipeline-recipes.md) composes these providers with
existing resource owners and explicit complete-route prices.

### Explicit finite INT4 ownership

`rns8_prepare_int4_recipe` prepares the real signed-mode IU4 WMMA candidate for
finite rings 2..16 and eligible prime fields. Mode 0 converts inside the tile;
mode 1 retains B as `[ceil(K/16)][N][2]` little-nibble uint32 fragments (lower K
in the low nibble). Odd/tail K is numerical zero, never uninitialized payload.
Both wave halves consume the documented duplicated B fragment. This is not an
unsigned-nibble path and does not truncate the ordinary RNS basis.

The shared arithmetic guard models four-bit signed inputs, signed prior residue
and every 65,536-product slab: `65536*64+15 < INT32_MAX`. Preparation probes the
actual device code object once; steady execution neither probes nor allocates.
The immutable owner binds B instance/generation, modulus, format, target, build
and certificate. Source mutation rejects before C invalidation. Device failure
invalidates C and retains borrowed storage until a checked drain; preparation
may return a non-null cleanup-only owner on an undrainable failure. Checked
destruction remains retryable. Default finite GEMM/AUTO are unchanged.

`rns8-int4-calibrate` and `collect_int4_calibration.py` provide matched ordinary
INT8, on-the-fly INT4 and packed-B routes, including full cold ownership, repeated
B and repeated A+B, padded complete export, setup/execute/export/cleanup phases,
sustained uninstrumented intervals and counterbalanced independent processes.
Fixtures cover tails, all slab boundaries and multiple slabs; authored regressions
cover every finite nibble alphabet, packed tails, stale generations and full
public output. Applicable gfx1100 regressions pass the current local batch;
no new campaign or speedup (including the specification's 1.25x gate) follows.
Public reachability does not qualify other targets.
