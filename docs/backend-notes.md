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
| `ck` | Opt-in public path; `RNS8_ENABLE_CK=ON` | Bounded including adaptive schedules, wide, finite; headers, shapes and static moduli constrain admission | [Implementation](../src/backend_ck/README.md) |
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

Ordinary GEMM/grouped calls are synchronous and operate on compatible current
resident resources. Public [prepared execution](prepared-execution.md) retains
resources and binds generations. Explicit fixed-prefix direct-HIP stream
submission, owned HIP capture/replay, same-device prepared groups and
[resident DAG submission](resident-graphs.md#explicit-asynchronous-direct-hip-dags)
have separate completion-token and reservation rules. Benchmark graph/streaming
flags are separate experiments; their larger combinations do not extend the
public API envelope.

## Internal prototypes and reserved APIs

| Family | Current boundary | Owner |
|---|---|---|
| Native-byte WMMA, grouped native conversion, queue and split-K | Internal checked candidates; public dispatch unchanged | [Native narrow](native-narrow.md), [queue](persistent-native-queue.md), [split-K](native-split-k.md) |
| Strassen / rank-23 | Private one-level modular prototypes; public `rns8_strassen_gemm_research` remains unsupported | [Strassen](strassen-device.md), [rank-23](rank23-device.md), [#48](https://github.com/doublemover/RNS8/issues/48) |
| Packed INT4/IU4 | Internal certified subset/prototypes; public research semantics rejected | [Specification](RNS8_RESEARCH_SPEC.md#171-int4), [#46](https://github.com/doublemover/RNS8/issues/46) |
| Floating encodings / Ozaki | Scalar certificates and research integration; no public floating backend | [Generated encodings](generated-encodings.md), [#45](https://github.com/doublemover/RNS8/issues/45) |
| Redundant checks / Freivalds | Private deterministic check-residue sentinel; public Freivalds entrypoint unsupported | [Integrity](device-integrity.md), [#49](https://github.com/doublemover/RNS8/issues/49) |
| Sparse/hybrid | Public explicit structured Sparse-A; private scalar block-sparse baseline | [Block sparse](block-sparse.md), [#51](https://github.com/doublemover/RNS8/issues/51) |
| Multi-GPU | Independent per-device shard scripts; true modulus-sharded GEMM remains work | [Linux helpers](platform-linux.md#existing-bounded-host-scripts), [#50](https://github.com/doublemover/RNS8/issues/50) |

The public Ozaki, Strassen and Freivalds research entrypoints return
`RNS8_UNSUPPORTED_BACKEND`; implemented private candidates do not change that
contract. The [research index](research/README.md) groups focused technical notes.
