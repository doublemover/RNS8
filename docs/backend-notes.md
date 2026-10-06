# Backend implementation and selection

All backends preserve explicit semantic contracts. Hardware support depends on
actual compiler/runtime support and successful execution on the selected device.
The [audit](audit-2026-10-06.md) and [qualification guide](gpu-qualification.md)
supersede earlier blanket "production" or "complete" labels.

| Backend | Contracts | Important limits |
| --- | --- | --- |
| CPU reference | bounded, exact-wide, finite | correctness reference; optional OpenMP |
| CPU wrap64 byte-limb | wrap64 | explicit separate backend |
| Direct HIP | bounded, exact-wide, finite, wrap64 | baseline for target qualification; no universal speed claim |
| Native HIP vector ALU | bounded signed/unsigned | native exact comparator, not finite/wide/wrap routing |
| hipBLASLt | bounded, exact-wide, finite | optional library, separate INT32 scratch/reduction; per-tile bounds unsupported |
| CK | bounded including scheduled work, exact-wide, finite | optional headers/generated configuration; target/shape/static-modulus limits |
| rocWMMA | bounded including scheduled work, exact-wide, finite | optional library; wrap byte-GEMM candidate is internal and separate |
| AMDGPU builtins | dense RNS/finite and explicit sparse-A | target-specific WMMA/MFMA/SMFMAC/SWMMAC; K cap and modulus restrictions |

## Feature detection

Enable optional backends with `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`,
`RNS8_ENABLE_ROCWMMA`, and `RNS8_ENABLE_AMDGPU_BUILTINS`. They require
`RNS8_ENABLE_HIP=ON`. Discovery-only `RNS8_PROBE_ACCELERATORS` does not enable a
backend. No optional accelerator is required for CPU or direct-HIP correctness.

The source contains RDNA3 WMMA, CDNA3 `gfx942` MFMA, and RDNA4 matrix/sparse
builtins. That is source coverage, not a claim that all such targets compile or
pass today. The recent scalar "WMMA skinny" candidates were removed: the dense
RDNA3 route again uses the actual `rdna3_dense_wmma_kernel<Finite>` for skinny
and ordinary shapes. It preserves finite/RNS semantics and leading dimensions.

## AUTO and reviewed caches

An AUTO context probes direct HIP and can fall back to CPU if HIP is unavailable.
Within a HIP AUTO context, an eligible plan can select a compiled backend from a
reviewed exact-match cache. Candidate checks include semantics, target/runtime
identity, selected kernel, workspace, and the full plan key. No matching entry
means the baseline context backend remains selected. There is no runtime timing
search and no general shape-neighborhood interpolation.

`implementation_revision=2` is now part of the exact plan key. Historical cache
entries cannot implicitly carry performance validation across the correctness
repairs. Rebuild, qualify, measure, review, and explicitly reinstall entries.
A cache label does not cryptographically attest to the source or evidence;
keep the original capture and build provenance outside the cache.

## Metadata and lifetimes

`rns8_get_backend_capability_info` reports compiled/source capability metadata;
it does not run an on-device validation campaign. `rns8_get_plan_backend_info`
adds the selected kernel, accumulation policy, workspace and autotune key.
`rns8_get_plan_packing_info` describes input/output domains, currentness, and
possible continuations. Distinguish these declarations from external hardware
qualification results.

Public grouped APIs operate on already-current resident tasks. Graph replay,
streaming overlap, several pack/export variants, and repeated-workload reports
have benchmark-specific contracts. The actual graph implementation remains in
the benchmark and graph-safe backend helpers; removed dead graph prototypes are
not public APIs.

For backend-specific details also see the implementation READMEs under
`src/backend_hipblaslt`, `src/backend_ck`, `src/backend_rocwmma`, and
`src/backend_wrap64`. [Performance policy](performance.md) defines evidence gates.
