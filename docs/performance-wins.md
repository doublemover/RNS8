# Performance Wins

This document records every measured performance improvement in RNS8 with
local evidence on Windows gfx1100. It is the durable short list: what is
winning now (reviewed local gfx1100 evidence), what it beat, and what still blocks promotion.

Scope: Windows HIP SDK on Radeon RX 7900 XTX / gfx1100. Release builds,
fixed seeds, 3 warmups, 9 measured repeats, schema-valid captures, exact
CPU differentials, required GPU events. Windows evidence does not imply
Linux ROCm, Instinct, RDNA4, or production profiling readiness.

## June 10, 2026 Optimization Campaign (reviewed local gfx1100 evidence)

Full sweep with all 8 backends: 257 captures, 0 failures, 0 checksum mismatches.
Direct HIP improvements from kernel optimizations and dispatch wiring.

### Direct HIP Gains

| Shape | Before | After | Gain |
|---|---:|---:|---:|
| bounded u64 256x256x256 | 3,235 us | 2,008 us | +37.9% |
| bounded i64 512x512x512 | 3,781 us | 2,456 us | +35.1% |
| bounded i64 512x4x512 | 2,381 us | 1,727 us | +27.4% |
| bounded i64 512x8x512 | 2,400 us | 1,769 us | +26.3% |
| finite field u8 512x512x512 | 2,269 us | 1,736 us | +23.5% |
| bounded u64 512x512x512 | 3,922 us | 3,275 us | +16.5% |

### Implemented Optimizations

| Phase | Item | Kernels |
|---|---|---|
| 1c | uint4 coalesced pack loads | `rns8_pack_i64_4wide_coalesced_kernel` |
| 2a | Persistent small-shape GEMM | `rns8_persistent_small_gemm_rns_kernel` |
| 2b | Persistent small-shape pack | `rns8_persistent_small_pack_i64_kernel` |
| 3a | WMMA-native pack | `rns8_amdgpu_builtin_pack_wmma_*_kernel` |
| 3c | WMMA tile-sweep 64t/128t/256t | Skinny GEMV variants |
| 4a | Status elision (prefix >= 9) | `hip_backend_export_bounded.inc` |
| 4b | Fused GEMM+export | `rns8_fused_gemm_export_i64_kernel` |
| 5a | Next-gen wrap64 v5 | `rns8_wrap64_byte_gemm_u64acc_fused_low64_export_v5` |
| 6a | Streaming overlap interleave | `hip_backend.cpp` |

## Backends Beating Direct HIP (reviewed local gfx1100 evidence)

June 10, 2026 sweep with all backends on gfx1100.

| Shape | Winner | Speedup vs Direct HIP | Note |
|---|---|---|---|
| Bounded u64 256x256x256 | AMDGPU builtin | 1.52x | Dense WMMA |
| Exact-wide signed 128x128x128 | AMDGPU builtin | 1.41x | Dense WMMA |
| Bounded u64 512x512x512 | rocWMMA | 1.17x | |
| Bounded u64 1024x1024x1024 | rocWMMA | 1.17x | |
| Finite field u8 512x512x512 | rocWMMA | 1.49x | |
| Finite field u8 512x512x512 | CK | 1.35x | |
| Finite field u8 512x512x512 | hipBLASLt | 1.35x | |
| Bounded i64 512x8x512 | AMDGPU builtin | 1.34x | Skinny GEMV |
| Bounded i64 512x4x512 | AMDGPU builtin | 1.31x | Skinny GEMV |
| Bounded i64 64x64x64 | CPU reference | 1.04x | Tiny shape |

## Direct HIP Production Baseline (reviewed local gfx1100 evidence)

| Semantics | Shape | E2E | Pack | GEMM | Export |
|---|---|---|---:|---:|---:|---:|
| Bounded i64 | 1024x1024x1024 | 6,301 us | 24% | 35% | 41% |
| Bounded i64 | 512x512x512 | 2,456 us | 51% | 24% | 25% |
| Bounded u64 | 1024x1024x1024 | 6,856 us | 27% | 50% | 23% |
| Bounded u64 | 512x512x512 | 3,275 us | 55% | 26% | 19% |
| Bounded i64 | 512x4x512 | 1,727 us | 31% | 10% | 59% |
| Bounded i64 | 512x8x512 | 1,769 us | 39% | 16% | 45% |
| Exact-wide signed | 512x512x512 | 6,390 us | 18% | 22% | 52% |
| Exact-wide unsigned | 512x512x512 | 4,730 us | 26% | 41% | 33% |
| Finite field u8 | 512x512x512 | 1,736 us | 52% | 11% | 37% |
| Strict wrap64 | 2048x2048x2048 | 41,538 us | n/a | n/a | n/a |

## hipBLASLt 4096 Wins (reviewed local gfx1100 evidence)

June 10, 2026 sweep. 43 captures, 0 failures. All groups have CPU reference baselines.

| Semantics | hipBLASLt | Direct HIP | Speedup vs Direct HIP | vs CPU |
|---|---|---|---:|---:|---:|
| Bounded i64 | 46,825 us | 129,734 us | 2.77x | 316x |
| Bounded u64 | 51,239 us | 128,598 us | 2.51x | 275x |
| Exact-wide signed | 125,862 us | 577,811 us | 4.59x | 675x |
| Exact-wide unsigned | 120,893 us | 632,243 us | 5.23x | 711x |
| Finite ring u8 | 8,443 us | 34,397 us | 4.07x | 539x |
| Finite field u8 | 9,369 us | 35,225 us | 3.76x | 484x |

## Installed Cache Snapshot (reviewed local gfx1100 evidence)

39 validated exact-key entries from prior sweeps. The June 10 sweep found no new
promotable accelerator entries (CK, rocWMMA, hipBLASLt lost to Direct HIP in all
comparable groups on gfx1100). AMDGPU builtin wins on skinny GEMV and small
exact-wide are compiled and benchmarked but not yet in the installed cache.

Key installed entries from prior validated sweeps:

| Contract | Shape | Winner | Speedup vs Direct HIP |
|---|---|---|---|
| Bounded i64 | 1024 | hipBLASLt v2 | 1.09x |
| Bounded i64 | 2048 | CK v2 | 1.57x |
| Bounded u64 | 2048 | rocWMMA v2 | 1.22x |
| Bounded i64 | 4096 | hipBLASLt | 2.77x |
| Bounded u64 | 4096 | hipBLASLt | 2.51x |
| Exact-wide signed | 4096 | hipBLASLt | 3.61x |
| Exact-wide unsigned | 4096 | hipBLASLt | 3.78x |
| Finite field u8 | 4096 | hipBLASLt | 5.25x |
| Finite ring u8 | 4096 | hipBLASLt | 4.73x |

## Promotion Boundaries

- Promote now: 39 installed cache entries from validated sweeps. Direct HIP is
  the production baseline for square bounded shapes >= 256. AMDGPU builtins
  lead skinny GEMV (N=1,4,8). hipBLASLt dominates 4096 shapes.
- Keep experimental: rocWMMA prepack-B reuse (needs event infrastructure).
  Persistent small GEMM dispatch (m*n <= 64 verified, broader threshold pending).
- Deprioritized: CK/rocWMMA on square bounded shapes (lose to Direct HIP).
  wrap64 matrix-engine candidates (lose to Direct HIP v4). K-block policy
  variants (no profiler evidence).
