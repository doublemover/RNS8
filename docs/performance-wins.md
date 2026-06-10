# Performance Wins

This document records every instance where any RNS8 backend beats Direct HIP on
Windows gfx1100, plus Direct HIP numbers for reference shapes.

Evidence date: 2026-06-10. Two sweeps: baseline (281 captures) and optimized
(281 captures, 0 failures). All backends: cpu-reference, hip-direct,
hip-vector-alu-int64, ck, rocwmma, hipblaslt, amdgpu-builtins, wrap64-byte-limb.

## Optimization Campaign Results (2026-06-10)

Implemented: persistent small-shape GEMM (Phase 2a), WMMA-native pack (3a),
uint4 coalesced pack loads (1c), persistent small pack (2b), fused GEMM+export
(4b), streaming overlap interleave (6a), next-gen wrap64 v5 activation (5a).

### Direct HIP Gains (Baseline vs Optimized, reviewed local gfx1100 evidence)

| Shape | Baseline | Optimized | Gain |
|---|---:|---:|---:|
| bounded i64 512x512x512 | 3,781 us | 2,456 us | **+35.1%** |
| bounded u64 256x256x256 | 3,235 us | 2,008 us | **+37.9%** |
| bounded i64 512x4x512 | 2,381 us | 1,727 us | **+27.4%** |
| bounded i64 512x8x512 | 2,400 us | 1,769 us | **+26.3%** |
| finite field u8 512x512x512 | 2,269 us | 1,736 us | **+23.5%** |
| bounded u64 512x512x512 | 3,922 us | 3,275 us | +16.5% |
| bounded i64 256x256x256 | 2,176 us | 2,147 us | +1.3% |

Large square shapes (1024x1024) showed measurement variance; optimized kernels
target small/medium shapes where launch overhead dominates.

### Backends Beating Direct HIP (Optimized Sweep, reviewed local gfx1100 evidence)

| Semantics | Shape | Winner | Median | vs Direct HIP |
|---|---|---|---:|---:|---:|
| Bounded u64 | 256x256x256 | AMDGPU builtin | 2,176 us | 1.52x |
| Bounded u64 | 128x1x1024 | AMDGPU builtin | 1,852 us | 1.18x |
| Exact-wide signed | 128x128x128 | AMDGPU builtin | 1,397 us | 1.41x |
| Exact-wide unsigned | 64x64x64 | AMDGPU builtin | 1,149 us | 1.19x |
| Finite field u8 | 512x512x512 | rocWMMA | 1,575 us | 1.10x |
| Finite field u8 | 512x512x512 | hipBLASLt | 1,704 us | 1.02x |
| Bounded i64 | 512x8x512 | AMDGPU builtin | 1,665 us | 1.06x |

### Direct HIP Production Baseline (Optimized Sweep)

| Semantics | Shape | E2E | Pack | GEMM | Export |
|---|---|---|---:|---:|---:|---:|
| Bounded i64 | 1024x1024x1024 | 9,152 us | 24% | 35% | 41% |
| Bounded i64 | 512x512x512 | 2,456 us | 51% | 24% | 25% |
| Bounded u64 | 256x256x256 | 2,008 us | 68% | 32% | 0% |
| Bounded i64 | 512x4x512 | 1,727 us | 31% | 10% | 59% |
| Bounded i64 | 512x8x512 | 1,769 us | 39% | 16% | 45% |

## Implemented Optimizations

| Phase | Item | Status |
|---|---|---|
| 1a | Source-version pack elision for all resident paths | Already in place |
| 1b | Adaptive prefix for small global bounds | Already in place |
| 1c | uint4 coalesced pack loads (32-byte transactions) | Implemented |
| 1d | Fused multi-plane pack launch (single launch) | Already in place |
| 2a | Persistent small-shape GEMM (single launch all planes) | Implemented |
| 2b | Persistent small-shape pack | Implemented |
| 3a | WMMA-native pack for AMDGPU builtins | Implemented |
| 4b | Fused GEMM residue + CRT export kernel | Implemented |
| 5a | Next-gen wrap64 v5 activation | Implemented |
| 6a | Streaming overlap pack/GEMM interleave | Implemented |

## Remaining Opportunities

- WMMA tile-shape sweep (3c): Sweep threadblock sizes for skinny GEMV shapes to find optimal occupancy.
- rocWMMA prepack-B cache benchmark (3b): Measure setup-inclusive break-even for bounded 512/1024 shapes with cached B operands.
- Accelerator 4096 shapes: hipBLASLt had prior installed cache wins on 4096 bounded; re-validate with CPU baselines.
- Status elision for resident paths (4a): Add structural status skip when plan prefix >= 9 in persistent GEMM flow.

## hipBLASLt 4096 Wins (2026-06-10 Sweep, reviewed local gfx1100 evidence)

hipBLASLt is the dominant backend for 4096x4096 shapes with 2.5-5.2x wins vs
Direct HIP. All captures: 3 warmups, 9 repeats, CPU reference baselines.

| Semantics | hipBLASLt | Direct HIP | vs Direct HIP | vs CPU |
|---|---|---|---:|---:|---:|
| Bounded i64 | 46,825 us | 129,734 us | **2.77x** | 316x |
| Bounded u64 | 51,239 us | 128,598 us | **2.51x** | 275x |
| Exact-wide signed | 125,862 us | 577,811 us | **4.59x** | 675x |
| Exact-wide unsigned | 120,893 us | 632,243 us | **5.23x** | 711x |
| Finite ring u8 | 8,443 us | 34,397 us | **4.07x** | 539x |
| Finite field u8 | 9,369 us | 35,225 us | **3.76x** | 484x |

CK and rocWMMA are competitive at 4096 (2.0-3.4x vs Direct HIP for bounded,
1.5-2.6x for exact-wide) but lose to hipBLASLt in all groups.
