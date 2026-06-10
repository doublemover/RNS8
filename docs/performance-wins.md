# Performance Wins

This document records every instance where any RNS8 backend beats Direct HIP on
Windows gfx1100, plus Direct HIP numbers for reference shapes.

Evidence date: 2026-06-10 full sweep. 281 captures, 0 failures, 0 checksum
mismatches. All backends: cpu-reference, hip-direct, hip-vector-alu-int64,
ck, rocwmma, hipblaslt, amdgpu-builtins, wrap64-byte-limb.

## Backends Beating Direct HIP

Sorted by speedup vs Direct HIP. Only includes groups where Direct HIP is a
reasonable baseline (excludes host-batch/grouped/chain scenarios where Direct
HIP numbers reflect extra launch overhead, not GEMM performance).

### AMDGPU Builtin Wins

AMDGPU builtins (RDNA3 WMMA) win primarily on skinny GEMV and small exact-wide
shapes where the dense matrix-core path avoids Direct HIP's tiled overhead.

| Semantics | Shape | AMDGPU Builtin | Direct HIP | Speedup | Note |
|---|---|---|---:|---:|---:|---|
| Bounded i64 | 512x8x512 | 1,794 us | 2,400 us | 1.34x | Skinny GEMV small-N |
| Bounded i64 | 512x4x512 | 1,811 us | 2,381 us | 1.31x | Skinny GEMV small-N |
| Bounded u64 | 1024x8x1024 | 2,262 us | 3,240 us | 1.43x | Skinny GEMV small-N |
| Bounded u64 | 256x256x256 | 1,947 us | 3,235 us | 1.66x | Small dense |
| Exact-wide signed | 128x128x128 | 1,517 us | 2,095 us | 1.38x | Dense WMMA |
| Exact-wide signed | 512x512x512 | 5,545 us | 6,390 us | 1.15x | Dense WMMA |
| Exact-wide unsigned | 512x512x512 | 4,651 us | 4,730 us | 1.02x | Marginal |
| Bounded i64 | 128x128x128 | 1,541 us | 1,599 us | 1.04x | Marginal |
| Bounded i64 | 256x256x256 | 2,102 us | 2,176 us | 1.04x | Marginal |
| Bounded u64 | 128x128x128 | 1,580 us | 1,606 us | 1.02x | Marginal |

### rocWMMA Wins

rocWMMA wins on larger bounded shapes and finite-u8 at 512, with clear margins.

| Semantics | Shape | rocWMMA | Direct HIP | Speedup |
|---|---|---|---:|---:|---:|
| Finite field u8 | 512x512x512 | 1,615 us | 2,413 us | 1.49x |
| Finite field u8 | 512x512x512 | 1,683 us | 2,413 us | 1.43x |
| Bounded u64 | 512x512x512 | 3,342 us | 3,922 us | 1.17x |
| Bounded u64 | 1024x1024x1024 | 5,869 us | 6,856 us | 1.17x |
| Bounded i64 | 512x512x512 | 3,653 us | 3,781 us | 1.04x |

### CK Wins

CK wins on finite-u8 at 512 (comparable to rocWMMA).

| Semantics | Shape | CK | Direct HIP | Speedup |
|---|---|---|---:|---:|---:|
| Finite field u8 | 512x512x512 | 1,783 us | 2,413 us | 1.35x |
| Finite field u8 | 512x512x512 | 1,835 us | 2,413 us | 1.32x |

### hipBLASLt Wins

| Semantics | Shape | hipBLASLt | Direct HIP | Speedup |
|---|---|---|---:|---:|---:|
| Finite field u8 | 512x512x512 | 1,784 us | 2,413 us | 1.35x |
| Finite field u8 | 512x512x512 | 1,791 us | 2,413 us | 1.35x |

### CPU Wins (Tiny Shapes)

On very small shapes (32x32, 64x64) GPU launch overhead exceeds computation.
CPU reference is the correct production route for these sizes.

| Semantics | Shape | CPU Ref | Direct HIP | Speedup |
|---|---|---|---:|---:|---:|
| Bounded i64 | 64x64x64 | 1,392 us | 1,443 us | 1.04x |
| Bounded u64 | 64x64x64 | 834 us | 39,654 us | 47.5x |
| Finite field u8 | 128x128x128 | 694 us | 1,266 us | 1.82x |

## Direct HIP Production Baseline

For reference, Direct HIP numbers on key shapes (all backends compiled).

| Semantics | Shape | Direct HIP | CPU Reference | vs CPU |
|---|---|---|---:|---:|---:|
| Bounded i64 | 1024x1024x1024 | 6,301 us | 219,837 us | 34.9x |
| Bounded i64 | 512x512x512 | 2,776 us | 121,634 us | 43.8x |
| Bounded i64 | 256x256x512 | 2,081 us | 29,063 us | 14.0x |
| Bounded u64 | 1024x1024x1024 | 6,856 us | 804,761 us | 117.4x |
| Bounded u64 | 512x1024x512 | 3,569 us | 61,169 us | 17.1x |
| Bounded u64 | 512x512x512 | 3,922 us | 129,674 us | 33.1x |
| Bounded u64 | 256x256x256 | 3,235 us | 28,137 us | 8.7x |
| Exact-wide signed | 512x512x512 | 6,390 us | 239,659 us | 37.5x |
| Exact-wide unsigned | 512x512x512 | 4,730 us | 212,806 us | 45.0x |
| Finite field u8 | 512x512x512 | 2,413 us | 27,925 us | 11.6x |
| Finite ring u8 | 512x512x512 | 2,512 us | 29,313 us | 11.7x |

## Wrap64 Direct HIP Baseline

| Shape | Direct HIP | CPU Byte-Limb | vs CPU |
|---|---:|---:|---:|
| 512x512x512 | 3,015 us | n/a | n/a |
| 1024x1024x1024 | 6,996 us | n/a | n/a |
| 2048x2048x2048 | 41,538 us | n/a | n/a |

## Vector-ALU Baseline

| Semantics | Shape | Vector ALU | Direct HIP | Ratio |
|---|---|---|---:|---:|---:|
| Bounded i64 | 1024x1024x1024 | 27,311 us | 6,301 us | 0.23x |
| Bounded u64 | 1024x1024x1024 | 17,665 us | 6,856 us | 0.39x |
| Bounded u64 | 128x128x128 | 1,315 us | 1,606 us | 1.22x |

## Summary

- **Direct HIP** is the production winner for square bounded shapes >= 256x256.
- **AMDGPU builtins (WMMA)** lead on skinny GEMV (N=1..8) and small exact-wide.
- **rocWMMA** has clear wins on bounded u64 512/1024 and finite-u8 512 (1.17-1.49x).
- **CK and hipBLASLt** are competitive on finite-u8 512 but lose elsewhere.
- **CPU reference** wins on tiny shapes (<128) where GPU launch dominates.
- No accelerator backend beat Direct HIP on bounded i64 1024 square shapes.
