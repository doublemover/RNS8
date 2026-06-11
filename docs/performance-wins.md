# Performance Wins

Local evidence on Windows gfx1100 (Radeon RX 7900 XTX, HIP SDK 7.1).
Release builds, fixed seeds, schema-valid captures, exact CPU differentials,
required GPU events. Windows evidence does not imply Linux/Instinct/CDNA/RDNA4.

## June 11, 2026 Final Sweep (reviewed local gfx1100 evidence)

243 captures, 0 failures across 5 backends (Direct HIP, rocWMMA, hipBLASLt, CK,
AMDGPU builtins). All accelerator presets build clean. All 281 tests pass.

### Active Optimizations

| Layer | Paths |
|---|---|
| **Pack** | Persistent (0-4096 cells), Coalesced 4-wide (>=256 cells, ld==cols), Standard (>4096). Non-temporal loads. |
| **GEMM** | DP4A finite-u8 (mod 256/255/251, `v_dot4_i32_iu8 neg_lo:[1,1,0]` fix). Persistent small (m*n <= 64). HIP graph replay. Plane parallelism via grid.z. |
| **Export** | VOPD DPP (prefix 1-8, status needed). Combined final-output (prefix 1-8, status elided). u192 CRT (prefix 9+). All paths use precomputed `__constant__` Garner weights for prefixes 1-8. Status elision all paths. |
| **AMDGPU** | WMMA skinny GEMV dispatch (N=1->64t, N<=4->128t, N<=8->256t). WMMA-native pack wrappers compiled. |
| **Wrap64** | Tiled u64acc dispatch (>=1024). Fused v5 wrappers compiled. |
| **Infra** | Zero-skip row/col detection. Adaptive prefix. Verification amortization. HIP graph replay. Scenario lint (0 errors). |

### Every Backend Beating Direct HIP (June 11 sweep, reviewed local gfx1100 evidence)

| Backend | Semantics | Shape | Winner | Direct HIP | Speedup |
|---|---|---|---|---|---|
| rocwmma | bounded u64 | 128x1x1024 | 1,981 us | 150,760 us | 76.1x |
| hipblaslt | bounded u64 | 128x1x1024 | 2,621 us | 150,760 us | 57.5x |
| rocwmma | bounded i64 | 32x32x32 | 1,436 us | 62,575 us | 43.6x |
| ck | bounded i64 | 32x32x32 | 1,687 us | 62,575 us | 37.1x |
| hipblaslt | bounded i64 | 32x32x32 | 2,528 us | 62,575 us | 24.8x |
| rocwmma | bounded u64 | 64x64x64 | 1,554 us | 35,375 us | 22.8x |
| hipblaslt | bounded u64 | 64x64x64 | 2,490 us | 35,375 us | 14.2x |
| rocwmma | exact-wide unsigned | 128 | 4,796 us | 34,611 us | 7.2x |
| rocwmma | exact-wide signed | 64 | 6,424 us | 44,194 us | 6.9x |
| ck | exact-wide signed | 64 | 7,661 us | 44,194 us | 5.8x |
| rocwmma | exact-wide unsigned | 64 | 5,825 us | 28,284 us | 4.9x |
| ck | exact-wide unsigned | 64 | 6,814 us | 28,284 us | 4.2x |
| ck | exact-wide unsigned | 128 | 8,519 us | 34,611 us | 4.1x |
| hipblaslt | exact-wide unsigned | 64 | 7,584 us | 28,284 us | 3.7x |
| hipblaslt | exact-wide unsigned | 128 | 9,744 us | 34,611 us | 3.6x |
| hipblaslt | exact-wide signed | 64 | 15,628 us | 44,194 us | 2.8x |
| rocwmma | bounded i64 | 512x512x512 | 1,791 us | 3,084 us | 1.72x |
| rocwmma | bounded i64 | 512x4x512 | 1,920 us | 3,092 us | 1.61x |
| ck | finite field u8 | 512 | 1,330 us | 2,093 us | 1.57x |
| rocwmma | finite field u8 | 512 | 1,391 us | 2,093 us | 1.51x |
| hipblaslt | finite field u8 | 512 | 1,441 us | 2,093 us | 1.45x |
| ck | bounded i64 | 512x4x512 | 2,370 us | 3,092 us | 1.30x |
| hipblaslt | bounded i64 | 512x4x512 | 2,475 us | 3,092 us | 1.25x |
| rocwmma | bounded u64 | 512x512x512 | 1,707 us | 2,107 us | 1.23x |
| rocwmma | bounded i64 | 1024x1024x1024 | 4,700 us | 5,202 us | 1.11x |

### Direct HIP Production Baseline (reviewed local gfx1100 evidence)

| Semantics | Shape | E2E | Notes |
|---|---|---|---|
| bounded i64 | 1024x1024x1024 | 5,202 us | Production baseline |
| bounded i64 | 512x512x512 | 3,084 us | VOPD DPP export active |
| bounded i64 | 256x256x256 | 2,147 us | Dominant backend for square bounded |
| bounded u64 | 1024x1024x1024 | 8,498 us | |
| bounded u64 | 512x512x512 | 2,107 us | Combined export active |
| strict wrap64 | 2048x2048x2048 | 41,538 us | |
| strict wrap64 | 1024x1024x1024 | 6,996 us | Tiled u64acc active |

### Installed Cache (39 entries, reviewed local gfx1100 evidence)

| Contract | Shape | Winner | Speedup |
|---|---|---|---|
| bounded i64 | 1024 | hipBLASLt v2 | 1.09x |
| bounded i64 | 2048 | CK v2 | 1.57x |
| bounded u64 | 2048 | rocWMMA v2 | 1.22x |
| bounded i64 | 4096 | hipBLASLt | 2.77x |
| bounded u64 | 4096 | hipBLASLt | 2.51x |
| exact-wide signed | 4096 | hipBLASLt | 3.61x |
| exact-wide unsigned | 4096 | hipBLASLt | 3.78x |
| exact-wide signed | 2048 | hipBLASLt | 2.23x |
| exact-wide unsigned | 2048 | hipBLASLt | 3.04x |
| finite field u8 | 4096 | hipBLASLt | 5.25x |
| finite ring u8 | 4096 | hipBLASLt | 4.73x |
| finite ring u8 | 1024 | rocWMMA v2 | 2.74x |
| finite ring u8 | 2048 | hipBLASLt v2 | 3.47x |

### Promotion Boundaries (reviewed local gfx1100 evidence)

- **Promoted**: 39 installed cache entries. Direct HIP is production baseline for square bounded shapes >= 256. rocWMMA dominates small bounded (32-64), finite-u8, and exact-wide. hipBLASLt leads on skinny shapes.
- **Ready**: Chain final-output wins (1.31-35.32x). Grouped dispatch wins (2-27x per task). Many-small per-task wins (10-27x vs independent).
- **Experimental**: Persistent small GEMM (m*n <= 64). Coalesced pack below 256. Fused GEMM+export kernel.
- **Deferred**: Streaming overlap (architecture already optimal via grid.z plane parallelism). 64-bit multi-precision GEMM (CDNA3-only).
- **Research**: INT4/IU4, Ozaki FP8, Strassen, Freivalds (API stubs declared, schema-gated).

### CDNA3 Status

MFMA, sparse SMFMAC, and CK/rocWMMA tuning kernels compiled and schema-registered.
No Linux ROCm validation sweep has been run. All CDNA performance evidence deferred
until `scripts/cdna_first_pass.sh` completes on Instinct hardware.
