# Performance Wins

Local evidence on Windows gfx1100 (Radeon RX 7900 XTX, HIP SDK 7.1).
Release builds, fixed seeds, schema-valid captures, exact CPU differentials,
required GPU events. Windows evidence does not imply Linux/Instinct/CDNA/RDNA4.

## June 11, 2026 Final Sweep

200 captures across 4 backends (Direct HIP, hipBLASLt, CK, AMDGPU builtins).
0 real failures, 0 checksum mismatches. rocWMMA excluded pending HIP event
recording in separate build target. All June 10 campaign wins carried forward.

### Active Optimizations Summary

| Layer | Paths |
|---|---|
| **Pack** | Persistent (0-4096 cells), Coalesced 4-wide (>=256 cells, ld==cols), Standard (>4096). Non-temporal loads on all. |
| **GEMM** | DP4A finite-u8 (mod 256/255/251, `v_dot4_i32_iu8 neg_lo:[1,1,0]` fix for ROCm 7.1 assembler bug). Persistent small (m*n <= 64). HIP graph replay (bounded + finite-u8). Per-plane parallelism via grid.z dimension. |
| **Export** | Garner fast CRT (i64+u64, prefix 1-8, precomputed `__constant__` weights). VOPD DPP export. Combined final-output. Status elision all paths (prefix >= 9). |
| **AMDGPU builtins** | WMMA skinny GEMV dispatch (N=1->64t, N<=4->128t, N<=8->256t). WMMA-native pack wrappers compiled. |
| **Wrap64** | Tiled u64acc dispatch (>=1024 per dimension). Fused v5 wrappers compiled. |
| **Infrastructure** | Zero-skip row/col detection. Adaptive prefix. Verification amortization. HIP graph full-path replay. Garner `__constant__` weight tables (prefix 1-8). |

### Direct HIP Production Baseline (June 11 sweep)

Key shapes with phase breakdown from the sweep with all backends.

| Semantics | Shape | E2E | Pack % | GEMM % | Export % |
|---|---|---|---|---|---|
| bounded i64 | 1024x1024x1024 | 5,198 us | 39% | 30% | 31% |
| bounded i64 | 512x512x512 | 2,456 us | 51% | 24% | 25% |
| bounded i64 | 256x256x256 | 2,147 us | 48% | 52% | 0% |
| bounded u64 | 1024x1024x1024 | 8,498 us | 28% | 49% | 23% |
| bounded u64 | 512x512x512 | 3,275 us | 55% | 26% | 19% |
| strict wrap64 | 2048x2048x2048 | 41,538 us | -- | -- | -- |
| strict wrap64 | 1024x1024x1024 | 6,996 us | -- | -- | -- |
| strict wrap64 | 512x512x512 | 3,015 us | -- | -- | -- |

### Backend Wins Over Direct HIP (June 11 sweep, reviewed local gfx1100 evidence)

| Backend | Semantics | Shape | Winner | Direct HIP | Speedup |
|---|---|---|---|---|---|
| ck | exact-wide unsigned | 128 | 5,833 us | 1,228,630 us | 210.6x |
| hipblaslt | exact-wide unsigned | 64 | 8,933 us | 1,310,000 us | 146.6x |
| ck | exact-wide unsigned | 64 | 9,081 us | 1,310,000 us | 144.3x |
| ck | exact-wide signed | 64 | 8,872 us | 1,271,300 us | 143.3x |
| hipblaslt | exact-wide unsigned | 128 | 8,999 us | 1,228,630 us | 136.5x |
| hipblaslt | exact-wide signed | 64 | 10,545 us | 1,271,300 us | 120.6x |
| ck | bounded i64 | 32 | 1,834 us | 93,183 us | 50.8x |
| hipblaslt | bounded i64 | 32 | 1,845 us | 93,183 us | 50.5x |
| ck | bounded i64 | 512x4 | 2,128 us | 43,827 us | 20.6x |
| hipblaslt | bounded i64 | 256x1x4096 | 3,138 us | 37,911 us | 12.1x |
| ck | bounded i64 | 256x256 | 14,111 us | 146,510 us | 10.4x |
| hipblaslt | bounded i64 | 256x256 | 14,455 us | 146,510 us | 10.1x |
| ck | exact-wide unsigned | 512 | 6,893 us | 44,361 us | 6.4x |
| ck | exact-wide signed | 512 | 7,647 us | 42,145 us | 5.5x |
| hipblaslt | exact-wide unsigned | 512 | 9,200 us | 44,361 us | 4.8x |
| hipblaslt | exact-wide signed | 512 | 11,597 us | 42,145 us | 3.6x |
| ck | finite field u8 | 512 | 1,445 us | 1,462 us | 1.01x |

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

### Workload-Level Wins (reviewed local gfx1100 evidence)

| Area | Evidence |
|---|---|
| Result cache | Direct-HIP: bounded-i64 dirty tile 1.21x, exact-wide signed 1.40x |
| CPU selector | 5 promotable thresholds: 4 CPU-favored (4.27x-21.77x), 1 GPU-favored |
| FHE proxy profiles | 13 profiles: Direct HIP 16-198x vs CPU |
| RNS chains | 15 fused device handoff wins (1.20-1.58x) vs host export/repack |
| Residue-current chains | 8 final-output chain wins (1.31-35.32x) vs independent |
| Vector-to-RNS | Direct-HIP chain3: signed 512 at 6,475 us, unsigned 512 at 5,817 us |
| Layout search | Exact-wide residue-current chain 35.36x vs independent |
| Many-small grouped | 10-27x per-task vs independent CPU/Direct HIP |
| Reuse/prepack | Direct HIP repeated-A 1024 1.45x, repeated-B 1024 1.47x |
| HIP graph replay | Wrap64 512 1.90x, Wrap64 1024 1.24x (graph vs ordinary) |

### Promotion Boundaries

- **Promoted**: 39 installed cache entries. Direct HIP is production baseline
  for square bounded shapes >= 256. hipBLASLt dominates 4096 shapes.
  AMDGPU builtins lead skinny GEMV.
- **Ready** (reviewed local gfx1100 evidence): AMDGPU builtin wins on small exact-wide (64/128, 1.03-15x).
  Grouped dispatch wins (2-27x per task). Chain final-output wins (1.52-35x).
- **Experimental** (reviewed local gfx1100 evidence): Persistent small GEMM (m*n <= 64). Coalesced pack below 256.
  Combined/VOPD/fused-GEMM export kernels.
- **Deferred**: Streaming overlap (architecture already optimal - kernel uses
  grid.z for plane parallelism). 64-bit multi-precision GEMM (CDNA3-only,
  RDNA3 lacks `v_dot2_i32_i16`).
- **Research**: INT4/IU4, Ozaki FP8, Strassen, Freivalds (API stubs declared,
  schema-gated behind `RNS8_ENABLE_*_RESEARCH`).

### CDNA3 Status

MFMA, sparse SMFMAC, and CK/rocWMMA tuning kernels are compiled and
schema-registered. No Linux ROCm validation sweep has been run. All CDNA
performance evidence is deferred until `scripts/cdna_first_pass.sh` produces
target-validation reports on Instinct hardware.
