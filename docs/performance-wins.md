# Performance Wins

This document is the durable short list of RNS8 performance improvements that
currently have local evidence.

Scope:

- Platform: Windows HIP SDK on Radeon RX 7900 XTX / gfx1100.
- Evidence date: 2026-06-10 full accelerator sweep (257 captures, 0 failures, 0
  checksum mismatches). Backends: cpu-reference, hip-direct, hip-vector-alu-int64,
  ck, rocwmma, hipblaslt, amdgpu-builtins, wrap64-byte-limb.
- Evidence standard: release builds, fixed seeds, three warmups, nine measured
  repeats, schema-valid captures, exact CPU differentials. Accelerator numbers
  require same-contract CPU baselines and GPU event reports.

## Installed Cache Snapshot (2026-06-10)

The reviewed cache currently contains accelerator wins from prior validated
sweeps. The June 10 full sweep found Direct HIP and AMDGPU builtins as the
dominant production backends; CK, rocWMMA, and hipBLASLt did not produce new
promotable entries.

## Production Route Winners (2026-06-10 Sweep)

Direct HIP wins for square bounded shapes:

| Family | Shape | Winner | Median E2E | vs CPU |
|---|---:|---:|---:|
| Bounded i64 | 1024 | Direct HIP | 6,301 us | 34.9x |
| Bounded i64 | 512 | Direct HIP | 2,776 us | 43.8x |
| Bounded i64 | 256 | Direct HIP | 2,081 us | 14.0x |
| Bounded u64 | 1024 rec | Direct HIP | 3,569 us | 17.1x |
| Bounded u64 | 512 | Direct HIP | 3,868 us | 33.0x |

AMDGPU builtin wins for skinny and specialized shapes:

| Family | Shape | Winner | Median E2E | vs CPU |
|---|---:|---:|---:|
| Bounded i64 | 512x4x512 | AMDGPU builtin | 1,773 us | 8.5x |
| Bounded i64 | 512x8x512 | AMDGPU builtin | 1,855 us | 9.1x |
| Bounded u64 | 1024x1x1024 | AMDGPU builtin | 2,271 us | 2.5x |
| Bounded u64 | 1024x8x1024 | AMDGPU builtin | 2,348 us | 4.2x |
| Exact-wide signed | 512 | AMDGPU builtin | 5,545 us | 43.2x |
| Exact-wide unsigned | 512 | Direct HIP | 4,634 us | 45.9x |
| Finite field u8 | 512 | AMDGPU builtin | 1,680 us | 16.6x |

Direct HIP wrap64 (no CPU baseline in this sweep):

| Family | Shape | Median E2E |
|---|---:|---:|
| Strict wrap64 | 512 | 3,015 us |
| Strict wrap64 | 1024 | 6,996 us |
| Strict wrap64 | 2048 | 41,538 us |

## Accelerator Competitiveness (2026-06-10)

AMDGPU builtins won 6 of 45 comparable accelerator groups, specifically on
skinny GEMV shapes (N=1, N=4, N=8) and small exact-wide/finite shapes. CK,
rocWMMA, and hipBLASLt lost to Direct HIP across all comparable groups in this
sweep (105 "not_faster_than_direct_hip" blockers). This is consistent with the
RDNA3 architecture: WMMA-based builtins have an advantage on narrow-N shapes
where the matrix-core tile overhead of CK/rocWMMA/hipBLASLt outweighs GEMM
gains, while Direct HIP wins on square and large shapes.

## Workload-Level Wins

| Area | Evidence |
|---|---|
| Skinny GEMV | AMDGPU builtin wins 512x4x512 (1,773 us, 8.5x vs CPU) and 512x8x512 (1,855 us, 9.1x vs CPU) |
| Exact-wide 512 | AMDGPU builtin wins signed 512 (5,545 us, 43.2x vs CPU); Direct HIP wins unsigned 512 (4,634 us, 45.9x vs CPU) |
| Finite-u8 512 | AMDGPU builtin wins field-251 512 (1,680 us, 16.6x vs CPU) across all accelerator backends |
| Grouped dispatch | Many-small hostbatch scenarios show Direct HIP wins; accelerator grouped paths under development |
| RNS chains | Direct HIP residue-current chain wins with 9-36x vs independent export; accelerator chain support in progress |

## Known Limitations

- Repeated-B and reuse-contract scenarios require prepack cache infrastructure
  (Ranks 84-85) -- excluded from this sweep pending schema validation updates.
- Large 4096 shapes require CPU baselines impractical on single machine --
  excluded from this sweep.
- Accelerator autotune cache entries from prior sweeps remain installed but
  were not re-validated in this sweep.
- CDNA3 sparse SMFMAC and RDNA4 SWMMAC paths skip on gfx1100 (expected).
- The 2026-06-10 sweep excluded repeated-b and reuse-contract scenario families
  due to rocWMMA prepack cache schema validation mismatch.
