# Performance Wins

This document is the durable short list of RNS8 performance improvements that
currently have local evidence. It is intentionally narrower than the research
roadmap and work queue: it records what is winning now, what it beat, and what
still blocks promotion.

Scope:

- Platform: Windows HIP SDK on Radeon RX 7900 XTX / `gfx1100`.
- Semantics: bounded i64/u64 square GEMM for the latest post-fix validation
  passes, plus same-backend strict wrap64 Direct-HIP implementation comparisons.
- Evidence standard: release builds, fixed seeds, three warmups, nine measured
  repeats, schema-valid captures, CPU reference checks, and required GPU event
  timing for GPU captures.
- Boundary: Windows `gfx1100` evidence does not imply Linux ROCm, Linux Radeon,
  Instinct, multi-GPU, or production profiling readiness.
- Durable evidence: [reviewed-local-evidence.md](reviewed-local-evidence.md)
  records the curated platform, command family, seed, shape, backend, result,
  review status, caveat, and reproduction command families. Raw captures stay
  ignored and are not durable docs.

## Current One-Shot Winners

The latest post-fix bounded-i64 validation pass covered 512 and 1024 after the
vector event-capture and hipBLASLt full A+B event fixes. It used seed
`20260603`; the durable summary lives in
[reviewed-local-evidence.md](reviewed-local-evidence.md).

| Shape | Current winner | Winner median end-to-end | Direct HIP median | Vector ALU median | Speedup | Decision |
|---:|---|---:|---:|---:|---:|---|
| 512 | Direct HIP `direct_hip_tiled_rns_gemm_v1` | 2986 us | 2986 us | 9232 us | No accelerator win | Keep direct HIP for this snapshot |
| 1024 | CK `ck_wmma_cshuffle_i8_i32_centered_epilogue_v1` | 9222 us | 9604 us | 23777 us | 1.04x vs direct HIP, 2.58x vs vector ALU | Promotable local candidate; cache not written in this run |

The 512 group had no missing required baselines, no duplicate backend records,
and release-review requirements were satisfied, but no accelerator beat direct
HIP. The 1024 group had the same clean review properties and selected CK as the
fastest promotable accelerator.

This differs from the earlier June 3, 2026 seed `20260602` four-shape
bounded-i64 matrix in [performance-model.md](performance-model.md), where
rocWMMA won 512 and hipBLASLt won 1024. Treat that as useful historical
release-reviewed evidence, not as proof that one winner is stable across local
driver/build/run conditions. Rerun the target shapes before installing a
durable cache.

## Direct-HIP Implementation Wins

These rows compare one Direct-HIP implementation against the previous
Direct-HIP implementation for the same public API and shape. They are not
cross-backend AUTO winners.

| Surface | Shape | New selected kernel | Average end-to-end speedup | Median end-to-end speedup | Event GEMM speedup | Status |
|---|---:|---|---:|---:|---:|---|
| Public bounded-u64 one-shot | 512 | `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2` | 1.09x | 1.21x | 1.06x | Routed only for Direct-HIP bounded-u64 `m/n/k >= 512`; i64 and smaller u64 remain on v1 |

The colpair one-shot kernel was not promoted for bounded i64 because the same
mapping regressed i64 release captures. It was also not routed for small
bounded-u64 shapes because 64/128 averages were spike-sensitive on Windows
`gfx1100`; they keep the prior v1 native-input grouped kernel.

The strict wrap64 Direct-HIP v4 kernel supersedes the previous v3 scalar path
for local `K <= 4096` shapes. It uses direct unsigned byte products, uint32
low-diagonal accumulation where safe, uint64 carry propagation, vectorized
compact byte-limb load/store where shape evidence permits, and scalar
pack/export fallbacks for 64-like shapes. These rows compare v4 against v3 with
release binaries, three warmups, nine measured repeats, seed `20260604`,
schema-valid/event-valid v4 captures, and checksum-matched before/after output.

| Surface | Shape | New selected kernel | Median end-to-end speedup | Event GEMM median speedup | Status |
|---|---:|---|---:|---:|---|
| Strict wrap64 Direct-HIP default pack/GEMM/export | 64 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.08x | 1.38x | Local implementation win; CPU byte-limb still faster at 64 |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 128 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.17x | 2.17x | Local implementation win |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 512 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.02x | 2.04x | Positive but pack/export-noisy |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 1024 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 5.60x | 8.94x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 64 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.22x | 1.45x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 128 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 4.67x | 3.16x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 512 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.07x | 1.99x | Positive but export-noisy |
| Strict wrap64 Direct-HIP reuse-packed inputs | 1024 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 6.74x | 7.83x | Local implementation win |

## Reuse And Prepack Wins

The latest reuse validation compares each reuse path against the same backend
with normal per-repeat packing. These are implementation wins, not default AUTO
promotion claims, because the comparison intentionally changes `pack_mode` and
reuse metadata. The headline speedup includes one-time setup over the measured
nine-repeat validation capture:

```text
setup_inclusive_speedup =
    (9 * non_reuse_avg_end_to_end_us) /
    (prepack_setup_us + 9 * reuse_avg_end_to_end_us)
```

Steady-state speedup is the per-repeat limit after setup has already been paid,
computed as `1 / end_to_end_ratio` from `tools\result_compare.py`.

| Backend | Shape | Reuse mode | Setup-inclusive speedup over 9 repeats | Steady-state per-repeat speedup | Saved per repeat | Setup | Break-even repeats | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| hipBLASLt | 512 | A | 1.47x | 1.55x | 8029.9 us | 7104 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | B | 4.72x | 5.05x | 18116.7 us | 2811 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | A+B | 5.84x | 7.68x | 19649.0 us | 8323 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | A | 3.36x | 3.57x | 20712.4 us | 4345 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | B | 1.74x | 1.80x | 12810.9 us | 4744 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | A+B | 4.32x | 4.81x | 22794.2 us | 5992 us | 1 | Event-valid experimental reuse win |
| Direct HIP | 1024 | B, uniform-small i8 A/B colpair v2, bounded i64 | 1.19x | 1.20x | 1379.7 us | 768 us | 1 | Event-valid explicit reuse-path win |
| Vector ALU | 512 | A | 1.54x | 1.56x | 3460.6 us | 883 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | B | 1.53x | 1.55x | 3412.6 us | 943 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | A+B | 1.01x | 1.10x | 838.8 us | 6357 us | 8 | Barely positive at 9 repeats |
| Vector ALU | 1024 | B | 1.21x | 1.23x | 4683.0 us | 3187 us | 1 | Event-valid local reuse win |
| Direct HIP | 512 | B, uniform-small i8 A/B colpair v2, bounded u64 | 1.34x | 1.39x | 981.0 us | 813 us | 1 | Event-valid explicit reuse-path win |
| Direct HIP | 1024 | B, uniform-small i8 A/B colpair v2, bounded u64 | 1.17x | 1.18x | 1282.9 us | 695 us | 1 | Conservative rerun result; export timing volatile |
| rocWMMA | 512 | B | 1.14x | 1.22x | 1210.3 us | 3428 us | 3 | Experimental; same-strategy review baselines still incomplete |

Non-winners from the same validation pass:

| Backend | Shape | Reuse mode | Setup-inclusive result over 9 repeats | Steady-state result |
|---|---:|---|---:|---:|
| Vector ALU | 1024 | A | 0.94x | 0.96x |
| Vector ALU | 1024 | A+B | 0.69x | 0.71x |
| Direct HIP | 512 | B, uniform-small i8 A/B colpair v2, bounded i64 | 1.00x | 1.02x |
| rocWMMA | 1024 | B | 0.64x | 0.68x |

The hipBLASLt full A+B reuse captures intentionally omit
`hipblaslt_pack_transpose_centered` from per-repeat event timing because the
operands are already packed. That absence is the corrected event contract, not
missing telemetry.

The Direct-HIP colpair v2 entries supersede the earlier single-column
uniform-small i8 A/B reuse kernel. In the first before/after release matrix,
v2 improved per-repeat end-to-end time against that v1 implementation by 2.10x
for bounded i64 512, 2.54x for bounded i64 1024, and 1.43x for bounded u64 512.
The same matrix showed bounded u64 1024 export volatility, so the table uses the
most conservative setup-inclusive speedup from three focused reruns against the
same-backend non-reuse baseline.

Direct-HIP reuse-A fixed-prefix reruns used release binaries, 3 warmups, and 33
measured repeats. The pre-change baseline came from a clean `a75b0a2` worktree
and the candidate used the `transient_uniform_small_i8_b_resident_i8_a_reuse`
benchmark path. All rows were schema-valid, event-valid, and checksum-matched.

| Backend | Shape | Semantics | Reuse mode | Setup-inclusive speedup over 33 repeats | Steady-state per-repeat speedup | Saved per repeat | Setup | Break-even repeats | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Direct HIP | 512 | bounded i64 | A, uniform-small i8 A/B colpair v1 | 3.04x | 2.93x | 6296 us | 618 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 1024 | bounded i64 | A, uniform-small i8 A/B colpair v1 | 1.32x | 1.26x | 1211 us | 580 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 512 | bounded u64 | A, uniform-small i8 A/B colpair v1 | 1.33x | 1.18x | 271 us | 536 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 1024 | bounded u64 | A, uniform-small i8 A/B colpair v1 | 1.30x | 1.24x | 1102 us | 544 us | 1 | Event-valid explicit fixed-prefix reuse-path win |

## Promotion Boundaries

- Promote now: no durable installed cache changes are made here. The latest
  one-shot 1024 CK result is a local promotable candidate, but this doc does
  not install it.
- Keep experimental for AUTO selection: Direct-HIP, hipBLASLt, vector ALU, and
  rocWMMA reuse/prepack wins. They are correct and event-visible, but they
  compare different reuse contracts and need workload-level promotion policy
  before AUTO selection. The mechanism split is summarized in
  [reviewed-local-evidence.md](reviewed-local-evidence.md).
- Deprioritize for now: vector 1024 repeated-A/full-reuse and rocWMMA 1024
  repeated-B, which regressed in the latest reuse comparisons.
