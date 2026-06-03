# Performance Wins

This document is the durable short list of RNS8 performance improvements that
currently have local evidence. It is intentionally narrower than the research
roadmap and work queue: it records what is winning now, what it beat, and what
still blocks promotion.

Scope:

- Platform: Windows HIP SDK on Radeon RX 7900 XTX / `gfx1100`.
- Semantics: bounded i64 square GEMM for the latest post-fix validation pass.
- Evidence standard: release builds, fixed seeds, three warmups, nine measured
  repeats, schema-valid captures, CPU reference checks, and required GPU event
  timing for GPU captures.
- Boundary: Windows `gfx1100` evidence does not imply Linux ROCm, Linux Radeon,
  Instinct, multi-GPU, or production profiling readiness.
- Storage: raw captures remain under ignored `temp/` paths. No durable
  autotune cache is installed by this document.

## Current One-Shot Winners

The latest post-fix bounded-i64 validation pass covered 512 and 1024 after the
vector event-capture and hipBLASLt full A+B event fixes. It used seed
`20260603` and wrote reviewed evidence under
`temp\vector-hipblaslt-event-fix\release-review\`.

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

## Reuse And Prepack Wins

The latest reuse validation compares each reuse path against the same backend
with normal per-repeat packing. These are implementation wins, not default AUTO
promotion claims, because the comparison intentionally changes `pack_mode` and
reuse metadata. Speedup is `1 / end_to_end_ratio` from
`tools\result_compare.py`; setup is the one-time prepack setup time reported by
the candidate capture.

| Backend | Shape | Reuse mode | Speedup vs own non-reuse path | Saved per repeat | Setup | Break-even repeats | Status |
|---|---:|---|---:|---:|---:|---:|---|
| hipBLASLt | 512 | A | 1.55x | 8029.9 us | 7104 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | B | 5.05x | 18116.7 us | 2811 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | A+B | 7.68x | 19649.0 us | 8323 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | A | 3.57x | 20712.4 us | 4345 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | B | 1.80x | 12810.9 us | 4744 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | A+B | 4.81x | 22794.2 us | 5992 us | 1 | Event-valid experimental reuse win |
| Vector ALU | 512 | A | 1.56x | 3460.6 us | 883 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | B | 1.55x | 3412.6 us | 943 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | A+B | 1.10x | 838.8 us | 6357 us | 8 | Only useful at higher repeat counts |
| Vector ALU | 1024 | B | 1.23x | 4683.0 us | 3187 us | 1 | Event-valid local reuse win |
| rocWMMA | 512 | B | 1.22x | 1210.3 us | 3428 us | 3 | Experimental; same-strategy review baselines still incomplete |

Non-winners from the same validation pass:

| Backend | Shape | Reuse mode | Result |
|---|---:|---|---|
| Vector ALU | 1024 | A | 0.96x vs own non-reuse path; regressed before setup cost |
| Vector ALU | 1024 | A+B | 0.71x vs own non-reuse path; regressed before setup cost |
| rocWMMA | 1024 | B | 0.68x vs own non-reuse path; regressed before setup cost |

The hipBLASLt full A+B reuse captures intentionally omit
`hipblaslt_pack_transpose_centered` from per-repeat event timing because the
operands are already packed. That absence is the corrected event contract, not
missing telemetry.

## Promotion Boundaries

- Promote now: no durable installed cache changes are made here. The latest
  one-shot 1024 CK result is a local promotable candidate, but this doc does
  not install it.
- Keep experimental: hipBLASLt, vector ALU, and rocWMMA reuse/prepack wins.
  They are correct and event-visible, but they compare different reuse
  contracts and need workload-level promotion policy before AUTO selection.
- Deprioritize for now: vector 1024 repeated-A/full-reuse and rocWMMA 1024
  repeated-B, which regressed in the latest reuse comparisons.

Evidence paths:

- `temp\vector-hipblaslt-event-fix\release-review\review_report.json`
- `temp\vector-hipblaslt-event-fix\reuse-release-review\reuse-packed-a\review_report.json`
- `temp\vector-hipblaslt-event-fix\reuse-release-review\reuse-packed-b\review_report.json`
- `temp\vector-hipblaslt-event-fix\reuse-release-review\reuse-packed-inputs\review_report.json`
- `temp\vector-hipblaslt-event-fix\reuse-release-review\comparisons\`
