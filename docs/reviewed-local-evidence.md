# Reviewed Local Evidence

This file is the durable summary of reviewed local evidence used by public
performance docs. Raw benchmark captures, cache candidates, comparison JSON,
and ISA/probe reports remain under ignored `temp/` paths and are not source
artifacts.

Scope:

- Platform: Windows HIP SDK on Radeon RX 7900 XTX / `gfx1100`.
- Review standard: release builds, fixed seeds, CPU reference checks,
  schema-valid captures, three warmups, and nine measured repeats unless noted.
- Boundary: Windows `gfx1100` evidence does not imply Linux ROCm, Instinct,
  RDNA4, multi-GPU, or production profiling readiness.

## Current Public Claim Summary

| Date | Command family | Seed | Shape | Backend | Result | Review status | Caveat |
|---|---|---:|---:|---|---|---|---|
| 2026-06-04 | bounded-i64 v2 one-shot release review | 20260604 | 512 | direct HIP | 1851 us median end-to-end; no accelerator win; rocWMMA v2 2591 us, vector ALU 6147 us, CK v2 7172 us, hipBLASLt v2 10101 us | release reviewed local matrix; required GPU events available | no cache entry; Direct HIP retained for this shape |
| 2026-06-04 | bounded-i64 v2 one-shot release review | 20260604 | 1024 | hipBLASLt | 4174 us median end-to-end; 1.09x vs Direct HIP; 8.13x vs vector ALU | release reviewed local matrix; required GPU events available; default local cache installed | current local cache contains this v2 entry only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 small current-v2 release review | 20260604 | 128 ring-251 | rocWMMA | 1136 us median end-to-end; 1.11x vs Direct HIP; CPU reference 1370 us | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 small current-v2 release review | 20260604 | 128 ring-256 | rocWMMA | 1132 us median end-to-end; 1.02x vs Direct HIP; CPU reference 1730 us | release reviewed local matrix; required GPU events available; default local cache installed | narrow win; explicit modulus/shape key only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-251 | rocWMMA | 1709 us median end-to-end; 2.74x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-255 | CK | 1938 us median end-to-end; 3.00x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 512 ring-256 | rocWMMA | 1365 us median end-to-end; 4.08x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-256 | hipBLASLt | 1792 us median end-to-end; 7.05x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 field-251 | CK | 1860 us median end-to-end; 5.68x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | field-251 512 not promoted because hipBLASLt GPU events were incomplete |
| 2026-06-04 | exact-wide current-v2 release review | 20260604 | 512 signed | rocWMMA | 7162 us median end-to-end; 1.02x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | narrow win; Windows `gfx1100` only |
| 2026-06-04 | exact-wide current-v2 release review | 20260604 | 1024 signed | hipBLASLt | 17092 us median end-to-end; 1.32x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit exact-wide signed key only; Windows `gfx1100` only |
| 2026-06-04 | exact-wide current-v2 release review | 20260604 | 1024 unsigned | CK | 20481 us median end-to-end; 1.22x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | unsigned 512 stayed on Direct HIP |
| 2026-06-04 | adaptive-bands current-v2 release review | 20260604 | 256 bounded i64, 1024 bounded i64, 512x1024 bounded u64 | Direct HIP | 1848 us, 4937 us, and 4224 us median end-to-end respectively; no accelerator beat Direct HIP | release reviewed local matrix; schema-valid captures; required GPU events available; corrected same-contract review grouping has no missing baselines or duplicate backends | no adaptive cache entry installed; older rocWMMA tiled-v1 adaptive winner is historical |
| 2026-06-03 | bounded-i64 one-shot release review | 20260603 | 512 | direct HIP | 2986 us median end-to-end; no accelerator win | release reviewed local snapshot | direct HIP retained for this snapshot; no cache installed |
| 2026-06-03 | bounded-i64 one-shot release review | 20260603 | 1024 | CK | 9222 us median end-to-end; 1.04x vs direct HIP; 2.58x vs vector ALU | release reviewed local snapshot | promotable local candidate; cache not written in this run |
| 2026-06-03 | bounded-i64 release matrix | 20260602 | 512 | rocWMMA | 2399 us median end-to-end; fastest promotable accelerator | release reviewed local matrix | same-day winner drift exists; rerun before durable cache install |
| 2026-06-03 | bounded-i64 release matrix | 20260602 | 1024 | hipBLASLt | 8326 us median end-to-end; fastest promotable accelerator | release reviewed local matrix | same-day winner drift exists; rerun before durable cache install |
| 2026-06-03 | bounded-u64 release matrix | 20260602 | 64, 128, 512, 1024 | vector ALU | fastest reviewed backend at all listed shapes | release reviewed local matrix | accelerator cache not promotable because vector baseline blocked every shape |
| 2026-06-03 | adaptive bounded release matrix | 20260602 | 1024 bounded i64 | rocWMMA | 5095 us median end-to-end with adaptive skip active | historical release reviewed local matrix | superseded by the 2026-06-04 current-v2 adaptive-bands review; do not install this tiled-v1 cache identity |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 64, 128, 512 | rocWMMA | winner across field-251, ring-251, and ring-255 groups | release reviewed local matrix | cache entries are explicit-modulus scoped |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 1024 ring | CK | 1428 us for modulus 251; 1354 us for modulus 255 | release reviewed local matrix | only same modulus/shape/contract keys are promotable |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 1024 field-251 | hipBLASLt | 2327 us median end-to-end | release reviewed local matrix | only same modulus/shape/contract keys are promotable |
| 2026-06-03 | strict wrap64 release matrix | 20260602 | 64, 128, 512, 1024 | direct HIP | 1828, 2090, 7757, 39359 us median end-to-end | release reviewed local baseline | no public wrap64 accelerator backend exists |
| 2026-06-03 | rocWMMA wrap64 candidate review | 20260603 | 64, 128, 512, 1024 | rocWMMA candidate | lost to direct HIP at every listed shape | release-shape candidate review | internal candidate only; not public or AUTO-eligible |
| 2026-06-03 | direct-HIP uniform-small reuse-B colpair v2 captures | 1 | 1024 bounded i64 | direct HIP | setup-inclusive 1.19x vs same-backend non-reuse; 2.62x vs prior v1 setup-inclusive path in before/after matrix | release local reuse capture; schema/event valid | explicit `--reuse-packed-b` path only; no AUTO/default routing change |
| 2026-06-03 | direct-HIP uniform-small reuse-B colpair v2 captures | 1 | 512 bounded u64 | direct HIP | setup-inclusive 1.34x vs same-backend non-reuse; 1.41x vs prior v1 setup-inclusive path in before/after matrix | release local reuse capture; schema/event valid | explicit `--reuse-packed-b` path only; no AUTO/default routing change |
| 2026-06-03 | direct-HIP uniform-small reuse-B colpair v2 reruns | 1 | 1024 bounded u64 | direct HIP | setup-inclusive 1.17x to 1.75x vs same-backend non-reuse across three focused reruns | release local reuse reruns; schema/event valid | export timing remains volatile; explicit `--reuse-packed-b` path only |
| 2026-06-03 | direct-HIP public one-shot colpair gate | 31 | 512 bounded u64 | direct HIP | 1.09x average end-to-end, 1.21x median end-to-end, and 1.06x average GEMM-event speedup vs prior one-shot v1 kernel | final release captures schema/event valid; before captures intentionally stale under new schema | routed only for bounded-u64 Direct-HIP one-shot `m/n/k >= 512`; i64 and smaller u64 stay on v1 |
| 2026-06-04 | direct-HIP uniform-small reuse-A colpair fixed-prefix captures | 1 | 512, 1024 bounded i64 | direct HIP | setup-inclusive 3.04x at 512 and 1.32x at 1024 vs clean `a75b0a2` same-contract repeated-A baseline over 33 repeats | release local reuse capture; schema/event/checksum valid | explicit fixed-prefix `--reuse-packed-a` path only; no AUTO/default routing change |
| 2026-06-04 | direct-HIP uniform-small reuse-A colpair fixed-prefix captures | 1 | 512, 1024 bounded u64 | direct HIP | setup-inclusive 1.33x at 512 and 1.30x at 1024 vs clean `a75b0a2` same-contract repeated-A baseline over 33 repeats | release local reuse capture; schema/event/checksum valid | explicit fixed-prefix `--reuse-packed-a` path only; no AUTO/default routing change |

## Reuse And Prepack Summary

| Mechanism | Public surface | What is reused | Evidence status | AUTO eligibility |
|---|---|---|---|---|
| Public prepack cache | `rns8_prepack_matrix`, `rns8_get_prepack_cache_key_info` | Backend-specific packed operand identity, currently narrow and backend-limited | Correctness and metadata surface exists; `production_prepack_cache_available` remains `0` for current reviewed captures | Not AUTO-promoted until workload-level policy and reviewed same-contract wins exist |
| Benchmark repeated-A mode | Benchmark-only reuse mode | A-side packing/setup across measured repeats | Event-visible local wins for selected hipBLASLt/vector/direct-HIP shapes; some regressions | Not AUTO-eligible because the benchmark changes `pack_mode` and reuse metadata |
| Benchmark repeated-B mode | Benchmark-only reuse mode | B-side packing/setup across measured repeats | Strongest current hipBLASLt local reuse wins; Direct-HIP uniform-small bounded i64/u64 now has event-valid same-backend wins; rocWMMA has narrow B-cache evidence | Not AUTO-eligible because the benchmark changes the workload contract |
| Benchmark repeated-A+B mode | Benchmark-only reuse mode | Both operands packed before the measured repeat loop | Event-visible hipBLASLt wins; vector 512 barely positive over nine repeats | Not AUTO-eligible until the public workload contract asks for reusable operands |
| Persistent matrix reuse | Public matrix/plan/workspace handles | Resident RNS, finite-u8, wrap64, and native vector storage across API calls | Correctness path; required for non-one-shot workflows | AUTO may select only reviewed same-contract cache hits for the explicit plan key |

## Reproduction Command Families

Current bounded-i64 one-shot claims:

```powershell
python tools\benchmark_sweep.py `
  --bench build\windows-msvc-hip-release\rns8-bench.exe `
  --bench-for hipblaslt=build\windows-msvc-hipblaslt-release\rns8-bench.exe `
  --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe `
  --bench-for rocwmma=build\windows-msvc-rocwmma-release\rns8-bench.exe `
  --out-root temp\perf-work-queue\bounded-rns-v2-release `
  --review-mode release `
  --warmups 3 `
  --repeats 9 `
  --seed 20260604 `
  --backend cpu `
  --backend hip-direct `
  --backend hip-vector-alu-int64 `
  --backend hipblaslt `
  --backend ck `
  --backend rocwmma `
  --semantics bounded-i64 `
  --case bounded-i64-512:512,512,512 `
  --case bounded-i64-1024:1024,1024,1024 `
  --write-autotune-cache `
  --autotune-cache temp\perf-work-queue\bounded-rns-v2-release\autotune-cache.json
```

Current finite-u8 v2 claims use the same backend/build/review settings as the
bounded-i64 command above, with `--backend cpu --backend hip-direct --backend
hipblaslt --backend ck --backend rocwmma`, `--review-mode release`, three
warmups, nine repeats, seed `20260604`, and one reviewed temp cache per
semantic/modulus root. The 64/128 follow-up used
`temp\perf-work-queue\finite-u8-v2-small-release\<contract>` and the same
command family with `--case ...-64:64,64,64` and
`--case ...-128:128,128,128`; ring-255 64 was not promoted because CPU reference
was faster than the rocWMMA accelerator result.

```powershell
python tools\benchmark_sweep.py `
  --bench build\windows-msvc-hip-release\rns8-bench.exe `
  --bench-for hipblaslt=build\windows-msvc-hipblaslt-release\rns8-bench.exe `
  --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe `
  --bench-for rocwmma=build\windows-msvc-rocwmma-release\rns8-bench.exe `
  --out-root temp\perf-work-queue\finite-u8-v2-release\ring256 `
  --review-mode release `
  --warmups 3 `
  --repeats 9 `
  --seed 20260604 `
  --backend cpu `
  --backend hip-direct `
  --backend hipblaslt `
  --backend ck `
  --backend rocwmma `
  --semantics finite-u8-ring `
  --modulus 256 `
  --case ring256-512:512,512,512 `
  --case ring256-1024:1024,1024,1024 `
  --write-autotune-cache `
  --autotune-cache temp\perf-work-queue\finite-u8-v2-release\ring256\autotune-cache.json
```

Current exact-wide v2 claims:

```powershell
python tools\benchmark_sweep.py `
  --bench build\windows-msvc-hip-release\rns8-bench.exe `
  --bench-for hipblaslt=build\windows-msvc-hipblaslt-release\rns8-bench.exe `
  --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe `
  --bench-for rocwmma=build\windows-msvc-rocwmma-release\rns8-bench.exe `
  --out-root temp\perf-work-queue\exact-wide-v2-release `
  --review-mode release `
  --warmups 3 `
  --repeats 9 `
  --seed 20260604 `
  --backend cpu `
  --backend hip-direct `
  --backend hipblaslt `
  --backend ck `
  --backend rocwmma `
  --semantics exact-wide-signed `
  --semantics exact-wide-unsigned `
  --case exact-wide-512:512,512,512 `
  --case exact-wide-1024:1024,1024,1024 `
  --write-autotune-cache `
  --autotune-cache temp\perf-work-queue\exact-wide-v2-release\autotune-cache.json
```

Reuse/prepack comparisons:

```powershell
python tools\benchmark_sweep.py `
  --semantics bounded-i64 `
  --case 512:512,512,512 `
  --case 1024:1024,1024,1024 `
  --backend hipblaslt `
  --backend hip-vector-alu-int64 `
  --backend rocwmma `
  --reuse-packed-a `
  --reuse-packed-b `
  --reuse-packed-inputs `
  --warmups 3 `
  --repeats 9 `
  --seed 20260603 `
  --release-review `
  --out-root temp\benchmark-sweeps\windows-gfx1100-release-reuse-current
```

Direct-HIP uniform-small bounded reuse-B colpair v2 captures:

```powershell
build\windows-msvc-hip-release\rns8-bench.exe `
  --backend hip-direct `
  --semantics bounded-i64 `
  --m 512 --n 512 --k 512 `
  --warmups 3 --repeats 9 `
  --reuse-packed-b

build\windows-msvc-hip-release\rns8-bench.exe `
  --backend hip-direct `
  --semantics bounded-u64 `
  --m 1024 --n 1024 --k 1024 `
  --warmups 3 --repeats 9 `
  --reuse-packed-b

python tools\benchmark_schema.py <captures>
python tools\gpu_event_report.py --fail-on-unavailable <captures>
```

Direct-HIP bounded-u64 public one-shot colpair gate:

```powershell
build\windows-msvc-hip-release\rns8-bench.exe `
  --backend hip-direct `
  --semantics bounded-u64 `
  --oneshot `
  --m 512 --n 512 --k 512 `
  --warmups 3 --repeats 9 `
  --seed 31

python tools\benchmark_schema.py <captures>
python tools\gpu_event_report.py --fail-on-unavailable <captures>
```

Wrap64 direct-HIP baseline plus internal rocWMMA candidate:

```powershell
python tools\benchmark_sweep.py `
  --semantics wrap-u64 `
  --case 64:64,64,64 `
  --case 128:128,128,128 `
  --case 512:512,512,512 `
  --case 1024:1024,1024,1024 `
  --backend cpu `
  --backend direct-hip `
  --include-rocwmma-wrap64-candidate `
  --warmups 3 `
  --repeats 9 `
  --seed 20260603 `
  --release-review `
  --out-root temp\benchmark-sweeps\windows-gfx1100-release-wrap64-current
```

Use `tools\benchmark_schema.py`, `tools\result_compare.py`, and
`tools\gpu_event_report.py` to validate captures and compare same-contract
groups before turning any local result into a durable claim.
