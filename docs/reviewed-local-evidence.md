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
| 2026-06-03 | bounded-i64 one-shot release review | 20260603 | 512 | direct HIP | 2986 us median end-to-end; no accelerator win | release reviewed local snapshot | direct HIP retained for this snapshot; no cache installed |
| 2026-06-03 | bounded-i64 one-shot release review | 20260603 | 1024 | CK | 9222 us median end-to-end; 1.04x vs direct HIP; 2.58x vs vector ALU | release reviewed local snapshot | promotable local candidate; cache not written in this run |
| 2026-06-03 | bounded-i64 release matrix | 20260602 | 512 | rocWMMA | 2399 us median end-to-end; fastest promotable accelerator | release reviewed local matrix | same-day winner drift exists; rerun before durable cache install |
| 2026-06-03 | bounded-i64 release matrix | 20260602 | 1024 | hipBLASLt | 8326 us median end-to-end; fastest promotable accelerator | release reviewed local matrix | same-day winner drift exists; rerun before durable cache install |
| 2026-06-03 | bounded-u64 release matrix | 20260602 | 64, 128, 512, 1024 | vector ALU | fastest reviewed backend at all listed shapes | release reviewed local matrix | accelerator cache not promotable because vector baseline blocked every shape |
| 2026-06-03 | adaptive bounded release matrix | 20260602 | 1024 bounded i64 | rocWMMA | 5095 us median end-to-end with adaptive skip active | release reviewed local matrix | bounded-u64 adaptive groups remain blocked by vector ALU |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 64, 128, 512 | rocWMMA | winner across field-251, ring-251, and ring-255 groups | release reviewed local matrix | cache entries are explicit-modulus scoped |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 1024 ring | CK | 1428 us for modulus 251; 1354 us for modulus 255 | release reviewed local matrix | only same modulus/shape/contract keys are promotable |
| 2026-06-03 | finite-u8 release matrix | 20260602 | 1024 field-251 | hipBLASLt | 2327 us median end-to-end | release reviewed local matrix | only same modulus/shape/contract keys are promotable |
| 2026-06-03 | strict wrap64 release matrix | 20260602 | 64, 128, 512, 1024 | direct HIP | 1828, 2090, 7757, 39359 us median end-to-end | release reviewed local baseline | no public wrap64 accelerator backend exists |
| 2026-06-03 | rocWMMA wrap64 candidate review | 20260603 | 64, 128, 512, 1024 | rocWMMA candidate | lost to direct HIP at every listed shape | release-shape candidate review | internal candidate only; not public or AUTO-eligible |
| 2026-06-03 | direct-HIP uniform-small reuse-B release captures | 1 | 512, 1024 bounded i64 | direct HIP | setup-inclusive 1.51x and 1.32x vs same-backend non-reuse | release local reuse capture; schema/event valid | explicit `--reuse-packed-b` path only; no AUTO/default routing change |
| 2026-06-03 | direct-HIP uniform-small reuse-B release captures | 1 | 512, 1024 bounded u64 | direct HIP | setup-inclusive 1.26x and 1.29x vs same-backend non-reuse | release local reuse capture; schema/event valid | explicit `--reuse-packed-b` path only; no AUTO/default routing change |

## Reuse And Prepack Summary

| Mechanism | Public surface | What is reused | Evidence status | AUTO eligibility |
|---|---|---|---|---|
| Public prepack cache | `rns8_prepack_matrix`, `rns8_get_prepack_cache_key_info` | Backend-specific packed operand identity, currently narrow and backend-limited | Correctness and metadata surface exists; `production_prepack_cache_available` remains `0` for current reviewed captures | Not AUTO-promoted until workload-level policy and reviewed same-contract wins exist |
| Benchmark repeated-A mode | Benchmark-only reuse mode | A-side packing/setup across measured repeats | Event-visible local wins for selected hipBLASLt/vector shapes; some regressions | Not AUTO-eligible because the benchmark changes `pack_mode` and reuse metadata |
| Benchmark repeated-B mode | Benchmark-only reuse mode | B-side packing/setup across measured repeats | Strongest current hipBLASLt local reuse wins; Direct-HIP uniform-small bounded i64/u64 now has event-valid same-backend wins; rocWMMA has narrow B-cache evidence | Not AUTO-eligible because the benchmark changes the workload contract |
| Benchmark repeated-A+B mode | Benchmark-only reuse mode | Both operands packed before the measured repeat loop | Event-visible hipBLASLt wins; vector 512 barely positive over nine repeats | Not AUTO-eligible until the public workload contract asks for reusable operands |
| Persistent matrix reuse | Public matrix/plan/workspace handles | Resident RNS, finite-u8, wrap64, and native vector storage across API calls | Correctness path; required for non-one-shot workflows | AUTO may select only reviewed same-contract cache hits for the explicit plan key |

## Reproduction Command Families

Current bounded-i64 one-shot claims:

```powershell
python tools\benchmark_sweep.py `
  --semantics bounded-i64 `
  --case 512:512,512,512 `
  --case 1024:1024,1024,1024 `
  --backend direct-hip `
  --backend hip-vector-alu-int64 `
  --backend hipblaslt `
  --backend ck `
  --backend rocwmma `
  --warmups 3 `
  --repeats 9 `
  --seed 20260603 `
  --release-review `
  --out-root temp\benchmark-sweeps\windows-gfx1100-release-bounded-i64-current
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

Direct-HIP uniform-small bounded reuse-B captures:

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
