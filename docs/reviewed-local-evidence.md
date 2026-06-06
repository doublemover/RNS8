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
| 2026-06-06 | rank 47 Direct-HIP exact-wide prefix18 fixed-limb export A/B | 20260606 | 1024 exact-wide signed, 4 limbs | Direct HIP | new prefix18 fixed-limb export path: 15481 us median end-to-end vs previous Direct HIP export path at 17537 us, 1.13x faster; export kernel median 2961.60 us vs 4846.72 us, 1.64x faster | same-backend selector-review captures under `temp/rank47-exact-wide-prefix18-fixed-20260606/`; schema v4 valid; required GPU events available; `result_compare.py --allow-export-selector-diff` confirms same shape, selected prefix, limb count, signedness, backend, and target | local Windows `gfx1100` implementation A/B only; no README claim, installed cache entry, default route change, or Linux/CDNA claim |
| 2026-06-06 | rank 47 Direct-HIP exact-wide prefix18 fixed-limb export A/B | 20260606 | 1024 exact-wide unsigned, 4 limbs | Direct HIP | new prefix18 fixed-limb export path: 12629 us median end-to-end vs previous Direct HIP export path at 15616 us, 1.24x faster; export kernel median 613.40 us vs 3940.24 us, 6.42x faster | same-backend selector-review captures under `temp/rank47-exact-wide-prefix18-fixed-20260606/`; schema v4 valid; required GPU events available; `result_compare.py --allow-export-selector-diff` confirms same shape, selected prefix, limb count, signedness, backend, and target | local Windows `gfx1100` implementation A/B only; no README claim, installed cache entry, default route change, or Linux/CDNA claim |
| 2026-06-06 | rank 47 Direct-HIP exact-wide prefix18 fixed-limb export A/B | 20260606 | 2048 exact-wide signed, 4 limbs | Direct HIP | new prefix18 fixed-limb export path: 96647 us median end-to-end vs previous Direct HIP export path at 105418 us, 1.09x faster; export kernel median 10539.60 us vs 16989.40 us, 1.61x faster | same-backend selector-review captures under `temp/rank47-exact-wide-prefix18-fixed-20260606/`; schema v4 valid; required GPU events available; `result_compare.py --allow-export-selector-diff` confirms same shape, selected prefix, limb count, signedness, backend, and target | local Windows `gfx1100` implementation A/B only; no README claim, installed cache entry, default route change, or Linux/CDNA claim |
| 2026-06-06 | rank 47 Direct-HIP exact-wide prefix18 fixed-limb export A/B | 20260606 | 2048 exact-wide unsigned, 4 limbs | Direct HIP | new prefix18 fixed-limb export path: 84122 us median end-to-end vs previous Direct HIP export path at 102559 us, 1.22x faster; export kernel median 2352.62 us vs 13917.90 us, 5.92x faster | same-backend selector-review captures under `temp/rank47-exact-wide-prefix18-fixed-20260606/`; schema v4 valid; required GPU events available; `result_compare.py --allow-export-selector-diff` confirms same shape, selected prefix, limb count, signedness, backend, and target | local Windows `gfx1100` implementation A/B only; no README claim, installed cache entry, default route change, or Linux/CDNA claim |
| 2026-06-06 | rank 47 padded-output compact-D2H selector A/B | 20260606 | 1024 exact-wide signed, padded output, 4 limbs | Direct HIP | compact-D2H candidate is slower: 17502 us median end-to-end vs padded default at 14931 us; export median 7814 us vs 5019 us; D2H event median 2011.8 us vs 60.0 us | same-backend selector-review captures under `temp/rank47-compact-d2h-20260606/`; schema v4 valid; required GPU events available; `result_compare.py --allow-export-selector-diff` confirms same shape, selected prefix, limb count, signedness, backend, target, and padded output contract | local Windows `gfx1100` deprioritization evidence only; compact-D2H is implemented as benchmark-only padded-output staging and is not a README claim, installed cache entry, default route, or Linux/CDNA claim |
| 2026-06-06 | gfx1100 pending-validation exact-wide export selector closeout | 20260606 | 1024 exact-wide signed, 4 limbs | CK | 14659 us median end-to-end; 1.20x vs Direct HIP at 17634 us; CPU reference 3261790 us | release-reviewed local matrix under `temp/gfx1100-pending-validation-20260606/`; CPU, Direct HIP, CK, and rocWMMA captures are schema-valid; required GPU events available; no missing required baselines or duplicate backend records | benchmark/evidence-only fixed-limb export selector candidate; no README claim, installed cache entry, default route, or Linux/CDNA claim |
| 2026-06-06 | gfx1100 pending-validation exact-wide export selector closeout | 20260606 | 1024 exact-wide unsigned, 4 limbs | CK | 15337 us median end-to-end; 1.05x vs Direct HIP at 16137 us; CPU reference 2512390 us | release-reviewed local matrix under `temp/gfx1100-pending-validation-20260606/`; CPU, Direct HIP, CK, and rocWMMA captures are schema-valid; required GPU events available; no missing required baselines or duplicate backend records | benchmark/evidence-only fixed-limb export selector candidate; no README claim, installed cache entry, default route, or Linux/CDNA claim |
| 2026-06-06 | gfx1100 pending-validation exact-wide export selector closeout | 20260606 | 2048 exact-wide signed, 4 limbs | hipBLASLt | 49161 us median end-to-end; 2.11x vs Direct HIP at 103762 us; CPU reference 19073800 us | release-reviewed local matrix under `temp/gfx1100-pending-validation-20260606/`; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA captures are schema-valid; required GPU events available; no missing required baselines or duplicate backend records | benchmark/evidence-only fixed-limb export selector candidate; no README claim, installed cache entry, default route, or Linux/CDNA claim |
| 2026-06-06 | gfx1100 pending-validation exact-wide export selector closeout | 20260606 | 2048 exact-wide unsigned, 4 limbs | hipBLASLt | 44964 us median end-to-end; 2.25x vs Direct HIP at 101018 us; CPU reference 15714000 us | release-reviewed local matrix under `temp/gfx1100-pending-validation-20260606/`; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA captures are schema-valid; required GPU events available; no missing required baselines or duplicate backend records | benchmark/evidence-only fixed-limb export selector candidate; no README claim, installed cache entry, default route, or Linux/CDNA claim |
| 2026-06-05 | large-release-validation bounded 2048 release review | 20260604 | 2048 bounded i64 | CK | 14220 us median end-to-end; 1.57x vs Direct HIP; CPU reference 67134800 us | release reviewed local matrix; required baselines and GPU events available; default local cache installed | explicit 2048 exact plan key only; repeated-B captures are workload-contract evidence, not AUTO cache entries |
| 2026-06-05 | large-release-validation bounded 2048 release review | 20260604 | 2048 bounded u64 | rocWMMA | 15128 us median end-to-end; 1.22x vs Direct HIP; CPU reference 50621600 us | release reviewed local matrix; required baselines and GPU events available; default local cache installed | explicit 2048 exact plan key only; repeated-B captures are workload-contract evidence, not AUTO cache entries |
| 2026-06-05 | large-exploratory bounded 4096 GPU-only classification | 20260605 | 4096 bounded i64/u64, non-reuse and repeated-B | hipBLASLt | 4096 winners: bounded i64 non-reuse 52259 us, bounded i64 repeated-B 40108 us, bounded u64 non-reuse 47467 us, bounded u64 repeated-B 45843 us; same-commit best-path 4096/2048 scaling ranges from 3.43x to 4.22x | exploratory GPU-only matrix; 32 schema-valid captures; required GPU events for every capture; no duplicate backends, git mismatch, or target mismatch | no CPU or runtime vector baselines, so all generic review groups intentionally report missing required baselines; no cache entry installed and no public 4096 promotion claim |
| 2026-06-05 | large-exploratory non-bounded 4096 GPU-only classification plus finite hipBLASLt event rerun | 20260605 | 4096 exact-wide signed/unsigned, finite ring/field u8, and strict wrap64 | CK, hipBLASLt, Direct HIP depending on contract | Event-valid 4096 winners: CK exact-wide signed 279231 us and unsigned 223816 us; hipBLASLt finite field-251 7970 us, ring-251 9101 us, and ring-256 13632 us; CK finite ring-255 10131 us; Direct HIP strict wrap64 352449 us; same-commit best-path 4096/2048 scaling ranges from 1.33x to 7.35x | exploratory GPU-only matrix; original 50 captures had 46 with required GPU events; timing-fallback rerun under `temp/perf-work-queue/finite-hipblaslt-event-reruns-all/` makes all 17 stale finite hipBLASLt reduce-label captures schema/event-valid | no CPU or runtime vector baselines, no cache entry installed, and no public 4096 promotion claim; focused event reruns are blocker cleanup, not same-contract cache review |
| 2026-06-05 | large-release-validation-4096-budgeted bounded gate | 20260605 | 4096 bounded i64 | hipBLASLt | 35303 us median end-to-end; 3.65x vs Direct HIP at 128995 us; 24.17x vs runtime vector at 853232 us; CPU reference 21244300 us | budgeted release-gate group; CPU, Direct HIP, runtime vector, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; release evidence does not imply Linux or Instinct readiness |
| 2026-06-05 | large-release-validation-4096-budgeted bounded gate | 20260605 | 4096 bounded u64 | hipBLASLt | 37543 us median end-to-end; 3.16x vs Direct HIP at 118674 us; 11.11x vs runtime vector at 416960 us; CPU reference 16592600 us | budgeted release-gate group; CPU, Direct HIP, runtime vector, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; release evidence does not imply Linux or Instinct readiness |
| 2026-06-05 | large-release-validation-4096-budgeted finite-u8 hot gate | 20260605 | 4096 field-251 | hipBLASLt | 6396 us median end-to-end; 5.25x vs Direct HIP at 33587 us; CPU reference 4782790 us | budgeted release-gate group; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; exact modulus/shape key only |
| 2026-06-05 | large-release-validation-4096-budgeted finite-u8 hot gate | 20260605 | 4096 ring-251 | hipBLASLt | 7284 us median end-to-end; 4.46x vs Direct HIP at 32508 us; CPU reference 4906360 us | budgeted release-gate group; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; exact modulus/shape key only |
| 2026-06-05 | large-release-validation-4096-budgeted finite-u8 hot gate | 20260605 | 4096 ring-255 | CK | 8786 us median end-to-end; 3.83x vs Direct HIP at 33643 us; CPU reference 4714350 us | budgeted release-gate group; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; exact modulus/shape key only |
| 2026-06-05 | large-release-validation-4096-budgeted finite-u8 hot gate | 20260605 | 4096 ring-256 | hipBLASLt | 6881 us median end-to-end; 4.73x vs Direct HIP at 32520 us; CPU reference 4685440 us | budgeted release-gate group; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA baselines are schema-valid with required GPU events and no missing baselines; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; exact modulus/shape key only |
| 2026-06-05 | large-release-validation-4096-budgeted wrap64 gate | 20260605 | 4096 strict wrap64 | Direct HIP | 295657 us median end-to-end; 348.06x vs byte-limb reference at 102905000 us; checksum `13518998852724169131` | budgeted release-gate group; byte-limb reference and Direct HIP are schema-valid with 3 warmups, 9 repeats, matching checksum, and required wrap64 GPU events | no cache entry installed; strict wrap64 Direct HIP is a correctness backend, not a public accelerator/cache candidate |
| 2026-06-05 | large-release-validation-4096-budgeted exact-wide gate | 20260605 | 4096 exact-wide signed, 4 limbs | hipBLASLt | current same-commit closeout: 176943 us median end-to-end; 3.61x vs Direct HIP at 639360 us; 639.1x vs CPU reference at 113085000 us; checksum `5508849193854467465` | current-commit rerun under `large-4096-signed-cache-closeout-current` plus `large-4096-signed-cpu-current`; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA captures are schema-valid with 3 warmups, 9 repeats, matching checksum, compatible git metadata, and required GPU events; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; supersedes the older mixed-commit signed 4096 gate row |
| 2026-06-05 | large-release-validation-4096-budgeted exact-wide gate | 20260605 | 4096 exact-wide unsigned, 4 limbs | hipBLASLt | 162382 us median end-to-end; 3.78x vs Direct HIP at 614116 us; 649.5x vs CPU reference at 105462000 us; checksum `9643325300233475427` | budgeted release-gate group; CPU, Direct HIP, hipBLASLt, CK, and rocWMMA captures are schema-valid with 3 warmups, 9 repeats, matching checksum, and required GPU events; promotion ledger installed the reviewed cache entry with zero blockers | local Windows `gfx1100` cache entry installed; exact semantic/shape/limb key only |
| 2026-06-05 | post-fix large-release-validation finite-u8 2048 release review | 20260605 | 2048 ring-251 | hipBLASLt | 3244 us median end-to-end; 3.47x vs Direct HIP; CPU reference 810162 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache refreshed | supersedes the earlier same-day rocWMMA 4216 us entry; explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-05 | post-fix large-release-validation finite-u8 2048 release review | 20260605 | 2048 ring-255 | hipBLASLt | 2425 us median end-to-end; 3.05x vs Direct HIP; CPU reference 637930 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache refreshed | supersedes the earlier same-day 2845 us hipBLASLt entry; explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-05 | post-fix large-release-validation finite-u8 2048 release review | 20260605 | 2048 ring-256 | hipBLASLt | 3017 us median end-to-end; 1.92x vs Direct HIP; CPU reference 854675 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache refreshed | supersedes the earlier same-day rocWMMA 5011 us entry after the finite hipBLASLt reducer-event fix; explicit modulus/shape key only |
| 2026-06-05 | post-fix large-release-validation finite-u8 2048 release review | 20260605 | 2048 field-251 | hipBLASLt | 3079 us median end-to-end; 2.97x vs Direct HIP; CPU reference 757623 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache refreshed | supersedes the earlier same-day 4432 us hipBLASLt entry; explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-05 | finite-u8 generic ring 2048 release refresh | 20260605 | 2048 ring-127 | rocWMMA | 3427 us median end-to-end; 1.59x vs Direct HIP; CPU reference 764044 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | explicit generic modulus/shape key only; stale hipBLASLt loser lacked required events and lost to Direct HIP; focused timing rerun cleared event capture only |
| 2026-06-05 | finite-u8 generic ring 2048 release refresh | 20260605 | 2048 ring-253 | rocWMMA | 4856 us median end-to-end; 1.12x vs Direct HIP; CPU reference 864589 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | explicit generic composite modulus/shape key only; Windows `gfx1100` only |
| 2026-06-05 | finite-u8 field release refresh | 20260605 | 2048 field-127 | CK | 3424 us median end-to-end; 1.57x vs Direct HIP; CPU reference 781139 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | explicit generic prime-field modulus/shape key only; stale hipBLASLt loser lacked required events and lost to Direct HIP; focused timing rerun cleared event capture only |
| 2026-06-05 | finite-u8 field release refresh | 20260605 | 512 field-251 | rocWMMA | 1241 us median end-to-end; 1.05x vs Direct HIP; CPU reference 16019 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | clears previous hipBLASLt event-debt note; hipBLASLt had required events but lost to Direct HIP and rocWMMA |
| 2026-06-05 | skinny-GEMV current release refresh | 20260605 | bounded-i64 512x1x512, bounded-i64 256x1x4096, bounded-u64 1024x1x1024 | Direct HIP | 1689 us, 2712 us, and 2307 us median end-to-end respectively; no accelerator or vector path beat Direct HIP | release reviewed local matrix; required baselines and GPU events available; no missing baselines, duplicate backends, or incompatible metadata | no cache entry installed; closes current N=1 selector refresh as Direct-HIP-favored scenario evidence |
| 2026-06-05 | many-small current release review plus host-batch comparison | 20260605 | bounded-i64 32/128, bounded-u64 64 and 128x1x1024, exact-wide signed 64, finite ring 64 mod 251/255, hostbatch32/64/128 variants | CPU, Direct HIP, or vector depending on proxy; Direct-HIP hostbatch32 for exact-wide signed 64 | Independent-call winners are CPU for bounded-i64 32, bounded-u64 64, and finite-u8 64; runtime vector ALU for bounded-i64 128; Direct HIP for bounded-u64 128x1x1024 and exact-wide signed 64; Direct-HIP exact-wide signed 64 hostbatch32 is 1903 us per task versus 3880 us independent Direct HIP, 2.04x faster | same-commit release matrix; 61 schema-valid captures; no missing required baselines, duplicate backend records, git mismatch, or target mismatch; required GPU events for host-batch GPU captures; no cache entries promoted | host-batch result is benchmark-only workload evidence; the other 19 host-batch candidates lose to same-backend or fastest independent-call baselines; no public batching route or AUTO cache entry installed |
| 2026-06-05 | many-small grouped-dispatch exact-wide follow-up | 20260605 | exact-wide signed 64, group32 | Direct HIP grouped-dispatch benchmark path | `tools/many_small_grouped_report.py` classifies the async export-slab grouped capture as a candidate win: 792.66 us per task versus 3880 us independent Direct HIP and 1902.97 us hostbatch32, 4.89x and 2.40x faster respectively; it is also 1.25x faster than the earlier 991.94 us grouped-dispatch capture | current branch grouped capture is schema-valid with three warmups, nine repeats, checksum parity with the earlier grouped capture, and required GPU events; comparison report reuses the reviewed many-small independent and hostbatch32 captures for the same seed/shape/contract | benchmark-owned persistent-task evidence only; async slab path is limited to contiguous exact-wide grouped export; not a device queue, public grouped API, AUTO cache entry, or Linux/Instinct claim |
| 2026-06-05 | many-small grouped-dispatch one-kernel export follow-up | 20260605 | exact-wide signed 64 group32; exact-wide unsigned 64 group32 smoke | Direct HIP grouped-dispatch benchmark path | signed one-kernel grouped export capture is a candidate win at 795.19 us per task, 4.88x faster than independent Direct HIP and 2.39x faster than hostbatch32; export average drops from 5670 us in the async slab capture to 1212 us, with event median `exact_wide_export_kernel=636.39 us` and `exact_wide_export_d2h=80.0 us`; unsigned smoke is schema/event-valid at 703.34 us per task | captures under `temp/perf-work-queue/many-small-grouped-dispatch-kernel-current/`; signed checksum matches the async slab capture; schema v4 and required GPU events pass; `many_small_grouped_report.py` compares signed against the reviewed many-small independent and hostbatch32 baselines | signed end-to-end median is effectively flat versus the 792.66 us async slab row; unsigned smoke is superseded by the focused unsigned baseline closeout row below; benchmark-owned exact-wide contiguous export evidence only, not a full device grouped dispatcher, generic public API, AUTO cache entry, or Linux/Instinct claim |
| 2026-06-05 | many-small grouped-dispatch grouped pack+export follow-up | 20260605 | exact-wide signed 64 group32; exact-wide unsigned 64 group32 smoke | Direct HIP grouped-dispatch benchmark path | signed grouped pack+export capture is a candidate win at 228.06 us per task, 17.01x faster than independent Direct HIP, 8.34x faster than hostbatch32, and 3.49x faster than the previous one-kernel grouped-export capture; host aggregate pack average drops from 16873 us to 778 us, and GPU event pack average drops from 12815 us to 461 us; unsigned smoke is schema/event-valid at 249.63 us per task | captures under `temp/perf-work-queue/many-small-grouped-pack-current/`; both captures are schema v4 and required-event valid; `result_compare.py` shows matching contract and GPU compatibility against the previous grouped-export captures; `many_small_grouped_report.py` compares signed against the reviewed many-small independent and hostbatch32 baselines | benchmark-owned same-shape grouped evidence only; pack and export are grouped but GEMM still loops through resident tasks host-side; unsigned smoke is superseded by the focused unsigned baseline closeout row below; no generic public grouped API, AUTO cache entry, device queue, or Linux/Instinct claim |
| 2026-06-05 | many-small grouped-dispatch grouped pack+GEMM+export follow-up | 20260605 | exact-wide signed 64 group32; exact-wide unsigned 64 group32 smoke | Direct HIP grouped-dispatch benchmark path | signed grouped pack+GEMM+export capture is a candidate win at 66.47 us per task, 58.37x faster than independent Direct HIP, 28.63x faster than hostbatch32, and 3.43x faster than the grouped pack+export capture; event median `rns_gemm` drops from 4701.94 us to 168.70 us; unsigned smoke is schema/event-valid at 63.56 us per task | captures under `temp/perf-work-queue/many-small-grouped-gemm-current/`; both captures are schema v4 and required-event valid; `result_compare.py` shows matching contract and GPU compatibility against the grouped pack+export captures; `many_small_grouped_report.py` compares signed against the reviewed many-small independent and hostbatch32 baselines | benchmark-owned same-shape grouped evidence only; pack, RNS GEMM, and contiguous exact-wide export are grouped, but there is still no generic public grouped API, generic descriptor queue, AUTO cache entry, or Linux/Instinct claim; unsigned smoke is superseded by the focused unsigned baseline closeout row below |
| 2026-06-05 | many-small grouped-dispatch unsigned baseline closeout | 20260605 | exact-wide unsigned 64 group32 | Direct HIP grouped-dispatch benchmark path | grouped pack+GEMM+export capture is a candidate win at 79.09 us per task, 18.70x faster than independent Direct HIP at 1479 us and 13.48x faster than Direct-HIP hostbatch32 at 1066 us per task; combined checksum is `666386315613239916` | captures under `temp/perf-work-queue/many-small-grouped-unsigned-current/`; five schema-v4 captures cover CPU and Direct-HIP independent baselines, CPU and Direct-HIP hostbatch32 baselines, and Direct-HIP grouped-dispatch; required GPU events pass for all Direct-HIP captures; `many_small_grouped_report.py` now requires same-task-count hostbatch checksum parity and reports one candidate win with no blockers | benchmark-owned same-shape exact-wide grouped evidence only; no generic public grouped API, generic descriptor queue, AUTO cache entry, or Linux/Instinct claim |
| 2026-06-05 | many-small grouped-dispatch descriptor-contract smoke | 20260605 | exact-wide unsigned 64 group32 | Direct HIP grouped-dispatch benchmark path | rebuilt grouped capture emits schema-v4 `task_descriptor_contract` with same-shape resident task triplets, single same-shape bucket, task count 32, shape/tile/prefix key, exact-wide limb host output domain, per-task source-version repack policy, one workspace per task under one shared plan, combined per-task checksum policy, fail-fast per-task status, and `device_pointer_tables_and_compact_slabs` descriptor policy | capture under `temp/perf-work-queue/grouped-descriptor-contract-smoke/`; `benchmark_schema.py` validates the contract; `gpu_event_report.py --require-events` passes for Direct-HIP pack, grouped GEMM, and exact-wide export phases; `windows-release` rebuilt `rns8-bench` before capture | contract/schema smoke only; no new speedup claim, no generic public grouped API, no AUTO cache entry, and no Linux/Instinct claim |
| 2026-06-05 | many-small bounded grouped pack+GEMM controls | 20260605 | bounded-i64 64 group32; bounded-u64 64 group32 | Direct HIP grouped-dispatch benchmark path | `device_grouped_pack_gemm_host_exports` grouped-dispatch captures are candidate wins for both bounded contracts: i64 is 544.56 us per task, 1.97x faster than the best independent CPU baseline at 1075 us and 2.19x faster than Direct-HIP hostbatch32 at 1190.03 us; u64 is 532.47 us per task, 1.46x faster than the best independent CPU baseline at 775 us and 2.11x faster than Direct-HIP hostbatch32 at 1124.94 us | captures under `temp/perf-work-queue/grouped-bounded-release-controls/`; eight schema-v4 captures cover CPU and Direct-HIP independent baselines, Direct-HIP hostbatch32, and Direct-HIP grouped-dispatch for both semantics; required GPU events pass for every Direct-HIP capture; `many_small_grouped_report.py` reports two candidate wins with checksum parity and schema-valid `device_pointer_tables_and_compact_slabs` descriptor contracts | historical benchmark-owned same-shape grouped evidence; this row grouped pack and RNS GEMM but still used per-task bounded CRT export, and is superseded by the 2026-06-06 bounded grouped export closeout below; no generic public grouped API, AUTO cache entry, generic descriptor queue, or Linux/Instinct claim |
| 2026-06-06 | many-small bounded grouped export closeout | 20260606 | bounded-i64 64 group32; bounded-u64 64 group32 | Direct HIP grouped-dispatch benchmark path | `device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h` grouped-dispatch captures are candidate wins for both bounded contracts after bounded CRT export is grouped into one device output slab plus one compact D2H: i64 is 53.625 us per task, 19.97x faster than the best independent CPU baseline at 1071 us and 23.32x faster than Direct-HIP hostbatch32 at 1250.59 us; u64 is 53.25 us per task, 10.95x faster than the best independent CPU baseline at 583 us and 22.95x faster than Direct-HIP hostbatch32 at 1222.19 us | captures under `temp/perf-queue-grouped-bounded-export-release/`; eight schema-v4 captures cover CPU and Direct-HIP independent baselines, Direct-HIP hostbatch32, and Direct-HIP grouped-dispatch for both semantics; required GPU events pass for every Direct-HIP capture; `many_small_grouped_report.py` reports two candidate wins with checksum parity and schema-valid `device_pointer_tables_and_compact_slabs` descriptor contracts | benchmark-owned same-shape grouped evidence only; this closes the bounded 64 group32 per-task export bottleneck but does not by itself validate generic public grouped routing, install an AUTO cache entry, prove a generic descriptor queue, or imply Linux/Instinct readiness |
| 2026-06-06 | broader many-small grouped-dispatch matrix | 20260606 | bounded-i64 64 group32; bounded-i64 128 group64; bounded-u64 64 group32; bounded-u64 128x1x1024 group128; finite-ring u8 mod251 64 group32 | Direct HIP grouped-dispatch benchmark path | `many_small_grouped_report.py` classifies all five grouped rows as candidate wins with no missing baselines: bounded-i64 64 at 57.41 us per task, 20.02x vs best independent and 19.98x vs hostbatch32; bounded-i64 128 at 55.88 us, 26.59x vs best independent and 23.09x vs hostbatch64; bounded-u64 64 at 59.38 us, 16.62x vs best independent and 15.87x vs hostbatch32; bounded-u64 128x1x1024 at 100.76 us, 16.14x vs best independent and 14.62x vs hostbatch128; finite-ring u8 mod251 64 at 33.53 us, 2.33x vs best independent and 26.41x vs hostbatch32 | captures under `temp/perf-queue-grouped-broader-release/`; 33 schema-v4 captures with seed 20260606, three warmups, nine repeats, CPU and Direct-HIP independent controls, matching Direct-HIP hostbatch controls, five Direct-HIP grouped rows, required GPU events for 19 GPU captures, and checksum parity in `many_small_grouped_report.py` | benchmark-owned same-shape grouped evidence only; finite row uses grouped finite pack, grouped finite GEMM, grouped finite canonical export, and one compact output D2H, but no generic public grouped ABI, AUTO route, cache entry, generic descriptor queue, or Linux/Instinct claim is made |
| 2026-06-06 | exact-wide 128 grouped-dispatch controls | 20260606 | exact-wide signed 64 and 128 group32; exact-wide unsigned 64 and 128 group32 | Direct HIP grouped-dispatch benchmark path | `many_small_grouped_report.py` classifies all four exact-wide grouped rows as candidate wins with no missing baselines: signed 64 at 68.12 us per task, 18.47x vs independent Direct HIP and 16.78x vs hostbatch32; signed 128 at 184.97 us, 7.93x vs independent Direct HIP and 7.59x vs hostbatch32; unsigned 64 at 75.53 us, 16.16x vs independent Direct HIP and 13.96x vs hostbatch32; unsigned 128 at 155.88 us, 11.66x vs independent Direct HIP and 8.94x vs hostbatch32 | captures under `temp/perf-queue-grouped-exact128-release/`; 20 schema-v4 captures with seed 20260606, three warmups, nine repeats, CPU and Direct-HIP independent controls, matching Direct-HIP hostbatch32 controls, four Direct-HIP grouped rows, required GPU events for 12 GPU captures, and checksum parity in `many_small_grouped_report.py` | benchmark-owned same-shape exact-wide grouped evidence only; no generic public grouped ABI, AUTO route, cache entry, generic descriptor queue, or Linux/Instinct claim is made |
| 2026-06-05 | RNS-chain final-output control report | 20260605 | bounded-i64 128 chain3 final output; exact-wide signed 128 chain3 final output | Direct HIP final-output RNS-chain benchmark path | `tools/rns_chain_report.py` classifies two Direct-HIP final-output chain rows as candidate same-contract GPU-vs-CPU wins: bounded-i64 128 at 1614 us versus CPU 15745 us, and exact-wide signed 128 at 1358 us versus CPU 41656 us. The same report deprioritizes Direct-HIP reusable-B after setup for both contracts: 3002 us versus 1614 us bounded and 3161 us versus 1358 us exact-wide signed | captures under `temp/perf-work-queue/rns-chain-final-output-current/`; six schema-v4 captures with three warmups and nine repeats; required GPU events pass for every non-CPU capture; report has no missing CPU baselines or missing same-backend non-reuse baselines | final-output chain evidence only; not an AUTO/cache entry, not an independent export/repack baseline, and not a Linux/Instinct claim |
| 2026-06-05 | RNS-chain bounded independent export/repack controls | 20260605 | bounded-i64 128 chain3; bounded-u64 256 chain3 | Direct HIP final-output RNS-chain benchmark path | `tools/rns_chain_report.py` classifies two Direct-HIP resident final-output chain rows as candidate same-contract wins against same-backend independent export/repack controls: bounded-i64 128 at 1805 us versus 3324 us independent, 1.84x faster, and bounded-u64 256 at 2240 us versus 4387 us independent, 1.96x faster | captures under `temp/perf-work-queue/rns-chain-independent-final-output-current/`; eight schema-v4 captures with three warmups and nine repeats; required GPU events pass for all Direct-HIP resident and independent captures; pairwise `result_compare.py` reports matching contracts and compatible GPU metadata | bounded benchmark-control evidence only; no AUTO/cache entry, no public lazy-output API, exact-wide controls are tracked in the separate row below, and no Linux/Instinct claim |
| 2026-06-05 | RNS-chain exact-wide independent export/repack controls | 20260605 | exact-wide signed 128 chain3; exact-wide unsigned 256 chain3 | Direct HIP final-output RNS-chain benchmark path | `tools/rns_chain_report.py` classifies two Direct-HIP resident final-output chain rows as candidate same-contract wins against same-backend independent export/repack controls: signed 128 at 1770 us versus 17344 us independent, 9.80x faster, and unsigned 256 at 2842 us versus 30687 us independent, 10.80x faster | captures under `temp/perf-work-queue/exact-wide-rns-chain-independent-final-output-current/`; eight schema-v4 captures with three warmups and nine repeats; required GPU events pass for all Direct-HIP resident and independent captures; pairwise `result_compare.py` reports matching contracts and compatible GPU metadata | exact-wide benchmark-control evidence only; no AUTO/cache entry, no public lazy-output API, no broader exact-wide chain matrix, and no Linux/Instinct claim |
| 2026-06-05 | reuse contract setup-amortized comparison reports | 20260605 | bounded repeated-B 2048 and 4096 | hipBLASLt, CK, rocWMMA depending on contract | CPU-backed 2048 report classifies 4/12 repeated-B captures as setup-inclusive workload candidates: hipBLASLt bounded-i64, CK bounded-u64, hipBLASLt bounded-u64, and rocWMMA bounded-u64; bounded 4096 exploratory report classifies 3/16 repeated-B captures as workload candidates: CK bounded-i64 2048, hipBLASLt bounded-i64 2048, and hipBLASLt bounded-i64 4096 | `tools/reuse_contract_report.py` report; computes setup-inclusive per-repeat medians, same-backend and fastest-non-reuse speedups, break-even repeats, source-identity metadata, and GPU event availability | explicit workload-contract evidence only; no AUTO cache entry installed; Direct-HIP large repeated-B is currently deprioritized against fastest non-reuse baselines |
| 2026-06-05 | reuse-contract A/B/A+B release matrix | 20260605 | bounded i64/u64 1024 and 2048, non-reuse plus stable-A, stable-B, and stable-A+B | hipBLASLt, Direct HIP, CK, and rocWMMA depending on explicit contract | 96 captures across 16 release groups; `reuse_contract_report.py` classifies 17/72 reuse comparisons as setup-inclusive workload candidates, 43 as deprioritized, 12 as experimental, and zero as missing a baseline. hipBLASLt 2048 stable-A and stable-A+B are explicit workload candidates for bounded i64 and u64; bounded-u64 2048 stable-A+B is strongest at 9198 us setup-inclusive per repeat, 3.99x faster than same-backend non-reuse and 2.35x faster than the same-run fastest non-reuse baseline | release reviewed same-commit matrix with CPU, Direct HIP, runtime vector ALU, hipBLASLt, CK, and rocWMMA where supported; no missing required baselines, duplicate backends, target mismatch, or git mismatch; required GPU events available for all non-CPU captures | explicit reusable-input workload evidence only; no AUTO cache entry installed; hipBLASLt 1024 stable-A and stable-A+B are deprioritized; bounded-u64 2048 stable-B has a same-run win but remains conservative because an older installed rocWMMA 2048 non-reuse row was faster than this same-run baseline |
| 2026-06-05 | bound-discovery setup-inclusive release matrix | 20260605 | bounded-i64 256/1024 adaptive bands and bounded-u64 512x1024 adaptive bands | none | 51 captures; 18 global input-scan candidates and 15 proof-mask per-tile candidates; 0 setup-inclusive candidate wins; hipBLASLt global input-scan improved over its own 256 bounded-i64 static baseline and CK global input-scan improved over its own rectangular bounded-u64 static baseline, but both lost to the fastest static workload baseline after scan setup cost | release reviewed local matrix; nine groups; no missing required baselines, duplicate backends, git mismatch, or target mismatch; required GPU events for all non-CPU captures; `tools/bound_discovery_report.py` compared scan setup plus median end-to-end timing | rank 21 closed as no-promotion evidence; no cache entry installed; per-tile proof masks are event-visible and correct but exact tile-bound scans dominate setup-inclusive timing |
| 2026-06-05 | large-release-validation wrap64 2048 release review | 20260605 | 2048 strict wrap64 | Direct HIP | 58331 us median end-to-end; 230.1x vs CPU byte-limb reference at 13423400 us | release reviewed local CPU/direct matrix; required Direct-HIP GPU events available; no missing baselines, duplicate backends, or incompatible metadata | not an AUTO cache entry because strict wrap64 Direct HIP is a correctness backend, not a public accelerator/cache candidate |
| 2026-06-05 | large-release-validation exact-wide 2048 release review | 20260605 | 2048 signed | hipBLASLt | 59074 us median end-to-end; 2.23x vs Direct HIP; CPU reference 19040900 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | explicit exact-wide signed 2048 key only; export-bound after GEMM acceleration |
| 2026-06-05 | large-release-validation exact-wide 2048 release review | 20260605 | 2048 unsigned | hipBLASLt | 40985 us median end-to-end; 3.04x vs Direct HIP; CPU reference 15742000 us | release reviewed local matrix; required baselines and promoted GPU events available; default local cache installed | explicit exact-wide unsigned 2048 key only; export-bound after GEMM acceleration |
| 2026-06-05 | exact-wide 64/128 current-v2 release review | 20260605 | 64 unsigned | hipBLASLt | 4611 us median end-to-end; 1.67x vs Direct HIP; CPU reference 9686 us | release reviewed local matrix; required baselines and GPU events available; default local cache installed | signed 64, signed 128, and unsigned 128 stayed on Direct HIP |
| 2026-06-05 | finite-u8 generic field-127 release refresh | 20260605 | 512 field-127 | CK | 1289 us median end-to-end; 1.10x vs Direct HIP; CPU reference 12367 us | release reviewed local matrix; required baselines and GPU events available; default local cache entry refreshed | hipBLASLt had required GPU events in this rerun but lost to Direct HIP and CK |
| 2026-06-04 | bounded-i64 v2 one-shot release review | 20260604 | 512 | direct HIP | 1851 us median end-to-end; no accelerator win; rocWMMA v2 2591 us, vector ALU 6147 us, CK v2 7172 us, hipBLASLt v2 10101 us | release reviewed local matrix; required GPU events available | no cache entry; Direct HIP retained for this shape |
| 2026-06-04 | bounded-i64 v2 one-shot release review | 20260604 | 1024 | hipBLASLt | 4174 us median end-to-end; 1.09x vs Direct HIP; 8.13x vs vector ALU | release reviewed local matrix; required GPU events available; default local cache installed | current local cache contains this v2 entry only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 small current-v2 release review | 20260604 | 128 ring-251 | rocWMMA | 1136 us median end-to-end; 1.11x vs Direct HIP; CPU reference 1370 us | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 small current-v2 release review | 20260604 | 128 ring-256 | rocWMMA | 1132 us median end-to-end; 1.02x vs Direct HIP; CPU reference 1730 us | release reviewed local matrix; required GPU events available; default local cache installed | narrow win; explicit modulus/shape key only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-251 | rocWMMA | 1709 us median end-to-end; 2.74x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-255 | CK | 1938 us median end-to-end; 3.00x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 512 ring-256 | rocWMMA | 1365 us median end-to-end; 4.08x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 ring-256 | hipBLASLt | 1792 us median end-to-end; 7.05x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | explicit modulus/shape key only; Windows `gfx1100` only |
| 2026-06-04 | finite-u8 current-v2 release review | 20260604 | 1024 field-251 | CK | 1860 us median end-to-end; 5.68x vs Direct HIP | release reviewed local matrix; required GPU events available; default local cache installed | field-251 512 was refreshed on 2026-06-05 and now has a separate rocWMMA cache entry |
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
| 2026-06-03 | direct-HIP public one-shot colpair gate | 31 | 512 bounded u64 | direct HIP | 1.09x average end-to-end, 1.21x median end-to-end, and 1.06x average GEMM-event speedup vs prior one-shot v1 kernel | final release captures schema/event valid; before captures intentionally stale under new schema | routed only for bounded-u64 Direct-HIP one-shot `m/n/k >= 512`; smaller u64 stays on v1 |
| 2026-06-05 | direct-HIP public one-shot colpair gate | 20260605 | 512 bounded i64 | direct HIP | 2.72x average end-to-end, 3.07x median end-to-end, 3.02x median one-shot GPU API event speedup, and matching checksum vs prior one-shot v1 kernel | final release capture schema/event valid; before capture valid under prior schema and intentionally stale under current large-i64 colpair gate | explicit Direct-HIP public one-shot route only; persistent resident Direct-HIP at the same shape remains faster |
| 2026-06-04 | direct-HIP uniform-small reuse-A colpair fixed-prefix captures | 1 | 512, 1024 bounded i64 | direct HIP | setup-inclusive 3.04x at 512 and 1.32x at 1024 vs clean `a75b0a2` same-contract repeated-A baseline over 33 repeats | release local reuse capture; schema/event/checksum valid | explicit fixed-prefix `--reuse-packed-a` path only; no AUTO/default routing change |
| 2026-06-04 | direct-HIP uniform-small reuse-A colpair fixed-prefix captures | 1 | 512, 1024 bounded u64 | direct HIP | setup-inclusive 1.33x at 512 and 1.30x at 1024 vs clean `a75b0a2` same-contract repeated-A baseline over 33 repeats | release local reuse capture; schema/event/checksum valid | explicit fixed-prefix `--reuse-packed-a` path only; no AUTO/default routing change |

## Reuse And Prepack Summary

| Mechanism | Public surface | What is reused | Evidence status | AUTO eligibility |
|---|---|---|---|---|
| Public prepack cache | `rns8_prepack_matrix`, `rns8_get_prepack_cache_key_info` | Backend-specific packed operand identity, currently narrow and backend-limited | Correctness and metadata surface exists; `production_prepack_cache_available` remains `0` for current reviewed captures | Not AUTO-promoted until workload-level policy and reviewed same-contract wins exist |
| Benchmark repeated-A mode | Benchmark-only reuse mode | A-side packing/setup across measured repeats | The release-contract matrix promotes explicit 2048 hipBLASLt stable-A candidates for bounded i64/u64, keeps selected Direct-HIP fixed-prefix wins as explicit paths, and deprioritizes hipBLASLt 1024 stable-A against the fastest non-reuse baseline | Not AUTO-eligible because the benchmark changes `pack_mode` and reuse metadata |
| Benchmark repeated-B mode | Benchmark-only reuse mode | B-side packing/setup across measured repeats | Mixed setup-inclusive evidence: hipBLASLt bounded-u64 1024/2048, CK/rocWMMA selected u64, and older Direct-HIP uniform-small paths have explicit wins, while hipBLASLt bounded-i64 2048 B and several Direct-HIP/vector rows lose the fastest-baseline gate | Not AUTO-eligible because the benchmark changes the workload contract |
| Benchmark repeated-A+B mode | Benchmark-only reuse mode | Both operands packed before the measured repeat loop | The release-contract matrix promotes explicit 2048 hipBLASLt stable-A+B candidates for bounded i64/u64 and deprioritizes hipBLASLt/vector 1024 full reuse after setup | Not AUTO-eligible until the public workload contract asks for reusable operands |
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

Direct-HIP bounded-i64 public one-shot colpair gate:

```powershell
build\windows-msvc-hip-release\rns8-bench.exe `
  --backend hip-direct `
  --semantics bounded-i64 `
  --oneshot `
  --m 512 --n 512 --k 512 `
  --prefix-policy fixed-requested --max-prefix 9 `
  --warmups 3 --repeats 9 `
  --seed 20260605

python tools\benchmark_schema.py <capture>
python tools\gpu_event_report.py --require-events <capture>
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
