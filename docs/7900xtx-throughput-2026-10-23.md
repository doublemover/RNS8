# RX 7900 XTX local throughput — 2026-10-23

Measured on AMD Radeon RX 7900 XTX (`gfx1100`), Windows, VS2026/MSVC
19.51.36260, HIP 7.1.51803 / Clang 21, runtime 71260602. The static calibration
executable was rebuilt from clean source `8a4d1219acad0288eb773c27221cae78796c1e32`,
RelWithDebInfo, optional rocWMMA enabled, generated WMMA and Garner table disabled.

Each row uses M=N=K=256,
seed 3135, five independently launched processes in randomized order, three
warmups and 15 measured calls per process. All 525 measured outputs passed
Boost reference and canary checks. The reported time is the median of five
process medians; the range is their minimum and maximum, not a confidence interval.

Effective useful-product TOPS = `2*M*N*K / (call_microseconds * 1e6)`. This
normalizes to logical GEMM work, without multiplying by RNS plane count or
counting epilogue arithmetic. Inputs are already device-resident. Timed host
calls include candidate-local allocation, validation/status work and
synchronization. Initial uploads, output poisoning and verification copies are
excluded. These small-workload measurements do not estimate peak hardware TOPS,
large-matrix throughput, transfer-inclusive performance, or automatic dispatch.

## Matched native integer output

All rows use signed byte-domain values stored in native i64 arrays and produce
bounded exact native i64 device output. WMMA and scalar calls discover actual
input bounds; vector ALU checks overflow status. The RNS baseline discovers
bounds, packs three residue planes, computes and reconstructs native output.

| Approach | Median call, µs | Process median range, µs | Effective integer TOPS |
|---|---:|---:|---:|
| Raw-byte native WMMA | 661.6 | 644.9–749.6 | 0.05072 |
| Native scalar | 664.8 | 582.1–736.8 | 0.05047 |
| Native vector ALU | 716.0 | 663.7–730.4 | 0.04686 |
| Direct HIP RNS pipeline, prefix 3 | 969.3 | 887.7–1020.7 | 0.03462 |

The WMMA and scalar ranges overlap; their small median difference is not a
demonstrated selection advantage. The RNS row is the direct-HIP baseline, not
a claim to use the fastest available RNS implementation.

## Matched resident residue epilogue

These rows compute `alpha*AB + beta*D` into nine residue planes, with original
integer coefficients alpha=INT64_MIN and beta=INT64_MAX. They do not export a
native integer or certify unique reconstruction of the original full-width
expression. All reserve the same intermediate footprint; only the unfused
route materializes it. This table must not be ranked against native-output rows.

| Approach | Median call, µs | Process median range, µs | Effective integer TOPS |
|---|---:|---:|---:|
| Direct HIP fused epilogue | 553.2 | 521.5–647.1 | 0.06066 |
| rocWMMA fused epilogue | 244.0 | 200.5–265.9 | 0.13752 |
| Direct HIP GEMM + separate epilogue | 480.0 | 404.0–569.5 | 0.06991 |

For this matched workload, rocWMMA reduces median call time by 1.97x versus
the unfused baseline and 2.27x versus direct fused. Its five process medians
were below every process median of either baseline. This is a local result;
held-out shapes, frequency/thermal control, counters and promotion gates remain.

## Reproduction and provenance

`temp/7900xtx-summary-cases.json` lists the four `narrow_*` and three
`epilogue_*` candidates above: M/N/K=256, group=1, unsigned_inputs=false,
prefix=1 (CLI placeholder) for narrow cases and prefix=9 for epilogues.

```powershell
python tools/windows_dev.py cmake --build build/issues-rocwmma --target rns8-native-calibrate -j 8
python tools/collect_native_calibration.py --exe build/issues-rocwmma/rns8-native-calibrate.exe --cases temp/7900xtx-summary-cases.json --output temp/7900xtx-summary-captures --runs 5 --repeats 15 --warmups 3
```

Raw records remain in ignored `temp/`, including commands, UUIDs, HIP event
components, reference checks, device identity and manifest v2 dependency hashes.
No captures are eligible for promotion. The unchanged whole-object rocWMMA
no-divide ISA gate and all 430 registered rocWMMA-build tests passed for the
source repair before this clean calibration build; captures do not separately
attest selected-symbol ISA or hardware counters.

- Campaign: `d64253d6-7b37-43a5-a487-67c421eff813` (35 independent records).
- Executable SHA-256: `24a5dc3028db732c251315ed01d9aa5992a17d6d9b1521625cbdfe05930246d4`.
- SHA-256 of sorted raw-record filename/hash index: `bdbaa934c7ff7929b1791f6805183ecfce17bc904a9884adb5d720a4aa82aa1d`.
- rocWMMA header-tree SHA-256: `4c6dc7334f7d4d7bc927db66dfef705379185455e2da473ae5902a9ef81ee714`.
