# RX 7900 XTX native WMMA and queue measurements

The subsequent reciprocal-conversion sweep at `436d235` reduces ordinary
group-two WMMA from **12.220 ms to 1.586 ms (7.71×)** and group four from
**7.630 ms to 1.559 ms (4.89×)** against the matching `6d8ab25` cases below.
All 70 new captures passed. Pack + direct GEMM remains faster at 0.784 ms for
this full-width 512³ product; the optimization does not change default dispatch.

The first sweep isolates worker count after uniform queue specialization.
Increasing the group-one queue from 32 to 1024 workers improves its sustained
call time by **19.1×**, from 55.70 ms to 2.91 ms. Ordinary grid WMMA still takes
2.05 ms, and packing followed by direct GEMM takes 0.729 ms for the same output
contract. This is a scheduling improvement within the private queue, not a reason
to replace ordinary launches for a large uniform product.

## Configuration and validation

- Clean compiled source: `6d8ab25`; collector and smoke manifest: `d18fef7`.
- RX 7900 XTX, `gfx1100`, Windows, HIP 7.1, VS2026. The all-feature configuration
  is described in [the local validation record](local-all-features-20261023.md).
- Calibration executable SHA256:
  `fdd62acb40049c48c7c5d54c6e271d5b684a6978e3e745c6532d0e89ab79c624`.
- Fresh full validation: 472 registered tests, **469 passed**, three
  target-inapplicable sparse builtin skips, zero failures; 164.03 seconds.
  The queue differential covers mixed tasks and all six uniform signedness/group
  combinations, padded tails, full-width inputs and K=65537.
- All 13 calibration modes passed a separate real-device v2 smoke collection.
- The worker sweep contains 14 cases × five independent processes, three warmups
  and nine paired instrumented/uninstrumented samples per process. All 1,260
  calls (630 pairs) and all 70 final burst outputs passed full exact checks. The bursts
  contain 4,159 calls in total. CPU reference checks occur outside timed calls.

Each case uses M=N=K=512, seed 3135, signed full-width native inputs and nine
default residue planes. Outputs are centered residues; this case performs no
integer reconstruction. Group denotes the number of modulus channels processed
by a tile. Split-K uses three fragments. Inputs remain resident; synchronous call
timing includes allocation, descriptor construction/upload where applicable,
capability checks and synchronization. Initial upload and reference construction
are excluded. Bursts disable timing hooks and run for at least 250 ms and 16 calls,
with a 4096-call cap. Every final output and canary is checked.

## Sustained call measurements

Times are the median of five process burst means. Ranges contain all five process
means, not individual call extrema. Requested workers equal effective workers
for every queued case in this table.

| Approach | Group | Workers | Median µs | Process range µs |
|---|---:|---:|---:|---:|
| Pack + direct GEMM | 1 | — | 729.07 | 720.00–797.58 |
| Ordinary native WMMA | 1 | — | 2,054.87 | 2,044.18–2,084.24 |
| Ordinary native WMMA | 2 | — | 12,220.01 | 12,184.14–12,315.30 |
| Ordinary native WMMA | 4 | — | 7,629.64 | 7,592.33–7,663.83 |
| Persistent queue | 1 | 32 | 55,699.45 | 54,997.34–56,016.22 |
| Persistent queue | 1 | 128 | 14,552.76 | 14,534.01–14,616.89 |
| Persistent queue | 1 | 512 | 4,566.13 | 4,543.16–4,627.49 |
| Persistent queue | 1 | 1024 | 2,913.87 | 2,874.10–3,007.00 |
| Persistent queue | 2 | 32 | 119,992.02 | 119,910.83–120,775.23 |
| Persistent queue | 2 | 512 | 13,992.46 | 13,873.59–14,231.78 |
| Persistent queue | 4 | 32 | 84,345.47 | 84,252.12–84,409.32 |
| Persistent queue | 4 | 512 | 9,424.39 | 9,372.11–9,471.52 |
| Split-K queue | 1 | 32 | 59,557.62 | 58,615.09–59,926.95 |
| Split-K queue | 1 | 512 | 5,112.51 | 5,038.72–5,124.71 |

Grouping two or four moduli remains slower than group one in both ordinary and
queued execution. The earlier [large-throughput report](7900xtx-large-throughput-2026-10-23.md)
preserves the generic-queue baseline. Uniform specialization alone does not show
a timing win at 32 workers across those campaigns; their warmup/repeat settings
also differ, so that cross-campaign comparison is descriptive.

## Compiled resource inspection

Inspection of the actual `6d8ab25` direct-HIP code object associates each queue
symbol with its group and signedness. All variants contain integer WMMA, use
wave32 and report zero scratch bytes and zero register spills.

| Queue kernel | VGPRs | Integer WMMA instructions in static body |
|---|---:|---:|
| Heterogeneous | 156 | 14 |
| Uniform unsigned, group 1 / 2 / 4 | 65 / 108 / 144 | 1 / 2 / 4 |
| Uniform signed, group 1 / 2 / 4 | 66 / 108 / 147 | 1 / 2 / 4 |

Specialization reduces register reservation, especially for group one. These
resource counts alone do not establish a latency gain. The grouped conversion
body still contains many constant-division paths. The subsequent reciprocal
conversion is validated and measured below.

## Reciprocal conversion follow-up

This sweep changes the native 64-bit conversion to exact 32-bit reciprocal
reductions. Case manifest, seed, signed full-width inputs, output planes,
worker counts, three warmups, nine paired samples, five independent processes,
and burst policy match the earlier sweep. The CPU oracle now uses proven-safe
Boost `int256_t` products plus independent `cpp_int` edge-column audits; this
reference construction remains outside GPU timing. Every output residue is
checked. Each before/after entry uses the same approach and worker count.

| Approach | Group | Workers | Before µs | After median µs | After process range µs | Before / after |
|---|---:|---:|---:|---:|---:|---:|
| Pack + direct GEMM | 1 | — | 729.07 | 784.13 | 750.24–806.71 | 0.93× |
| Ordinary native WMMA | 1 | — | 2,054.87 | 1,601.42 | 1,556.77–1,637.60 | 1.28× |
| Ordinary native WMMA | 2 | — | 12,220.01 | 1,585.68 | 1,559.48–1,628.86 | 7.71× |
| Ordinary native WMMA | 4 | — | 7,629.64 | 1,559.02 | 1,514.13–1,931.72 | 4.89× |
| Persistent queue | 1 | 32 | 55,699.45 | 35,110.30 | 34,204.44–38,246.78 | 1.59× |
| Persistent queue | 1 | 128 | 14,552.76 | 8,507.69 | 8,274.77–9,191.25 | 1.71× |
| Persistent queue | 1 | 512 | 4,566.13 | 2,830.69 | 2,772.58–3,707.97 | 1.61× |
| Persistent queue | 1 | 1024 | 2,913.87 | 2,199.27 | 2,116.93–2,203.39 | 1.32× |
| Persistent queue | 2 | 32 | 119,992.02 | 22,447.68 | 22,284.57–26,817.96 | 5.35× |
| Persistent queue | 2 | 512 | 13,992.46 | 2,375.61 | 2,346.33–2,444.34 | 5.89× |
| Persistent queue | 4 | 32 | 84,345.47 | 19,551.48 | 19,209.39–20,675.62 | 4.31× |
| Persistent queue | 4 | 512 | 9,424.39 | 2,345.25 | 2,289.81–2,397.04 | 4.02× |
| Split-K queue | 1 | 32 | 59,557.62 | 32,429.37 | 31,501.11–33,348.39 | 1.84× |
| Split-K queue | 1 | 512 | 5,112.51 | 3,488.52 | 3,386.48–3,739.07 | 1.47× |

Every native WMMA/queue before-and-after process range is separated. The direct
control ranges overlap, so its median change is not attributed to this reducer.
The three ordinary WMMA groups now overlap one another; group four's smallest
median is not evidence that it wins over group one. No queue beats ordinary
WMMA here. These are residue-output timings, not exact-integer export timings.

Compiled resource inspection connects the change to smaller conversion bodies:

| Signed kernel | Group 1 VGPRs before → after | Group 2 | Group 4 |
|---|---:|---:|---:|
| Ordinary native WMMA | 49 → 41 | 106 → 64 | 143 → 104 |
| Uniform persistent queue | 66 → 64 | 108 → 69 | 147 → 109 |

All examined variants retain integer WMMA, wave32, zero scratch bytes and zero
register spills. The mixed queue drops from 156 to 118 VGPRs. Resource counts
describe the compiled artifact; the timing table establishes the measured gain.

- Clean compiled source: `436d23545e08d0c6c1da68ee269865ce43c0306f`.
- Executable SHA256: `34e4857ccb22be4fefaf7a4e5961e587023dc68a9023efe99ebc5a53ac3b7248`.
- Fresh all-feature validation: **472 passed, three expected sparse-target skips,
  zero failures** across 475 registered tests, 184.52 seconds.
- All 13 modes passed the separate device smoke collection for that revision.
- All 1,260 paired calls and 70 final burst outputs passed; the bursts contain
  **6,844 calls**. CPU reduction tests cover every modulus 2..256 against Boost,
  including full-width boundaries and deterministic random values.
- All 18 Linux CPU push/PR checks passed at `436d235`; this is CPU CI, not Linux
  GPU qualification.
- Captures: `temp/native-reciprocal-captures/`, with sibling `.summary.json` and
  `.sha256.txt`; smoke: `temp/native-reciprocal-v2-smoke/`; resources:
  `temp/native-resources-436d235/resources.json`; logs:
  `temp/native-reciprocal-build.log` and `temp/native-reciprocal-tests.log`.

Reproduce with the same commands below, replacing output directories with the
reciprocal paths above and using the clean `436d235` all-feature executable.

## Reusable preparation at 51ae8d8

The fresh all-feature pass at `51ae8d8` completed 492 registered tests: 489
passes, three target-inapplicable sparse-instruction skips and no failures.
This includes queue mutation/replacement differentials, complex 3M comparisons
and CPU/HIP public workload examples rebuilt against an installed package.
All 14 native calibration modes passed their separate smoke collection.

The ten-case prepared-queue collection uses five independent processes per
case, nine measured pairs, three warmups, full-width signed inputs, prefix nine,
group one and seed 3135. The 50 records contain 900 individually verified calls
and 49,252 sustained calls with complete final-output verification. Sustained
timing disables event hooks and performs no reference readback between calls.
The same RX 7900 XTX / gfx1100, VS2026 and HIP 7.1 configuration applies.

| Approach | 32³ median µs [process range] | 512³ median µs [process range] |
|---|---:|---:|
| Pack + direct GEMM | 164.08 [156.80, 177.40] | 750.09 [729.49, 761.84] |
| Already-packed direct GEMM | 73.91 [71.24, 77.14] | 654.65 [646.03, 665.29] |
| Ordinary native WMMA | 250.37 [238.55, 292.89] | 1555.41 [1489.42, 1556.64] |
| One-shot queue, requested 1024 workers | 491.51 [432.74, 499.40] | 2092.83 [2028.99, 2120.02] |
| Prepared queue, requested 1024 workers | 104.36 [94.99, 138.77] | 1509.74 [1491.55, 1518.27] |

Preparation reuse improves queue steady latency by **4.71× at 32³** and
**1.39× at 512³**, with separated observed ranges. Device descriptor storage
is 668 bytes and 147,548 bytes respectively. Median preparation costs are
949.0 µs [709.6, 1185.8] and 748.3 µs [582.5, 959.8]. Charging each process's
preparation across 15 calls gives 177.86 µs [146.51, 214.44] and 1558.61 µs
[1541.43, 1574.66] per call: **2.76× and 1.34×** versus the one-shot queue.
This charges the descriptor lifecycle; common operand/output allocations and
initial native uploads are outside both candidates' measured boundary.

The large prepared queue overlaps ordinary WMMA and loses to pack + direct
GEMM. Already-packed direct GEMM starts at a different input boundary. None
of these observations establishes a queue scheduling advantage: ordinary WMMA
still includes allocation/capability checks per call. The new `grid_prepared`
control reuses identical preparation with ordinary grids; its subsequent
measurement is below.

Executable SHA-256:
`24c620d120d0332cc16e6aa0b4ed577c0c825a47d7479b48e561221c1e707f3e`.
Raw records are in `temp/prepared-queue-captures/`, with sibling `.summary.json`
and `.sha256.txt` files. The command used `prepared-queue.json` at `51ae8d8`,
`--runs 5 --repeats 9 --warmups 3 --timeout 900`; that tracked manifest now
also includes the two new ordinary-grid controls. Validation logs are
`temp/complex-workloads-build.log` and `temp/complex-workloads-tests.log`.

## Matched prepared ordinary grid at 1bb5e33

Five independent processes for each of twelve cases retain the same seed,
prefix nine, group one, three warmups and nine measured pairs. All 1,080 paired
calls and the final outputs of 60 sustained bursts passed exact comparisons;
the bursts contain 61,179 calls. This uses the pre-upgrade HIP 7.1 stack.

| Approach | 32³ median µs [process range] | 512³ median µs [process range] |
|---|---:|---:|
| Pack + direct GEMM | 177.10 [160.73, 187.06] | 747.62 [736.25, 754.58] |
| Already-packed direct GEMM | 78.25 [72.05, 81.42] | 663.18 [649.51, 959.47] |
| Ordinary native WMMA | 257.91 [232.46, 303.86] | 1547.47 [1521.77, 1558.13] |
| One-shot queue, requested 1024 workers | 519.09 [452.21, 564.34] | 2143.28 [2115.76, 2253.17] |
| Prepared queue, requested 1024 workers | 102.47 [100.43, 119.71] | 1510.66 [1506.74, 1530.65] |
| Ordinary grid with identical prepared state | 101.69 [91.90, 113.81] | 1391.68 [1384.09, 1528.95] |

Prepared queue reuse improves steady latency by **5.07× and 1.42×** versus
the one-shot queue. The matched ordinary grid has a lower median at both
sizes, with overlapping process ranges. There is no demonstrated persistent
queue scheduling advantage. Direct GEMM remains faster on the large case.

The fresh validation retry passed 491 tests, with three expected sparse-target
skips and no failures across 494 registered tests. All seventeen native modes
also passed a separate smoke collection. Raw records, SHA-256 index and
summary are under `temp/1bb5e33-prepared-queue`; the command uses
`prepared-queue.json --runs 5 --repeats 9 --warmups 3 --timeout 900`.
Validation logs are `temp/1bb5e33-tests-retry.log` and
`temp/1bb5e33-native-smoke.log`.

## Reproduction and retained evidence

From a matching all-feature build:

```powershell
python tools/collect_native_calibration.py --exe build/issues-all/rns8-native-calibrate.exe --cases benchmarks/native-calibration/all-modes-smoke.json --output temp/native-v2-smoke --runs 1 --repeats 1 --warmups 1 --timeout 120
python tools/collect_native_calibration.py --exe build/issues-all/rns8-native-calibrate.exe --cases benchmarks/native-calibration/queue-workers.json --output temp/queue-worker-captures --runs 5 --repeats 9 --warmups 3 --timeout 900
```

Tracked manifests provide every case without a private scratch prerequisite.
Local raw records are in `temp/native-v2-smoke/` and `temp/queue-worker-captures/`;
the latter has a sibling `.summary.json` and `.sha256.txt` index. Resource details
are in `temp/native-resources-6d8ab25/resources.json`; build/test logs are
`temp/queue-specialization-build.log` and `temp/queue-specialization-tests.log`.
Records retain executable, collector/validator hashes and compiled provenance.
Experimental Garner-table configuration remains ineligible for production
promotion; hardware qualification beyond this local GPU remains deferred.
