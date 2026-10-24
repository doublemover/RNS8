# RX 7900 XTX native queue worker sweep

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
body still contains many constant-division paths; a subsequent reciprocal
conversion change must be validated and measured independently.

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
