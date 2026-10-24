# RX 7900 XTX: reviewed partial campaign, October 24, 2026

## Disposition and scope

**Stopped; do not resume without a new user request.** The original process ended
without a final receipt during the 512-cubed block-sparse cohort. Its last saved
capture was October 24 at 11:46 a.m. Eastern. The reason for process termination
is not established by the available logs. A resume launched during status
inspection was stopped immediately when requested; its separate directory is
excluded from this report.

The original campaign saved **5,382 correctness- and provenance-admitted process
captures** out of 16,513 planned runs. Sixteen cohorts finished successfully,
accounting for 5,333 captures, including 23 single-run preflight cases. The final
49 block-sparse captures belong to an interrupted cohort and are retained as
partial evidence, excluded from performance conclusions. **11,131 planned runs
were not completed.** There are no recorded failed execution captures; this is
not a claim that the unfinished campaign passed.

Completed cohorts: native preflight (23), adaptive native (230), extended native
(1,080: tiled tails/512, encoded-B tails/512, complex, prepared queue and worker
counts), conversion (3,500: smoke/every-prefix/large), finite verifier (150),
and block-sparse smoke/tails (350). The corrected 512/1024 headline set, wrap64,
word modulus, instruction roof, split-K, ragged, stream-panel, additional CRT
layout, public-workload and optional-reference campaigns remain unmeasured in
this collection. Initial headline input validation rejected 2048-cubed before
execution because it exceeded the existing exact-reference budget. No admission
limit was relaxed.

## Qualified configuration

Measured source: `ad71df518f3ed9bb4b2180d3841c7d1e310ecfbd`, clean at build time.
RX 7900 XTX, `gfx1100`, Windows HIP 7.1, VS2026 18.11.1 / MSVC 19.51.36260,
Python 3.15.1; RelWithDebInfo, all optional accelerators enabled, experimental
Garner tables **disabled**, `OMP_NUM_THREADS=3`. The competing NuLARRY workload
finished before collection began. Collection was serial on the GPU.

The production GPU build passed 747 tests with three target-inapplicable skips
and zero failures. The all-feature profile at `02f0f03` also passed 747/750;
CPU AddressSanitizer passed 428 with 16 HIP-only skips. Three libFuzzer harnesses
passed 1,000 runs each. Installed static/shared ordinary and native-device
consumers, shared workloads and eight ISA extractions passed. Linux CI at
`ad71df5` passed all nine jobs. These correctness checks cover more paths than
the completed performance campaign.

## Measurement and comparison rules

Performance rows use five independent processes, three warmups and nine measured
calls per process, plus sustained production batches targeting 250 ms (subject
to each executable's documented call cap). The reported time is the median of
five sustained process means. Timing hooks and independent reference audits are
outside the sustained interval; required arithmetic admission/status work and
synchronization stay inside it. Full outputs/canaries are checked afterward.
Native cases use seed 3135; conversion uses seed 3903. Captures record exact
commands, artifacts, external-header identities, semantic contracts and timing
scopes. Host wall time is used for these complete call/batch comparisons; event
breakdowns are separate diagnostics.

Only beneficial matched pairs whose observed process ranges do not overlap are
listed below. These are selected observed wins, not confidence intervals or
universal speedup claims. Groups, signedness, prefix, shape and reuse count match
within each pair. Native rows return device residues, without integer export;
conversion rows return canonical three-limb values. No backend default changes.

## Beneficial results

| Addition | Matched workload and boundary | Baseline | Candidate | Speedup |
|---|---|---:|---:|---:|
| Encoded-B reuse, preparation included | 512 cubed, unsigned, prefix 9, group 4, 32 GEMMs per batch | Native WMMA 42.052 ms/batch | **22.575 ms/batch** | **1.86x** |
| Prepared grid, preparation charged over 15 calls | 32 cubed, signed, prefix 9, group 1 | Native WMMA 0.223 ms/call | **0.135 ms/call** | **1.65x** |
| Prepared grid, same amortization | 512 cubed, signed, prefix 9, group 1 | Native WMMA 1.493 ms/call | **1.412 ms/call** | **1.06x** |
| Prepared queue, preparation charged over 15 calls | 32 cubed, signed, prefix 9, group 1, 1024 workers | One-shot queue 0.451 ms/call | **0.159 ms/call** | **2.83x** |
| Prepared queue, same amortization | 512 cubed, same controls | One-shot queue 1.985 ms/call | **1.556 ms/call** | **1.28x** |
| Complex 3M composition | 512 cubed, signed, prefix 20, paired residue output | Four-product composition 5.105 ms | **4.136 ms** | **1.23x** |
| LDS tiling, K-depth 32, one buffer | 512 cubed, signed, prefix 20, group 2 | Native WMMA 2.868 ms | **2.814 ms** | **1.02x** |
| LDS tiling, K-depth 32, one buffer | M=17, N=19, K=257, unsigned, prefix 20, group 1 | Native WMMA 0.320 ms | **0.280 ms** | **1.14x** |
| Direct CRT | 1,048,576 values, prefix 20, allocation/upload/reconstruction/readback/free included | Garner 10.731 ms | **6.845 ms** | **1.57x** |
| Fixed dot4 CRT | Same conversion contract | Garner 10.731 ms | **7.033 ms** | **1.53x** |
| Fixed WMMA CRT | Same conversion contract | Garner 10.731 ms | **6.891 ms** | **1.56x** |

For encoded-B, the setup-inclusive per-GEMM time is **0.705 ms / 0.381 effective
TOPS**, versus **1.314 ms / 0.204 TOPS** for the matched native WMMA baseline.
Effective TOPS counts `2*M*N*K` useful integer operations once per GEMM, without
multiplying by residue count. The warm-only encoded-B result is 0.698 ms/GEMM,
0.385 TOPS and 1.88x, with preparation reported separately; the headline uses
the more complete cold boundary. A batch of 32 here is explicitly measured.

Prepared rows instead amortize each process's measured one-time preparation over
15 calls and add its sustained per-call mean before computing medians/ranges.
They are cost projections from measured setup and steady components, not a
separately timed 15-call batch. Complex and CRT rows are compared by latency;
their operation contracts differ from an ordinary integer GEMM.

### Representative observed ranges

| Comparison | Baseline range | Candidate range |
|---|---:|---:|
| Encoded-B cold, 32-GEMM batch | 41.856-43.029 ms | 22.516-22.800 ms |
| Prepared grid, 512 cubed, setup/15 included | 1.481-1.503 ms/call | 1.401-1.446 ms/call |
| Prepared queue, 512 cubed, setup/15 included | 1.980-2.067 ms/call | 1.542-1.585 ms/call |
| Complex 3M versus four products | 5.058-5.397 ms | 4.102-4.202 ms |
| LDS tiling, 512 cubed | 2.867-2.958 ms | 2.796-2.849 ms |
| Direct CRT, largest cold case | 10.213-11.481 ms | 6.367-7.368 ms |
| Fixed dot4 CRT, same case | 10.213-11.481 ms | 6.482-7.832 ms |
| Fixed WMMA CRT, same case | 10.213-11.481 ms | 6.540-7.831 ms |

Direct, dot4 and WMMA CRT all beat Garner here; their ranges overlap each other,
so this evidence does not establish a winner among those three implementations.
Likewise, the 512-cubed tiling improvement is small and specific to its controls.

## Negative and incomplete findings

- Adaptive bound discovery/completion did not beat matched fixed-prefix WMMA in
  any of 30 complete comparisons on the full-width datasets. Extra proof,
  compaction and suffix work outweighs saved computation in these cases.
- Register tiling had no range-separated win in 12 comparisons. LDS tiling had
  11 such wins in 48 comparisons, mostly small/tail cases; it is not a general
  large-matrix acceleration.
- Encoded-B cold had 28 range-separated wins in 36 comparisons; warm-only had
  35/36. Preparation and reuse count materially change the benefit.
- Scalar native import was not clearly beaten by dot4 or WMMA import in any of
  the 50 comparisons per signedness/implementation. Balanced CRT and tile
  conversion plus reconstruction also had no range-separated wins over Garner
  in their 50 comparisons each. Pre-tiled inputs have a different boundary and
  are not credited as eliminating conversion cost.
- Completed sparse smoke/tail comparisons produced no setup-inclusive,
  range-separated sparse-over-dense win. The interrupted 512-cubed sparse cohort
  is excluded from conclusions.
- Finite verifier modes passed their capture checks. Verification-only and
  compute-plus-verification have different contracts from full exact reference
  computation; verification-only latency is not advertised as an exact-GEMM
  speedup or a replacement for deterministic exact defaults.
- Fresh public host-output, 1024-cubed headline and remaining candidate timings
  are unavailable. October 23 results retain their original source/configuration
  and are historical, not requalified by this partial collection.

The performance review supports scoped reuse and composition improvements, but
not automatic promotion of adaptive scheduling, conversion matrix engines or
new defaults. No implementation changes or additional benchmark runs were used to establish
these results. Missing hardware qualification and ROCm 10.1 stay deferred.

## Evidence and reproduction

Raw captures remain ignored under
`temp/validation-20261024/benchmarks-production-ad71df5/`; durable docs contain
only this reviewed summary. `receipt.json` SHA-256:
`8db689fcf071dd0e046a530a074fff8c313940216266a93269238d94c04ad0a9`.
The local analysis files are `reviewed-case-statistics.json`,
`reviewed-matched-comparisons.json` and `adaptive-reviewed-comparisons.json`.

| Artifact | SHA-256 |
|---|---|
| Native calibration executable | `e737317474c1ae7035419f9c1650812633327a49b9f0cf0d9897538e89b74f87` |
| CRT calibration executable | `7ad6d563dfe4a99c3ab746a652f54315d2eb4de7df2f98ab383691d5b3519290` |
| Finite verifier executable | `4965e3c0a6cdfce45d9f9e88998291032134480024bc09ef2795cbf8be387aa9` |

The following are reproduction commands, **not a request to restart**. Use a new
output directory, the qualified configuration and the recorded Python runtime:

```powershell
python tools/collect_native_calibration.py --exe build/final-msvc315-production/rns8-native-calibrate.exe --cases benchmarks/native-calibration/encoded-b-512.json --output temp/reproduction-encoded-b --runs 5 --repeats 9 --warmups 3 --timeout 1800
python tools/collect_crt_calibration.py --exe build/final-msvc315-production/rns8-crt-calibrate.exe --cases benchmarks/conversion-calibration/large.json --output temp/reproduction-conversion --runs 5 --repeats 9 --warmups 3 --timeout 1800
```

Other completed case manifests are recorded verbatim in the campaign receipt;
all source manifests remain in `benchmarks/`. Earlier experimental-Garner-ON
captures and the aborted resume are excluded.
