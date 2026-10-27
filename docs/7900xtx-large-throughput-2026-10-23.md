# RX 7900 XTX larger-workload measurements — 2026-10-23

Larger jobs show useful gains. At 1024³, the checked native-byte WMMA candidate
reaches **1.72 effective TOPS**, **2.23×** faster than the matched minimal-prefix
RNS pipeline and **7.26×** faster than full-width vector ALU. The same WMMA route
rises from 0.595 TOPS at 512³ to 1.718 TOPS at 1024³.

These measurements replace the small 256³ collection as the current scaling
evidence. Each table compares a stated input/output contract; native device
output, resident residues and host-exported output remain separate.

## Method

Source `c7f7e05cbade0e827b6e952f4798012824090e98`, clean when compiled. Windows,
VS2026/MSVC 19.51.36260, HIP 7.1.51803 / Clang 21, runtime 71260602, gfx1100.
The RelWithDebInfo build enables all locally available optional backends,
OpenMP, generated native WMMA and experimental Garner tables. See the
[complete validation record](local-all-features-20261023.md).

The main campaign comprises 17 cases × five independent processes, randomized
with seed 3135. Each process performs ten warmups and 100 instrumented /
uninstrumented call pairs, verifying every output outside the timed interval.
It then performs at least 16 consecutive checked calls, targeting 250 ms and
capped at 4,096 calls, without verification readbacks or poisoning between
calls. The final output is verified afterward. Native/product outputs include
tail-canary checks; packing-only cases compare the complete operand planes.
**All 17,000 paired calls, 850 warmups and 85 burst-final outputs passed.**
The bursts executed 31,272 calls in total.

The native microcase tables report the median of five process burst means; ranges are their minimum
and maximum. Timing hooks are disabled for those bursts. Production admission,
status reads, candidate-local allocations and synchronization remain included.
The repeated inputs are resident and cache-hot; initial uploads and verification
are excluded. Clocks/power were not pinned and hardware traffic counters were
not collected. This is repeated-call throughput, not a long-duration thermal test.

`Effective TOPS = 2*M*N*K / (call_microseconds * 1e6)` counts the logical product
once; it does not multiply useful work by the number of residue planes.

## Native bounded integer device output

Signed byte-domain inputs are stored as i64. Every approach produces the same
exact i64 device matrix. WMMA/scalar/RNS discover current bounds; vector ALU
checks overflow. RNS includes both operand packs and reconstruction, using
the minimal four-plane prefix at both sizes. The CPU oracle computes every
output in range-proven int64 arithmetic and independently audits each row’s
boundary columns with Boost.

| Approach | Shape | Sustained call, µs | Process range, µs | Effective TOPS | Speedup vs RNS |
|---|---|---:|---:|---:|---:|
| Direct HIP RNS pipeline | 512³ | 921.75 | 875.1–1000.8 | 0.2912 | 1.00× |
| Full-width vector ALU | 512³ | 1417.12 | 1391.2–1445.3 | 0.1894 | 0.65× |
| Native scalar | 512³ | 543.14 | 488.2–582.9 | 0.4942 | 1.70× |
| Native-byte WMMA | 512³ | 450.95 | 426.8–457.5 | 0.5953 | 2.04× |
| Direct HIP RNS pipeline | 1024³ | 2782.54 | 2775.8–2794.1 | 0.7718 | 1.00× |
| Full-width vector ALU | 1024³ | 9072.63 | 8996.4–9093.1 | 0.2367 | 0.31× |
| Native scalar | 1024³ | 2763.45 | 2664.1–2782.3 | 0.7771 | 1.01× |
| Native-byte WMMA | 1024³ | 1249.69 | 1199.0–1286.5 | 1.7184 | 2.23× |

The 1024³ WMMA improvement remains present across the observed process ranges.
Scalar and RNS ranges overlap at 1024³; their small median difference does not
establish a winner. Native-byte WMMA remains an explicitly checked research
candidate, not an automatically promoted public dispatch route.

## Resident residue epilogue

`alpha*AB + beta*D`, prefix nine, alpha=INT64_MIN and beta=INT64_MAX,
resident residue inputs and residue output without integer export. Boost
evaluates the original full-width integer expression for every output.

| Approach | Shape | Sustained call, µs | Process range, µs | Effective TOPS | Speedup vs unfused |
|---|---|---:|---:|---:|---:|
| Direct GEMM + separate epilogue | 512³ | 760.91 | 741.2–791.5 | 0.3528 | 1.00× |
| Direct fused | 512³ | 1970.10 | 1804.4–2128.5 | 0.1363 | 0.39× |
| rocWMMA fused | 512³ | 459.58 | 458.8–466.6 | 0.5841 | 1.66× |

## Conversion and scheduling controls

Full-width signed native inputs; nine default-basis output residue planes.
The prepacked row starts after conversion. All other product rows start from
native device inputs. Queue/split-K use 32 workers; split-K uses three fragments.

| Approach | Shape | Sustained call, µs | Process range, µs | Effective TOPS | Speedup vs pack + GEMM |
|---|---|---:|---:|---:|---:|
| Pack + direct GEMM | 512³ | 749.42 | 723.5–754.5 | 0.3582 | 1.00× |
| Already-packed direct GEMM | 512³ | 646.13 | 630.4–656.8 | 0.4155 | — |
| Fused native WMMA, group 1 | 512³ | 2017.27 | 2003.1–2577.3 | 0.1331 | 0.37× |
| Persistent queue, group 1 | 512³ | 55819.62 | 53585.3–56377.7 | 0.0048 | 0.01× |
| Persistent split-K, group 1 | 512³ | 58367.46 | 56238.6–58722.1 | 0.0046 | 0.01× |

Packing both operands alone costs 100.45 µs. It is not a GEMM and receives no product TOPS figure.
The prepacked control illustrates the recurring conversion cost but excludes
initial preparation. The group-one queue and split-K configurations are clear
regressions on this dense case; they are excluded from the beneficial-additions
table. These fixed-worker results do not select an optimal queue configuration.

## Modulus grouping follow-up

A second randomized campaign used the same executable, shape (512³), signed
full-width dataset and nine-plane output for groups two and four. Each of the
six cases has five independent processes, ten warmups, 100 verified pairs and
a sustained burst. All 6,000 paired calls, 300 warmups and 30 burst-final
outputs passed; bursts executed 589 calls.

| Approach | Group | Sustained call, ms | Process range, ms |
|---|---:|---:|---:|
| queue | 2 | 123.639 | 121.167–123.665 |
| queue | 4 | 85.604 | 83.661–85.855 |
| split_k | 2 | 126.176 | 125.516–127.382 |
| split_k | 4 | 88.397 | 86.391–88.591 |
| wmma | 2 | 12.250 | 12.191–12.352 |
| wmma | 4 | 7.657 | 7.623–7.828 |

Every group-two/four configuration is slower than its group-one control.
Sharing native loads did not pay off in these implementations on this case.
Register/scratch inspection and worker-count tuning remain necessary before
attributing the cause or selecting a better variant.

The second collection command substitutes `temp/grouped-throughput-cases.json`
and `temp/grouped-throughput-captures` in the command below. Its case list is
`wmma`, `queue`, `split_k`, each at groups two and four, all 512³/prefix nine/signed.
Capture-index SHA-256: `2253cf5bd46f5a9ee04298fc352cfc1c66e94e20dafbb3dabb6156b1a390e4de`.

## Instrumented components

The paired diagnostic calls retain labeled HIP-event intervals. For 1024³
native-byte WMMA, the median GEMM interval is 1,076.3 µs and admission is
165.0 µs. For the matched RNS pipeline, GEMM is 2,039.9 µs, operand packing
80.3 µs and native export 251.8 µs. These are instrumented component intervals,
including their wrapper boundaries; they are not isolated instruction issue
rates and must not be summed with overlapping aliases or host call timings.

## Public bounded-i64 pipelines and reuse

This separate campaign exercises all seven public bounded-i64 backends at
1024³, signed uniform [-16,16] inputs, seed 3135, fixed requested prefix nine,
and native i64 **host** output. The vector route keeps native storage. It uses
three independent randomized processes per backend/mode, five warmups and
15 measured calls. All 39 captures passed the benchmark and schema validator;
every capture has output checksum `9903994594253216464`.

The following table includes per-call packing, GEMM and host export. Plans,
workspaces and matrices are already allocated. Unlike the native microcase
bursts, these public benchmark timings include event instrumentation. Reported
times are medians of three process medians; ranges show those process medians.

| Public backend | Repack + GEMM + host export, ms | Process range, ms | Effective TOPS | Speedup vs Direct HIP |
|---|---:|---:|---:|---:|
| CPU reference (3 threads) | 718.763 | 518.510–721.474 | 0.00299 | 0.01× |
| Direct HIP | 8.127 | 7.934–8.342 | 0.26425 | 1.00× |
| Vector ALU | 24.931 | 23.965–25.910 | 0.08614 | 0.33× |
| hipBLASLt | 10.928 | 10.250–11.346 | 0.19650 | 0.74× |
| CK | 8.581 | 8.564–9.478 | 0.25027 | 0.95× |
| rocWMMA | 8.546 | 8.520–9.338 | 0.25127 | 0.95× |
| AMDGPU builtins | 6.736 | 6.684–6.980 | 0.31879 | 1.21× |

AMDGPU builtins are the fastest measured repacking GPU backend here. The
native-byte device-output candidate above is a different output/measurement
boundary and must not be ranked against these host-exported public calls.

Packed-input reuse retains both operands. Its steady call time is shown
separately from the initial preparation cost. For the 15-call comparison,
each process contributes `(sum(measured call times) + initial packing setup)/15`;
the table takes the median across three processes. Common plan/workspace/matrix
creation remains outside both modes. This charges setup instead of presenting
a free prepacked input as a complete pipeline.

| Backend | Reuse steady call, ms | One-time packing setup, ms | Repack 15-call cost/call, ms | Reuse 15-call cost/call, ms | 15-call speedup |
|---|---:|---:|---:|---:|---:|
| CPU reference (3 threads) | 485.771 | 18.795 | 701.735 | 490.355 | 1.43× |
| Direct HIP | 5.866 | 41.259 | 8.225 | 8.626 | 0.95× |
| hipBLASLt | 5.365 | 41.434 | 10.979 | 8.237 | 1.33× |
| CK | 5.816 | 38.521 | 8.915 | 8.562 | 1.04× |
| rocWMMA | 5.875 | 41.664 | 8.929 | 8.749 | 1.02× |
| AMDGPU builtins | 4.668 | 39.790 | 6.818 | 7.480 | 0.91× |

hipBLASLt reuse yields **1.33×** lower setup-inclusive per-call cost across
15 calls, with separated process ranges: repack 10.255–11.090 ms versus reuse
7.392–8.415 ms. CPU/CK/rocWMMA process ranges overlap in this setup-inclusive
comparison, so their median differences are not counted as established gains.
Direct HIP and AMDGPU-builtin reuse have lower steady latency but have not
recovered their measured initial preparation cost after 15 calls.

The public campaign records all commands under `temp/all-backend-throughput/`
and validated raw captures under its `raw/` directory. Commands use
`build/issues-all/rns8-bench.exe --backend <backend> --semantics bounded-i64
--m 1024 --n 1024 --k 1024 --warmups 5 --repeats 15 --seed 3135
--cpu-threads 3 --prefix-policy fixed-requested --max-prefix 9`, with
`--reuse-packed-inputs` for the reuse controls. Vector reuse is not included.
Executable SHA-256: `fff7abcc53e654e04fe52f34ce037f8ed11603c6d4afb7c7ea0c37789446cb29`.
Capture-index SHA-256: `4386a12073f5d73e20f3275c67fe9170a33a455a0d05e78d1fef123e8a43d090`.

## Reproduction and retained evidence

```powershell
python tools/collect_native_calibration.py --exe build/issues-all/rns8-native-calibrate.exe --cases temp/large-throughput-cases.json --output temp/large-throughput-captures --runs 5 --repeats 100 --warmups 10 --timeout 900
```

The case list contains the four `narrow_*` approaches at 512³ and 1024³
(prefix/group=1), followed by the three `epilogue_*` approaches, `pack_pair`,
`packed_gemm`, `pack_gemm`, `wmma`, `queue` and `split_k` at 512³
(prefix=9, group=1). All use signed inputs. Raw commands, process identities,
compiled manifests, output verification flags and timings are retained in
the ignored collection directory.

Executable SHA-256: `94cacb1551c038f91d74eeb12c8ffee5a271dd5cb306c4d2c7fab6220d0ed77d`.
Capture-index SHA-256: `8a74f99e3c03b22255cfaebef16c9e31213d1086bd5aa5f3c85a31ec291cef1f`.
