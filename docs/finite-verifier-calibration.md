# Public finite verifier calibration

`rns8-finite-verify-calibrate` measures the synchronous public finite-ring API.
The benchmark, collector and fixtures are authored for the final combined
validation batch. They have not been executed or qualified. No performance
result, default route or integer-exactness claim follows from these sources.
The [verifier contract](finite-freivalds.md) owns the arithmetic and sampling model.

## Matched cost boundaries

All modes start with the same resident A, B and independently computed correct C.
Every mode records context/plan/workspace/matrix creation and packing as setup;
dataset generation and the independent oracle have a separate untimed duration.
The benchmark does not substitute a raw projection kernel for the public API.

| Mode | Work inside each wall interval |
|---|---|
| `verify_only` | Public verification, including scratch allocation/free, challenge generation, transfer, projection, synchronization and report construction |
| `compute_verify` | Public finite GEMM followed by the complete public verification call |
| `full_reference` | Allocate host buffers, export resident A/B/C, compute the complete blocked modular product and compare every output |

The reference uses a serial blocked uint64 GEMM with 32-row/32-column/64-K blocks.
Its full dot products fit: `K <= 65537` and each byte product is at most 65025.
A separate scalar row/column/K implementation builds the oracle before setup.
Every warmup and sample receives a complete output and export-padding audit
outside its interval. The blocked reference checks its candidate inside its
interval as part of the operation being compared. Host exports, allocations and
all public call overhead belong to the stated wall cost. These are application
costs, not HIP-event kernel durations or a comparison to an optimized CPU BLAS.
Both input matrices are exported and checked for preservation after the sustained
interval, outside its timing.

A sustained interval invokes the same operation at least 16 times and stops
when elapsed time reaches 250ms or 4096 calls, whichever condition is reached
after the minimum. There is no additional independent audit between these calls;
the last output receives the complete audit afterward. `full_reference` still
performs its intrinsic full comparison on every call. The record distinguishes
elapsed duration, count, mean, target and call-cap termination. Single-call
samples use their actual median, averaging the two central values for even counts.
The rate is calls per second, not operations per second.

## Reproducible input and replay

`splitmix64_finite_u8_seed3135_extrema_v1` generates row-major A, then B with
SplitMix64 starting at 3135; each word is reduced modulo the explicit modulus.
After generating each operand, byte 0 becomes `modulus-1` and byte 1 becomes zero
when present. The seed stream continues across operands. Reports contain full
16-digit FNV-1a checksums for A, B and the independent expected output. The
collector reconstructs A/B, checks their checksums and records SHA256 of their
concatenated bytes. It does not mistake a checksum for an independent oracle.

The challenge seed is separate from the dataset seed. Verifier modes require
1..256 trials and record sampler version, replay seed, completed trial count,
all instance IDs/content generations, logical scratch and the ideal exponent
and probability. The reported generations describe the final sustained call;
IDs/generations are process-local bindings, not comparison keys across processes.
No expected-generation override is supplied; the public report captures current
bindings. Repeated calls replay the same challenges; repetitions do not accumulate
independent confidence. Deterministic replay supplies no fresh entropy. The `2^-trials` value
belongs only to the independent uniform binary-bit theorem. A pass tests a
finite-ring congruence and never certifies integer equality. `full_reference`
requires zero trials and zero challenge seed and emits no probabilistic report.

Shapes are bounded by M,N=1..512, K=1..65537 and M*N*K<=134217728. Moduli 2..256
include composite rings and the full-byte ring. Repeats and warmups are explicit
1..1000 controls. No implicit backend/device selection occurs: CPU requires
`cpu-reference -1`; HIP requires `hip-direct` and a device ordinal 0..63. The
actual device name, architecture and runtime/driver versions are captured.
Unsupported selection exits 77 and cannot become a passing collector record.

The 256MiB logical live-payload cap conservatively includes retained host inputs
and oracle, both possible resident copies, all reference export buffers, blocked
uint64 sums/results, padded audit output, verifier scratch and the queried plan
workspace. With S=MK+KN+MN, the bound is `5*S + 16*MN + 3*M + verifier_scratch +
plan_workspace`. Actual resident allocation is checked against 2*S. The bound
covers simultaneous buffers, including the extra arrays in the timed reference.
Allocator bookkeeping, context/compiler runtime and fixed metadata are excluded;
the record does not claim to cap process RSS or free device memory requirements.

## Commands and evidence receipts

Enable `RNS8_BUILD_BENCHMARKS` in the existing project configuration. The target
uses public API headers and `rns8_static`, and receives the same immutable
build manifest as other calibrators. It includes no private runtime headers.
For a bounded CPU run from the repository root:

```powershell
build/issues-all/rns8-finite-verify-calibrate.exe cpu-reference -1 verify_only 2 3 65 250 8 9 3 3135 cpu-replay-1
build/issues-all/rns8-finite-verify-calibrate.exe cpu-reference -1 compute_verify 2 3 65 250 8 9 3 3135 cpu-compute-1
build/issues-all/rns8-finite-verify-calibrate.exe cpu-reference -1 full_reference 2 3 65 250 0 9 3 0 cpu-reference-1
```

Use `hip-direct 0` only when deliberately selecting that available device. The
tracked [matched cases](../benchmarks/finite-verification/matched.json) contain
prime/composite/power-of-two rings for both explicit backends and all three modes.
The [K-boundary cases](../benchmarks/finite-verification/k-boundary.json) exercise
65537. Select the actual HIP ordinal by editing a copied manifest under ignored
`temp/` when needed; do not silently fall back or filter failed cases.

```powershell
python tools/collect_finite_verify.py --exe build/issues-all/rns8-finite-verify-calibrate.exe --cases benchmarks/finite-verification/matched.json --output temp/finite-verification --runs 5 --repeats 9 --warmups 3
```

Each case/run is a fresh process. Within-process samples are not independent
observations. The collector retains stdout/stderr even when metadata fails,
rejects duplicate cases, checks all requested controls, sample/sustained
verification, checksums, probability/replay and resource metadata, and rejects
artifact or runtime/build identity changes. Records bind executable SHA256,
case-file SHA256, collector and manifest-validator SHA256, the compiled source
manifest, commands, process indices and selected device-visibility variables.
Dirty, unavailable-dependency and experimental-Garner captures may remain
structurally valid diagnostics with `provenance_complete=false`; all records
retain `promotion_eligible=false`. This thin microcase adapter does not assert a
pass under the production benchmark schema or verified ISA evidence.

Compare matched artifact/backend/device/shape/modulus/dataset/output cohorts.
Treat full-reference-versus-verification and full-reference-versus-compute-plus-
verification as different questions. Report each process observation and its
spread; never turn trials or repetitions into independent run counts. Broader
check-plane controls, large-workload measurements and promotion remain open.
