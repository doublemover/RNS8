# Performance evidence and promotion

## Current evidence boundary

The [October 23 larger-workload report](7900xtx-large-throughput-2026-10-23.md)
is the current RX 7900 XTX scaling evidence, built at `c7f7e05` with the
[recorded all-feature configuration](local-all-features-20261023.md). It compares
native device output, resident residue epilogues and public host-output
pipelines in separate tables. The [256³ diagnostic](7900xtx-throughput-2026-10-23.md)
is retained as earlier small-workload evidence. Old June summaries remain in the
[archive](archive/README.md).

The public 1024³ bounded-i64 repack/export comparison records AMDGPU builtins
at 6.736 ms versus direct HIP at 8.127 ms (1.21×). Internal native-byte WMMA
and private fused residue epilogues have different output/measurement boundaries;
the report states their matched baselines and ranges. These observations do not
install AUTO routing or fill missing clock/counter/held-out calibration gates.
Linux/Instinct and other devices require their own captures.

Current [backend availability](backend-notes.md), dated correctness results and
performance promotion are separate records. A new source/configuration identity
must be checked before reusing an earlier result for routing decisions.

## Measurement contract

A comparable capture must identify:

- Exact commit and dirty-tree state; compiler, HIP/ROCm and accelerator versions
- GPU architecture/device identity, OS, relevant visibility variables, and clocks
- Semantic contract, dimensions and strides, bounds, selected/requested prefix,
  tile schedule, source/input profile, seed, and output representation
- Selected backend and actual kernel, warmups/repeats, timing source, allocations,
  setup/reuse lifetime and workspace requirements
- Pack/H2D, GEMM/reduction, CRT/export/D2H, scheduling and end-to-end costs
- Exact CPU/reference result comparison and required HIP event coverage

Schema v4 validates capture structure and consistency; it is not a proof that a
GPU instruction was executed or that a declaration is true. Keep raw JSON,
stdout/stderr, build logs, exact commands, and disassembly together.

## Commands

A bounded smoke is prepared by [gpu_qualification.py](../tools/gpu_qualification.py).
It performs no promotion. For a later authorized release campaign, start with
reviewable scenario families instead of an unbounded "all" run:

```sh
python tools/benchmark_sweep.py --list-scenarios
python tools/benchmark_sweep.py --scenario release-candidates \
  --bench build/qualification/gfx942/hip-direct/rns8-bench \
  --bench-for amdgpu-builtins=build/qualification/gfx942/amdgpu-builtins/rns8-bench \
  --bench-for hipblaslt=build/qualification/gfx942/hipblaslt/rns8-bench \
  --bench-for ck=build/qualification/gfx942/ck/rns8-bench \
  --bench-for rocwmma=build/qualification/gfx942/rocwmma/rns8-bench \
  --review-mode release --warmups 3 --repeats 9 --seed 20261006 \
  --capture-timeout-seconds 600 --progress --dry-run \
  --out-root temp/mi300x-release-plan
```

Inspect the resulting plan, resource requirements, selected scenarios and CPU
reference costs before removing `--dry-run`. On Windows, use `gfx1100` build
paths, `.exe` suffixes, and PowerShell continuations. Do not use smoke captures
as release timing evidence. At least three warmups and nine repeats are required
by the existing release-review policy.

Extract ISA from the actual build, not from an unrelated compiler probe:

```sh
python tools/gpu_isa_report.py --build-tree build/qualification/gfx942/amdgpu-builtins \
  --backend amdgpu-builtins --target gfx942 --out-dir temp/mi300x-isa
```

Use `--backend direct-hip`, `ck`, `rocwmma`, `wrap64`, or `vector-alu` as
appropriate. Preserve the tool output and compiler identity. Matrix-instruction
and no-divide rules must be interpreted for the specific kernel family, rather
than applied to every helper indiscriminately.

After collecting matching independent runs, use `perf_variance_report.py` with
`--require-multi-run`, `target_validation_report.py` with real target-status
records, and the relevant workload report. Review their blockers. Reuse, graph,
grouped, sparse, result-cache, and chain workloads require matching lifetime and
output contracts; their measurements cannot be substituted for ordinary one-shot
latency.

## Promotion boundaries

- CPU success and static/host arithmetic tests do not qualify GPU code.
- Header discovery and a compiled object do not imply runtime correctness.
- A reported kernel name does not prove the selected machine instruction.
- Correctness success is not a throughput/latency claim.
- Windows Radeon results do not qualify Linux Instinct.
- A measured helper improvement does not prove an end-to-end improvement.
- Exact-shape cache entries do not authorize nearby-shape routing.
- Historical entries omit this revision's implementation key and cannot hit it.

`benchmark_sweep.py` and the cache install/review tools retain their explicit
promotion workflow. The qualification runner never passes
`--write-autotune-cache`. Keep broad tuning, paid hardware, publishing, and
installed-cache changes outside a source-preparation run.

### Synchronous transfer payload accounting

The benchmark allocation snapshots also record `h2d_calls`, `h2d_bytes`,
`d2h_calls`, `d2h_bytes`, `d2d_calls` and `d2d_bytes`. Their scope is successful
synchronous copy APIs inside the direct HIP implementation, including its
packing, descriptor, status and export helpers. A 2D copy counts only width
multiplied by height, excluding pitch gaps. Setup and measured-repeat deltas
use the same boundaries as allocation counters. The counters are process-wide;
measure a single workload without concurrent counter reset or other activity.

These are observed API payload counts, not profiler memory-transaction counts,
physical PCIe traffic, host RSS or transient device peak. Asynchronous copies,
graph replay, copies internal to vendor libraries and other backend translation
units are outside this scope; zero here cannot establish that such a route
performed no transfer. Existing retained-native signed/unsigned differential
fixtures check one compact logical upload for both compact and padded sources.
The new counters and fixtures await the final validation campaign.

## Capture resume and retired reports

A resumed sweep satisfies a requested capture only when its versioned request
receipt matches the complete arguments (including seed and sampling), relevant
runtime environment and compiled executable digest. Matching captures are not
relabelled or rewritten. Historical captures without this receipt remain valid
review inputs, but are not a substitute for a new requested observation. Stale
observations are retained beside a recollected capture; the sweep reports matched,
stale, attempted and deferred counts separately. RNS8 linkage is static, as
specified by the compiled build manifest; no checkout revision replaces this
artifact identity.

The golden suite records retired reports separately from attempted/tested cases.
Its three retired legacy reports do not execute and do not contribute passes.
Current sweep admission, scheduler evidence and autotune installation keep their
own executable coverage.
