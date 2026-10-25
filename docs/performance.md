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

### Selected static ISA and proof-work identity

ISA inventory is no longer joined to a timing capture by backend name alone.
Qualified static resource fields require the concrete target, unchanged captured
executable/library hash, build-manifest hash, code-object hash and selected
kernel/symbol mapping. `gpu_isa_report.py --capture ... --artifact ...
--selected-symbol ...` checks the artifact hash and extracts the target code
object to prove byte-identical membership. Repeat `--selected-symbol` for an
explicit static symbol set. Unbound reports remain separately available under
`isa_inventory`; missing target/build/symbol associations return unknown, never
another target's VGPR/occupancy numbers. Reports from different artifacts keep
distinct hash-suffixed filenames, and duplicate copies of one symbol are not
summed. A static mapping is not a dispatch trace or an executed-instruction count.

Strict comparison, sweep grouping, repeatability cohorts and timing eligibility
now share normalized proof-policy and host-batch work identities. Historical
omission of the proof policy means the documented eager-moments behavior.
`result_compare.py --allow-proof-policy-diff` is an explicitly labeled A/B
review, not repeated measurements of one proof workload. Batch task counts and
inner repetitions describe work per observation, not the number of observations.
The host-batch scaling report intentionally compares separately labeled aggregate
and per-task values; ordinary strict comparisons do not call batch4 and batch8
the same work. Regression fixtures are authored; their execution is deferred.

## Independent observations and profiler partitions

Adaptive, finite-modulus, finite-distribution, layout, bound-discovery and FHE
proxy reports share `observation_cohorts.py`. They aggregate **process medians**
within an exact arithmetic/work, implementation, artifact and execution cohort
before selecting an implementation. Copied captures do not add weight; reused
execution IDs with conflicting records block qualification. Reports retain all
submitted hashes/paths, distinct-run counts and process spread. Raw capture
arrays are not rewritten. Legacy observations remain descriptive, with no
qualified winner or promotion implication.

The default admission requires two independent processes, three warmups, nine
within-process observations, measured uncensored end-to-end samples, at most
15% within-process p95/median spread and at most 10% process-median range. Those
checks are necessary, not sufficient, for scientific qualification. Same-source
CPU arithmetic controls are matched by complete work contract; GPU performance
controls additionally match device/runtime/configuration/timing policy. Explicit
layout or bound-discovery A/B reports name their permitted contract differences.
Bound-discovery setup-inclusive comparisons sum setup and execution **within
each process before taking the median**. Host batch tasks are work per sample,
not independent observations. FHE recommendations carry shape/work/environment
identity and retain their proxy-only status.

Profiler imports retain per-metric totals, present counts, ranges and units,
partitioned by kernel, phase, device and execution identity. Uneven export-file
partitions no longer give small files extra statistical weight. Mixed-kernel
means are diagnostic inventories, not a kernel's register requirement; occupancy
observation means are not duration-weighted occupancy. Selected scalar resources
require execution/symbol association and an unambiguous phase/unit group. Raw
ISA inventories likewise cannot supply selected-kernel resources without the
artifact/target/symbol association described above.

## CI smoke is not a regression gate

The retained `perf-regression.yml` entrypoint is named **Non-promoting Benchmark
Smoke**. It builds Release and collects six bounded 512³ CPU/direct-HIP captures
(bounded signed, bounded unsigned, finite field 251), with per-capture and job
budgets and serialized device use. `ci_benchmark_smoke.py` generates both the
plan and execution through the supported sweep parser. Missing or invalid
captures fail the collection result; partial collection cannot pass by omission.
Its one warmup/three repeats are diagnostic only. There is deliberately no
invented baseline artifact, numeric regression threshold, cache write or speed
claim. A later true regression gate needs a separately versioned, same-contract
qualified baseline and independent-process threshold review. The CPU workflow
owns a no-executable parser/plan integration regression for this entrypoint.

## Paired residue-dot scheduler controls

`rns8-residue-dot-calibrate` and `tools/collect_residue_dot.py` isolate two axes:
scalar versus packed signed-byte arithmetic **inside the same production
scheduler**, then independent versus grouped/scheduled execution **with the same
arithmetic build**. Region cases explicitly recompute a nonzero `(1,1)` rectangle
and check the cache's recomputation receipt; their full-output export cost is
reported, but their reduced compute domain is never called a batching speedup.

Build the same revision/dependencies/compiler twice. One build enables
`RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL`; the other enables
`RNS8_EXPERIMENTAL_RESIDUE_DOT4`. They are mutually exclusive. OFF/OFF preserves
legacy arithmetic and is **not** the scalar control: the legacy uniform finite
256/255/251 kernels already use packed dot. The scalar switch overrides those
explicit packed calls inside the shared accumulation helper without changing
kernel dispatch, tile geometry, reduction or slab carries. The collector rejects
legacy/mislabeled builds and only compares manifests identical after removing
these two explicit switches. Different artifact hashes remain recorded.

The bounded shortlist includes every default ladder plane, prefix 1/3/9/18/20,
other finite rings, group size one/several, sparse nonzero/zero/centered/full-width
native sources, padded host rows, K tails through 65,535/65,536/65,537, and larger
sustained grids. Every output and every residue plane is checked with an
independent Boost product outside timing, including export padding. Setup,
first call, complete compute+export p50/p95 and sustained intervals remain
separate. Work/payload admission bounds reference cost before allocation.

Optional diagnostic companions run in separate processes against the same
immutable artifacts. Their final hook-enabled invocation reports actual HIP
**event group spans**, not inferred launch counts; production comparison samples
remain hook-disabled. `gpu_isa_report.py` accepts these captures, binds explicitly
selected symbols to the captured target code object and reports static dot4,
scalar multiply/add, byte-extract/permute, LDS load/store, VGPR/SGPR, LDS and spill
metadata. Static instruction inventory does not prove dynamic execution, sign
interpretation or speedup. Actual launch counts remain unknown without a dynamic
profiler receipt. The existing arithmetic slab certificate is unchanged:
`65536*128*128+128 < INT32_MAX`; numerical-zero tails never read inactive bytes.

All authored regressions and experiments remain unexecuted; no target is newly
qualified and no default is promoted by these source-level update counts.
