# Performance evidence and promotion

## Current evidence boundary

The repaired source has no new GPU timing claim. Previous Windows `gfx1100`
summaries are retained in the [archive](archive/README.md), including the June
243-capture/281-test assertions. The raw campaigns and installed user cache are
not in this checkout, and the audit found executable defects that ordinary small
random inputs did not establish away. Do not extrapolate those summaries to this
revision, MI300X, Linux, or every matrix shape.

The next step is hardware correctness and compiler/ISA qualification. Only then
should an accelerator be compared with the corrected baseline. A slower checked
path is preferable to an unchecked, incorrect result.

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
