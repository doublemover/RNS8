# Hardware qualification: MI300X and RX 7900 XTX

The preparation tool is [tools/gpu_qualification.py](../tools/gpu_qualification.py).
Default behavior is **plan-only**. It records current source kernel definitions,
source locations, commit/diff identity, backend builds, exact correctness/capture
commands, and remaining gates. A source kernel listing does not mean every
specialization has been compiled or executed.

## Generate the manifests anywhere

```sh
python tools/gpu_qualification.py --target mi300x --out-dir temp/qualification-mi300x-plan
python tools/gpu_qualification.py --target 7900xtx --out-dir temp/qualification-7900xtx-plan
```

By default each plan covers direct HIP, AMDGPU builtins, hipBLASLt, CK, and
rocWMMA. Use repeatable `--backend` arguments to prepare a smaller lane. No
commands from the manifest are executed without `--execute`.

Each backend gets a separate `build/qualification/<gfx>/<backend>` directory,
explicit accelerator flags, Release configuration, and one target architecture.
The current plans select `gfx942` on Linux for MI300X and `gfx1100` on Windows
for RX 7900 XTX. See AMD's [compiler/target documentation](https://rocmdocs.amd.com/projects/HIP/en/latest/understand/compilers.html)
and the installed release's support matrix; repository presets do not establish
support for a different runtime or target.

## Prerequisites and execution

Use a clean committed checkout with native build dependencies and an already
available supported GPU. Verify HIP/compiler paths and optional libraries first.
The runner does not rent hardware, install dependencies, upgrade drivers, push
source, or promote caches. Windows uses `tools/windows_dev.py` for configure,
build, and CTest. The default Windows preset paths are documented in the README;
use a matching installation or configure your local preset before running.

First execute only the direct-HIP lane on the corresponding host:

```sh
# MI300X, native Linux; choose one visible physical GPU before launching.
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python tools/gpu_qualification.py \
  --target mi300x --backend hip-direct --execute \
  --out-dir temp/mi300x-direct-qualification
```

```powershell
# RX 7900 XTX, Windows PowerShell
$env:HIP_VISIBLE_DEVICES = '0'
python tools\gpu_qualification.py --target 7900xtx --backend hip-direct --execute `
  --out-dir temp\7900xtx-direct-qualification
```

After that succeeds, qualify each optional backend with the same command and a
new `--backend` / `--out-dir`, for example `--backend amdgpu-builtins`. Omit the
backend filter only when you intend to run all five lanes. Every command has a
wall-clock timeout (default 1800 seconds); progress shows stage/count/elapsed
with a ten-second heartbeat. Full CTest uses a 600-second per-test timeout.
Use a fresh output directory for each execution to preserve previous results.

## What the runner actually checks

1. Configure and compile the selected backend with explicit target/flags.
2. Inspect the requested runtime backend and reject the wrong GPU architecture.
3. Run `rns8-verify --hip-smoke`; this fails when HIP or the device is missing.
4. Run Catch2 `[qualification]` tests with JUnit output. Missing mandatory GPU
   tests, failures, errors, and any skips fail this stage.
5. Run full CTest, including the backend's existing differential and ISA checks.
   Review intentional target-inapplicable skips; this stage alone cannot prove
   every source kernel was exercised.
6. Capture bounded signed/unsigned, exact-wide signed/unsigned, finite ring 255,
   and finite field 251 smoke benchmarks. Direct HIP additionally covers wrap64,
   finite ring 256, and the existing full-path bounded graph replay lane.
7. Validate each raw benchmark JSON with the existing schema validator.

Smoke shape is 32×32×64, seed 20261006, one warmup and three repeats. These are
small reproducibility checks. The result explicitly remains
`smoke_passed_release_qualification_pending`; it is never a release/performance
promotion. The runner isolates `RNS8_AUTOTUNE_CACHE` to a nonexistent path in its
output directory and never writes a reviewed cache.

## Required arithmetic cases

The new tagged regressions cover the defects found in this audit:

- Pack dispatch thresholds 255/256/257/258/259 and 4095/4096/4097, padded strides,
  full-width signed values, prefixes 1/8/9/20
- Bounded export of `2` at prefix 8, checked errors at prefixes 1/7/8/9/20,
  failure in every lane of a 65-element output, and padding/sentinel preservation
- Persistent compact GEMM at K=65535/65536/65537/131073/262145
- Optional AMDGPU finite-field tile/tail test expanded to N=1/2/4/5/8/19

The full suite adds full-width i64/u64 extrema, signed cancellation, all wrap64
byte pairs, prime/composite finite moduli, scheduled/zero-proof metadata, exact
limb boundaries, stale handles, and optional sparse differentials.

## Existing kernel families and remaining target checks

| Family | Source surface | Hardware evidence still required |
| --- | --- | --- |
| RNS and finite Direct HIP | `hip_direct_gemm_*`, `hip_direct_finite_*` | dot4 signedness and emitted ISA, K-splits, tile tails, finite reducer variants |
| Packing and export | `hip_direct_pack_*`, `hip_direct_export_*` | repaired pack tails, all selected prefixes, checked status, 192-bit CRT/limb export |
| Wrap64 | `wrap64_hip_kernels.hip` | carry-heavy full-width data, tiled/colpair variants, no semantic downgrade |
| Native integer comparator | `vector_alu_kernels.hip` | signed/unsigned multiprecision accumulation and small-N tails |
| hipBLASLt | backend wrappers and reduction kernels | actual algorithm selection, signed layout, scratch/reduction and version identity |
| CK / rocWMMA | backend kernels and included fragments | target-specialized matrix instructions, pack/reduction, cap/tail handling |
| AMDGPU builtins | `amdgpu_builtins_kernels.hip` | real RDNA3 WMMA, CDNA3 MFMA variants, explicit sparse-A paths on supported targets |
| Graph/reuse/grouping | benchmark graph buffers and resident launch helpers | capture legality, replay parity, allocation/source-version lifetime, cleanup |

On MI300X explicitly inspect wave64 and MFMA/SMFMAC; on RX 7900 XTX inspect
wave32, WMMA and the DP4A path. Cross-wave reduction assumptions must be checked
against [AMD's HIP porting guidance](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html).
A kernel name containing WMMA, DPP or VOPD is not ISA proof.

## After smoke, before promotion

Run the bounded [release plan](performance.md), then selected larger/padded,
long-K, graph/reuse/grouped/chain/sparse campaigns. Record which concrete kernels
and specializations execute, compare final outputs exactly, inspect ISA/resources,
and measure setup-inclusive and steady-state timings separately. Required
profiler/counter and target-status evidence must come from that actual machine.
The old 2048/4096 summaries do not qualify the repaired baseline.

Stop at correctness or device errors. Preserve `manifest.json`, `results.json`,
stdout/stderr logs, JUnit, raw captures, CTest logs, and disassembly together.
No GPU session was run to create this preparation work.
