# Hardware qualification: MI300X and RX 7900 XTX

The preparation tool is [tools/gpu_qualification.py](../tools/gpu_qualification.py).
Default behavior is **plan-only**. It records current source kernel definitions,
source locations, commit/diff identity, backend builds, exact correctness/capture
commands, and remaining gates. A source kernel listing does not mean every
specialization has been compiled or executed.

## Generate the manifests anywhere

```sh
python tools/gpu_qualification.py --target mi300x --devices 0 --out-dir temp/qualification-mi300x-plan
python tools/gpu_qualification.py --target 7900xtx --devices 0 --out-dir temp/qualification-7900xtx-plan
```

By default each plan covers direct HIP, AMDGPU builtins, hipBLASLt, CK, and
rocWMMA. Use repeatable `--backend` arguments to prepare a smaller lane. No
commands from the manifest are executed without `--execute`.
Both the target and device list are required. A plan records the requested
architecture while `observed_architecture` remains null; a requested MI300X
does not establish that an MI300X exists. There is no all-device, device-zero,
or `gfx942` fallback. `--devices 0` is an explicit operator selection.

Each backend gets a separate `<out-dir>/build/<backend>` directory,
explicit accelerator flags, Release configuration, and one target architecture.
The current plans select `gfx942` on Linux for MI300X and `gfx1100` on Windows
for RX 7900 XTX. See AMD's [compiler/target documentation](https://rocmdocs.amd.com/projects/HIP/en/latest/understand/compilers.html)
and the installed release's support matrix; repository presets do not establish
support for a different runtime or target.

## Prerequisites and execution

Use a clean committed checkout with native build dependencies and an already
available supported GPU. Verify HIP/compiler paths and optional libraries first.
The runner does not rent hardware, install dependencies, upgrade drivers, push
source, or promote caches. The write-free [shared resolver](dependency-discovery.md)
selects SDK components and supplies explicit configure bindings. Use repeatable
`--set KEY=VALUE` dependency overrides when necessary; an invalid explicit root
or ambiguous stack is an error, including in a plan. Missing prerequisites can
remain pending in a plan. Windows uses `tools/windows_dev.py` for configure,
build, and CTest. Existing user presets are never overwritten. Package installation
and accelerator probes are disabled during configure.

First execute only the direct-HIP lane on the corresponding host:

```sh
# MI300X, native Linux; explicitly select an unmasked parent runtime ordinal.
python tools/gpu_qualification.py \
  --target mi300x --devices 0 --backend hip-direct --execute \
  --set RNS8_HIP_ROOT=/opt/rocm \
  --total-seconds 7200 --timeout-seconds 1800 --jobs 4 \
  --out-dir temp/mi300x-direct-qualification
```

```powershell
# RX 7900 XTX, Windows PowerShell
python tools\windows_dev.py python tools\gpu_qualification.py --target 7900xtx --devices 0 --backend hip-direct --execute `
  --out-dir temp\7900xtx-direct-qualification
```

After that succeeds, qualify each optional backend with the same command and a
new `--backend` / `--out-dir`, for example `--backend amdgpu-builtins`. Omit the
backend filter only when you intend to run all five lanes. Every command has a
wall-clock timeout (default 1800 seconds), bounded by the remaining session
budget (default 7200 seconds, with 30 seconds reserved for receipts). Progress
shows stage/count/elapsed with a ten-second heartbeat. Full CTest uses a
600-second per-test timeout. A free-space threshold is checked before execution
and each command (`--min-free-gib`, default 2); it is a stop threshold, not a
promise that a complete build will fit. `--jobs` bounds build/CTest concurrency.
The runner does not impose an OS RSS, CPU affinity, GPU memory, thermal or power
limit. Record and enforce allocation limits separately on the actual host.

Execution rejects inherited `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
`GPU_DEVICE_ORDINAL` and `CUDA_VISIBLE_DEVICES`, including empty masks. It also
rejects inherited diagnostic HSA overrides. A scheduler-provided mask must not
be cleared blindly: request an approved unmasked allocation or adapt and validate
that scheduler's namespace first. This runner does not infer that mapping.
For each explicit parent runtime ordinal it inspects that device, then sets only
`HIP_VISIBLE_DEVICES=<ordinal>` in the owned child. The child inspector must
identify the requested architecture/backend and available logical device 0 with
matching name, memory and runtime/driver evidence before smoke work proceeds.
The inspector exposes no PCI BDF or UUID. Those remain unknown; identical GPU
descriptions do not prove a physical topology mapping.

## What the runner actually checks

1. Configure and compile the selected backend with explicit target/flags.
2. Inspect the explicit parent ordinal and isolated logical device; reject a
   wrong architecture, backend, ordinal, unavailable GPU or inconsistent evidence.
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

## Native Linux session receipts

The existing shell entrypoints dispatch to the same runner. They default to a
direct-HIP **plan**, require the same explicit selection, and do not bootstrap
Python packages, Catch2 or CMake. `--accelerators` explicitly requests all five
backend lanes. Arbitrary `--bench-args`, rank scenario strings and `--skip-build`
were removed from these bounded sessions; run a separately reviewed campaign
after smoke instead of changing its semantics or trusting an unrelated build.

```bash
# No build, hardware probe, package installation or GPU execution.
bash scripts/cdna_env_probe.sh --target mi300x --devices 4,7 \
  --out-dir temp/cdna-environment-plan
bash scripts/cdna_first_pass.sh --target mi300x --devices 4 \
  --set RNS8_HIP_ROOT=/opt/rocm --out-dir temp/cdna-first-pass-plan
bash scripts/cdna_smoke.sh --target mi300x --devices 4 \
  --out-dir temp/cdna-smoke-plan
bash scripts/cdna_multigpu_smoke.sh --target mi300x --devices 4,7 \
  --out-dir temp/cdna-two-lanes-plan
```

Add `--execute` only on the intended host after reviewing prerequisites and the
allocation. The environment entrypoint writes filesystem discovery and an
allowlisted environment receipt; it does not run profiler/topology commands.
Multiple device lanes execute sequentially within one budget, as independent
processes. This is neither concurrent scaling evidence nor distributed GEMM.
No RCCL collective is called. Optional benchmark rank/world-size metadata records
the lane list; it does not establish physical device identity.

`manifest.json` v2 records command dependencies, requested device namespace,
budgets, source commit/diff and untracked-file hashes, selected tool hashes,
dependency-resolution identity and policy hash. `dependency-resolution.json`
retains selected/rejected prerequisite evidence. `environment-allowlist.json`
contains only named runtime settings; authenticated URLs/control characters are
redacted. Child build commands receive an explicit local build/runtime allowlist,
not cloud credentials, SSH agents or arbitrary parent environment variables.
Tool output is retained as evidence; review it before public sharing. Historical
`env.log` files from older scripts may contain a full environment and must not be
published without review.

Each completed command checkpoints logs, validation status and artifact hashes.
Failure stops dependents. Interrupt/timeout cleanup uses the existing owned
process-group runner. A fresh directory preserves earlier evidence. Explicit
`--execute --resume` accepts only the same source/dependencies/device/environment
and ordered successful prefix with unchanged artifacts; failed work is rerun and
its consumed time stays charged to the original budget. A changed input requires
a new directory. Receipts are reproducibility records, not signed attestations.
No fake benchmark captures or successful hardware results are written by a plan.

The provided-log normalizer `tools/cdna_env_summary.py` v2 preserves explicitly
indexed SMI fields, but never sorts rocminfo architectures/BDFs into guessed device
records or copies one architecture to every requested GPU. Runtime-to-SMI mapping,
node/visible GPU counts, profiler availability without captured evidence and
missing topology remain unknown. This source work and its no-GPU fixtures await
the consolidated validation run. Native Linux rehearsal, scheduler integration,
physical UUID/BDF correlation, profiler/ISA collection and actual MI300X execution
remain acceptance gaps under [#58](https://github.com/doublemover/RNS8/issues/58).

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

### External accelerator build identity

Build manifest v2 binds enabled rocWMMA, CK and hipBLASLt include trees, plus
CK generated/patch trees, by relative filename, byte length and SHA256 of actual
contents. Trees may live outside the RNS8 checkout. Every build rehashes them;
header edits/additions/deletions change the forced build identity and COFF
mixed-object link guard. Directory links are followed with cycle rejection;
missing, empty and unreadable configured trees fail the manifest build.

Release comparison requires v2 external-tree records consistent with the enabled
backend configuration. Shared dependencies must have matching content identities
and vcpkg metadata digests; relocations alone do not change tree contents.
Historical v1 captures remain readable but cannot qualify new promotions.
This is build-input identity, not a claim that runtime-loaded shared libraries
have been hashed. Execution artifact/environment checks remain separate gates.
Authored isolated-tree and release-admission regressions await execution.
