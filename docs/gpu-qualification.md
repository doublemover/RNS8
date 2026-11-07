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

Positive smoke cases consume the same versioned source semantic inventory
compiled into native backend admission and operation-provider queries. It
records backend names, signed/unsigned/wide/wrap/finite support, finite modulus
bounds and required target/provider detection. The planner retains that source
digest and separately lists inapplicable cases. In particular, CK bounded-u64
is excluded from positive captures with expected `RNS8_UNSUPPORTED_BACKEND`;
the native rejection tests remain required and later supported CK cases remain
scheduled. This never treats arbitrary nonzero exits as expected rejection.
Compilation, observed target/device availability and exact numerical completion
remain independent gates. Altered source decisions or mismatched backend,
semantic or modulus arguments stop the campaign before commands launch.
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

Optional `--stage-forecast FILE` binds a complete list of exact command names
to operator-supplied lower, expected and upper durations. The JSON schema is
`rns8_session_stage_forecast_v1`; each `stages` row has `name`, `lower_seconds`,
`expected_seconds` and `upper_seconds`, with durations written as decimal
strings. All selected stages plus closeout must fit the upper forecast. Before
each stage, the runner checks its upper duration against actual remaining time;
it stops before launching work that no longer fits. Forecasts are unqualified
estimates, and existing timeouts remain hard command limits.

An operator may also provide `--price-per-hour`, `--maximum-cost` and `--currency`
together. Quotes use explicit decimal strings and a three-letter currency;
`--price-scope allocation` is the whole allocation rate, while
`--price-scope per-selected-device` multiplies the rate by the explicitly selected
device count. `--already-incurred-cost` records a user-supplied initial amount.
Exact decimal arithmetic caps owned execution time at affordable whole seconds
and retains the requested budget, rate basis and forecast cost separately.
Quoted-cost execution requires a complete stage forecast. These are linear
operator estimates, without price lookup, provisioning, reservation extension
or automatic shutdown. Provider rounding, idle time between resumptions and
actual billing remain outside owned-command accounting. The operator still owns
reservation closeout. Forecast inputs and quote decisions are replayed on resume;
changed data invalidates the old session budget. No new session or budget
fixture has executed during the source frontload.
The runner does not impose OS RSS, CPU affinity, GPU memory, thermal or power
limits. Record and enforce allocation limits separately on the actual host.

Native Linux plans begin with a read-only resource/access receipt before any
configure/build stage. CPU preparation checks host resources without requiring
GPU access; environment-only execution performs just this preflight. It reads
the current cpuset, visible cgroup-v2 hierarchy and ancestor CPU/memory limits,
MemAvailable, available disk bytes and inodes. Requested jobs cannot exceed the
observed affinity/quota ceiling. Optional `--minimum-host-memory-gib`, measured
`--peak-build-job-memory-gib` and `--host-memory-reserve-gib` supply the declared
headroom model; absent measurements stay explicitly unavailable. Available
memory is a point-in-time observation, not a reservation or build-fit guarantee.

GPU sessions additionally require a loaded amdgpu module, character-device
KFD/render access and a positive native AMD gfx942 KFD inventory. The field
serialization and target value follow the pinned Linux
[topology producer](https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/drivers/gpu/drm/amd/amdkfd/kfd_topology.c)
and [device mapping](https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/drivers/gpu/drm/amd/amdkfd/kfd_device.c).
KFD node IDs and GPU IDs do not establish selected HIP ordinals, physical GPU
count, usable HBM or a supported driver/OS tuple. The post-build native inspector
still binds each selected runtime device. Missing hierarchy/permission/target
facts stop the dependent build with operator guidance; no permissions, driver,
affinity or cgroup settings are changed. The runner checks the preflight's actual
PID and reconstructs its policy decisions from retained observations. New native
resource, cgroup and orchestration controls remain authored and unexecuted.

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

The 17 byte-LDS wrappers in `hip_direct_gemm_tiled_kernels.cuh` and
`hip_direct_gemm_grouped_scheduled_kernels.cuh` share the portable cooperative
loader in `residue_tile.hpp`. Sixteen share `hip_direct_tiled_residue_core.cuh`;
the 128-thread column-pair entry retains its scalar two-output arithmetic in a
portable helper. Their original named entrypoints retain the concrete
16×64 A / 64×17 B shared arrays and uniform task/prefix admission. The kernel
registry's `direct_residue_tile_dataflow` records each input conversion,
reducer, dot policy, region, schedule clipping, output count and task/plane
addressing tuple. Scheduled entries rebase input pointers to their own bounds,
keep global output coordinates and share the uniform zero-axis skip predicate.
The ordinary
default-ladder wrapper retains its selected-modulus dot4 policy; the region
and task-prefix wrappers retain their configured arithmetic policy.

`tools/direct_residue_dataflow.py` checks the current lexical source/registry
contract and reports its source hashes. CPU controls compare cooperative loads,
tail zeroing, LDS padding, signed-byte dots and carry reduction with a scalar
Boost specification across the ordinary tuples, scheduled rebases and the
independent scalar column pair. Zero-axis controls independently enumerate
clipped output cells, including empty CTA windows. The GPU differential extends actual
resident/native/native-A/region routes through padded tails and K=65/65537;
existing grouped/default-prefix differentials remain part of the batch. The
uniform-small/column-pair route includes K=9/65 with odd columns and padding;
the per-tile zero-mask differential also crosses the K=64 tile boundary.
These controls are authored and syntax-checked, with execution deferred.
Qualification still requires actual launched-symbol/ISA/resource attribution,
exact outputs and matched before/after complete-call measurements. The shared
dataflow has no new measured speedup or resource-regression result yet.

The CK/rocWMMA checkers require matrix instructions for every selected kernel
symbol. They parse the exact native function label and do not borrow work from
a healthy sibling or an unrelated function in whole-object output. Explicitly
resolved direct device calls can contribute their reachable helper bodies;
unresolved register-indirect calls, recursive graphs and exceeded attribution
budgets remain unqualified. Static instruction counts and forbidden-instruction
checks cover that attributed set. They establish instruction presence, not
arithmetic execution, dynamic reachability or measured performance. New native-
text controls are authored for both target families; their batch execution and
real compiled-object checks remain deferred in the current implementation pass.

Generated native WMMA inspection uses the same per-entry/resolved-helper
attribution. Each generated variant requires its own IU8 WMMA instructions and
complete native register/LDS/scratch/spill metadata. Native symbol-table extents
report reachable code bytes per variant and unique function bytes across the
family, counting a shared helper once in the family total. Missing extents stay
unknown and fail inspection; source-only instruction counts cannot supply them.
Object hashes must remain stable throughout inspection. General ISA summaries
also slice the exact selected native label before counting instructions.
The generated report distinguishes variant coverage, native code size and
compile cost; it does not infer compile time from artifact size or instruction
count. These authored reporting controls have syntax/lint checks;
no native object inspection or fixture execution has run for the new slice.

Configure `RNS8_CAPTURE_HIP_COMPILE_COST=ON` in a named build configuration
(for example `CMAKE_BUILD_TYPE=Release`) alongside explicit HIP integration
to record the actual configured compiler command. Default builds use the direct
compiler command. Recording uses the same argv and working directory, bounded
process/output/object budgets and source/compiler/prerequisite bookends. It
requires 5 GiB free before compilation. A stale old object is removed before
issue; successful exit without a new nonempty object cannot supply a receipt.
Each attempt retains logs and a sealed receipt beneath
`<object>.compile/<attempt>/`; `<object>.compile.json` points to the latest attempt.
Failure, interruption or changed inputs remain incomplete evidence.

Generated-native ISA CTest automatically consumes that pointer when recording
is enabled. Standalone inspection accepts `--compile-receipt` for the same
pointer or retained receipt. Reported seconds cover the owned host/device
compiler-process capture, including startup and completion observation; generator
and outer receipt bookkeeping are excluded. Driver elapsed time is recorded
separately through artifact bookending. Hash coverage includes configured CMake
prerequisites, selected compiler binaries and receipt tooling; transitive system
SDK headers remain part of the separate build/dependency provenance. Missing
receipts report unrecorded cost. New process/configure/tamper controls are
authored; no build, fake-compiler fixture or CMake configuration was executed
during this source pass.

The family report also joins source-current logical/memory coverage by variant
and exact configuration. Reused checks increase the record count while each
configuration is counted once, with unknown/refuted checks preventing a proven
summary for that configuration. Full structural equality identifies duplicate
configurations and work counters; hashes serve as recorded identities. The
summary checks final-writer counts against prefix × M × N and reports qualified
full-pitch storage envelopes. These are logical ownership/model claims under
the named trusted contracts, rather than device traffic or emitted barrier
evidence. Physical lane/layout correctness remains unknown. Raw logical reports
stay linked by path/hash instead of being repeated in the ISA summary. Logical
ingestion rejects duplicate/nonfinite JSON and oversized reports before parsing.
The added inventory controls are authored with syntax/lint checks only.

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


## No-GPU Linux preparation lane

`bash scripts/cdna_prepare.sh --target mi300x --out-dir temp/linux-preparation`
plans a CPU-only rehearsal using the same dependency discovery, source identity,
command runner, budgets and checkpoint logic as the GPU session. It accepts no
device list or accelerator backend lanes. Planning is the default; `--execute`
explicitly performs configuration, compilation and CPU CTest on a Linux host.
This lane has not been rehearsed during the current development frontload.

The lane uses the existing `linux-cpu-release` preset, disables HIP and accelerator
probes, disables vcpkg manifest installation, and sets CMake FetchContent fully
disconnected. Required packages must already exist in the selected local prefixes;
it does not download, bootstrap or install them. `--set` uses the same dependency
overrides as the doctor and live session, with selected tools and policy identities
recorded in the manifest. This is a receipt for the selected environment, not a
newly qualified Linux/ROCm stack lock. The ROCm upgrade remains deferred.

CPU tests run serially and produce a retained JUnit artifact. Missing, empty,
all-skipped or failed reports cannot complete preparation; expected unsupported
GPU skips are preserved. Successful execution reports
`cpu_preparation_passed_hardware_pending`, never GPU smoke success. Matching
`--execute --resume` reuses successful stages only when the source, environment,
plan and artifact hashes still match. Changed inputs require a fresh directory.

A later GPU session uses its own output directory and explicit `--devices` list.
Preparation proves neither HIP compilation nor device/runtime/ISA correctness.
Actual Linux rehearsal, image/driver compatibility, physical device identity,
artifact export and hardware qualification still require their own receipts.

## Offline artifact cache and explicit staging

### Frozen native Linux tuple

Live Linux GPU execution requires `--platform-lock PATH`. Plan-only and CPU-only
preparation remain useful without that input. An operator selects the exact
tuple from AMD's **versioned** compatibility matrix; the runner does not choose
an image, driver or supported tuple from version numbers. The strict
`rns8_linux_platform_lock_v1` format is defined by
[`session_platform_contract.py`](../tools/session_platform_contract.py):

| Field | Required content |
|---|---|
| `target` | `gfx942` for this initial native Linux lane |
| `matrix_url`, `reviewed_date_utc` | Versioned official AMD compatibility-matrix URL and ISO review date |
| `image_digest` | Exact `sha256:` image identity, or null for a host installed without an image |
| `expected` | Exact `os_id`, `os_version`, `kernel`, `machine`, `glibc`, `driver_version` and `driver_srcversion`; one driver identity may be null, but at least one must be observed |
| `files` | 1–256 absolute native paths, each with `role`, exact positive `bytes` and lowercase `sha256` |

File roles distinguish host/HIP compilers, C++ standard library, OpenMP, HIP
runtime, device library, references, firmware and native package inventories.
Host compiler, standard library, HIP compiler/runtime and device-library roles
are mandatory. Include every selected optional/runtime input in the maintained
lock. Per-file and total limits are 2GiB and 16GiB. Loaded firmware and complete
transitive ABI correctness cannot be inferred from files on disk.

Native preflight compares OS/kernel/glibc/module observations and hashes actual
regular files before configure. The child process identity and exact lock bytes
are checked in its receipt. Closeout repeats this bounded check; resume rechecks
the native inputs before reusing successful stages. Changed or missing inputs
stop dependents. Image identity stays operator-declared; matching the lock is
reproducibility evidence, never a platform-support or hardware certificate.

```bash
bash scripts/cdna_first_pass.sh --target mi300x --devices 4 \
  --platform-lock /approved/linux-platform-lock.json \
  --set RNS8_HIP_ROOT=/selected/rocm --out-dir temp/cdna-live
```

Review the plan and add `--execute` only on the intended allocation. The lock
does not admit arbitrary prebuilt binaries: sessions still configure and build
fresh owned trees with compiled content provenance.

### Verified private export

After a completed, failed or interrupted session, use the separate export mode:

```bash
python tools/gpu_qualification.py --target mi300x --out-dir temp/cdna-live \
  --export-session --export-dir /approved/mounted-artifacts
```

This command performs no discovery, install, build or GPU work. It validates
the stopped manifest/result association and every declared session-owned file,
then streams an atomic content-addressed bundle to the explicit local or mounted
destination and verifies destination readback. External executables remain
hashed input identities rather than copied system installations. The default
8GiB export limit is explicit through `--export-maximum-bytes` (maximum64GiB).
Active sessions, changed/missing artifacts, links, conflicting existing bundles
and insufficient destination space fail without deleting source evidence.
Identical exports verify and reuse the existing bundle.

The bundle is private: allowlisted environment capture does not guarantee that
raw tool logs lack private details. Review and sanitize a separate shareable
copy before public submission. The export receipt includes a shutdown/release
checklist; this tool does not remove local evidence, terminate allocations or
assert that billing stopped. Native transfer, interruption and platform controls
are source-authored and await the consolidated validation batch.

The existing session runner accepts `--artifact-inventory inventory.json
--artifact-cache /path/to/cache`. It checks every declared artifact before session
execution and records all missing, corrupt or unsafe entries in `artifact-cache.json`.
There is no network fallback. Planning reads the cache and writes only the declared
session output directory. A supplied incomplete cache blocks execution before any
build or GPU command. The inventory is explicit: its completeness does not imply
that an operator has listed every dependency needed by the selected stack.

Use `--stage-artifacts` instead of `--execute` to copy the verified inventory to
`OUT/artifacts`. For a no-GPU packet, combine it with `--target mi300x
--session-kind prepare`. Staging is separate from dry-run, installation and runtime
qualification. It needs no selected GPU and does not extract archives, install
packages, modify a working SDK or authorize cached binaries for reuse.

The strict `rns8_session_artifacts_v1` JSON object contains only `schema` and an
`artifacts` array. Each entry requires `id`, `kind`, relative POSIX `path`, exact
`bytes`, lowercase `sha256`, `origin_url` (credential-free HTTPS or null), and
`authentication`. Kinds distinguish `package_source`, `compiler_cache`,
`binary_code_object`, `runtime_library` and `autotune_evidence`; these categories
remain separate staged files and are never activated as runtime or AUTO caches.
Authentication is a declared provenance value: `local_digest_only`,
`upstream_checksum_verified` or `upstream_signature_verified`. The staging tool
checks content integrity only and explicitly reports that it did not verify an
upstream signature or authenticate the supplied digest.

Paths cannot escape the cache, traverse symbolic links/junctions, overlap, use
Windows device names, or alias by case. Staging checks free disk space including
the session reserve, streams large files with bounded buffers, rehashes the copy,
publishes it exclusively and atomically, and performs final readback. Existing
mismatched destinations are preserved. A second run reuses verified files without
copying. Interrupted copies remove their private temporary files; complete earlier
files remain resumable. `artifact-staging.json` distinguishes complete, interrupted
and incomplete preparation. These source additions and fixtures await the final
validation batch; no preparation or hardware qualification has run here.

### Combined accelerator lane

`--all-enabled` adds a separate combined optional-accelerator build after the
requested isolated lanes. Dependency resolution includes every optional backend
in that combined build. Each selected device receives one complete regression/
CTest pass for the combined binary plus explicit smoke and semantic captures
for every backend. Parent/child device inspection and artifact names remain
backend-specific. GPU CTest runs serially; `--jobs` controls build parallelism.
The combined lane is explicit and is not part of CPU-only preparation. It tests
interactions among the configured accelerator options, not availability of
unsupported target-specific instructions or automatic backend promotion.


## Compiled binary association

The benchmark, verifier, inspector and Catch test runner accept --build-identity
as a standalone CPU-only query before any GPU or workload initialization. It
emits the executable's embedded manifest, the actual linked library's embedded
manifest and that library's ABI declaration digest. Missing provenance returns
a nonzero status. The session runner queries all four binaries after manifest
admission and before runtime device probes; differing, dirty or incomplete
executable/library manifests block the session.

Each query receipt binds the executable and raw JSON file by content, verifies
the executable did not change during the query, and records the compiled
manifest digest. Session checkpoints retain these receipts and their files.
Benchmark captures now use the same request/environment/executable annotation
helper as the existing sweep collector, outside benchmark timing intervals.
The request must describe the actual child environment, and the executable
must remain unchanged. Runtime architecture and requested backend must agree
with the session's selected lane. Resume still requires unchanged checkpoint
artifacts. These identities are prerequisites for candidate-specific phase
qualification; a successful query or smoke session alone does not complete
release, profiler, physical-device or cache-promotion gates.

The [candidate target qualification policy](target-qualification.md) explains the
versioned phase adapter, unskipped GPU/JUnit controls and explicit requalification.
