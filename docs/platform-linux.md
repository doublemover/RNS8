# Native Linux ROCm setup

Use native Linux packages and a supported GPU/OS/ROCm pairing. Common dependencies
and their sources are in the [dependency guide](../third_party/README.md).
The [readiness reference](platform-readiness.md) records the difference between
source coverage and actual device evidence. Linux CPU CI runs are recorded
separately from Linux GPU qualification, which remains open.

## Prerequisites

Use the selected Python 3.15.1 and exact non-GPU tool/package candidates from the
[version policy](dependency-versions.md). Their upgrade qualification remains
pending; the user has deferred the separate ROCm10.1 migration.

Check AMD's [release-specific GPU and OS matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html),
then follow its [Linux driver/runtime/development installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/install/install-methods/package-manager/package-manager-ubuntu.html)
for your distribution. Radeon graphics workloads also need the matching
[Radeon guidance](https://rocm.docs.amd.com/projects/radeon/en/latest/).
Install the HIP development compiler/headers/libraries as well as a compatible
driver. Profilers and distributed libraries are separate optional components.

Install native CMake, Ninja, C++17 compiler, Git, Python, Boost headers and
nlohmann-json; Catch2 3.x is needed when building tests. Ubuntu's package names
for the non-GPU development dependencies are:

```sh
sudo apt install cmake ninja-build g++ git python3 libboost-dev nlohmann-json3-dev catch2
```

This is an explicit package installation, not a configure prerequisite check.
Verify the distribution's Catch2 version; a 2.x package does not satisfy the
project. Use `CMAKE_PREFIX_PATH` for a separately installed native dependency.
Keep Windows vcpkg roots and triplets out of Linux/WSL search paths; CMake rejects
them. The [dependency guide](../third_party/README.md) owns current constraints.

Inspect installed tools before configuration:

```sh
python3 tools/check_dependencies.py
```

The checker reports `hipcc`, `hipconfig`, `rocminfo`, `rocm-smi`/`amd-smi` and
optional profiling/topology tools. Discovery is not a correctness test.
`rocprofv3`/`rocprofv3-avail` are needed for counter/resource qualification,
not a basic compile. `numactl` and `lstopo` help record topology. RCCL readiness
is relevant to future distributed paths; today's shard smoke has no collectives.

## Radeon build and first execution

Choose the actual ROCm root and one target supported by that compiler/device.
For an RX 7900 XTX, the target is `gfx1100`:

```sh
rns8_rocm_root=/opt/rocm
cmake --preset linux-rocm-debug "-DRNS8_HIP_ROOT=$rns8_rocm_root" -DRNS8_AMDGPU_TARGETS=gfx1100
cmake --build --preset linux-debug --parallel 4
./build/linux-rocm-debug/rns8-bench --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

Set `rns8_rocm_root` to a nondefault installation if needed, including a path
with spaces. The preset retains its target policy while the shared pre-project
[selection contract](dependency-discovery.md) resolves SDK locations. Explicit
roots, environment/prefix sources and bounded native installations are ranked;
multiple equally eligible SDKs require a choice. For a configured tree use
`python3 tools/check_dependencies.py --resolve-only --json --cmake-cache
build/linux-rocm-debug/CMakeCache.txt` to inspect the same selected context.
Actual native-Linux fresh-prefix execution and broader API/ABI/probe acceptance
remain gaps under[#57](https://github.com/doublemover/RNS8/issues/57).

## Instinct build

The CDNA preset keeps native package discovery and empty workstation dependency
roots. Select the target for the actual device; this example is MI300X/MI325X:

```sh
rns8_rocm_root=/opt/rocm
cmake --preset linux-cdna-debug "-DRNS8_HIP_ROOT=$rns8_rocm_root" -DRNS8_AMDGPU_TARGETS=gfx942
cmake --build --preset linux-cdna-debug --parallel 4
./build/linux-cdna-debug/rns8-bench --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

MI210/MI250(X) use `gfx90a`; MI350X/MI355X use `gfx950`, subject to the selected
release's GPU/OS/compiler support. These Linux GPU commands are statically
checked instructions, unexecuted on the Windows development host. Device
validation is requested per SKU, OS and backend.

## Optional backends and profiling

Enable hipBLASLt, CK, rocWMMA or AMDGPU builtins explicitly with the switches in
[backend capabilities](backend-notes.md). Linux integration exists; supported
hardware and compatible packages must still execute the differentials.
`linux-cdna-accelerators-release` enables all four and discovery probes for a
real Instinct host. Override its `RNS8_HIP_ROOT` and `RNS8_AMDGPU_TARGETS` as needed.
Native package discovery remains active; optional `RNS8_CK_ROOT` and
`RNS8_ROCWMMA_ROOT` select explicit compatible header installations.

Discovery-only probes are separate:

```sh
python3 tools/check_dependencies.py --accelerator-probes --json
cmake --preset linux-rocm-accelerator-probe -DRNS8_HIP_ROOT=/opt/rocm -DRNS8_AMDGPU_TARGETS=gfx942
```

Probe artifacts stay under ignored output directories. Probe presets keep
backend enable flags disabled; explicitly enabled backends compile real kernels
and retain target restrictions. See [qualification](gpu-qualification.md) for
correctness/ISA checks and [performance](performance.md) for recorded comparisons.

## Existing bounded host scripts

The CDNA entrypoints use the existing
[qualification runner](gpu-qualification.md#native-linux-session-receipts).
They default to plan-only and require a target plus explicit unmasked parent
HIP runtime ordinals. No missing-device or architecture fallback is permitted.
The same shared dependency resolver supplies configure bindings; the scripts do
not install packages or dump the complete process environment.

```bash
bash scripts/cdna_first_pass.sh --target mi300x --devices 0 \
  --set RNS8_HIP_ROOT=/opt/rocm --out-dir temp/cdna-first-pass-plan
bash scripts/cdna_multigpu_smoke.sh --target mi300x --devices 4,7 \
  --set RNS8_HIP_ROOT=/opt/rocm --out-dir temp/cdna-selected-lanes-plan
```

Review `manifest.json`, `dependency-resolution.json` and the allowlisted
environment receipt before adding `--execute` on the actual host. Select
`--total-seconds`, `--timeout-seconds`, `--jobs` and `--min-free-gib` for that
allocation. `--accelerators` explicitly selects optional backend lanes.
Inherited visibility masks and diagnostic HSA overrides block execution; a
scheduler's device namespace requires explicit integration rather than guessing
or blindly clearing its mask. Logical device 0 is used only inside the child
isolated from an explicitly selected parent ordinal.

Multiple lanes run sequentially as independent processes within the bounded
session. Their runtime ordinals are not physical SMI indices; PCI/UUID/NUMA
correlation remains unknown. No distributed GEMM, RCCL collective, scaling
qualification or profiler readiness is implied. Arbitrary benchmark/rank strings
and reuse of unrelated build trees were removed from this bounded path.
Run separately reviewed follow-up campaigns after smoke.

These source changes and no-GPU fixtures await the consolidated validation run.
Native Linux rehearsal, scheduler integration, physical topology, profiler/ISA
collection and real Instinct execution remain explicit
[#58](https://github.com/doublemover/RNS8/issues/58) acceptance gaps.

GCC may emit fortified `memcpy` warnings from Boost.Multiprecision `cpp_int`
internals on some optimization levels. Treat those as non-blocking compiler
noise unless a project-owned callsite is implicated; isolate or suppress them
locally in a later warning-cleanup pass rather than globally weakening Linux
diagnostics here.

The Linux preset keeps two target lists separate:

- `RNS8_AMDGPU_TARGETS` is the active offload list used by explicit HIP
  compilation. It should be set to targets supported by the active ROCm release
  and the validation machine.
- `RNS8_ROCM_COVERAGE_TARGETS` is source-level readiness metadata. It records
  the documented RDNA2/RDNA3/RDNA4 and CDNA2/CDNA3/CDNA4 coverage families:
  `gfx1030`, `gfx1100`, `gfx1200`, `gfx1201`, `gfx90a`, `gfx942`, and
  `gfx950`. It does not add compiler offload architectures by itself.


GCC can emit Boost.Multiprecision fortified `memcpy` warnings. Investigate any
project-owned callsite, and isolate compiler/library noise rather than globally
weakening diagnostics.

## WSL

Treat WSL as its own driver/runtime environment. Follow
[AMD's WSL support instructions](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
for a listed GPU and release. Native Windows HIP execution does not establish
WSL support, and native Linux presets do not validate WSL's GPU stack. Keep all
Linux dependencies native to WSL and use a separate build tree; Windows vcpkg
paths remain invalid. No RNS8 WSL GPU validation is recorded.

## Developer checks

[Contributing](../CONTRIBUTING.md) owns CPU reference and sanitizer workflows,
CTest, package installation and downstream-consumer commands. The
[hardware qualification guide](gpu-qualification.md) owns GPU acceptance. Avoid
using a fresh configure, discovered library or source target as evidence of a
passed device test.
