# Linux ROCm Platform Notes

Linux ROCm remains the production, profiling, multi-GPU, and Instinct
validation path. The scaffold keeps Linux ROCm presets and toolchain variables
represented, but Windows validation does not validate Linux ROCm, Instinct
CDNA, profiling, power, or cluster production gates.

`tools/check_dependencies.py` reports Linux readiness separately from Windows
readiness. On non-Linux hosts, `E003` Linux ROCm detection and the Linux platform
matrix gates are reported as not applicable rather than as Windows blockers.
On Linux, `hipcc`, `hipconfig`, `rocminfo`, and either `rocm-smi` or `amd-smi`
are treated as the ROCm capability-inspection command set.
`rocprofv3` and `rocprofv3-avail` are required for the counter/resource audit
lane, but not for a basic build-only smoke. `numactl` and `lstopo` provide
topology evidence. RCCL and `rccl-tests` are future multi-GPU platform
readiness signals; the current multi-GPU smoke runs independent per-GPU
benchmark shards and does not use collectives.
The JSON readiness report also carries `exact_wide_platform_validation` so
Windows `gfx1100` exact-wide limb evidence remains separate from Linux ROCm,
Radeon Linux, and Instinct CDNA validation. Those fields stay unvalidated until
the exact CPU differential suite is run on real supported Linux ROCm hardware.
In particular, Windows reports keep `windows_evidence_validates_linux_rocm` and
`windows_evidence_validates_instinct` false; Linux and Instinct fields are
validation claims only after a real Linux ROCm host runs the relevant parity
suite.
The same report carries `readiness.correctness_backend_validation`, which
records that dependency discovery does not validate CPU, direct-HIP, wrap64, or
accelerator correctness backends. Its Linux/Instinct records are representation
and readiness metadata on Windows, not validation evidence.

Expected Linux configure path:

```bash
cmake --preset linux-cpu-debug
cmake --build --preset linux-cpu-debug
ctest --preset linux-cpu-debug --output-on-failure

cmake --preset linux-rocm-debug
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

For Ubuntu 24.04 hosts with AMD ROCm package repositories already configured,
the intended native-package install surface is:

```bash
sudo apt install cmake ninja-build g++ python3 python3-venv python3-pip catch2 nlohmann-json3-dev libboost-all-dev rocm rocm-hip-sdk rocm-developer-tools rocminfo rocm-smi-lib amd-smi-lib rocprofiler-sdk rocprofiler-compute rocm-bandwidth-test rccl-dev rocwmma-dev hipblaslt-dev composablekernel-dev
```

AMD's relevant readiness surfaces are the ROCm package-manager install flow,
HIP visibility variables, `rocprofv3`/`rocprofv3-avail`, AMD SMI,
hipBLASLt/rocWMMA native packages, and RCCL/`rccl-tests`:
[ROCm Ubuntu install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html),
[HIP env vars](https://rocmdocs.amd.com/projects/HIP/en/latest/reference/env_variables.html),
[rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html),
[rocprofv3-avail](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3-avail.html),
[AMD SMI](https://rocmdocs.amd.com/projects/amdsmi/en/latest/install/install.html), and
[RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/).

Linux presets use system packages and native CMake package discovery only.
Install native Linux development packages for dependencies such as Catch2,
nlohmann-json, Boost headers, and optional GMP/FLINT references, or point
`CMAKE_PREFIX_PATH` at native Linux installs. Do not point Linux/WSL
configures at Windows vcpkg roots such as `/mnt/c/vcpkg`, and do not use a
Windows vcpkg triplet; CMake fails at configure time if those paths enter the
Linux prefix or include search.

Clean CDNA real-host presets avoid repo-local generated accelerator dependency
roots and target the supported CDNA families directly:

```bash
cmake --preset linux-cdna-debug
cmake --build --preset linux-cdna-debug
ctest --preset linux-cdna-debug --output-on-failure
```

The one-command first-pass path captures the host environment and topology
before any benchmark evidence:

```bash
bash scripts/cdna_first_pass.sh --out-dir temp/cdna-first-pass-real
```

Optional CDNA follow-up scenario groups can be queued after the first smoke
without making them default smoke blockers. The current CDNA-prep closeout
groups are `wrap64-carry`, `k-block-tile-variants`, `layout-search`,
`finite-distributions`, and `vector-to-rns-chain`:

```bash
bash scripts/cdna_first_pass.sh \
  --out-dir temp/cdna-first-pass-rank-followup \
  --rank-scenarios wrap64-carry,k-block-tile-variants,layout-search,finite-distributions,vector-to-rns-chain
```

For an independent per-GPU shard smoke on an eight-GPU host:

```bash
bash scripts/cdna_multigpu_smoke.sh --devices 0,1,2,3,4,5,6,7 --out-dir temp/cdna-multigpu-smoke-real
```

The scripts always run `scripts/cdna_env_probe.sh` first and write raw logs plus
`cdna-env-summary.json`. The summary records `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, `ROCM_PATH`, `HIP_PATH`,
`LD_LIBRARY_PATH`, rocminfo `gfx*` targets, SMI device names, ROCm/HIP versions,
visible and node GPU counts, a `physical_devices` array keyed by physical GPU
index with target arch/name/BDF/NUMA/visibility metadata where discoverable,
`rocprofv3` readiness, RCCL readiness, and `rccl-tests` readiness.

The multi-GPU smoke remains embarrassingly parallel: one process per physical
device with `ROCR_VISIBLE_DEVICES=<physical_id>` and mirrored
`HIP_VISIBLE_DEVICES=<physical_id>`. `scripts/cdna_multigpu_smoke.sh` writes
per-shard captures under `shards/gpu*/` and then runs
`tools/multigpu_shard_report.py`; the report lists rank, world size, physical
device id, target arch, device name, BDF, NUMA node, schema status, checksum,
timing outliers, missing shards, profiler readiness, and RCCL/`rccl-tests`
readiness. Partial lists such as `--devices 4,5,6,7` are recorded by physical
device id, not by rank position.

Raw `rns8-bench` captures also record optional `RNS8_MULTI_GPU_MODE`,
`RNS8_RANK`, and `RNS8_WORLD_SIZE` runtime environment fields when a shard
launcher sets them. The `target-status.json` sidecar remains the richer source
for BDF, NUMA, profiler, RCCL, and physical-device topology.

`scripts/cdna_smoke.sh` is the minimal single-device smoke wrapper:

```bash
bash scripts/cdna_smoke.sh --devices 0 --out-dir temp/cdna-smoke-real
```

Every CDNA script supports `--out-dir`, `--preset`, `--devices`,
`--bench-args`, `--skip-build`, and `--dry-run`. `--accelerators` selects the
clean `linux-cdna-accelerators-release` preset unless `--preset` is supplied.
That preset enables `RNS8_PROBE_ACCELERATORS`, hipBLASLt, CK, and rocWMMA for a
real Instinct host while keeping `RNS8_ENABLE_AMDGPU_BUILTINS=OFF`,
`RNS8_HIP_ROOT=/opt/rocm`, `RNS8_AMDGPU_TARGETS=gfx90a;gfx942;gfx950`, and no
Windows vcpkg or local workstation paths. It is a configure/build surface only;
CDNA performance readiness still requires target-validation and release-review
captures.

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

Linux-specific accelerator paths are intentionally not required for core
correctness:

- hipBLASLt INT8 GEMM remains a later feature-detected backend.
- CK grouped/fused kernels remain a later feature-detected backend.
- rocWMMA and AMDGPU builtins remain target-specific hot paths.

The checker and CMake find modules report hipBLASLt, CK, and rocWMMA discovery
as candidate evidence only. The hipBLASLt shallow probe looks for headers,
libraries, and the optional `hipblaslt-bench` utility; CK and rocWMMA shallow
probes look for their headers. AMDGPU builtin readiness has no shallow
discovery-only pass; it remains not ready until target-specific exact kernels
exist. Optional compiled evidence can be requested with:

```bash
python tools/check_dependencies.py --accelerator-probes --json
cmake --preset linux-rocm-accelerator-probe
```

Component probes write scratch evidence under `temp/` or configure an
evidence-only CMake preset. AMDGPU builtin probes report
`NOT_RUN_NO_CORRECTNESS_KERNEL` until a real target-specific exact kernel
exists. They keep `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`,
`RNS8_ENABLE_ROCWMMA`, and `RNS8_ENABLE_AMDGPU_BUILTINS` disabled. Linux
production readiness still requires target-supported components, compiled
capability probes, exact CPU differential coverage, and measured performance
before enabling advanced backend stages. AMDGPU builtin hot kernels follow the
same rule: compiler or architecture availability is only candidate evidence
until a target-specific kernel has exact CPU differential coverage.
Checker records mark this as
`evidence_class=candidate_accelerator_evidence_only` and
`candidate_evidence_is_correctness_validation=false`.
The same hard-cut enable policy is covered by CTest configure-negative cases:
each accelerator enable flag must fail while only evidence probes exist.

Before claiming Linux production readiness, run direct HIP parity tests on the
target ROCm release and actual supported Radeon or Instinct hardware. Until
that happens, the Linux and Instinct entries are represented roadmap targets,
not validated substitutes for the Windows `gfx1100` direct HIP evidence.
Exact-wide signed/unsigned limb export must be validated on the same real Linux
host before claiming Linux or Instinct parity; Windows fixed-width limb ABI
evidence is not portable validation evidence.
