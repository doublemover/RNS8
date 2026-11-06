# Native ROCm device-ASan candidate

The authored `tools/device_asan_preflight.py` plans by default. No preflight,
tool version probe, native agent query, configuration, build, artifact extraction
or device test has run during this source pass. Windows gfx1100 is outside this
device-instrumentation lane; host ASan and ordinary device differential checks
retain separate scopes.

The initial package policy is native Linux x86-64, ROCm **10.1.0**, and one
`gfx942:xnack+` or `gfx950:xnack+` target. AMD's
[10.1 ASan packages](https://rocm.docs.amd.com/en/docs-10.1.0/install/asan.html)
cover gfx942/gfx950 and require HMM, XNACK and instrumented runtimes. The
[GPU sanitizer guide](https://rocmdocs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html)
specifies address instrumentation, a shared sanitizer runtime, explicit
`xnack+`, and instrumented HIP/HSA libraries. Its older architecture examples
do not expand this release's packaged target policy.

```sh
python tools/device_asan_preflight.py --rocm-root /opt/rocm/core-asan-10.1 \
  --target gfx942:xnack+ --agent-index 1 --out temp/device-asan-preflight
```

The agent index is the **rocminfo agent index**, not a HIP ordinal. The plan
requires existing SDK tools but launches none and creates no output directory.
SDK-local compiler/hipcc/rocminfo/LLVM tools are discovered automatically from
known prefix locations; explicit overrides must stay inside that SDK.

`--observe` collects the running kernel's native configuration, checks kernel
version/HMM build prerequisites, existing `HSA_XNACK=1`, absence of a fabricated
target override, loaded AMDGPU driver identity, KFD access and available host
memory. It binds the actual SDK version file, matching AMD LLVM tools and HIP
component version matching the selected HIP header, native agent UUID and
advertised `xnack+` ISA. The ROCm Core release and HIP component version are
independent identities; Core10.1.0 does not imply HIP10.1. Missing
kernel configuration or runtime identities block the preflight. WSL and
unreviewed release/target tuples cannot pass as native supported tuples.

HIP/HSA runtime paths are discovered only when unambiguous. Use `--hip-runtime`,
`--hsa-runtime` or `--asan-runtime` to disambiguate a prepared SDK. The observer
requires native ASan instrumentation interfaces in the selected HIP/HSA ELF
files and a defined ASan interface in the shared sanitizer runtime. File names
or an `asan` directory alone are insufficient. The report binds immutable file
identities and the explicit child-process runtime search directories; it does
not infer that a future application loaded these files.

Add `--compile --cmake /absolute/path/to/cmake` to the command to plan a fresh
build under its owned output directory. Add `--observe` to execute that plan.
Ninja is discovered from PATH or selected with `--ninja`, then bound explicitly.
No installer, kernel setting, driver switch, retry or deletion is performed.

The fresh Debug candidate uses `RNS8_ENABLE_DEVICE_ASAN=ON` and
`RNS8_DEVICE_ASAN_ROCM_VERSION=10.1.0`. CMake applies `-fsanitize=address`,
`-shared-libsan`, nonrecovering checks and debug information to first-party host
objects **and** explicit HIP compiler argv; matching final/consumer link flags
are carried by the library interface. Separate host sanitizer, coverage, fuzzer
and OpenMP lanes are rejected. The preflight disables optional external
accelerators/transports and CPU instruction providers, tools and benchmarks.
Instrumentation selectors/flags are recorded in compiled provenance and are
rejected by production timing admission.

The observer checks every configured first-party command and object against
the selected compiler/target/flags. Shared ISA extraction routes all commands
through bounded native captures, preserves the inspected input object and
rejects duplicate/ambiguous bundles. Actual extracted AMDGPU ELF objects must
contain observed ASan access/report symbols; direct HIP, wrap64 and vector ALU
sources must all appear. Host symbols, selected bundle, raw device symbols,
ELF notes, source/generated inputs, artifact files and the current source
manifest are retained separately. The test executable must directly depend on
the shared ASan runtime. Ignored/unsupported instrumentation diagnostics block
acceptance.

`prerequisites_observed` and `artifact_preflight_ready` are **preflight states**.
They are not successful GPU tests or detector qualification. Symbol sites do
not prove every access was instrumented; external backend instrumentation,
actual application/consumer loader resolution, ordinary bounded CPU-differential
correctness and actual free device-memory admission remain separate gates.
The report is non-promotable and supplies no production timing evidence.

The command has no workload-launch or fault-injection option. Existing stopped
device-failure campaigns remain stopped. A detector-sensitivity device fault
would require its own bounded authorization and receipt; host controls cannot
stand in for it. Deferred offline controls are in `tools/test_device_asan.py`.
