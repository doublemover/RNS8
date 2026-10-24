# Dependencies and third-party policy

RNS8 uses vcpkg-managed C++ dependencies declared in `vcpkg.json` on Windows.
Linux builds use distro packages or native CMake package prefixes instead of
Windows vcpkg roots.

## Shared prerequisites

This page owns the current dependency requirements; the
[Windows](../docs/platform-windows.md) and [Linux](../docs/platform-linux.md)
guides own OS-specific commands. The current source uses C++17, CMake 3.22+,
Ninja, Git and Python 3.10+. Python's annotation syntax and standard-library APIs
set that existing tooling floor; this is not a newly qualified version matrix.
[#60](https://github.com/doublemover/RNS8/issues/60) owns compatibility-gated
upgrades/locks, which are not yet implemented. Use the versions recorded in a
[dated build](../docs/local-all-features-20261023.md) when reproducing its evidence.

| Role | Dependency | Installation source |
|---|---|---|
| Required C++ headers | Boost.Multiprecision, nlohmann-json | [Windows vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started), or native Linux development packages |
| GPU compiler/runtime | AMD driver plus HIP SDK/ROCm development stack | [Windows HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/how-to/install.html), [Linux ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/install/install-methods/package-manager/package-manager-ubuntu.html) |
| Windows host compiler | VS2026 C++ desktop workload + Windows SDK for the recorded local build | [Microsoft C++ installation](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation) |
| Build tooling | CMake, Ninja, Git, Python | [CMake](https://cmake.org/download/), [Ninja](https://ninja-build.org/), [Git](https://git-scm.com/downloads), [Python](https://www.python.org/downloads/) |
| Tests only | Catch2 3.x | vcpkg or native Linux package/prefix; `RNS8_BUILD_TESTS=OFF` omits test targets |
| Optional exact references | GMP, FLINT | vcpkg `optional-exact-libs` feature or native packages, with explicit enable flags |
| Optional accelerators | hipBLASLt, CK, rocWMMA | Compatible AMD packages or the pinned source headers below |
| Optional CPU parallelism | OpenMP runtime/compiler support | Native compiler toolchain; serial reference remains available |

The Windows manifest currently includes Catch2 even when test targets are
disabled. Disabling tests changes the build requirement, not the manifest's
package list. Required C++ packages are in [vcpkg.json](../vcpkg.json); its
`builtin-baseline` pins the registry snapshot. RNS8 does not provide a system installer. The vcpkg toolchain can install
manifest packages according to its own configuration. To perform that package
step explicitly before CMake, install the manifest packages
from the repository root before configuring:

```powershell
$env:VCPKG_ROOT = 'D:/Development Tools/vcpkg'
& "$env:VCPKG_ROOT/vcpkg.exe" install --triplet x64-windows
```

This command installs packages. `python tools/check_dependencies.py` only reports
existing dependencies; optional compile probes require `--accelerator-probes`.
[#57](https://github.com/doublemover/RNS8/issues/57) tracks root autodetection;
current GPU presets accept explicit `RNS8_HIP_ROOT`, `CMAKE_TOOLCHAIN_FILE`,
`CMAKE_PREFIX_PATH`, `RNS8_CK_ROOT` and `RNS8_ROCWMMA_ROOT` overrides.

## Optional exact references and accelerator pins

Boost.Multiprecision supplies exact CRT and wide-integer reference behavior. GMP and FLINT are optional
differential references on Windows and must not be required for core
correctness. hipBLASLt, CK, rocWMMA, and AMDGPU builtins remain
feature-detected accelerator paths and are not correctness requirements.

Repo-local CK and rocWMMA source is tracked through pinned Git submodules;
Linux can alternatively use compatible native package headers:

- `third_party/rocm/composable_kernel`
- `third_party/rocm/rocWMMA`

Record pin state with `git submodule status --recursive`; the exact descriptive
suffix can vary with upstream tags, so the durable contract is the path, branch,
URL, and commit SHA:

| Component | Path | Branch | Commit |
|---|---|---|---|
| Composable Kernel | `third_party/rocm/composable_kernel` | `release/rocm-rel-7.1` | `d9272218c4c59a58e41d3d346362cdaa707c30ce` |
| rocWMMA | `third_party/rocm/rocWMMA` | `release/rocm-rel-7.1` | `1ab208f49945c38626b79e3f0c284d65ac44a781` |

Initialize and probe them with:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
python tools\bootstrap_rocm_accelerators.py --dry-run --init --probe --target gfx1100 --json
```

The bootstrap command compile-probes CK and rocWMMA headers and object-compiles
RNS8-owned int8 matrix-engine primitive probes for `gfx1100`. That output is
dependency readiness evidence only; it is not backend execution, differential
correctness validation, ISA evidence, or performance evidence.

Do not clone these repositories under `C:\` or commit upstream build products.
Generated dependency artifacts belong under ignored `temp/`, `out/`, or
`build/` paths.


## Attribution and provenance

[NOTICE](../NOTICE) and upstream licenses retain attribution. Preserve submodule
commit identities, configured external include-tree hashes and build manifests
when comparing results. Configure-time CK compatibility overlays are generated
in the build tree and exact-match guarded; they do not edit installed/upstream
headers. The LLVM HIP math wrapper compatibility overlay likewise remains
build-local. Automatic upgrades and lock regeneration belong to #60.
