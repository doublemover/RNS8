# Shared dependency selection

The existing dependency checker and the pre-project CMake hook use
`tools/check_dependencies_lib/resolution.py` to select installations. Selection
is a filesystem operation: it does not run a compiler, package manager,
device query, installer or version probe. A discovered path is not evidence
of API, ABI, instruction, runtime or arithmetic compatibility.

## SDK initialization

`tools/bootstrap_rocm_accelerators.py` also provisions exact ROCm Core SDK
tarballs from the [AMD stable index](https://stable.repo.amd.com/rocm/core/tarball/).
The default is Core SDK 10.1.0. This opt-in path is separate from the read-only
doctor and from `--init`/`--probe` source-submodule actions. It does not install
a driver, change system PATH, install vcpkg packages or qualify a GPU backend.
AMD documents the [tarball layout and extraction](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html).

```powershell
# Preview detected targets and one SDK/build lane per family; no network/writes.
python tools/bootstrap_rocm_accelerators.py --sdk-install --sdk-build --dry-run --json

# Explicit target also works before any SDK is installed, or on a headless host.
python tools/bootstrap_rocm_accelerators.py --sdk-install --sdk-target gfx1100 --json

# Reuse an existing extraction; explicitly select the vcpkg installation already prepared.
python tools/bootstrap_rocm_accelerators.py --sdk-root C:/TheRock/build --sdk-target gfx1100 --sdk-build --sdk-cmake-arg=-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake --sdk-cmake-arg=-DVCPKG_INSTALLED_DIR=C:/Users/sneak/Development/RNS8/vcpkg_installed --json
```

Native Linux uses the same Python entry point. AMD architecture enumeration is
preferred; Windows also recognizes common desktop RX 7000/9000 adapter names
without an SDK. Unknown AMD adapters require an exact `--sdk-target`; there is
no silent gfx1100 fallback. Repeat that option for multiple devices/families.
Only exact known targets are mapped; actual OS/family/release availability is
checked against AMD's index before download. Available archives do not imply
official driver support or arithmetic/instruction qualification.

`--sdk-prefix`, `--sdk-cache` and `--sdk-build-root` select managed directories;
defaults are ignored `out/third_party/rocm/sdk`, `temp/rocm-sdk-downloads` and
`build/rocm-sdk`. Family/version directories are never overlaid. Downloads use
HTTPS, bounded streaming and atomic publication; extraction uses Python's tar
data filter plus path/member/expanded-size checks in a new staging directory.
Internal Linux library links are permitted, escaping links and special files
are rejected. An interrupted extraction never becomes an installed prefix.
Links resolve after streamed payloads, including forward references; Windows
file links can become internal hard links/copies when symlink privileges are
unavailable. Expanded-byte limits also cover fallback copies.
Successful installations record source URL, archive SHA-256, family, version
and SDK-relative root in `rns8-sdk.json`. Reuse rechecks SDK coherence/version.

Supply `--sdk-sha256 gfx110X-all=HEX_DIGEST` to enforce an independently obtained
64-digit SHA-256. The observed AMD index has no checksum sidecar for the local
archive; an unpinned install records `https_observed_sha256`, not publisher hash
verification. Cached archives are rehashed before reuse; corrupt/unknown cache
entries and existing inconsistent prefixes fail without being overwritten.
An adopted `--sdk-root` records unverified archive origin instead of inventing
a download receipt. Archive hashes describe the archive, not ongoing installed
file tamper detection.

`--sdk-build` configures Release/Ninja and builds each selected family separately,
feeding explicit compiler, runtime, bitcode, optional SDK-header roots and
offload targets through shared discovery. `--sdk-cmake-arg=-DKEY=VALUE` supplies
feature/package choices and `--sdk-jobs` bounds compilation parallelism. SDK
bindings take precedence over extra options. Build commands inherit only
process-local SDK paths; Windows loads the VS developer environment through
`windows_dev.py`. No tests or benchmark campaigns are started by provisioning.
The JSON output records the actual commands; in a dry run the archive's internal
SDK subdirectory is not yet known, so the planned root is the destination prefix.

Run Windows CTest through the developer wrapper as well, so installed-consumer
checks inherit the Windows SDK linker/resource-compiler environment:

```powershell
python tools/windows_dev.py ctest --test-dir build/rocm-sdk/windows-gfx110X-all-10.1.0 --output-on-failure -j 1
```

## Inspect or configure

Inspect environment candidates without executing the full doctor's command/
device inventory or optional probes:

```powershell
python tools/check_dependencies.py --resolve-only --json
```

Use the same CMake variable names for explicit choices. This example selects
nondefault Windows roots with spaces:

```powershell
python tools/check_dependencies.py --resolve-only --json --set "RNS8_HIP_ROOT=D:/AMD SDK/ROCm/7.1" --set "VCPKG_ROOT=D:/Development Tools/vcpkg" --set RNS8_ENABLE_HIP=ON
python tools/windows_dev.py cmake --preset windows-msvc-hip-debug "-DRNS8_HIP_ROOT=D:/AMD SDK/ROCm/7.1" "-DVCPKG_ROOT=D:/Development Tools/vcpkg" -DRNS8_AMDGPU_TARGETS=gfx1100
```

The Windows wrapper loads the Visual Studio developer environment and preserves
the CMake command, preset and `-D` choices. CMake calls the same checker in
resolve-only mode **before `project()`**, so the selected vcpkg toolchain is
available before compiler and package discovery. The checked-in presets no
longer require a particular workstation HIP/vcpkg root. An unambiguous existing
installation can be selected without root overrides:

```powershell
python tools/windows_dev.py cmake --preset windows-msvc-hip-debug -DRNS8_AMDGPU_TARGETS=gfx1100
```

On native Linux use the existing native-package presets. For example:

```sh
python3 tools/check_dependencies.py --resolve-only --json --set RNS8_HIP_ROOT=/opt/rocm-7.1.0 --set RNS8_ENABLE_HIP=ON
cmake --preset linux-rocm-debug -DRNS8_HIP_ROOT=/opt/rocm-7.1.0 -DRNS8_AMDGPU_TARGETS=gfx1100
```

These commands describe implemented selection paths. Fresh nondefault-prefix
Windows/Linux GPU quickstart qualification remains separate from filesystem
fixtures. Choose versions and targets supported by the actual platform; this
resolver does not select or upgrade to a purported latest release. The
[Windows](platform-windows.md) and [Linux](platform-linux.md) guides retain the
actual build/test boundaries.

Each configure writes `dependency-discovery/request.json` and `resolution.json`
inside that build directory. The typed request records root/component overrides,
feature policy and target context; the result records every bounded candidate,
its origin/priority, canonical paths, rejection reasons and the selected reason.
An effective identity binds selected files/paths, header version or null, source
revision where readable, targets, feature policy and manifest identity. Candidate
history and generated output filenames do not change that identity merely
because a subsequent configure cached an automatic choice. Compiled provenance
records `dependency_resolution_sha256`; without this path it explicitly records
`unavailable_no_shared_resolution`.

Inspect a configured build with the same cache choices:

```powershell
python tools/check_dependencies.py --resolve-only --json --cmake-cache build/windows-msvc-hip-debug/CMakeCache.txt
```

`--set NAME=VALUE` overrides that cache for inspection only. It does not rewrite
the cache, a user preset, PATH, registry, package sources or installed files.
The ordinary checker still includes its existing tool/device inventory;
`--accelerator-probes` explicitly requests its existing compile/runtime probes.
Resolve-only cannot be combined with those probes. Bootstrap remains a separate
explicit workflow; its selected toolchain report reuses the shared resolver.

## Selection rules

HIP selection uses these ordered sources:

1. Explicit CMake/cache/`--set` `RNS8_HIP_ROOT`, or an SDK inferred from explicit
   compiler/include/library/runtime/device-library component paths.
2. Project environment `RNS8_HIP_ROOT`, then `HIP_PATH` and `ROCM_PATH`.
3. Declared sysroot/find-root candidates and `CMAKE_PREFIX_PATH` candidates.
4. The SDK owning the `hipcc` found on PATH.
5. Bounded platform roots: immediate version directories under
   `Program Files/AMD/ROCm` on Windows; `/opt/rocm`, immediate `/opt/rocm-*`,
   `/usr` and `/usr/local` on Linux.

An invalid explicit choice fails even when another complete SDK exists. Two
valid candidates at the same best priority require an explicit selection;
directory names and highest-version guesses do not break ties. Symlink aliases
to the same canonical root count as one installation. Lower-priority rejected
candidates remain visible. Missing unrequested optional components do not fail
an otherwise valid selection; explicitly requested missing components do.

A HIP candidate needs its compiler driver, Clang driver, public headers,
development link library, runtime artifact and device bitcode under one
canonical SDK root. Header/library/compiler component overrides are honored
only when coherent with that root. Runtime/device-library overrides can resolve
ambiguity within it. A runtime DLL/header major disagreement, known binary
architecture mismatch, missing component or path escaping the selected root
rejects the candidate. The HIP version comes from its selected `hip_version.h`;
unknown remains null. Selected compiler/header/library/runtime file hashes are
identity artifacts, not a hash of every SDK header or proof of binary loading.

The HIP compiler child processes receive the selected `HIP_PATH`, `ROCM_PATH`,
`HIP_CLANG_PATH` and `HIP_DEVICE_LIB_PATH`, including component overrides.
These paths pass through the child environment because the Windows HIP driver
can lose spaces when forwarding joined path flags. These are process settings;
the user's system environment remains unchanged. See the upstream
[Clang HIP path rules](https://clang.llvm.org/docs/HIPSupport.html) and
[HIPCC environment variables](https://rocm.docs.amd.com/projects/HIPCC/en/docs-6.0.2/env.html).
The STL overlay probe uses the selected Clang driver rather than assuming it
lives beside `hipcc`.

Host Python selection preserves an explicit interpreter or virtual environment.
Windows autodiscovery tries `python` before `python3`. Unreadable Windows Store
app-execution aliases are unsupported: select a readable interpreter explicitly,
for example `-DPython3_EXECUTABLE=C:/path/to/python.3.15.1/tools/python.exe`
for a verified portable CPython installation, or the selected project venv executable. Discovery fails rather than switching interpreter versions.
A virtual environment's invocation path stays intact even when its executable
is a symlink to a base interpreter. The selected Python version, prefixes and
readable executable digest are recorded in discovery identity.

Windows vcpkg selection prefers an explicit `CMAKE_TOOLCHAIN_FILE` or
`VCPKG_ROOT`, then environment `VCPKG_ROOT`, the executable's PATH root, and the
bounded system-drive `vcpkg` root. The root must contain both the toolchain and a
bootstrapped executable. An explicitly selected external toolchain is preserved;
discovered vcpkg is not injected over it. Conflicting explicit vcpkg roots fail.
The declared Windows triplet is checked against the selected host/system
processor. Package presence, baseline identity and triplet spelling do not prove
the ABI or lock compatibility of an existing binary installation.

Native Linux and WSL retain native package policy. They reject vcpkg toolchains,
Windows triplets and Windows package paths in include/library/prefix/sysroot
inputs. No Linux-vcpkg lane is introduced. `lib`, `lib64` and immediate native
multiarch directories are considered. Declared sysroot/find-root candidates
are recorded; package mode `ONLY` rejects candidates outside those roots.
The resolver does not execute or interpret arbitrary user toolchain scripts.
If a toolchain computes roots only when CMake loads it, supply the intended
`CMAKE_SYSROOT`/find-root and SDK values explicitly for the early selection.
This is a selection contract, not general cross-compilation qualification.

## Optional components and core packages

Explicit `RNS8_CK_ROOT`, `RNS8_ROCWMMA_ROOT`, `RNS8_HIPBLASLT_ROOT`,
`RNS8_GMP_ROOT` and `RNS8_FLINT_ROOT` are honored. Their existing include/library
overrides are also retained or rejected for mixed-root conflicts; the handoff
does not silently overwrite them. Without overrides, candidates include CMake
prefixes, the selected SDK, selected Windows vcpkg/manifest triplet, pinned
repository CK/rocWMMA sources, and native prefixes as appropriate. Optional
header/library candidates remain `discovered`; their existing CMake finders and
explicit probe workflows own API/target/link evidence. Merely discovering them
does not change any `RNS8_ENABLE_*` flag or default backend dispatch.

Boost.Multiprecision and nlohmann-json remain required CMake dependencies;
Catch2 remains conditional on test targets and OpenMP on the existing CPU option.
Their actual `find_package`/header checks consume the selected toolchain and
prefix context. The shared resolver does not replace CMake's native package
solver. Unrelated Python/scientific packages remain inventory entries. GMP/FLINT
package inventory versions are reported separately from usability. Explicit
enablement now compiles and links a tiny arithmetic API program through the
selected imported targets, including builds with project tests disabled. Configure
does not execute it. The separately registered smoke test checks runtime arithmetic
when tests are enabled. Those authored checks do not establish complete transitive
ABI or upgraded-stack qualification.

## Installation and remaining gates

The shipped presets and early handoff default `VCPKG_MANIFEST_INSTALL=OFF`.
Install declared packages explicitly using the existing
[dependency guide](../third_party/README.md) command. An old cache or deliberately
supplied vcpkg setting remains visible; use `-DVCPKG_MANIFEST_INSTALL=OFF` when
reconfiguring such a tree to prohibit its toolchain's install behavior. Root
selection itself never downloads, bootstraps or upgrades anything. It never
overwrites `CMakeUserPresets.json`. Remove a stale build cache or use an explicit
fresh build when switching compiler generations or installations; invalid cached
paths fail rather than silently selecting another release.

This slice does not close all of[#57](https://github.com/doublemover/RNS8/issues/57).
Existing CK XDL/WMMA probes instantiate the actual DeviceGemm, MakeArgument and
Invoker path; rocWMMA probes compile the required MMA primitive. These are
source/target compile checks, not complete backend differentials. Remaining gates
include broader target/release API compatibility, builtin signature coverage,
hipBLASLt runtime/transitive ABI checks, optional-reference platform qualification,
core-package version/lock compatibility, disposable installer qualification, and
real fresh-prefix Windows/Linux/WSL runs.
[#60](https://github.com/doublemover/RNS8/issues/60) owns upgrade qualification and the maintained version policy;
[#58](https://github.com/doublemover/RNS8/issues/58) owns Linux session rehearsal.
Filesystem fixtures and a CMake script-mode handoff test do not compile a kernel
or qualify a GPU, compiler, platform, or supported-release combination.

The [dependency version policy](dependency-versions.md) now supplies exact
candidate pins and offline consistency/observed-version gates. Filesystem
resolution still does not establish qualification; source declarations, installed
artifacts and actual runtime compatibility remain distinct.
