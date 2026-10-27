# Dependency versions and upgrade gates

[metadata/dependencies.json](../metadata/dependencies.json) is the dated version
policy. It distinguishes the latest stable release checked against primary
sources, the exact selected candidate, observations from a particular machine,
and qualification evidence. The current selection is a candidate awaiting
qualification. A version match or successful package install does not qualify
RNS8 correctness, ABI compatibility, GPU execution or performance.

## Selected sources and artifacts

The Windows manifest selects Boost.Multiprecision 1.92.0, Catch2 3.16.1,
nlohmann-json 3.12.0, optional GMP 6.3.0, FLINT 3.6.0 and MPFR 4.2.2 through
the immutable vcpkg baseline and explicit port revisions in
[vcpkg.json](../vcpkg.json). The policy retains the conservative declared closure
of 41 ports, including host dependencies, platform conditions, licenses,
manifest/recipe SHA256 values and source SHA512 values from the registry.
This declaration is separate from the actual installed feature/triplet/ABI
receipt. Conditional ports are not all required on every platform.

The baseline comes from the [official vcpkg registry](https://github.com/microsoft/vcpkg/tree/b42e8ed051d17a7f95b9d2a3140c1f9296423712).
The changed upstream releases are [Boost 1.92.0](https://www.boost.org/releases/1.92.0/),
[Catch2 3.16.1](https://github.com/catchorg/Catch2/releases/tag/v3.16.1) and
[FLINT 3.6.0](https://github.com/flintlib/flint/releases/tag/v3.6.0).
FLINT's selected source requires GMP at least 6.2.1 and MPFR at least 4.1.0;
Windows ARM64/UWP are excluded by the selected vcpkg port. Linux uses native
packages or native prefixes; this lock does not introduce Linux vcpkg.

[.python-version](../.python-version) selects stable CPython 3.15.1. The local
Windows installation uses the official CPython NuGet package after the official
MSI failed with Windows Installer errors 2502/2503. Its package SHA256 and the
parent's successful NuGet repository-signature verification are recorded in the
policy. This is installation evidence; Python tool qualification remains a
separate gate. CI selects the same interpreter version explicitly through an
immutable `setup-python` action. [Python 3.10 is end of life](https://devguide.python.org/versions/) and is not the
selected development interpreter; minimum-version support must be tested
separately from the latest interpreter lane.

[scripts/requirements-build-tools.txt](../scripts/requirements-build-tools.txt)
pins CMake 4.4.4, Ninja 1.13.2 and pip 26.2.2 with publisher-index SHA256 values
for Linux glibc x86_64 and Windows AMD64 wheels. Those hashes are integrity
identities, not signature verification. Other architectures require a reviewed
artifact set; installation must fail rather than silently build an unpinned
source distribution. The CMake project minimum remains 3.22; it is distinct
from this selected candidate toolchain.

The user authorized ROCm Core SDK10.1 migration on 2026-10-26. Its extracted
local candidate supplies HIP7.16.26385, AMD Clang24, CK1.3.0 and rocWMMA2.2.1.
Fresh candidate builds explicitly select those installed SDK headers and
compiler paths through shared discovery. Keep the exact HIP7.1-era CK1.1.0 and
rocWMMA2.1.0 source pins as the prior source fallback in the
[dependency guide](../third_party/README.md#optional-exact-references-and-accelerator-pins).
The newer bundle's monorepo commit remains recorded separately from those
unchanged submodule pins. SDK qualification does not silently move source pins.
Generated CK version metadata now comes from the selected source or installed
header instead of a hardcoded version. Matching AMD compiler/runtime/driver and
target qualification remain required for each candidate lane; stock upstream LLVM
is not a replacement for AMD's HIP compiler.

The [October 26 Windows/gfx1100 Release record](platform-windows.md#october-26-core-sdk-101-local-release-validation)
now qualifies the applicable local registered suite with this exact SDK and
manifest stack: 1,080 passes, 30 feature/target skips and zero failures, plus
four installed native-device consumer checks. Python3.15.1 runs the registered
tooling checks. Runtime module hashes, optional GMP/FLINT prefix consistency,
compiler identities and compiled-source hashes are recorded. This local result
does not promote the full platform/configuration matrix or replace matched
performance qualification; retained source fallback pins remain unchanged.

The [configuration follow-up](platform-windows.md#october-26-configuration-follow-up)
also passes HIP Debug, HIP Release with OpenMP OFF and CPU LLVM23.1.4 ASan
RelWithDebInfo with `/MD`, static/shared packages and installed consumers.
The supported final lanes total 2,480 passes and 96 explicit skips; the eight
primary Release checks are artifact integration checks, not a second complete
Release arithmetic run. LLVM rejects Debug-CRT `/MDd` ASan; its attempted build
is preserved separately. [Bounded SDK10.1 measurements](performance.md#october-26-sdk101-bounded-measurements)
compare approaches within the new stack. They do not establish a matched
prior-SDK versus new-SDK regression result or global platform qualification.

NumPy, pandas, Matplotlib, pytest, SciPy, NTL, FFLAS-FFPACK and LinBox remain
unused optional inventories. PyYAML only enables one optional documentation
schema test. Their latest releases are recorded without selecting or installing
them as core dependencies. Actions' Node 24 runtime belongs to those actions;
RNS8 has no project Node package set.

## Inspect and enforce the selection

Check repository consistency without network, package installation or probes:

```sh
python tools/check_dependencies.py --check-lock --json
```

Capture observations through the existing doctor. A configured cache identifies
the actual vcpkg installation used by that build, rather than an unrelated
classic installation:

```sh
python tools/check_dependencies.py --json --cmake-cache build/issues-all/CMakeCache.txt
python tools/check_dependencies.py --require-locked --json --cmake-cache build/issues-all/CMakeCache.txt
```

The explicit `--require-locked` gate fails when selected core/build/test or
enabled optional-reference versions are missing or differ, including vcpkg port
revisions. It is expected to fail on an old environment. Native Linux package
and image locks, compiler/STL/CRT/OpenMP ABI receipts, and loaded-library
identities are still pending; unknown values do not become matching versions.
Ordinary doctor output records version drift without relabeling installed paths
as qualified.

Install build tools only into a deliberately selected environment. For example,
after selecting the desired interpreter:

```sh
python -m venv temp/dependency-tools
temp/dependency-tools/bin/python -m pip install --require-hashes --only-binary=:all: --no-deps -r scripts/requirements-build-tools.txt
```

On Windows use `temp/dependency-tools/Scripts/python.exe`. CI creates this
job-local environment and adds its tools only to the job's PATH. The existing
CDNA fallback uses the same exact requirements rather than a moving pip/CMake
range. Its Catch2 fallback verifies the locked source commit and refuses a
modified checkout. CDNA session safety, explicit device mapping, offline staging
and installation opt-in remain [#58](https://github.com/doublemover/RNS8/issues/58)
work; these dependency pins do not close that issue.

Windows C++ packages are installed explicitly through the selected vcpkg
toolchain. Configure defaults continue to disable manifest installation. Use a
fresh build/install prefix for the candidate stack and select its GMP/FLINT
roots explicitly if an older cache contains classic-install overrides. Preserve
the old prefix and source revision when reproducing an earlier measurement.

## Qualification and maintenance

The policy keeps the prior vcpkg baseline and the recorded 1bb5e33 validation
separate from this candidate. Do not attach those prior results to upgraded
dependencies. Existing build manifests bind the policy, wheel requirements,
selected dependency resolution, installed vcpkg status, selected Boost/json and
enabled Catch2 header subtrees, and enabled accelerator include contents.
A dependency change invalidates prior binary/cache evidence
where its identity differs; it does not automatically update a reviewed cache.

Required gates include clean Windows and Linux Debug/optimized and static/shared
builds, OpenMP ON/OFF, exact arithmetic and optional-reference checks, installed
consumers, Python tooling, and actual target-specific HIP correctness/ISA/resource
checks. Matched independent measurements must assess regressions before a
performance/default promotion. No new GPU or Linux qualification is claimed by
the lock files.

The Windows Git candidate is the separately published [2.56.0.windows.3 security release](https://github.com/git-for-windows/git/releases/tag/v2.56.0.windows.3),
not merely any binary reporting upstream2.56.0. Its portable artifact hash is
recorded; installation and qualification remain pending. No successful Git
update is claimed.

The workflows pin checkout 7.0.1, setup-python 7.1.0 and upload-artifact 7.0.2 to
full commit SHAs. Their Node 24 runtime requires Actions Runner 2.327.1 or newer.
The self-hosted performance job requires the `node24` label: an operator must
verify the runner before assigning that label. Hosted and self-hosted workflow
execution with the new actions remains a separate validation result.

Review official stable releases monthly and before a release or GPU-session
lock refresh. Update the policy, exact manifests/hashes, bootstrap inputs and
workflow pins together; run the offline consistency check and then the applicable
compatibility gates. A failed candidate needs a concrete exception and revisit
condition. There is no automatic install, merge, driver change or backend/cache
promotion.


Explicit GMP/FLINT enablement now performs a compile/link API probe against the
selected imported targets, including when project tests are disabled. The probe
uses the same small signed multiprecision arithmetic program as the runtime smoke;
configure never executes it. Compiler/link diagnostics are retained in
`exact-reference-link-probe.log` under the build directory. The selected-prefix
compile/link probes and runtime arithmetic checks pass in the local SDK10.1
Release record. Broader OS/compiler/configuration and matched-performance
qualification remain required for the upgraded stack.
