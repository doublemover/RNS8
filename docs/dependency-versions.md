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

[.python-version](../.python-version) selects stable [CPython 3.15.2](https://www.python.org/downloads/release/python-3152/).
The November5 Windows candidate uses a side-by-side managed NuGet extraction;
`dotnet nuget verify --all` verified its repository signature, and the package
and executable SHA256 identities are recorded in the policy. The prior3.15.1
installation/MSI exception remains in its dated observation. This is installation
evidence; the historical3.15.1 runtime passes do not qualify3.15.2. CI selects
the same interpreter version explicitly through an
immutable `setup-python` action. [Python 3.10 is end of life](https://devguide.python.org/versions/) and is not the
selected development interpreter; minimum-version support must be tested
separately from the latest interpreter lane.

Optional source-quality tools have their own hash lock under this same policy:
[quality-tools.json](../metadata/quality-tools.json) selects Ruff 0.17.4,
clang-format 23.1.5, actionlint 1.7.12 and ShellCheck 0.11.0 for Windows AMD64
and Linux x86_64. The formatter is independent of both host compiler selection
and AMDClang. Registry/release artifact identities are recorded; native archives
were retrieved and hashed without executing tools. Tool behavior and the
repository formatting baseline still require the deferred validation batch.
The same lock includes mypy 2.4.1 and its exact six-package CPython 3.15 wheel
closure. The five explicitly migrated evidence-boundary modules pass the narrow
Windows source type-check; checker controls and cross-platform gate validation
remain pending. This does not qualify the rest of the Python tree.
Optional analysis also locks clang-tidy 23.1.4 and ruamel.yaml 0.19.1, using
the safe pure-Python YAML decoder without native parser extras. Their registry
artifacts are recorded; analyzer and decoder execution remain pending. The
diagnostic adapter follows the pinned LLVM producer's native fields and
retains proposed replacements without applying them.
Use the explicit developer initializer described in
[CONTRIBUTING](../CONTRIBUTING.md#source-quality); normal quality checks never
download or install tools.

[scripts/requirements-build-tools.txt](../scripts/requirements-build-tools.txt)
pins Ninja 1.13.2 and pip 26.3 with publisher-index SHA256 values
for Linux glibc x86_64 and Windows AMD64 wheels. Those hashes are integrity
identities, not signature verification. Other architectures require a reviewed
artifact set; installation must fail rather than silently build an unpinned
source distribution. The CMake project minimum remains 3.22; it is distinct
from this selected candidate toolchain.

The November5 selection uses the official [CMake4.4.5](https://github.com/Kitware/CMake/releases/tag/v4.4.5)
Windows AMD64 ZIP and Linux x86_64 tarball while PyPI still publishes4.4.4.
`python tools/init_build_tools.py` installs the matching archive from the native
artifact lock, verifies its SHA256, and atomically publishes the complete prefix
including CTest, CPack and CMake modules. Reuse checks the receipt and hashes of
all prefix files; incomplete or modified installations fail without an overlay.
Downloads and extraction have byte, member and path limits shared with SDK
provisioning. This source implementation still awaits offline controls and
project qualification. The policy also records Git for Windows2.56.0.windows.4 and host LLVM23.1.5 candidates
separately from installed tools and AMD's matched compiler.
The standalone host LLVM candidate now has explicit Windows AMD64 and Linux
x86_64 archive locks with official release asset SHA256 and byte counts. Its
bootstrap shares bounded download, extraction, complete-prefix receipts,
verified reuse and atomic publication with the native CMake adapter. Gzip/xz
archives use the same path/link/member/expanded-byte admission. The default
host-LLVM command is a write-free plan; installation and integrity discovery
are separate explicit modes, with no compiler/runtime execution or automatic
PATH/registry/SDK changes. Setup commands belong to the
[Windows](platform-windows.md#standalone-host-llvm-candidate) and
[Linux](platform-linux.md#standalone-host-llvm-candidate) guides. Published hashes
are integrity inputs; release signatures and project compatibility remain
unqualified. The retained clang-tidy23.1.4 analysis wheel remains a separate
native-analysis selection.

Fresh quality environments install the shared hashed build-tool requirements
as well as formatter/type/analyzer locks, including selected pip26.3. The
November5 managed CPython3.15.2 and isolated quality environment are installed
candidates; only version/integrity and necessary source checks have been made.
The integrated project validation batch remains deferred.

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
python tools/init_build_tools.py
```

On Windows use `temp/dependency-tools/Scripts/python.exe`. CI creates this
job-local environment and adds its tools only to the job's PATH. CI explicitly
requests `--github-path` to add the native CMake bin directory as well. Outside
CI the initializer prints exact executable paths without changing PATH; use
those paths for direct CMake/CTest commands. `tools/init.py` selects them
automatically for a requested SDK build. Its `--dry-run` and `--no-build` modes
do not install build tools. The managed root is
`%LOCALAPPDATA%/RNS8/dependencies/build-tools` on Windows or
`${XDG_CACHE_HOME:-~/.cache}/rns8/build-tools` on Linux, overridden explicitly by
`RNS8_BUILD_TOOLS_ROOT` or the standalone installer's `--root` option.
CDNA session entrypoints remain dispatch-only; provisioning requires the
explicit installation workflow. These dependency pins do not close
[#58](https://github.com/doublemover/RNS8/issues/58).

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

The Windows Git candidate is the separately published [2.56.0.windows.4 security release](https://github.com/git-for-windows/git/releases/tag/v2.56.0.windows.4),
not merely any binary reporting upstream2.56.0. Its portable artifact hash is
recorded and checked against its exact platform patch in policy; the doctor
retains the selected artifact/source/hash. The prior `.3` candidate is recorded
separately. Installation and qualification remain pending. No successful Git
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
