# Packaged CPU continuation checkpoint - 2026-10-08

## Lineage and scope

This slice continues accepted allocation/scratch checkpoint
`d5049dffd5ebcda68ed19a78a18f4e7deb0d7e92` in
`C:/Users/sneak/Development/RNS8/temp/worktrees/row-column-range-20261008`, branch
`codex/row-column-range-20261008`. Accepted opt-in continuation parent is
`8626a548b4663f6c870d592036559792765f0b18`; retained source is
`80b9ffe1f8439ad0254918152746a77feb8035de`. No library API, ABI, arithmetic,
allocation policy or range-proof behavior changed in this slice.

Owner main remains `80791bd3d2900bd80b2ac6ba957d15329771043a`. The untracked
recovery ZIP retains SHA256
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`. Origin
identifies `https://github.com/doublemover/RNS8.git`. No publication, PR, merge,
new toolchain/dependency install or GPU execution occurred. Package installation
was local staging beneath this worktree's ignored build directories.

## Deliverables

The new self-contained C++17 program
`examples/downstream-cmake/exact_wide_cpu_continuation.cpp` uses only the installed
public C/C++ headers and a private process-local error-reporting helper. It is
also built as `rns8-example-exact-wide-cpu-continuation` by RNS8_BUILD_EXAMPLES.
It constructs signed/unsigned wide operands through public native packing and
ordinary GEMM, rather than injecting private matrix proofs or raw CRT residues.

Both semantics verify ordinary wider-product rejection, two consecutive
one-byte-short opt-in budget rejections, preserved selected-prefix output and
public input/output integer/storage/version/currentness state, exact-budget
success, unchanged input versions/values, rejected original-prefix export with
destination preservation, a matching FORCE_FIXED_PREFIX export plan, full
three-limb results and subsequent input-plane reuse. The source compares against
analytically derived power-of-two limb constants, including a negative signed
input, without Boost or reconstruction helpers.

Observed deterministic fixture results with the current default ladder:

| Semantics | Original floor | Returned prefix | Exact initial payload | Rejected payload | Reuse payload | Product limbs, low to high |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| signed | 17 | 19 | 23 | 22 | 19 | `0, 0, 0x40000` |
| unsigned | 17 | 20 | 26 | 25 | 20 | `0x100000, 0xffffffffffe00000, 0xfffff` |

Payload sizes are bytes for distinct scalar operands. This payload limit still
excludes metadata, allocator overhead, CRT scratch and row scratch. The shipped
guide explains general checked sizing, using the returned prefix for export,
fixed-width signed limb representation and compiler/runtime configuration.

The Development component now ships five standalone consumer files under
`${CMAKE_INSTALL_DATADIR}/RNS8/examples/downstream-cmake` (default share):
CMakeLists.txt, README.md, bounded smoke main.cpp, the continuation source and
noninteractive_errors.hpp. The consumer defaults to rns8::rns8_static;
RNS8_DOWNSTREAM_USE_SHARED=ON selects the available rns8::rns8 target and copies
the imported DLL next to its Windows executables. Missing shared targets fail
at configuration. Both programs use strict compiler warnings and 10-second
CTest limits. Windows runtime errors/assertions use captured stderr with
process-local settings; assertions and nonzero failures are retained.

The install smoke copies sources from the staged package into a separate source
directory, disables CMake package registries, and configures fresh static and
available shared consumer build directories without a vcpkg toolchain. It
records the selected package directory and verbose compile/link commands.
Install/configure/build/test phases have 30-second limits, builds use two jobs,
consumer tests have 10-second limits and the outer smoke has a 120-second limit.
README and current correctness/roadmap docs link the shipped workflow and keep
qualification boundaries explicit.

## Focused evidence

Installed MSVC 19.44.35228.0 built the in-tree example in the existing Debug and
RelWithDebInfo configurations. Installed clang-cl 22.1.6 built the existing ASan
configuration. Each in-tree example run passed its signed and unsigned workflow:
three passing CTest example runs. ASAN_OPTIONS=detect_leaks=0 was used, with no
unexpected sanitizer diagnostic. The prior full CPU/fault campaigns were not
repeated; library implementation files were unchanged.

Fresh clean consumers of the staged package passed six executable tests:

| Package/compiler configuration | Linkage | Bounded smoke | Signed/unsigned continuation |
| --- | --- | --- | --- |
| MSVC Debug | static | PASS, 111 | PASS |
| MSVC Debug | DLL | PASS, 111 | PASS |
| MSVC RelWithDebInfo | static | PASS, 111 | PASS |

The Debug aggregate therefore passed 4/4 downstream tests; the Release aggregate
passed 2/2. Verbose commands show only staged package public include paths and
staged rns8_static.lib/rns8.lib imports. Consumer caches identify their copied
source directory and the staged package config, with no CMAKE_TOOLCHAIN_FILE.
Five installed source/guide files matched the checked-in files byte-for-byte in
both packages; four compiled copied source files matched in both consumer trees.
The guide was added after runtime checks and its install/byte identity was
verified directly. The source manifest records exact SHA256 values.

Representative bounded commands:

```powershell
python tools/windows_dev.py cmake --build build/row-column-cpu --target rns8-example-exact-wide-cpu-continuation rns8 rns8_static --parallel 2
ctest --test-dir build/row-column-cpu -R '^example_exact_wide_cpu_continuation$' --verbose
python tools/windows_dev.py ctest --test-dir build/row-column-cpu -R '^install_downstream_cmake_smoke$' --verbose
python tools/windows_dev.py cmake --build build/row-column-release --target rns8-example-exact-wide-cpu-continuation rns8_static --parallel 2
python tools/windows_dev.py ctest --test-dir build/row-column-release -R '^install_downstream_cmake_smoke$' --verbose
$env:ASAN_OPTIONS = 'detect_leaks=0'
ctest --test-dir build/row-column-asan -R '^example_exact_wide_cpu_continuation$' --verbose
```

## Failures, cleanup and remaining gates

No compile, link, runtime, sanitizer or installed-consumer failure occurred.
Expected budget/range statuses are explicit asserted checks. A tools-only patch
formatting attempt was rejected before modifying files. A documentation helper
then encountered README's UTF-8 bytes under Windows' default cp1252 decoding;
explicit UTF-8 reading/writing fixed it. These were not product failures. The
helper's partially added guide/install entries were reconciled without duplicates.
No dialog or debugger was opened.

Temporary consumer build outputs (executables, objects, PDBs and copied DLLs) were
removed using native PowerShell after verifying each exact generated directory
resolved inside this worktree and was not a reparse point. Copied sources, local
staged packages, project example/library builds and raw evidence remain. No
consumer/example process was left running. No other workload was stopped.

The owner allowed ordinary Blendslop work to coexist while these bounded scalar
checks/two-file consumer builds ran. Test elapsed times are not performance
measurements. The previous row-buffer allocation measurements remain scoped to
the measured MSVC Debug profiles in the allocation receipt. Debug STL OOM remains
unqualified and its fault sweep remains explicitly skipped; this slice made no
OOM claim and injected no allocation failure.

This closes the requested packaged CPU usability/install-consumer batch. It
qualifies only the above CPU Windows/MSVC configurations and the in-tree ASan
example. Release DLL, pure C installed consumers, Linux, other compilers/CRT
combinations, OpenMP, ASan package consumption, UBSan/LeakSanitizer and hardware
execution remain broader gates. There is no new API expansion. Integration or
publication requires owner authorization and review of the entire inherited
main-to-continuation lineage; no latest-commit-only route was substituted.

## Raw evidence

Ignored worktree temp files:

| Evidence | SHA256 |
| --- | --- |
| `continuation-example-msvc-configure.log` | `5b6ad360a04a36b9fdd33d028f9b159e54044ea05e18284c55c0d0e71275c40a` |
| `continuation-example-msvc-build.log` | `3b29410ab62fe3e2e0fb98cc532b359fbe164491eb5c89d267a19bce8282c307` |
| `continuation-example-msvc.log` | `98c3355522383fb40cc5aa0ff6013a860d4849c85f0d1e29be319d347f440fa4` |
| `continuation-example-release-build.log` | `3b29410ab62fe3e2e0fb98cc532b359fbe164491eb5c89d267a19bce8282c307` |
| `continuation-example-release.log` | `2b9aeb4f1e7719234ee6dcc63c1b6ce4055459cc10ab76207834000a2815fb24` |
| `continuation-example-asan-build.log` | `3b29410ab62fe3e2e0fb98cc532b359fbe164491eb5c89d267a19bce8282c307` |
| `continuation-example-asan.log` | `3d7296de4ee6b01d270deedb89973cb364787ea4cc8384e495f471b9900221bb` |
| `continuation-package-msvc-final-build.log` | `bae11df3d81630c9e82daebf621d04ae8c175e57c3da8ca147862352bd56e4f4` |
| `continuation-package-msvc-final.log` | `6287fdc83d6c2ab672aa163db213aeb84d01f9b4782223eb16958c310073590d` |
| `continuation-package-release-build.log` | `f8dc5f8f51b4834d456523bc4471c43090b23428f05f226e55d370c5365f0d1b` |
| `continuation-package-release.log` | `68cd9cfb5b206e23907bd85d193f94f2b6c914dfc0ac4b0e328c5de50223cedf` |
| `continuation-package-documentation-msvc-install.log` | `13347ce5c3d76aeda7096555742a052a26ea380d674c1e8623a41e1784704e64` |
| `continuation-package-documentation-release-install.log` | `45971152827ccd361be28b5b22cb0a1a2585bee240212d9cc814af001abb0cde` |
| `continuation-package-manifest.json` | `350d99a9141767bf1900793287a91facec036bc24547c74a481cc97c422319c6` |
| `continuation-package-manifest-check.log` | `b71ffea81eff945558daa4dc5bd20eaa3b456a8c5a37de60f4285693f8cd00aa` |
| `continuation-package-cleanup.log` | `56b61071200625dcc8e5a530f2d0a59e22144fe15c28f8026a390f2c1d570f7e` |
