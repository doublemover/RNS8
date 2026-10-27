# Contributing

RNS8 needs exact arithmetic, inspectable kernels and reproducible measurements.
Use [Windows setup](docs/platform-windows.md) or [Linux setup](docs/platform-linux.md)
for GPU development. [Dependencies](third_party/README.md) owns requirements/pins;
this guide owns developer checks. [AGENTS.md](AGENTS.md) adds repository work rules.

## Make a focused change

Check `git status --short` first. State the arithmetic contract and ownership
behavior being changed. Add CPU reference coverage before GPU performance work
counts; kernel changes need CPU differentials plus a minimal GPU correctness
case. Use Boost.Multiprecision for exact-wide and CRT references. GMP/FLINT are
optional comparisons. Public API or emitted-schema changes include docs and
regressions in the same slice.

Cover boundaries before random inputs: signed/unsigned 64-bit extremes,
cancellation and worst-case partial sums, composite/even moduli and primes,
padding/aliases and K splits around 65536. Keep deterministic exactness as the
default. Optional accelerators must report unsupported cases explicitly.
Scratch sources, binaries, raw captures, downloaded references and ISA dumps
belong in ignored `temp/`, `build/` or `out/` paths.

## CPU oracle development

A GPU is unnecessary for portable reference, API and tooling work. On Windows,
set `VCPKG_ROOT` to an installation containing the manifest packages, then use
the VS wrapper from an ordinary PowerShell:

```powershell
$env:VCPKG_ROOT = 'D:/Development Tools/vcpkg'
python tools/windows_dev.py cmake --preset cpu-debug "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
python tools/windows_dev.py cmake --build --preset cpu-debug --parallel 4
python tools/windows_dev.py ctest --preset cpu-debug --output-on-failure
.\build\cpu-debug\rns8-verify.exe
```

On native Linux, use installed Boost, JSON and Catch2 3.x development packages:

```sh
cmake --preset linux-cpu-debug
cmake --build --preset linux-cpu-debug --parallel 4
ctest --preset linux-cpu-debug --output-on-failure
./build/linux-cpu-debug/rns8-verify
```

For a library-only build, `linux-cpu-release` with `-DRNS8_BUILD_TESTS=OFF`
omits Catch2-dependent tests. OpenMP is optional; enable/disable it with
`RNS8_ENABLE_CPU_PARALLEL`, and use `RNS8_REQUIRE_OPENMP=ON` when validating that
specific path. Record `OMP_NUM_THREADS` with results.

## Focused tooling and documentation checks

Run checks relevant to the change. Common tooling checks are:

```sh
python tools/test_check_dependencies.py
python tools/test_benchmark_schema.py
python tools/test_result_compare.py
python tools/test_benchmark_sweep.py
python tools/check_release_tree.py
```

Documentation has its own bounded checks:

```sh
python tools/test_documentation.py
python tools/check_documentation.py
python tools/check_documentation.py --external --official-only
```

The default check validates the exhaustive inventory, internal paths/anchors,
HTML links, templates and referenced preset/target/script names. External checks
are opt-in; timeouts, 429 responses and skipped checks remain unresolved results.
See [documentation maintenance](docs/documentation-maintenance.md) for scope,
command evidence and migration rules. Documentation-only changes do not need an
unrelated CMake build or GPU campaign.

## Sanitizers

Linux Clang/GCC CPU hardening uses:

```sh
cmake --preset cpu-asan-ubsan-debug
cmake --build --preset cpu-asan-ubsan --parallel 4
ctest --preset cpu-asan-ubsan --output-on-failure
```

Windows LLVM CPU hardening uses `windows-clang-asan-debug`, with
`LLVM_ROOT` pointing to the selected LLVM installation, and the
`windows-clang-asan` build/test preset. It links the static runtime and enables
fuzzers. The [dated local record](docs/local-all-features-20261023.md) scopes the
executed sanitizer evidence; a preset is an available configuration.

## GPU correctness and measurements

Build the selected GPU/backend as described in the platform guide, then run its
matching CTest preset and `rns8-verify --hip-smoke` from that build directory.
[GPU qualification](docs/gpu-qualification.md) enumerates target-specific cases
and ISA checks. Stop on the first arithmetic mismatch, out-of-bounds access,
device fault, wrong-target error or unexplained skip. Preserve the exact build
and logs before diagnosis.

[Performance methodology](docs/performance.md) defines matched contracts,
phase boundaries, independent processes and promotion. Record full commands,
seed, shape, bounds, prefixes, backend/kernel, warmups/repeats, timing source,
compiler/runtime and artifact identity. Include packing/reconstruction and setup
costs at the claimed boundary. Reviewed dated reports retain results; raw records
stay ignored. A measured regression is useful evidence too.

Keep full machine reports in durable artifacts with a checked-in source/hash
manifest; dated repository records summarize reviewed conclusions and point to
the complete evidence. See [research artifacts and case catalogs](docs/performance.md#research-evidence-and-benchmark-inputs).
Before invoking collectors directly, expand their inputs with
`python tools/benchmark_cases.py --output temp/benchmark-cases/current` and use
the resulting family paths. CMake/CTest expands its input corpus automatically.

## Install and test a consumer

Use an absolute installation prefix. After building the selected configuration,
Linux's CPU example is:

```sh
cmake --install build/linux-cpu-debug --prefix "$PWD/temp/install-rns8"
cmake -S "$PWD/temp/install-rns8/share/RNS8/examples/downstream-cmake" -B temp/downstream-rns8 -G Ninja "-DCMAKE_PREFIX_PATH=$PWD/temp/install-rns8" -DCMAKE_BUILD_TYPE=Debug
cmake --build temp/downstream-rns8 --parallel 2
ctest --test-dir temp/downstream-rns8 --output-on-failure
```

On Windows use `build/cpu-debug`, a matching Debug consumer, and the VS wrapper
or a developer shell. Match the installed MSVC runtime. The installed
[consumer guide](examples/downstream-cmake/README.md) covers static/shared targets,
DLL placement and exact-wide continuation using public headers only.

## Pull requests and compute offers

Describe the final behavior, exact validation commands/results and unavailable
hardware/dependencies. Keep commits focused; their bodies explain material
choices, checks and deferred work. Run `git diff --check` and inspect status
before committing. Keep generated captures out of durable docs unless reviewed.

Use the [hardware-offer form](https://github.com/doublemover/RNS8/issues/new?template=hardware_validation_offer.yml)
for volunteer checks, temporary compute or credits. Never post passwords, keys,
tokens, credentials, private endpoints or billing data. Maintainers should agree
on a bounded plan, duration/storage estimate, commands, sanitization and artifact
list before any run. The complete remote-session bundle is still tracked by
[#58](https://github.com/doublemover/RNS8/issues/58); submitting an offer grants no
provisioning, access or spending authorization. Financial support belongs on
[GitHub Sponsors](https://github.com/sponsors/doublemover).
