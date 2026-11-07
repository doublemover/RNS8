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
regressions in the same slice. Review public C changes with the
[ABI inventory and installed-consumer checks](docs/public-c-abi.md).
Target qualification follows the [candidate-bound phase receipt policy](docs/target-qualification.md);
platform history alone cannot authorize a current cache entry.

Cover boundaries before random inputs: signed/unsigned 64-bit extremes,
cancellation and worst-case partial sums, composite/even moduli and primes,
padding/aliases and K splits around 65536. Keep deterministic exactness as the
default. Optional accelerators must report unsupported cases explicitly.
Scratch sources, binaries, raw captures, downloaded references and ISA dumps
belong in ignored `temp/`, `build/` or `out/` paths.

## Source quality

`python tools/quality.py inventory` prints the tracked first-party C/C++/HIP,
Python, Bash and workflow inventory. Generated outputs, imported dependencies
and immutable certificate data have explicit exclusions in
`metadata/first-party-inventory.json`; generator sources remain included.

Install the optional, separately locked developer environment explicitly:

```shell
python tools/init_quality.py --out temp/quality-tools
```

Then run `python tools/quality.py check --tool-bin temp/quality-tools/Scripts`
on Windows, or use `temp/quality-tools/bin` on Linux. Checking performs no
downloads, compiler configuration or workloads. It reports the selected tool
versions, executable hashes, effective configuration hashes and argv. The
explicit `fix` mode changes C++/HIP and Python formatting only; lint remains
check-only, with no automatic import deletion or unsafe fixes. Include sorting
is disabled to preserve semantic fragment order. The Ruff syntax target is a
conservative Python 3.12 floor, not a claim of interpreter qualification.

The shared dependency policy owns exact versions and Windows/Linux artifact
hashes in `metadata/quality-tools.json`. These are optional developer tools,
separate from AMD's compiler and runtime dependencies. The canonical baseline is
recorded in `.git-blame-ignore-revs`; subsequent changes use the same check-only
gate. CI checks the inventory on Windows and Linux before native builds, retaining
the existing functional suites. The `self-test` mode takes the same `--tool-bin`
directory and runs tiny checker controls in ignored scratch space:
formatting drift, an undefined Python name, an unsafe shell expansion and an
invalid workflow expression must fail. It never compiles or runs fixture workloads.
The source gates and controls still need their final qualification batch and
cold/warm wall-time and memory measurements. Schema and sweep contract modules now
run independently through ordinary unittest classes and fresh fixture inputs;
their aggregate commands retain all mapped cases without shared fragment globals.

Production benchmark-schema modules also use explicit imports from the owning
predicate/metadata modules. They do not inherit a wildcard namespace or forward
unrelated helpers through the validator class module. Keep these imports explicit
when adding validation contracts; the structure suite checks fresh-process imports.
Semantic dispatch uses a required, slotted field record. Raw capture fields stay
`object` until their validators admit them; constructing the record is not admission.
Sweep command controls assert visibility masks, unique shard destinations and
metadata-to-argv agreement; a tiny authored catalog checks that removing masks
or reuse flags, or aliasing outputs, is rejected without launching a workload.

The same environment also locks mypy 2.4.1 and its complete CPython 3.15 wheel
closure. `python tools/quality.py types --tool-bin temp/quality-tools/Scripts`
(Linux: `bin`) checks the five strict modules listed in
`metadata/python-typecheck-scope.json` and requires deliberate wrong-command,
missing-artifact-key and unresolved-name inputs to produce their expected
diagnostics. Those three named inputs are excluded only from passing lint and
format inventories. The gate follows imports normally, reports sources outside
its explicit strict scope, and does not claim that the entire Python tree is
typed. The Python 3.12 source target remains separate from the selected 3.15
developer interpreter. Negative controls and integrated gate execution remain
part of final validation.

## CPU oracle development

First-party library, tool, test and benchmark targets share private compiler
warnings. `-DRNS8_WARNINGS_AS_ERRORS=ON` opts into `/WX` or `-Werror`; those
options are not exported to installed consumers or applied to vendor targets.
HIP custom compilations write `rns8-hip-compile-commands.json` from the same
argv list passed to the compiler, including selected environment overrides,
targets and required generated inputs. Per-configuration files use the suffix
`-<CONFIG>.json`; generated-header expressions are resolved during generation.
This is configured-command inventory, not proof that the compiler or an
analyzer ran.
The `windows-analysis-cpu` and `linux-analysis-cpu` presets export non-unity
host commands into separate build directories. Their generated target/source
inventory lets the reader reject missing or duplicated static/shared/tool
contexts rather than treating a shortened database as full coverage. Native
`command` strings are retained unchanged; they are not reparsed with a foreign
shell's quoting rules. `rns8_analysis_inputs` prepares real generated inputs
without depending on HIP object compilation.

`python tools/quality.py analyze --build-dir build/analysis-cpu --out-dir
temp/analysis-session --tool-bin temp/quality-tools/Scripts` prints a plan
without executing a compiler or analyzer. Use the actual selected preset build
directory and a fresh output directory. On Linux the tool directory is
`temp/quality-tools/bin`. Add `--execute --prepare` to prepare required generated
inputs and analyze the configured contexts with the locked native clang-tidy.
`--clang-tidy` can explicitly select the native executable instead of locating
it inside the initialized environment; the runner still requires the pinned
version and diagnostic decoder.

The default `--lane host` analyzes each actual CXX context separately.
`--lane hip-analyzer` projects the configured HIP command into a host pass and
one device pass per configured architecture. `--lane hip-syntax` instead runs
the configured HIP compiler with `-fsyntax-only`; this never counts as analyzer
coverage. Unsupported flags, missing inputs, failed parsing, timeouts and absent
branch diagnostics remain incomplete coverage. No target substitution or fake
SDK header is used.

`--changed-paths-json '["src/core/api.cpp"]'` can select changed translation
units while retaining every configured ABI, preprocessor and GPU target context
for those files. A header, fragment, configuration, unknown, missing or deleted
input conservatively selects the full configured lane; empty inventories also
fall back to the full lane. The bounded path list, fallback reasons and selected
versus configured pass counts are retained and recomputed during receipt replay.
Unselected contexts are not claimed as analyzed, and their dispositions are not
retired by an incremental run. Changed or expired policy evidence still requires
review. Gated PR analysis uses the actual Git change inventory through a quoted
environment value; oversized inventories, scheduled runs and manual runs select
the full lane. Periodic full-lane analysis remains necessary.

Each analyzer pass force-includes `tests/analysis/branch_control.hpp`, which
contains an intentional null dereference in its host or HIP kernel branch.
It is never executed or linked into production. Exactly one expected native
diagnostic must be present before a pass receives analysis credit; other
diagnostics remain findings. A separate missing-header control must fail with
its native parsing diagnostic. `--advisory` permits findings at the CLI exit
boundary, but cannot accept incomplete analysis.

The runner retains the plan, original/projected commands, allowlisted compiler
environment, observed first-party include footprint, native YAML, process logs,
partial interrupted output and compact pass receipts. Per-process/session time,
output, artifact and free-space budgets are bounded; unavailable peak-memory
observations remain explicit. `python tools/quality.py analyze
--verify-receipt temp/analysis-session/results.json` reopens the current configured
compiler inventory, exact per-pass databases/commands and native YAML, then
reconstructs findings, include footprints, branch visibility, missing-header
rejection and coverage counts. Re-sealing derived summaries cannot substitute
for those native observations. This is retained-file custody, not cryptographic
execution attestation, and it launches no analyzer or compiler. External dependency trees are not
fully rehashed, and static analysis does not qualify GPU execution or performance.
The orchestration and controls are authored; their deferred batch execution,
real platform pilots, reviewed diagnostic dispositions and CI activation remain open
under #99.

`tools/analysis_dispositions.py --receipt temp/analysis-session/results.json
--output temp/analysis-dispositions.json` applies the reviewed
`metadata/analysis-dispositions.json` policy after native receipt reconstruction.
The initial policy has no waivers. Findings require an exact check, compiler
context, host/device branch, target, message and unique source-byte anchor;
line-number churn outside that anchor does not erase a finding. Every allowed
intentional-arithmetic or tool/third-party false-positive disposition needs an
owner, reason, digest-bound reduced control or source-invariant evidence, and a
revisit date. Tool/config changes, ambiguous/changed anchors, expired evidence,
new findings and stale entries require review. Parse failures, errors, unknown
locations and external-only diagnostics cannot be waived through a tree glob.
Reachable defects need repairs rather than baseline entries. The gate neither
applies replacements nor generates an accepted baseline automatically.

Strict mode accepts only complete native analysis with no new findings and
valid narrow dispositions. `--advisory` retains unclassified findings explicitly;
it cannot accept incomplete/syntax-only coverage, stale entries or invalid
evidence. The `Native CPU Analysis Candidate` workflow uses shared locked
tools, the non-unity CPU preset, native sensitivity controls and bounded serial
passes, preserving raw failures and gate receipts. Its first native pilot,
finding cleanup and measured cost must establish required-check activation.
`RNS8_NATIVE_ANALYSIS_CI=enabled` opts into affected same-repository PRs and the
weekly native CPU analysis job; automatic runs use the strict disposition gate.
The shared changed-tree classifier skips expensive checks for documentation-only
edits without hiding deleted code through rename detection. Workflow invocation
and schedule hooks are authored; no repository variables are set by source
development and no native pilot is implied. Windows/HIP configurations keep
their separate pilot gates.

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

## Optional scalar proofs

The optional [scalar arithmetic proof pilot](docs/scalar-arithmetic-proofs.md)
keeps source-bound CBMC properties and independent native counterexample replay
separate from ordinary correctness and sanitizer tests. Its contract controls
are registered as `scalar_proof_contract_self_test`; they do not run a solver.
Real proof acquisition requires explicit `--execute` and compatible selected
tools. The current source workflow has not been executed.

## Sanitizers

The native Linux Clang CPU hardening preset uses:

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

The [host instrumentation source workflow](docs/host-instrumentation.md)
describes the separate Linux ASan/UBSan, TSan and branch-coverage presets,
disposable detector controls, legal concurrency corpus and production-timing
exclusion. Detector drivers, reports and CI acceptance remain source work;
none of these newer lanes has executed during the frontload pass.

The [fuzzing contract](docs/fuzzing.md) describes the separate instrumented
library, versioned exact-arithmetic input format and ignored corpus expansion.
New discovery/replay controls remain part of the deferred validation batch.

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

Installed HIP package checks include a compiler-free SDK-package policy lane.
Windows static consumers use the package's matching HIP import library and DLL;
install both Development and Runtime components. Linux consumers supply their
current AMD HIP package through `RNS8_HIP_ROOT`, `hip_DIR` or CMake prefixes.
These configure fixtures verify dependency naming, relocation and rejection;
the existing installed static/shared workload lanes still own real link/runtime
qualification. Neither fixture generation nor package discovery qualifies GPU code.

Numerical report gates use tools/reference_verdict.py; review-sidecar consumers
use tools/review_reference.py. A CPU backend name or matching checksum alone
does not establish a full output comparison. Preserve explicit mismatches and
pending coverage, bind decisions to capture contents and mathematical inputs,
and keep research promotion restrictions separate from numerical completion.
Current schema3 sidecars must resolve their candidate and CPU-reference captures;
stale, conflicting or legacy sidecars require a fresh review. Synthetic protocol
fixtures exercise these rules and are never hardware qualification evidence.
The standalone scheduler, CPU selector, FHE proxy, vector-chain, tile-shape,
incremental-cache and layout-search reports use the same numerical contract.
CPU anchors are indexed by mathematical/input identity, independently of timing
selection; conflicting observations remain blockers. Pending policy statuses
remain incomplete even when other metadata is valid. The adapter regression
suite is tools/test_report_numerical_gates.py.
Production/diagnostic binding v2 uses the reviewed observation projection in
tools/companion_contract.py. Execution timings may differ between processes;
requested budgets, repeat counts, reuse policies and unknown fields remain
contractual. Complete original file/content hashes still make each receipt
immutable. Older binding receipts must be regenerated from their original
captures before they can qualify under the current projection.
