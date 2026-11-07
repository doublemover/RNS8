# Native production mutations

The authored `tools/native_mutation_session.py` lane builds real CPU library/test
variants for six exact-source seeds in `metadata/native-production-mutations.json`:
checked unsigned addition, signed CRT correction, published value generations
and retry ownership after failed handle close, signed export rejection status
and stale prepared-input generation admission. The shared native arithmetic
replay is exercised through actual public CPU calls for wrong-result,
wrong-status and stale-state sensitivity. These witnesses mutate production
operations in disposable source copies; their assertions receive no mutant flag.
The checked-add mutant also faces a real integer-split public query whose
element product fits but whose final logical extent overflows. Its assertion
requires both range rejection and an unchanged resource record before a valid
control query. Native coverage uses the same seed/test registry and a separate
reviewed public-route map; it cannot turn branch reachability into a mutant kill.
The ownership case instantiates
the maintained production template with fake host operations; it does not call
a HIP driver or qualify a device failure. No source snapshot, compiler, build,
Catch discovery/test, baseline or mutant has run in this source pass.

```sh
python tools/native_mutation_session.py --cmake /path/to/cmake \
  --ctest /path/to/ctest --compiler /path/to/cxx --ninja /path/to/ninja \
  --python /path/to/selected/python --out temp/native-mutation-candidate
```

Windows also requires the explicit vcpkg `--toolchain` and the VS developer
environment. The installed vcpkg directory defaults to this checkout's existing
`vcpkg_installed`; `--vcpkg-installed` can select its explicit replacement. No
dependency installation is attempted. Planning performs source/operation checks
and prints the commands without creating snapshots or launching subprocesses.

`--execute` first requires the actual clean committed source worktree. It copies
bounded tracked regular files into fresh owned source trees; Git metadata,
external gitlinks, SDKs and old build objects are excluded. One unchanged CPU
baseline precedes six separate fresh mutant builds of `rns8_tests`. HIP,
accelerators, transports, OpenMP, optional CPU instructions, sanitizers, coverage,
shared libraries, tools and benchmarks are explicitly disabled. Every source
copy and test input is content-bound; mutations preserve unaffected file bytes
and newline style. The original checkout is unchanged. No automatic retry or
artifact deletion occurs.

The actual compiler/database/dependency observations must contain the changed
TU/header/fragment. Test and static-library files, native compiled identity,
source digest, compiler/configuration and dependency domain bind each variant.
The maintained streaming source digest is shared with build-manifest generation.
Copied trees remain unknown/dirty for general target provenance, and these
correctness-only receipts are explicitly non-promotable. They cannot become
production performance or source-checkout qualification evidence.

Complete discovery and an exact ten-case CTest selection are retained. A filter
mismatch stops **before** execution. The baseline must pass every selected case;
the publication seed requires both the reserved publication assertion and the
real public CPU concurrency route to fail as expected. A mutant only counts as
`killed_by_assertion` when the complete CTest process reports ordinary failed
cases containing the required native source file and intended assertion marker.
Timeouts, signals, missing outputs, changed inputs, skips, compile/link failures,
unrelated assertions and survivors remain distinct unaccepted results. A user
interrupt stops the batch and preserves existing process records.

These curated native seeds and the separate
[Python production gate mutations](production-mutations.md) are source-authored.
Actual execution, reviewed coverage/kill baselines and
continuous CI integration remain work under
[#103](https://github.com/doublemover/RNS8/issues/103). No mutation score or GPU
qualification is inferred from this implementation.

The same driver accepts an optional `--mull-runner` and `--mull-plugin` pair on
native Linux. The authored pilot pins
[Mull0.37.0](https://github.com/mull-project/mull/releases/tag/0.37.0) and checks
the runner's observed plugin path/full LLVM version against the selected Clang.
The release's packaged LLVM23 plugin is for macOS; a matching native Linux pair
must be supplied explicitly. This does not replace the selected host LLVM or
install another compiler. Windows and HIP mutation are outside this pilot.

After the ordinary baseline and mandatory curated mutations pass, the existing
snapshot/build owner creates a separate unchanged instrumented CPU baseline.
Only checked arithmetic and the integer-split production TU admit boundary and
comparison mutants. Tests/third-party sources are excluded, compilation records
must contain the selected plugin/debug flags, and compilation and execution use
the same retained config. The driver explicitly runs the actual Catch binary
with the selected literal cases, one worker and bounded time/artifacts; ordinary
CTest execution alone cannot run Mull. See the upstream
[setup](https://mull-project.com/getting-started/setup/) and
[CTest contract](https://mull.readthedocs.io/en/latest/tutorials/CTestIntegration.html).

The read-only native SQLite adapter follows the pinned producer's actual
[schema](https://github.com/mull-project/mull/blob/04f5c409878d602e1527ffae4c8d21c61a33d8ea/rust/mull-reporters/src/sqlite_reporter.rs).
It preserves failures, survivors, timeouts, crashes, abnormal exits, dry runs,
fail-fast omissions and uncovered mutants separately. A native failure remains
pending assertion review; no optional pilot result earns an intended-assertion
kill or a mutation score. Partial/live databases, unknown versions/schemas,
duplicate mutants and out-of-scope source locations reject. Compiler-plugin
builds and `MULL_` injection cannot supply production profiling baselines. The
pilot and its controls are source-written and unexecuted.
