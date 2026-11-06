# Native production mutations

The authored `tools/native_mutation_session.py` lane builds real CPU library/test
variants for six exact-source seeds in `metadata/native-production-mutations.json`:
checked unsigned addition, signed CRT correction, published value generations
and retry ownership after failed handle close, signed export rejection status
and stale prepared-input generation admission. The shared native arithmetic
replay is exercised through actual public CPU calls for wrong-result,
wrong-status and stale-state sensitivity. These witnesses mutate production
operations in disposable source copies; their assertions receive no mutant flag.
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

Complete discovery and an exact nine-case CTest selection are retained. A filter
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
Actual execution, reviewed coverage/kill baselines, compatible optional Mull and
continuous CI integration remain work under
[#103](https://github.com/doublemover/RNS8/issues/103). No mutation score or GPU
qualification is inferred from this implementation.
