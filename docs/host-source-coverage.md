# Native host source coverage

The authored native Linux Clang lane collects source-based coverage for the
actual `rns8_tests` Catch/API executable. It does not qualify HIP, Windows,
OpenMP, Python subprocess coverage or mutation detection. No coverage build,
profile collection, fixture or native tool has run in this source frontload.

```sh
python tools/host_coverage_session.py --out temp/host-coverage-candidate \
  --cmake /path/to/cmake --ctest /path/to/ctest --compiler /path/to/clang++ \
  --nm /path/to/llvm-nm --llvm-cov /path/to/llvm-cov \
  --llvm-profdata /path/to/llvm-profdata
```

The default prints a plan without creating directories or running tools.
`--execute` uses a fresh ignored build with `cpu-coverage-debug`. All selected
native LLVM tools must match the compiler's complete version; CMake/CTest must
match. The initial JSON reader follows the pinned LLVM23
[3.1.0 exporter](https://github.com/llvm/llvm-project/blob/llvmorg-23.1.5/llvm/tools/llvm-cov/CoverageExporterJson.cpp).
Other producer formats require explicit reader qualification.

Compile-command/object/target inventories reuse the host instrumentation owner.
Coverage mapping and profile generation are selected together, including final
executable linking and a native ELF build ID. Discovery/setup profiles use a
different directory from the actual selected test profiles. Collision-safe
`%m-%p` filenames retain module/process separation; empty, corrupt, oversized,
changed or incompatible profiles block collection. The merge explicitly fails
on **any** invalid input. Profile objects are supplied explicitly, including
the static archive for compiled but unlinked code. Binary-ID checks are enabled
and debuginfod lookup is disabled. See LLVM's
[source coverage workflow](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html),
[profile merge options](https://llvm.org/docs/CommandGuide/llvm-profdata.html#merge)
and [object/export options](https://llvm.org/docs/CommandGuide/llvm-cov.html#export-command).

The lane retains full CTest discovery, an exact-name selection file, selected
discovery, JUnit and separate executed/pass/fail/skip/disabled inventories.
Selection includes every discovered case of the exact hashed native Catch
binary. Other CTest entrypoints remain explicit exclusions from this coverage
lane. Both legal-concurrency cases must pass. An optional `--allowed-skips`
file maps exact selected names to reviewed reasons; historical skip counts
and broad HIP patterns provide no approval.

`coverage-summary.json` preserves function, region, line and branch counts per
mapped source, detailed branch outcomes, expansion-record counts and source
digests. Native JSON retains complete function/region/expansion mappings, and
`html/index.html` provides source inspection. Production, test/tool and generated
metrics stay separate. The complete first-party denominator retains unselected
translation units, headers/fragments without mapping and compiled sources
without observed mapping. An empty native row for an uncompiled GPU file is
not evidence it compiled. Native `.inc` locations remain their own source files.

`collected` means the selected test inventory and source-bound collection
completed, with real production branches and required core modules observed.
It does not establish a reviewed coverage baseline or assertion strength.
No arbitrary percentage threshold, current measured percentage or mutation
success is invented. Python
broader branch/subprocess qualification, curated production mutations and CI policy
remain separately tracked work under
[#103](https://github.com/doublemover/RNS8/issues/103).

The same coverage summary now contains an initial invariant mapping tied to the
six mandatory native production mutations. The reviewed
[route map](../metadata/native-invariant-routes.json) must cover exactly that
registry. It names public integer-split queries, signed export, packing and
prepared execution; failed-close ownership remains explicitly a host template
with fake operations. Source hashes and unique operation/anchor spans bind the
reviewed source prefix and its actual native branch outcomes. Each row retains
the real discovered/selected/executed/passed assertion inventory, plus the
matching public C symbol's whole-suite native count and source mapping.

Unmapped or untaken public entrypoints, missing assertions and uncovered branch
outcomes remain gaps. A source operation without a mapped branch remains
unresolved, even if its file has coverage. These whole-suite counts do not prove
a per-test call chain or assertion strength; the separate native mutation lane
must detect the actual wrong behavior. The new public overflow witness reaches
checked addition after a valid near-maximum extent product and requires range
rejection without query-output publication. Mapping/control execution remains
deferred; no coverage number or mutation kill is claimed.

The authored Python inventory wrapper, `tools/python_test_inventory.py`, uses
the existing native `unittest` loader/runner and each module's `load_tests`
hook. It rejects empty/duplicate selections and disabled Python assertions,
records individual cases rather than a module-level exit alone, and preserves
subtest/class-fixture failures, skipped/expected-failure/unexpected-success
states and unexecuted cases. These controls do not count as ordinary passes.
`--discover-only` records discovery without running cases. This wrapper and the
bounded parent/child tracing control are source-authored. The separate
[Python branch collector](python-source-coverage.md) binds them to the selected
tool package, actual process profiles and source/test contexts; it is unexecuted.
