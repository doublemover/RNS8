# Curated production mutations

The authored `tools/seeded_mutation_session.py` lane evaluates the exact Python
production operations in `metadata/production-mutations.json`. The seven
mandatory seeds bypass executable/library identity equality, request environment
seals, mandatory native-test admission, instrumented-production rejection,
complete exact-reference admission, stale qualification-artifact rejection and
the actual candidate consumer's refusal of stale qualification. The last seed
uses the real synthetic producer/receipt/consumer chain, independently of the
direct artifact-store guard control; reaching a helper does not prove consumer
admission.
They change real production modules, not test assertions or a mutant-enabled
flag. No baseline, mutation, test or worker has executed in this source pass.

```sh
python tools/seeded_mutation_session.py --python /path/to/selected/python \
  --out temp/production-mutation-candidate
```

The default checks source/operation identity and Python syntax, then prints a
plan without imports, overlays or subprocesses. `--execute` creates isolated
baseline/mutant copies of each production module under the owned output tree.
Each runs in a fresh selected-version Python process through the existing real
unittest module. Dependencies and tests remain source-bound to this checkout;
the original production file is never edited. Both variants have the same
relocation behavior, so a broken relocated baseline cannot count as detection.

Every seed binds the full original source SHA256, unique function/operation,
replacement, required real test ID and intended assertion marker. Source drift
requires re-review. Scopes name one module function or one exact `Class.method`;
nested definitions are excluded. Operations replace one complete guard, return
or named argument of a directly returned call. Argument mutations retain the
same field name. Repeated guards in other functions cannot redirect a seed.
All passing baselines precede the mutant batch, with one
attempt per variant and no automatic retry. A narrow main-thread trace records
whether the required test reached the exact changed source line. The actual
unittest case observer scopes the trace; it does not tell tests to fail. A
covered line in a different case or an unexecuted operation cannot qualify a
kill. Dynamic self-comparison/membership seeds retain actual source-line
execution rather than relying on an optimizer preserving a constant-false
branch. The reference-result field seed remains subject to the same actual-line
trace: if a selected interpreter omits that line, it is `not_covered`, not a kill.

An accepted mandatory result is `killed_by_assertion`: a complete captured
process, unchanged native test inventory, observed required-test source line and
that test's intended assertion failure. Passing wrong behavior is `survived`;
unreached sites, failed baselines, invalid compilation, timeouts, infrastructure
errors, sanitizer findings and unrelated assertions remain separate outcomes.
Compiler/startup crashes and unsupported tools never become assertion kills.
Raw process logs, source/module overlays, native case inventories, trace
observations, budgets and final digests remain available for reproduction.

The lane measures these Python production gates only. It does not qualify
execution of the separate authored [native C++ mutation lane](native-production-mutations.md), optional Mull, generated
device proof mutations or GPU faults. Those owners and native invariant/CI
integration remain separately tracked work under
[#103](https://github.com/doublemover/RNS8/issues/103). Existing generated-WMMA
`unknown`/`qualified=false` policy is unchanged. Reviewed baseline coverage and
mutation detection remain complementary to exact differential and sanitizer
checks, and no current mutation score is invented.
