# Python production branch coverage

`tools/python_coverage_session.py` is an authored, plan-by-default collector for
the production modules and real native unittest entrypoints in
`metadata/python-coverage-policy.json`. No package installation, process-control
run, test discovery/execution, profile collection or coverage number is claimed
by the current source pass.

```sh
python tools/python_coverage_session.py \
  --python temp/quality-tools/bin/python --out temp/python-coverage-candidate
```

Windows uses `temp/quality-tools/Scripts/python.exe`. The isolated tool prefix
comes from the existing `tools/init_quality.py` bootstrap and its shared
Coverage.py7.17.2 CPython3.15 wheel lock. Add `--execute` only when running the
validation batch. The selected interpreter must match the shared Python version,
have assertions enabled and load Coverage.py from its own isolated prefix.
Interpreter/base interpreter, venv configuration and installed package files
are hashed; source inputs, policy, process logs and artifacts are checked again
at completion. Invocation paths retain venv/compiler driver names while artifact
references hash their resolved files.

The collector first runs a bounded parent/child control with a changed child
working directory. It checks actual captured parent/child PIDs, independent
native process-data files and execution of each process's **function body**,
not merely module import or function definitions. Control data stays separate
from application coverage and contributes zero application passes. Coverage.py's
versioned [subprocess patch](https://github.com/nedbat/coveragepy/blob/7.17.2/doc/subprocess.rst)
and `_exit` patch are selected explicitly. The C tracer supports the selected
test-function contexts; tracing warnings or invalid/missing/nonbranch raw data
block collection. Every raw SQLite profile is inspected before combine, which
prevents a corrupt input being silently skipped by the reporting tool.

Native unittest discovery and execution go through the existing loader,
`load_tests` hooks and runner. Complete case IDs and source identities must
match between discovery and execution; every selected case and each required
real evidence test must pass. Skips, missing cases, class/subtest failures,
stripped assertions, tracing warnings or incomplete process capture block the
lane. Exact test IDs and production-function body/context observations bind
three initial evidence invariants: executable/library manifest admission,
runtime environment identity and complete mandatory native test execution.

The version-bound native JSON retains executed/missing lines and branch arcs,
function/class regions and real per-test contexts. `branch-summary.json` keeps
the explicit production denominator and missing outcomes; `html/index.html`
supports inspection. Every other tracked Python source remains a named scope
exclusion. This initial lane is neither a repository-wide percentage nor proof
that every possible subprocess path is instrumented. See the selected native
[JSON producer](https://github.com/nedbat/coveragepy/blob/7.17.2/coverage/jsonreport.py)
and [context producer](https://github.com/nedbat/coveragepy/blob/7.17.2/coverage/context.py).

`collected` means source-bound collection and its required inventory/control
checks completed. Reviewed baseline thresholds, broader process/OS paths,
production mutation detection, native C++ invariant mapping and continuous CI
remain separate work under [#103](https://github.com/doublemover/RNS8/issues/103).
Coverage startup is scoped to these owned child invocations and must not leak
into benchmark or performance evidence jobs.
