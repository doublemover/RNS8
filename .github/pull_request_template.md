## Behavior and scope

Describe the problem, resulting behavior and arithmetic/output contract.
Link the owning issue and call out public versus internal changes.

## Validation performed

List exact commands and results, with source/configuration and device identity
where relevant. State unavailable dependencies/hardware and explain skips.
For docs-only work use the [documentation checks](../CONTRIBUTING.md#focused-tooling-and-documentation-checks).

- [ ] Relevant CPU/reference regressions cover changed semantics.
- [ ] Kernel changes include CPU differentials and an applicable GPU correctness case.
- [ ] Public API, schema or backend changes include documentation and tests.
- [ ] `git diff --check` passes; ignored scratch output stays out of git.

## Evidence and remaining work

Keep raw captures/disassembly out of git. Link reviewed dated summaries for
hardware or performance claims, with matched baselines and output boundaries.
A build, probe or model is identified as such. State material limitations and
follow-up work needed for review or qualification.
