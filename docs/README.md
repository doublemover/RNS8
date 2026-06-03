# RNS8 Documentation

Start here when evaluating the repository.

- [RNS8_RESEARCH_SPEC.md](RNS8_RESEARCH_SPEC.md): source of truth for
  architecture, roadmap, exact semantics, backend policy, and acceptance gates.
- [roadmap-status.md](roadmap-status.md): current implementation status and
  remaining gaps against the spec.
- [correctness.md](correctness.md): test coverage, semantic guardrails, and
  exactness rules.
- [backend-notes.md](backend-notes.md): implemented backend families,
  accelerator enablement policy, and AUTO selection boundaries.
- [performance-model.md](performance-model.md): benchmark schema, review
  requirements, release evidence summaries, and performance-promotion policy.
- [platform-windows.md](platform-windows.md): Windows HIP SDK setup and local
  `gfx1100` validation path.
- [platform-linux.md](platform-linux.md): Linux ROCm and Instinct validation
  scope.
- [platform-readiness.md](platform-readiness.md): dependency/readiness report
  policy and validation-boundary classes.
- [design.md](design.md): compact design notes for the implemented scaffold.

Tracked documentation should summarize reviewed facts. Raw benchmark captures,
probe output, downloaded installers, and smoke-test scratch files belong under
ignored `temp/`, `build/`, or `out/` paths.
