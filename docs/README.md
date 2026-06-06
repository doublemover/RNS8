# RNS8 Documentation

Start here when evaluating the repository.

- [RNS8_RESEARCH_SPEC.md](RNS8_RESEARCH_SPEC.md): source of truth for
  architecture, roadmap, exact semantics, backend policy, and acceptance gates.
- [roadmap-status.md](roadmap-status.md): current implementation status and
  remaining gaps against the spec.
- [public-roadmap.md](public-roadmap.md): compact public roadmap for pre-1.0
  validation and implementation gates.
- [release-checklist.md](release-checklist.md): release-candidate metadata,
  package, validation, evidence, and documentation gate.
- [glossary.md](glossary.md): concise terminology map for semantic modes,
  backend names, and evidence terms.
- [prior-art.md](prior-art.md): related compute and exact-arithmetic systems
  plus project scope boundaries.
- [correctness.md](correctness.md): test coverage, semantic guardrails, and
  exactness rules.
- [backend-notes.md](backend-notes.md): implemented backend families,
  accelerator enablement policy, and AUTO selection boundaries.
- [performance-wins.md](performance-wins.md): current Windows `gfx1100`
  winning improvements, compact speedup tables, and promotion boundaries.
- [reviewed-local-evidence.md](reviewed-local-evidence.md): durable summaries
  of reviewed local benchmark claims and reproduction command families.
- [performance-model.md](performance-model.md): benchmark schema, review
  requirements, release evidence summaries, and performance-promotion policy.
- [resident-output-api-draft.md](resident-output-api-draft.md): ABI-neutral
  draft for future resident matrix and residue-current output handles.
- [platform-windows.md](platform-windows.md): Windows HIP SDK setup and local
  `gfx1100` validation path.
- [platform-linux.md](platform-linux.md): Linux ROCm and Instinct validation
  scope.
- [platform-readiness.md](platform-readiness.md): dependency/readiness report
  policy and validation-boundary classes.
- [design.md](design.md): compact design notes for the implemented scaffold.

Cleanup and drift-control helpers:

- `metadata/`: checked-in registry for benchmark modes, grouped-dispatch
  strategies, output policies, event phases, kernel identities, epilogues,
  workspace modes, and claim labels.
- `tools/metadata_registry.py`: validates the registry and regenerates tracked
  Python/C++ constants.
- `tools/repo_hygiene_report.py`: reports large files, duplicated metadata
  strings, direct currentness writes, and raw HIP resource calls.
- `tools/claim_validation.py`: checks durable docs for unsupported target
  readiness wording and unqualified speedup claims.
- `tools/golden_regression_suite.py`: runs compact metadata/schema/report smoke
  checks without broad release performance sweeps.
- `tools/perf_variance_report.py`: groups reviewed captures by same
  contract/backend/kernel and reports whether observed timing variance leaves a
  claimed speedup outside the repeatability margin.
- `tools/shape_family_shadow_report.py`: builds non-routing AUTO
  shape-family recommendations from reviewed exact cache entries and records
  why each recommendation remains advisory.
- `tools/auto_shape_family_gate.py`: verifies shape-family shadow reports stay
  exact-cache, non-routing, non-promotable, and boundary-safe before any AUTO
  selector work can use them.

Tracked documentation should summarize reviewed facts. Raw benchmark captures,
probe output, downloaded installers, and smoke-test scratch files belong under
ignored `temp/`, `build/`, or `out/` paths.
