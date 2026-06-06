# Large-File Refactor Notes

This cleanup split first-party implementation, tooling, and test files that were
over the rough 800 LOC target into focused include/package fragments.

The remaining over-target files are source-of-truth documents:

| File | Reason It Remains Whole |
|---|---|
| `docs/performance-gain-work-queue.md` | Active performance execution queue plus retained backlog/archive notes; splitting it would make the queue harder to scan as the working control panel. |
| `docs/RNS8_RESEARCH_SPEC.md` | Architecture and roadmap source of truth per `AGENTS.md`; keeping it single-file preserves reviewability of semantic and platform policy. |
| `docs/performance-model.md` | End-to-end performance model with connected assumptions, formulas, and evidence notes; the sections depend on shared notation. |

If any of these documents grow into unrelated concerns, split the archive or
appendix portions first and leave durable source-of-truth status in place.
