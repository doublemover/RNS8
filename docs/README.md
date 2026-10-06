# Documentation map

## Start here

- [README](../README.md): purpose, contracts, builds, examples, and current limits
- [Correctness](correctness.md): mathematical guarantees and caller obligations
- [Design](design.md): implemented storage, ownership, and execution model
- [Backends](backend-notes.md): backend capabilities and selection limits
- [GPU qualification](gpu-qualification.md): exact RX 7900 XTX / MI300X run plan
- [October 2026 audit](audit-2026-10-06.md): defects, changes, evidence, open risks
- [Performance evidence](performance.md): measurement and promotion policy
- [Roadmap and release gates](roadmap.md): the single current execution checklist

## Build and reference

- [Windows](platform-windows.md), [Linux](platform-linux.md), and
  [readiness terminology](platform-readiness.md)
- [Research specification](RNS8_RESEARCH_SPEC.md): intended architecture,
  mathematical requirements, long-term research, and acceptance rules
- [Glossary](glossary.md), [prior art](prior-art.md), and [contributing](../CONTRIBUTING.md)

The specification describes both existing and proposed behavior. For an
implementation claim, consult the source and the audit; a design requirement or
registered kernel name is not proof that a path is executable or qualified.

## Research and history

[Research notes](research/README.md) preserve application-alignment studies and
the resident-output proposal. They are not implemented API guarantees.

[Historical records](archive/README.md) preserve prior queues, research backlogs,
work logs, and local performance summaries. The previous independent status,
release, and promotion documents have been consolidated into the active roadmap
and performance pages above. Historical claims do not override current evidence.

Raw logs, captures, manifests, and disassembly belong in ignored `temp/`, `build/`,
or `out/`. The checked-in audit records what was actually tested and what remains.
