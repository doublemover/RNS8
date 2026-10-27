# Documentation ownership and maintenance

The [task index](README.md) is the reader's entry point.
[documentation-inventory.json](documentation-inventory.json) is the exhaustive
migration checklist for first-party Markdown, HTML, reStructuredText, legal files
and issue/PR templates. Git's file list excludes generated output and submodule
contents. Every record has a purpose/audience, disposition, canonical destination
and section, active capability/evidence owner, inbound links, anchors and reason.

## Ownership map

| Canonical owner | Reconciled material | Disposition |
|---|---|---|
| [Product README](../README.md), [task index](README.md) | Root overview and former linear doc map | Rewrite for hardware, arithmetic, GPU setup, use and navigation |
| [Windows](platform-windows.md), [Linux](platform-linux.md), [dependencies](../third_party/README.md) | Root setup, platform/readiness and package pointers | One path per OS; explicit current overrides; one prerequisite/pin owner |
| [Contributing](../CONTRIBUTING.md) | Root CPU/test/package recipes, PR checklist, AGENTS pointers | Merge developer workflows outside the product introduction |
| [Correctness](correctness.md), [glossary](glossary.md) | Arithmetic, exact-wide ABI/readiness duplication | Keep normative semantics/export/error owner; readiness links to it |
| [Architecture](design.md) and focused plan/prepared/DAG/epilogue guides | Public lifecycle and composition material | Keep distinct detailed contracts, reconcile stale implementation claims |
| [Backend reference](backend-notes.md) | Backend table and embedded implementation READMEs | Rewrite shared capabilities; local READMEs retain layout/admission details |
| [Performance](performance.md), [qualification](gpu-qualification.md) | Calibration/timing methodology and dated results | Separate methodology, applicable correctness evidence and dated measurements |
| [Research index](research/README.md) | Representation/CRT, kernels/scheduling, planning/generation guides | Group navigation, preserve focused proofs/private-public boundaries |
| [Roadmap](roadmap.md), [specification](RNS8_RESEARCH_SPEC.md) | Current gates, chronological checkpoints and normative proposals | Concise current gates; archive chronology intact; retain normative requirements |
| [Research applications](research/README.md#application-context-and-primary-sources), [examples](../examples/README.md) | Four alignment surveys, prior art and consumer guide | Retain distinct dated insights/bibliographies; separate runnable source from proposals |
| [History](archive/README.md) | Audit/local checkpoints, dashboard, old resident proposal | Preserve dates/commits; historical navigation and proposal redirect |
| [Community templates](../.github/ISSUE_TEMPLATE/hardware_validation_offer.yml) | Bug/backend/performance forms and PR template | Focused prompts, hardware-offer form, public-safe coordination |
| [LICENSE](../LICENSE), [NOTICE](../NOTICE), [security](../SECURITY.md), [changelog](../CHANGELOG.md) | Legal/security/release metadata | Preserve legal/attribution content and historical release facts |

Distinct proof and source detail remains in topic guides. The resident-output
proposal moved to the archive with a redirect at its original path. Chronological
roadmap entries moved to a dated archive with links/anchor redirects from the
current roadmap. Dated local evidence retains its original path and scope,
including the 256³ diagnostic; the 512³/1024³ report owns current scaling claims.

## Check the inventory and navigation

```sh
python tools/check_documentation.py
python tools/test_documentation.py
python tools/check_documentation.py --json
```

The check validates complete inventory coverage, canonical destinations/sections,
owner paths, local Markdown/HTML/image references, heading/explicit anchors,
reference-style links, and current preset/example-target/script names. Historical
commands are preserved as historical; their file links and anchors are still
checked. Inbound links/anchor lists are derived per record; refresh them after
intentional navigation changes:

```sh
python tools/check_documentation.py --refresh-inventory
```

This updates only inventory metadata, not document prose. Review dispositions
and canonical owners manually when adding or moving a document. The checker
cannot establish mathematical correctness, external page truth or command
execution from matching text.

## External links and templates

```sh
python tools/check_documentation.py --external --official-only
python tools/check_documentation.py --external
```

The first network scope checks official dependency/installer domains; the second
checks all linked HTTP sources/issues/bibliography targets. Requests have bounded
timeouts, four workers and at most one retry for transient responses, honoring a
bounded `Retry-After`. HTTP 429, blocked authentication and timeouts remain
unresolved with exit status 2. A local-only pass reports external checks as
unrequested. A partial network run is not an all-links pass. External fragments
are retained but browser-generated headings are not inferred by the static check.

The hardware form uses GitHub issue-form YAML with distinct field IDs. Authored
tests check its schema, required fields/notice and optional parser rendering
structure. Final GitHub rendering is an external UI check. The form accepts
"not sure" and asks for a safe coordination method; credentials and private
endpoints are never requested. Monetary sponsorship remains a separate Sponsors
link. Existing funding configuration, when present on the integration branch,
must be preserved; this change creates no replacement funding file.

## Command evidence and dependent acceptance

Preset/target/script/CLI names are checked against repository declarations and
help where available. This documentation change runs focused static/tooling
checks; it performs no dependency installation, configure/build or GPU campaign.
The dated validation and measurement reports identify previously executed builds,
commands and hardware. Illustrative nondefault-root/paths-with-spaces commands
remain unexecuted clean-install scenarios; native Linux/WSL/Instinct GPU commands
require matching supported hosts. They are not labeled tested by this editorial
work.

Automatic root discovery ([#57](https://github.com/doublemover/RNS8/issues/57)),
bounded remote-session preparation ([#58](https://github.com/doublemover/RNS8/issues/58)),
compatibility-gated dependency locks ([#60](https://github.com/doublemover/RNS8/issues/60))
and practical GPU application examples ([#62](https://github.com/doublemover/RNS8/issues/62))
remain dependent acceptance gaps. The current guides describe explicit existing
controls. Backend promotion, new APIs/results, rentals, provisioning and spending
are outside this documentation slice.
