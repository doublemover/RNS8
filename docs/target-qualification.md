# Candidate target qualification

Current execution results and remaining qualification are owned by the
[current roadmap](roadmap.md#current-validation-and-remaining-work).

Target report schema 3 keeps platform/architecture/toolchain history as
descriptive inventory. Its phase counts and historical_inventory_readiness
are not a cache admission decision. Candidate verdicts identify an exact
compiled artifact, source/configuration/dependency manifest, backend/kernel,
mathematical work, device/runtime/driver and execution environment.

Generated manifests claim a Git commit/clean status only when the actual source
directory is Git's reported worktree root. Nested source copies and extracted
snapshots retain their content digest but report an unknown commit and dirty
state; they cannot borrow the containing checkout's identity for qualification.
Linked worktrees retain their own root. Git identity probes have bounded time,
ignore inherited `GIT_*` repository/configuration selection, disable fsmonitor
and avoid optional index writes. This source check adds no runtime qualification.

A new timing process is a new observation. It cannot hide a failed phase for
the same candidate artifact and work. A passing historical capture alone also
cannot qualify a newly supplied capture without an explicit phase association.
Unrelated builds and work contracts remain separate, so a historical failure
does not permanently invalidate a platform.

## Existing session evidence

The bounded GPU session runner records a stable session-manifest identity and
an explicit run ID, retained on an allowed resume. Legacy checkpoints without
that run identity are not retroactively labeled. Compiled identity queries
establish that test, verifier and benchmark binaries use the candidate's actual
library/build manifest. The selected phase must reference its actual command,
binary query, raw artifacts and process result in that same session/run.

Build evidence targets the complete configured build. CTest evidence includes
nonempty executed JUnit results and the required unskipped GPU qualification
case through the bound Catch runner. Empty/all-skipped selections, failed
processes, missing tests and unrelated binaries remain unqualified.
Release evidence names the exact candidate capture and request, has production
instrumentation disabled, and meets the shared release warmup/repeat policy.
Profiler evidence requires native selected-dispatch attribution; tool presence,
static opcode inventory or a successful unrelated command is insufficient.

The default bounded smoke plan still leaves release/profiler qualification
incomplete. These code paths do not retroactively qualify earlier campaigns.

## Emitting a receipt without running work

tools/target_qualification.py adapts retained session files; it launches no
commands. Supply --candidate, --session-manifest, --session-results and a new
ignored --output file. Select each phase with:

    --phase PHASE=STAGE,BINARY_IDENTITY_STAGE

PHASE is build, ctest, smoke, release_capture or profiler. CTest also needs
--correctness-stage; profiler needs --dispatch-receipt. All names refer to
existing entries in that session's manifest/results, not arbitrary status
labels. A missing phase is recorded as incomplete. The emitter exits 0 only
for a qualified receipt, 2 for a written incomplete receipt and 1 for malformed
inputs. It never overwrites a receipt.

For an explicit same-candidate requalification, pass each replaced receipt as
--supersedes PATH and retain its raw evidence. Unknown/self/cyclic supersession
or a still-current failed/incomplete receipt blocks qualification. A repaired
build has its own artifact identity and must supply coherent evidence for that
build.

Pass --qualification-receipt PATH to tools/target_validation_report.py alongside
the capture. Each candidate verdict retains the original receipt inputs and
explains excluded historical records. The promotion ledger merges all supplied
same-candidate inputs and recomputes qualification from raw files. The installer
recomputes it again and matches the actual candidate with the proposed cache
entry. Target-ready booleans and platform summaries alone cannot satisfy the
mandatory CDNA gate or an explicitly requested RDNA target gate.

Raw files are hashed at read and review completion, with shared per-invocation
caching so large executables are not repeatedly hashed for every phase. JSON
is parsed from the same bytes whose digest is checked. Changed/missing raw
captures, identities, commands, outputs or summaries fail closed.

## Native profiler boundary

The dispatch readers accept the maintained ROCprofiler-SDK native JSON
kernel-dispatch structure and finalized rocpd 3.0.4 databases. They bind the profiled benchmark's actual process ID
to process-local agent/queue/dispatch/kernel/code-object records and the existing
selected static ISA receipt. Exact mangled symbols and bytes from the loaded
file-backed code-object URI must match the recorded executable/artifact.
Repeated/ambiguous IDs, wrong agents, missing launch geometry and unsupported
memory-backed loads cannot qualify by string similarity or timestamp proximity.
See AMD's [JSON schema](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
and [code-object URI specification](https://rocm.docs.amd.com/projects/ROCdbgapi/en/docs-5.1.3/doxygen/docBin/html/group__code__object__group.html).

Profiler observations have their own capture ID and explicit external
instrumentation marker. Only declared profiler environment changes are allowed
in the production/profile relation. Completed exact-reference coverage and
matching results remain required. The profiled host interval cannot enter a
production timing claim. The session must establish native outputs as newly
created for that acquisition; stale files with a recycled process ID are not
accepted.

These readers establish selected dispatch identity, not complete-run counter
coverage, occupancy, measured traffic or a performance gain. Native acquisition and
capability discovery are implemented under #104. Supported machine formats and Windows
export capabilities are limited to the explicit contracts in the profiler guide. Raw RGP
and unsupported formats
remain inventory-only; Linux traces do not qualify a Windows target. Current native
reader/runtime execution scope is recorded in the roadmap. The
[profiler guide](amd-profiler.md) owns supported contracts, feature-level
preflight, database metric interpretation and remaining qualification.
