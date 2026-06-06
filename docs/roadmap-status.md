# RNS8 Roadmap Status

Status date: 2026-06-06

This file summarizes live implementation status against
[RNS8_RESEARCH_SPEC.md](RNS8_RESEARCH_SPEC.md). The research spec remains the
architecture and roadmap source of truth when details disagree.

## Implemented And Verified

- Phase 0 host foundation: C ABI, C++ wrapper, CMake targets, CPU reference,
  tests, dependency checker, benchmark schema, and result comparison tooling.
- Windows direct-HIP bring-up on Radeon RX 7900 XTX / `gfx1100`: explicit HIP
  SDK compiler integration, device inspection, residue conversion,
  one-modulus ring GEMM, K-block splitting, and CPU differential tests.
- Persistent RNS matrices: direct-HIP matrices own device residue, upload,
  export, and status buffers; resident GEMM/export consume device-current
  residues rather than silently uploading stale host state.
- Bounded i64/u64: CPU and direct-HIP fixed-prefix and per-tile bounded paths
  support exact CRT export with range-error preservation.
- Native vector ALU: the Windows HIP bounded i64/u64 backend supports explicit
  bounded contracts and remains separate from exact-wide, finite, and wrap64
  semantics.
- Exact-wide signed/unsigned: persistent RNS output and fixed-width
  little-endian limb export are implemented for CPU and direct HIP.
- Strict wrap64: CPU byte-limb reference and direct-HIP byte-limb correctness
  paths implement explicit `mod 2^64` semantics without odd-modulus CRT.
- Finite u8: CPU and direct-HIP finite-ring and finite-field paths use explicit
  modulus contracts, prefix-zero storage, and canonical byte export.
- Benchmark schema v4 and review tooling: captures record command line, git
  commit, compiler/HIP/device metadata, backend metadata, phase timings, event
  timing availability, and comparison-baseline status.
- Optional accelerators: hipBLASLt, CK, and rocWMMA have opt-in Windows
  `gfx1100` correctness backends with exact differential and ISA/schema
  coverage where applicable.
- AUTO selection: reviewed release cache hits can select compiled HIP-resident
  accelerator candidates for supported contracts; missing or rejected hits stay
  on the configured correctness path.

## Backend Promotion Status

| Backend | Correctness status | Performance status |
|---|---|---|
| CPU reference | Required reference backend | Baseline only |
| Direct HIP | Required Windows GPU correctness path | Production correctness baseline; not a matrix-engine speed claim |
| Native vector ALU | Bounded i64/u64 correctness backend | Useful explicit backend and long-K `n == 1` microkernel evidence, but the current skinny-GEMV release review keeps reviewed N=1 scenario routing on Direct HIP; not a current AUTO cache winner |
| hipBLASLt | Opt-in correctness baseline | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including eligible 4096 bounded, finite hot-modulus, and exact-wide entries |
| CK | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including bounded-i64 2048, generic finite-field 2048, and finite ring-255 4096 |
| rocWMMA | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-u64, finite-u8, and exact-wide shapes, including bounded-u64 2048, generic finite-ring 2048 entries, finite-field 512 entries, and smaller hot finite-u8 shapes; post-fix hot finite-u8 2048 winners are hipBLASLt |
| AMDGPU builtins | Not implemented | Fail-fast until real exact kernels exist |
| Wrap64 matrix-engine candidate | Internal rocWMMA harness only | Not public, not AUTO-selected, and not faster than direct HIP in current reviewed shapes |

Detailed benchmark policy, current wins, and reviewed release summaries live in
[performance-model.md](performance-model.md),
[performance-wins.md](performance-wins.md), and
[reviewed-local-evidence.md](reviewed-local-evidence.md).

## Cleanup Consolidation Status

The current cleanup program is an internal consolidation lane, not a public API
or routing change. The branch now has a checked-in metadata registry under
`metadata/`, generated Python/C++ constants, stale-label registry tests, a repo
hygiene reporter, a compact golden regression runner, and durable documentation
claim validation for target-readiness and speedup wording.

Cleanup adoption has progressed beyond the guardrail slice: benchmark scenario
families live in data under `benchmarks/scenarios/` with explicit review-mode
and promotion-eligibility contracts backed by the metadata registry. Benchmark
schema validation has a compatibility wrapper over a package entrypoint, with
GPU event, semantic-contract, reuse-timing, execution-mode,
contract-metadata, helper/output-policy, and backend metadata validators split
into focused modules and CTest-backed self-tests. Shared benchmark support,
argument parsing, backend selection, grouped-dispatch descriptor contracts, and
large semantic lane bodies have been moved behind internal helpers or include
units while preserving `rns8-bench` flags and schema output. Grouped-dispatch
benchmark lanes now share an internal Direct-HIP same-shape bucket-plan builder
and grouped resource helper for A/B/C slabs, optional status storage, residue
pointer tables, and descriptor matrix vectors instead of duplicating descriptor
and device-resource assembly per semantic.

Core currentness transitions for output setup, benchmark exact-wide pack
materialization, Direct-HIP output stamping, native-to-RNS bridge paths, and
test-owned stale/currentness mutations are helper-routed. Workspace identity,
schedule metadata, backend metadata, accelerator scratch, and prepack/resource
teardown now flow through named internal helpers while keeping the public
`rns8_workspace*` handle unchanged. Export/reconstruction paths now use a
shared internal selector module that records output layout, limb count, status
policy, D2H policy, selected export kernel, and tiled/all-zero metadata needs
before touching the documented mutable export cache; benchmark captures expose
the same selector source, layout, D2H policy, status policy, selected export
kernel, and tiled/all-zero decisions through schema-validated
`export_variant` fields.

HIP event/stream/pinned-staging/temporary-device-buffer ownership uses internal
RAII wrappers, including CK/rocWMMA event timing helpers. Direct-HIP host code
is split into resource, pack, GEMM, and export include units behind the same
translation unit; the Direct-HIP `.hip` source is split into common device
helpers plus pack, GEMM, and export kernel include units while preserving the
existing compiled object and launch wrappers. Hardening now includes the
portable non-Windows CPU ASan/UBSan preset, a Windows clang-cl CPU-only
ASan/libFuzzer preset, three deterministic fuzz harnesses, and a non-GUI
`cdb.exe` WinDbg triage helper. The hygiene report filters intentional
helper/RAII implementation sites so remaining findings point at real drift.
Performance evidence hardening now also includes
`tools/perf_variance_report.py`, which groups same-contract capture reruns by
backend/kernel, records within-capture and run-to-run timing spread, derives
the minimum speedup margin needed to clear observed repeatability noise, and
can feed `tools/promotion_ledger.py` through `--variance-report`.
The non-routing `tools/shape_family_shadow_report.py` can now explain which
reviewed exact cache entry a shape-family AUTO policy would choose and which
blocker prevents that advisory recommendation from changing routing. Future
CDNA validation infrastructure now also includes schema-visible export selector
keys, export selector reports, Direct-HIP workspace arena allocation-delta
gates, and K-block/tile-K scenario/report metadata; these are readiness
surfaces only, not Linux/CDNA performance claims.
Pending-validation infrastructure now uses `tools/pending_validation.py` as the
target-generic command-planning, review-indexing, post-report, and summary core;
`tools/gfx1100_pending_validation.py` is the local Windows `gfx1100`
release-control wrapper. Multi-GPU readiness infrastructure includes
physical-device topology records in `cdna-env-summary.json` and
`tools/multigpu_shard_report.py` for independent per-GPU shard aggregation.
These do not change AUTO routing, installed cache entries, or durable
performance claims.

Remaining cleanup is now mostly validation and follow-through: run the full
final gate on the current host, fix any sanitizer/fuzzer/HIP failures it
exposes, and keep future optimization lanes using the registry, split schema
modules, currentness helpers, descriptor contracts, and export plan surface.
These changes preserve existing public ABI compatibility, reviewed cache
behavior, existing benchmark CLI compatibility, and Windows `gfx1100`
validation boundaries. Grouped
benchmark lanes now also route bounded, finite, and exact-wide pack/GEMM/export
phase execution through the internal Direct-HIP grouped descriptor/resource
helpers. `rns8_get_grouped_dispatch_contract_info` exposes the read-only
contract surface, and `rns8_gemm_rns_grouped` plus
`rns8_gemm_finite_u8_grouped` expose narrow same-shape Direct-HIP resident
grouped GEMM. Public grouped pack/export, AUTO routing, and broader generic
dispatch remain unexposed.

After PR #12, the active performance queue was refreshed so grouping, export,
residency, reuse policy, variance gates, and target validation drive the next
work before narrower kernel-tuning lanes.

## Not Yet Implemented

- Linux ROCm direct-HIP parity on a real supported Linux host.
- Instinct CDNA validation, profiling, power runs, and cluster reproducibility.
- Multi-GPU modulus-split experiments.
- Public optimized matrix-engine strict wrap64 backend.
- AMDGPU builtin hot kernels.
- Durable packed-layout/prepack-cache production paths.
- Optimized finite-field algorithms beyond current explicit-modulus CPU,
  direct-HIP correctness, and narrow reviewed accelerator-cache paths.
- Broader production performance gates beyond reviewed Windows `gfx1100`
  shape-scoped evidence.
- Linux/Instinct promotion gates remain unvalidated. Exact-wide 2048 and
  eligible exact-wide 4096 rows now have Windows `gfx1100`
  CPU/direct/accelerator release review and installed local cache entries;
  strict wrap64 2048/4096 has CPU/direct-HIP release review but no public
  optimized matrix-engine backend.

## Validation Boundary

Windows `gfx1100` proof does not validate Linux ROCm, Linux Radeon, Instinct
CDNA, profiling, power, or cluster production gates. Dependency discovery does
not validate correctness backends. Accelerator probes remain candidate evidence
until a dedicated preset builds real kernels and exact differentials pass.
The June 6, 2026 CDNA-readiness tooling pass adds target-validation,
counter/resource, promotion-ledger, cache-history, and bounded-i64 1024 review
control surfaces, but it does not add Linux/CDNA validation evidence.
The pending-validation follow-up adds the Windows `gfx1100` validation driver
and independent multi-GPU shard report, but any outputs stay local or
infrastructure-only until a later explicit review/promotion pass.

Raw captures, historical smoke files, and review scratch output stay under
ignored `temp/` paths. Durable docs summarize only reviewed facts.
