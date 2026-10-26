# October 26 selected implementation pass

The requested development scope comprises 25 issue IDs. The bounded research
implementations and their fixtures are present for all 25; this is a source
completion record, not a hardware qualification or release decision. #28 and #29
were already closed when the issues were refreshed. The other 23 remain open
for their acceptance gates. No issue is closed by this pass.

The [source inventory](../metadata/selected-implementation-20261026.json) records
files, hashes, implemented boundaries, syntax checks and deferred qualification.
The [roadmap](roadmap.md) retains current release gates; older test receipts
retain their original compiled sources.

## New source work

`ae1b322` adds the [real modulus-sharded HIP owner](distributed-hip.md), extending
the existing immutable plan and CPU mock into actual device allocation,
replication, packing/GEMM, peer/portable-pinned transfers and owner-device CRT.
It includes expected coverage, checked publication, deadlines/cancellation,
resident reuse/materialization, bounded memory and cleanup quarantine, plus
independent exact fixtures and a counterbalanced future-host collector.

The subsequent source review also retains initialization DMA endpoints and
the lowered host tile directory when preparation cannot prove completion.
Quarantine observes initialization streams before reclaiming those allocations.

[#31's system probes](system-roof-calibration.md) fill the remaining standalone
LDS/global-memory and launch/event/descriptor/graph calibration source gap.
They include independent references, output guards, all-source/displacement
audits, checked live payloads, instrumented/production scopes, censored sustained
intervals and immutable independent-process collection. No timing is collected.

## Implementation audit

| Issue | Implemented scope | Remaining acceptance |
|---|---|---|
| #28 | Authoritative expected/observed shard inventory, uniqueness and missing-result rejection | Already closed; no distributed execution inferred |
| #29 | Exact arithmetic/composition certificates with separate algebra, floating, layout and target verdicts | Already closed; target evidence stays separate |
| #31 | Typed calibration/rates, instruction/conversion/CRT/system probes, paired interactions and held-out discrepancy tools | Named-target ISA/counters/clocks, independent collection and accepted prediction tolerance |
| #32 | Certified bounded joint basis/form/group/slab search and explicit ordered-basis runtime pack/GEMM/export/lift | Measured hardware weights, profitability and supported-provider qualification |
| #34 | Device summaries/buckets, adaptive consumers, retained summary/schedule facts and explicit proof-policy planning | Benefit against strongest controls, fallback overhead and measured selection |
| #35 | Multi-channel/multi-wave native WMMA, register/one-/two-buffer LDS and lane/CTA conversion ownership | Launched mapping, resources/traffic and complete-route benefit |
| #36 | Checked locality permutations, cost buckets, real queue/split-K lowering and displacement controls | Real topology/cache/XCD evidence and profitable scheduling |
| #37 | Exact split-K producers, hierarchical/fused merges, bounded windows and explicit split/merge controls | Independent same-contract crossover and profiler evidence |
| #40 | Executable fragment algebra, tile-major consumers, ragged adapters, generation-bound panels and retention choices | Launched layout/ISA/resource evidence and profitable end-to-end reuse |
| #41 | Sign-correct device exact extension, shared mixed-radix digits, prepared/graph and explicit-basis conversion | Broader target execution, transfer/extension cost and amortization |
| #42 | Native/uniform/ragged/packed materializations, chain DP, bounded DAG search, shared B and lifetime/arena execution | Accepted measured prices, held-out choice, peak/capacity and provider qualification |
| #43 | Public direct-HIP state/tokens/dependencies, bounded/finite/wrap semantics, pinned transfers/export/lift, graphs and failure ownership | OS/device failure/overlap/lifetime qualification; typed unsupported providers/caller capture remain explicit |
| #46 | Explicit finite signed INT4 owner, packed-B generation identity and shared arithmetic/resource admission | Actual target opcode/fragment/resource evidence and profitable crossover |
| #47 | Shared coefficient lowering and dead-term elimination across CPU/HIP prepared/graph and explicit rocWMMA | Compiled resource evidence and complete-expression comparisons |
| #48 | One-level Strassen, coefficient rank-23 and retained/recomputed straight-line DAG matrix subproducts | Independent large-shape benefit and memory gates; optional rank48 stays conditional |
| #50 | Real M1/M2 modulus-sharded HIP ownership, explicit transports, weighted work, bounded pipelines and complete-result reuse | Actual 2/4/8-GPU Linux/Windows execution, faults/topology/traffic and 1.55x release gate |
| #51 | Explicit Sparse-A/hybrid owners/examples, real target builtins, prepared sparse storage and shared event registry | Actual gfx942/gfx1200 arithmetic/ISA and sustained crossover |
| #53 | Proof-selected scalar/vector/dot/matrix native paths, fused RNS output and public typed planning | Independent tiny/skinny/dense controls and target selection |
| #54 | Actual Linux AMX-INT8 kernel, per-worker OS permission, packing/reuse and public scalar/AMX controls | Supported Intel/Linux hardware execution, energy and profitability |
| #55 | Bounded generated kernels and source-bound symbolic/mutation/ownership/layout checks | Compiled target equivalence, resources and measured pruning/promotion |
| #59 | Atomic paired complex 3M/4M, transpose/conjugation, reduced dead-slot scratch and graph nodes | Independent complete-route and failure/lifetime qualification |
| #63 | Prepared Gram/SYRK native panel reuse, transpose/mirror and reconstruction-once export | Physical mapping/ISA and profitability; optional five-product extension stays gated |
| #64 | Signed-radix wrap64 with distinct RDNA3 WMMA and CDNA2/3/4 MFMA maps/slabs and native controls | Actual target mapping/ISA, sustained performance and route admission |
| #65 | Lazy host payloads, canonical native upload, byte-identical mod256 transfer, trim/retention and allocation observations | Matched storage/latency, physical peaks and broader platform qualification |
| #66 | Exact word splitting/recombination, width-specialized dot4, FP64 MFMA asymmetric/concatenated schedules and collectors | Actual CDNA execution/ISA and same-contract crossover |

This audit preserves each issue's bounded initial envelope. It does not turn
optional, separately admitted experiments into implicit support: floating
hardware (#45), distributed M/N/K/RCCL/multi-process M3, arbitrary experimental
basis providers, unsupported library capture and conditional algebraic extensions
remain explicit. A typed unsupported result is not a simulated successful kernel.

## Check boundary

MSVC and HIP-off Clang parse the new runtime/driver; new CPU fixtures and
collector sources parse. The maintained direct-HIP translation unit parses for
gfx1100 plus explicit device-only gfx90a/gfx942/gfx950/gfx1030/gfx1200. The sparse
builtin translation unit parses for gfx942/gfx1200; new system probes parse for
gfx1100 and device-only gfx942/gfx1200. These checks emit no objects and establish
neither linking nor executed instruction behavior.

No new unit test, GPU fixture, benchmark, stopped campaign or residue-dot
allowance is executed. Newly authored cases await a later validation batch.
The preceding [full/targeted validation](local-validation-20261025.md#october-26-frontloaded-source-and-combined-retest)
remains attributed to `f406c29`/`84b4d9c`; it does not validate this new source.
