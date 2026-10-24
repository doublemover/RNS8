# Research topics and application context

Topic guides preserve concrete arithmetic contracts, source boundaries and
remaining acceptance work. A private candidate can be implemented and tested
without becoming a public backend. The [backend reference](../backend-notes.md)
and [public headers](../../include/rns8/rns8.h) identify public support; the
[roadmap](../roadmap.md) owns current issue gates.

## Representation and reconstruction

| Topic | Boundary / owner |
|---|---|
| [Device bounds](../device-bound-discovery.md) and [adaptive native product](../adaptive-native-product.md) | Device scans and opt-in public retained-native exact products; new ownership validation pending, #34/#35 |
| [Ragged materialization](../ragged-residues.md) | Private mixed-prefix storage and exact completion; #40 |
| [Mixed-radix lift](../mixed-radix-lift.md) | Private suffix arithmetic alternative; public lift keeps its default; #41 |
| [Balanced CRT](../balanced-crt.md) | Private reconstruction candidate; public Garner remains default; #39 |
| [Integrity sentinel](../device-integrity.md) | Private deterministic redundant-residue consistency checks; public Freivalds reserved; #49 |

[Correctness](../correctness.md) owns uniqueness, centering and failure semantics.
The [glossary](../glossary.md) defines shared terms; topic pages specialize those
rules instead of creating independent public contracts.

## Kernels and scheduling

- [Native narrow](../native-narrow.md) and [public narrow owner](../narrow-native-owner.md): checked scalar/byte products with complete persistent RNS publication; new owner validation pending, #53
- [Encoded B](../encoded-b-native.md): generation-bound materialization and explicit WMMA consumers; cold/warm reuse calibration authored, #54
- [Native tiled WMMA](../native-tiled-wmma.md): private four-wave register/LDS candidates and matched calibration; validation pending, #35
- [Instruction-chain probes](../instruction-chain-calibration.md): private dependent/independent INT8 WMMA and centered-reduction controls; modeled work and unqualified event normalization, #31/#35
- [Wide-index scalar control](../wide-index-gemm.md): private 64-bit offsets; large-span descriptors and small-allocation GPU fixtures authored, #24
- [Native split-K](../native-split-k.md) and [persistent queues](../persistent-native-queue.md): private producer/merge scheduling, #37/#38
- [Logical locality](../locality-schedules.md): offline permutations/model, #36
- [Block sparse](../block-sparse.md): private scalar control, generation-bound public block materializations and separate structured Sparse-A, #51
- [Strassen](../strassen-device.md) and [rank-23](../rank23-device.md): opt-in public one-level Strassen, private coefficient control and pinned alternative-basis SLP; new SLP validation pending, #48
- [Exact epilogues](../exact-epilogues.md): public exact-wide expression and separate private rocWMMA candidate, #47

Packed INT4/IU4 work remains restricted to internal proof-certified subsets;
public research semantics are rejected. Floating FP4/FP6/FP8 and Ozaki retain
separate arithmetic/instruction/ISA gates under #45/#46. The
[specification](../RNS8_RESEARCH_SPEC.md#17-technique-decisions-and-retirement-rules)
keeps their retirement and production criteria.

## Planning and generation

- [Typed planning recipes](../planning-recipes.md), #29/#30
- [Runtime product arena](../runtime-product-arena.md): internal CPU/HIP exact product DAG execution with strict last-reader reuse; validation pending, #29/#30
- [Distributed modulus plan and CPU mock](../distributed-mock.md), #50 M0; no GPU transport or qualification
- [Joint basis/schedule search](../joint-scheme-search.md), #32
- [Generated scalar encoding certificates](../generated-encodings.md), #44
- [Floating rewrite oracle](../floating-rewrite-oracle.md): host rational/binary32 checks and separate proof verdicts; no hardware qualification, #29/#45
- [Generated native WMMA family](../generated-native-wmma.md), #55
- [Calibration identities/rates](../calibration.md) and [native microcases](../native-calibration.md), #31

Models, supplied resource costs and synthetic fixtures are distinguishable from
compiled instructions and measured hardware. Representation/certificate identities
constrain admissibility; they do not establish measured performance.

## Application context and primary sources

| Study | Distinct contribution |
|---|---|
| [Computational algebra](computational-algebra-alignment.md) | Modular dense GEMM beneath elimination/reconstruction; kernels versus full algebra systems |
| [Exact linear and symbolic algebra](exact-linear-symbolic-alignment.md) | Rank/solve/determinant phases, rational reconstruction and structured workflows |
| [Computer algebra systems](computer-algebra-systems-alignment.md) | Domain/coercion boundaries, libraries, workload and oracle sources |
| [FHE and lattice work](fhe-lattice-alignment.md) | RNS/base-extension inspiration; large-prime polynomial/NTT workloads have different arithmetic |

These June studies remain research context, with their dated procedures and
primary-source bibliographies preserved. Their implementation status prose is
historical context; current API/capability/evidence owners are linked above.
[Prior art](../prior-art.md) provides the short library orientation.
[Runnable examples](../../examples/README.md) are separate from proposed
application integrations. The former resident-output proposal has moved to
[history](../archive/resident-output-api-draft.md).

## Evidence

[Finite-ring verifier calibration](../finite-verifier-calibration.md) defines
public verification, compute-plus-verify and complete modular-reference controls.
Its authored fixtures and deterministic replay metadata await consolidated
validation; no timing result or integer equality guarantee is implied.

The [expanded all-feature record](../local-all-features-20261023.md) supersedes
earlier pending-runtime notes for the registered target-applicable tests it ran
at `51ae8d8` on Windows/gfx1100. Topic-local mathematical claims and authored
fixtures describe their source; later changes, other targets, resources/ISA and
performance need their own scoped records. Current matched scaling results are
in the [larger-workload report](../7900xtx-large-throughput-2026-10-23.md).
