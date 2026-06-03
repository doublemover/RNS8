# Computer Algebra Systems Alignment Notes

Date: 2026-06-03

This pass refines RNS8's targeting against real computer algebra systems
(CAS), CAS libraries, exact-algebra kernels, benchmark corpora, and GPU/CUDA
artifacts. It builds on
[computational-algebra-alignment.md](computational-algebra-alignment.md) and
[exact-linear-symbolic-alignment.md](exact-linear-symbolic-alignment.md).

The practical answer is stable: CAS systems are semantic, orchestration,
oracle, and workload ecosystems. RNS8 should target dense exact/RNS GEMM,
explicit finite-u8 ring/prime-field GEMM, reconstruction/export, and
residue-current chains. It should not become a CAS, a coercion engine, a
symbolic algebra runtime, or a full exact-linear-algebra package.

## Research Method

The pass used six exploration commanders. Each commander worked read-only from
`C:\Users\sneak\Development\RNS8-cas-alignment`, managed its own
`gpt-5.3-codex-spark` citation-chaining subprocesses through `codex exec`, and
stored raw details under `temp\cas-alignment\`. The lead consolidated only the
commander syntheses into durable docs.

Commander lanes:

- CAS domain model: parent/category/domain/coercion models and exact versus
  approximate boundaries.
- CAS kernel decomposition: dense modular GEMM, exact LA, symbolic phases,
  CRT/CRA, factorization, interpolation, and polynomial/tree phases.
- CAS library stack: FLINT, NTL, LinBox, FFLAS-FFPACK, Givaro, Singular,
  PARI/GP, Normaliz, GAP packages, Nemo/Hecke, and related roles.
- CAS workload corpus: benchmark families, examples, matrix/polynomial/domain
  shapes, and scenario metadata.
- CAS GPU/HPC translation: CUMODP, Linac, GBLA, Magma dense-F4/CUDA notes,
  CUDA-to-HIP risks, and AMD evidence gaps.
- CAS correctness and evidence: deterministic/probabilistic modes,
  certificates, bad-prime handling, licensing, reproducibility, and wording
  boundaries.

No second commander wave was needed. The lanes agreed on the main boundary, and
none found a real `docs/RNS8_RESEARCH_SPEC.md` mismatch.

## Executive Findings

- RNS8 is best positioned as a dense exact integer/RNS GEMM backend and small
  explicit finite-u8 GEMM backend. CAS systems can supply workload vocabulary,
  oracle outputs, and phase labels; they do not define RNS8 product scope.
- CAS domain models strongly support RNS8's explicit-semantics stance. Plans
  should be keyed by semantic contract, modulus/ring/field descriptor,
  reconstruction mode, and currentness, not by C++ type or storage shape alone.
- Dense modular GEMM is a real kernel under exact rank, determinant, solve,
  nullspace, characteristic polynomial, F4, and FGLM workflows. Those workflows
  also need controller, symbolic preprocessing, factorization, triangular
  solve, sparse reduction, certificate, and reconstruction phases that raw GEMM
  timings do not prove.
- CRT/CRA is not just export formatting. It has residue selection, controller,
  termination, builder, and reconstruction costs that should remain visible in
  scenario evidence.
- External libraries should be recorded by role: CPU exact oracle, finite-field
  reference, algorithm reference, benchmark comparison, CAS semantic oracle,
  CUDA translation study, workload source, or explicit non-goal.
- CUDA CAS artifacts are useful translation studies only. CUMODP, Linac, and
  Magma CUDA dense-F4 notes do not become AMD HIP, Windows `gfx1100`, Linux
  ROCm, or Instinct evidence until RNS8 has target-specific compiled kernels,
  exact comparisons, and phase timings.
- Wording matters. RNS8 evidence should not say "CAS-correct", "certified exact
  LA", or "secure probabilistic verification" when it only proves GEMM under an
  explicit RNS8 contract.

## Source-Ranked Evidence Map

| Source | Type | RNS8 implication | Classification |
|---|---|---|---|
| Sage coercion model: https://doc.sagemath.org/html/en/reference/coercion/sage/structure/coerce.html | Official CAS docs | Parent/common-parent/coercion separation supports explicit semantic descriptors and rejection of incompatible domains. | Direct metadata model |
| Magma structures and coercion: https://www.math.ru.nl/magma/text40.html | Official CAS docs | Unique parent object plus automatic/forced coercion maps to: AUTO backend selection is not semantic coercion. | Direct metadata model |
| Singular basering model: https://www.singular.uni-kl.de/Manual/4-4/sing_3.htm | Official CAS docs | Basering-centered polynomial objects reinforce explicit finite-domain stamps and mismatched-handle rejection. | Direct finite-domain guard |
| GAP type/family/category model: https://docs.gap-system.org/doc/ref/chap13.html | Official CAS docs | Family/category metadata supports plan-cache keys by algebraic family and semantics, not type alone. | Direct metadata model |
| Oscar/AbstractAlgebra rings: https://docs.oscar-system.org/stable/AbstractAlgebra/ring/ | Official CAS docs | Parent, base ring, exactness, and domain metadata map to inspectable scenario fields. | Direct metadata model |
| Wolfram exactness/precision and finite fields: https://reference.wolfram.com/language/howto/ControlThePrecisionAndAccuracyOfNumericalResults.html and https://reference.wolfram.com/language/ref/FiniteField.html | Official CAS docs | Exact/default versus approximate modes support deterministic default APIs and explicit research metadata. | Adjacent model |
| Macaulay2 rings and numeric types: https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/Macaulay2Doc/html/_basic_springs_spof_spnumbers.html | Official CAS docs | Rings, promotion/lift, exact `ZZ`/`QQ`, and approximate `RR`/`CC` support explicit export/lift boundaries. | Adjacent model |
| Dense Linear Algebra over Word-Size Prime Fields: https://arxiv.org/abs/cs/0601133 | Primary paper | Finite-field GEMM is a real exact-LA kernel, but word-size prime fields exceed current finite-u8 contracts. | Adjacent kernel evidence |
| FFLAS-FFPACK docs: https://linbox-team.github.io/fflas-ffpack/ | Official artifact | `fgemm` is a useful finite-field BLAS reference; factorization, TRSM, rank, determinant, and nullspace are larger workflows. | Adjacent/reference |
| LinBox: https://linalg.org/linbox/linbox/ | Official artifact | Exact LA taxonomy for rank, determinant, solve, Smith form, CRA, Dixon, Wiedemann, and certificates. | Workload/reference |
| Generic Chinese remaindering design: https://arxiv.org/abs/1005.0830 | Primary paper | CRA is a controller/build/reconstruction design surface, not a hidden export detail. | Direct reconstruction input |
| FLINT docs: https://flintlib.org/doc/ | Official artifact | Strong optional CPU oracle for integer, rational, modular, CRT, matrix, and polynomial edge cases. | Optional oracle |
| NTL docs: https://libntl.org/doc/tour-modules.html | Official artifact | Optional semantic/differential oracle for integer, modular, polynomial, and finite-field cases. | Optional oracle |
| Givaro: https://linalg.org/linbox/givaro/ | Official artifact | Prime-field, extension-field, finite-ring, integer, and rational domains clarify RNS8's finite-u8 boundary. | Finite-domain oracle |
| Nemo/Hecke paper: https://arxiv.org/abs/1705.06134 and Nemo matrix docs: https://nemocas.github.io/Nemo.jl/stable/matrix/ | Primary plus official artifact | CAS stack and matrix workflows over `Z`, `Z/nZ`, `Q`, finite fields, and number fields are workload/oracle inputs only. | CAS oracle/source |
| F4: https://doi.org/10.1016/S0022-4049(99)00005-5 | Primary paper | Dense finite-field matrix phases are adjacent scenarios; symbolic preprocessing is not RNS8 functionality. | Adjacent symbolic phase |
| F5: https://doi.org/10.1145/780506.780516 | Primary paper | Signature/controller logic is not a GEMM kernel. | Non-goal |
| FGLM: https://doi.org/10.1006/jsco.1993.1051 | Primary paper | Multiplication-matrix/order-conversion phases can be adjacent dense-LA scenarios with explicit phase boundaries. | Adjacent phase |
| GBLA: https://arxiv.org/abs/1602.06097 | System paper | F4/F5 matrix reduction structure informs scenario labels; sparse/structured reductions are not plain dense GEMM evidence. | Adjacent/non-goal boundary |
| CUMODP: https://www.cumodp.org/ | Official artifact | CUDA modular polynomial, matrix, tree, interpolation, and solver artifacts are workload and port-risk studies. | CUDA translation study |
| Linac: https://arxiv.org/abs/2605.25863 and https://github.com/GDeLaurentis/linac | Primary plus official artifact | CUDA finite-field elimination/RREF is a horizon workload; it is not AMD/RNS8 evidence. | CUDA translation study |
| Magma Groebner handbook: https://magma.maths.usyd.edu.au/magma/handbook/text/62 | Official CAS docs | Dense F4/CUDA notes are phase vocabulary and port-risk input, not RNS8 product proof. | CAS phase classifier |
| M4RI and M4RIE: https://arxiv.org/abs/0811.1714 and https://arxiv.org/abs/1111.6900 | Primary papers | Bit-packed `GF(2)` and extension-field `GF(2^e)` are not finite-u8 `Z/qZ` or `GF(p <= 251)` evidence. | Mismatch guard |
| Sage matrix benchmarks: https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/benchmark.html | Official corpus/docs | Useful shape/domain templates for `ZZ`, `QQ`, finite fields, determinant, rank, and Smith form. | Workload corpus |
| PARI/GP benchmarks: https://pari.math.u-bordeaux.fr/bench.html | Official corpus/docs | Useful CAS-level categories across exact integer, rational, finite-field, polynomial, and matrix-normal-form workloads. | Workload corpus |
| AMD HIP porting guide: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html | Official AMD docs | CUDA artifacts need explicit HIP translation, lane/library/memory review, and target-specific validation. | Port-risk input |

## Domain And Coercion Implications

CAS domain systems point in one direction: RNS8 should preserve explicit
semantic descriptors and reject ambiguity. It should not attempt implicit CAS
coercion.

Recommended scenario and plan metadata:

- `domain_family`: integer, bounded integer, exact-wide integer, finite ring,
  prime field, wrap64 byte limb, rational export, polynomial-lowered scenario,
  or CAS oracle.
- `parent_domain_id`: stable descriptor for the semantic parent/ring/field used
  to generate, compare, or consume the workload.
- `coercion_policy`: none, explicit convert, explicit lift/export, or external
  CAS/oracle coercion. AUTO backend selection must not be recorded as algebraic
  coercion.
- `exactness_mode`: deterministic exact, approximate external oracle,
  probabilistic diagnostic, or needs verification.
- `finite_modulus`, `prime_field_flag`, `extension_degree`,
  `characteristic`, and `composite_modulus_flag`.
- `output_domain` and `reconstruction_mode`: native integer, RNS-current,
  finite-current, wrap-byte-current, rational reconstruction, or external CAS
  export.

The key guardrail is that RNS8 may use CAS metadata to describe a workload, but
should not inherit CAS semantics. In particular, `GF(2^e)`, bit-packed
`GF(2)`, word-size prime fields, approximate numeric promotion, and symbolic
polynomial rings remain unsupported unless future APIs add them explicitly.

## Kernel Decomposition Map

| Phase | Fit | RNS8 stance |
|---|---|---|
| Dense exact integer/RNS GEMM | Direct | Current core. Measure as RNS8 production evidence under explicit contracts. |
| Finite-u8 `Z/qZ` GEMM | Direct | Current finite-ring target for `2 <= q <= 256`. |
| Finite-u8 `GF(p)` GEMM | Direct | Current prime-field target only for explicit prime `p <= 251`. |
| CRT/Garner/MRS export | Direct | Reconstruction backend and lazy export are first-class performance surfaces. |
| CRA/early termination | Adjacent | Useful controller research; deterministic defaults must remain separate. |
| Rank/determinant/solve/nullspace | Adjacent | Dense GEMM may accelerate phases, but RNS8 does not provide solver/certificate semantics today. |
| PLUQ/CUP/PLE/echelon/TRSM | Adjacent | Scenario and future-kernel vocabulary, not current public functionality. |
| F4 dense finite-field matrix phase | Adjacent | Valid scenario when symbolic preprocessing and sparse reduction are timed separately. |
| FGLM multiplication matrices | Adjacent | Scenario vocabulary for explicit dense/sparse LA phases. |
| F5 signature control | Non-goal | Controller logic is not a GEMM kernel. |
| Sparse F4/F5 reductions | Mismatch | Sparse structured elimination is not dense-GEMM evidence. |
| Wiedemann/Lanczos black-box solve | Mismatch | Sparse matvec/Krylov work should not rank dense GEMM backends. |
| Resultants/subresultants | Adjacent/non-goal | Dense Sylvester lowering may be a scenario; full polynomial algorithms are outside scope. |
| NTT/FFT/product/remainder trees | Mismatch | Important algebra phases, but not current RNS8 dense GEMM proof. |
| Polynomial factorization/interpolation | Adjacent/non-goal | Workload source only unless explicit polynomial APIs are added. |
| `GF(2)`, `GF(2^e)`, word-size prime fields | Mismatch | Different representations and semantics than current finite-u8 contracts. |

## Library And Oracle Role Matrix

| Source family | Role in RNS8 | Boundary |
|---|---|---|
| Boost.Multiprecision | Existing exact CPU reference | Production CPU-side reference remains local and deterministic. |
| FLINT, NTL | Optional CPU exact/modular/polynomial oracles | Optional comparison only; do not make hidden dependencies. |
| FFLAS-FFPACK, Givaro | Finite-field/ring reference and performance comparison | Useful for finite semantics and workload taxonomy; does not define RNS8 APIs. |
| LinBox, IML | Exact-LA workflow, certificate, CRA/Dixon/Wiedemann taxonomy | Workload and algorithm references; no solver dependency. |
| Sage, Magma, Maple, Wolfram, Singular, Macaulay2, GAP, Oscar/Nemo/Hecke | CAS semantic oracles, phase classifiers, scenario sources | External CAS semantics must not be implied by RNS8 GEMM evidence. |
| PARI/GP, Normaliz, GAP packages | Workload/corpus/oracle sources | Mostly non-goal except for integer matrix shapes and external comparisons. |
| M4RI/M4RIE | Small-characteristic mismatch guard | Do not treat bit-packed or extension-field results as finite-u8 proof. |
| CUMODP, Linac, Magma CUDA notes, GBLA | GPU/HPC or dense symbolic translation studies | CUDA or CPU-side evidence only until HIP-native AMD proof exists. |

## Scenario Corpus Implications

The CAS pass strengthens the case for scenario-driven benchmarking. RNS8
should report winners by scenario family, not only by matrix shape.

Direct scenario families:

- bounded `i64` and `u64` dense GEMM;
- exact-wide RNS-output GEMM;
- finite-u8 `Z/qZ` and `GF(p <= 251)` GEMM;
- CRT/Garner/MRS export-heavy GEMM;
- residue-current GEMM chains with lazy export;
- repeated-A, repeated-B, repeated-A/B, and many-small grouped GEMMs.

Adjacent scenario families:

- modular rank, determinant, solve, nullspace, inverse, echelon, PLUQ/CUP/PLE,
  and TRSM-like update phases;
- characteristic/minimal polynomial phases;
- deterministic CRT, early-terminated CRA, p-adic/Dixon solve, rational
  reconstruction, redundant residues, and error-detecting CRT;
- Freivalds-style product checks and certificate workflows as diagnostic or
  adjacent exact-LA metadata;
- F4 dense finite-field matrix phases and FGLM multiplication-matrix phases;
- polynomial-matrix and modular-composition lowerings when the dense phase is
  isolated.

Mismatch/non-goal scenario families:

- full CAS workflows;
- full exact-LA package behavior;
- implicit CAS coercion;
- word-size prime fields as finite-u8 proof;
- `GF(2^e)` extension fields and bit-packed `GF(2)`;
- sparse Wiedemann/Lanczos as dense GEMM evidence;
- NTT/FFT/product-tree/remainder-tree/resultant workflows as current dense
  GEMM proof;
- CUDA artifacts as AMD evidence.

Recommended metadata additions for CAS-oriented scenarios:

- source metadata: `source_role`, `cas_system`, `oracle_role`,
  `artifact_lineage`, `license_role`;
- domain metadata: `domain_family`, `parent_domain_id`, `coercion_policy`,
  `algebra_family`, `coefficient_ring`, `finite_modulus`,
  `prime_or_composite`, `characteristic`, `extension_degree`,
  `exactness_mode`;
- phase metadata: `workflow_name`, `phase_label`, `phase_id`,
  `dense_kernel_extracted`, `symbolic_precompute`, `controller_mode`,
  `certificate_mode`, `verification_mode`;
- shape metadata: `shape_signature`, `M`, `N`, `K`, `batch`, `density`,
  `sparsity`, `structure_id`, `structure_declared`, `reuse_profile`;
- RNS8 metadata: `bound_kind`, `prefix_count`, `prefix_budget`,
  `reconstruction_mode`, `output_domain`, `backend`, `target_arch`,
  `toolchain_version`;
- phase timings: pack, raw GEMM, reduction, sparse/controller work,
  reconstruction/export, verification, D2H, and end-to-end.

These are evidence and scenario labels, not new promotion gates.

## GPU And HPC Translation

GPU CAS artifacts are useful for architecture ideas, but none found in this
pass directly validates RNS8's byte-sized RNS/INT8 design on AMD GPUs.

- CUMODP shows CUDA modular polynomial, matrix, subproduct-tree,
  interpolation, and solver workloads. It is a design input and port-risk
  study, not HIP proof.
- Linac shows a modern CUDA finite-field elimination/RREF artifact. Its finite
  fields and NVIDIA stack are useful horizon context, not finite-u8 or AMD
  evidence.
- Magma's dense-F4 CUDA notes are useful phase vocabulary. CAS orchestration
  and CUDA executables must not be folded into RNS8 backend claims.
- GBLA and F4/F5 papers clarify that symbolic matrix reductions can be sparse,
  structured, and controller-heavy. Dense block timing must be isolated before
  using them as dense-GEMM scenarios.
- AMD int8 library or matrix-core capability is only a primitive possibility.
  RNS8 evidence still needs selected kernel metadata, signedness handling,
  reducer/export correctness, and target-specific timing.

## Correctness And Evidence Wording

Use precise wording:

- "RNS8 provides exact dense GEMM under explicit RNS8 semantic contracts."
- "This scenario is adjacent to exact-LA/CAS workflows because it isolates a
  dense modular matrix phase."
- "This external source is a CPU oracle, CAS semantic oracle, algorithm
  reference, benchmark comparison, CUDA translation study, workload source, or
  non-goal."

Avoid:

- "CAS-correct" when RNS8 only proved GEMM;
- "certified exact LA" unless the certificate workflow itself is implemented
  and checked;
- "secure probabilistic verification" for Freivalds, CRA early termination,
  redundant residues, or sampled checks;
- "AMD evidence" for CUDA-only artifacts;
- "finite-field coverage" when the source is `GF(2^e)`, bit-packed `GF(2)`,
  or word-size prime fields outside finite-u8.

Default exact APIs remain deterministic. Probabilistic, bad-prime, redundant
residue, or certificate-style features are research/diagnostic metadata unless
future work adds explicit APIs, semantics, tests, and documentation.

## PGWQ Alignment

The PGWQ already contains the right architecture hooks: scenario corpus,
roofline/evidence database, toolchain matrix, reconstruction backend, lazy
export, finite data specialization, RNS-native chains, plan-level lowering,
verification-cost reduction, and generated kernel search.

Targeted PGWQ refinements from this pass:

- Make CAS domain/coercion metadata explicit in scenario and evidence rows.
- Record external libraries by role, including optional oracle, CAS semantic
  oracle, workload source, CUDA translation study, and non-goal.
- Add CAS workload families around dense-kernel extraction, controller timing,
  symbolic preprocessing, certificate timing, and reconstruction/export timing.
- Strengthen non-goal language for full CAS semantics, implicit coercion,
  extension fields, bit-packed fields, sparse symbolic reductions, and CUDA as
  AMD proof.
- Keep the changes as guidance and prioritization notes only. Do not add new
  gates, proof requirements, public APIs, or dependencies.

## Spec Alignment

No `docs/RNS8_RESEARCH_SPEC.md` mismatch was found.

The current spec already aligns with this pass because it says:

- RNS8 is exact integer GEMM on AMD GPUs, not a CAS runtime.
- Semantics must be explicit and not inferred from C++ type alone.
- `RNS8_FINITE_RING_U8` is `Z/qZ`; `RNS8_FINITE_FIELD_U8` is prime-field
  `GF(p)` for explicit `p <= 251`.
- `Z/2^eZ` is not `GF(2^e)`.
- Strict `mod 2^64` wraparound uses the byte-limb backend.
- Default exact APIs are deterministic.
- External CAS/exact-algebra libraries are optional comparison/oracle sources,
  not production correctness backends.

No spec edit is needed for this alignment pass.

## Unresolved Research Gaps

- No HIP-native CAS or exact-LA artifact was found that directly validates
  RNS8's byte-sized RNS/INT8 design on AMD GPUs.
- No optional CPU/CAS library was built or run as a live differential oracle.
- No CUDA artifact was cloned, hipified, compiled, or benchmarked.
- Linux ROCm and Instinct CAS-adjacent evidence remain separate future work.
- Word-size prime fields, extension fields, bit-packed `GF(2)`, polynomial
  storage, NTT domains, sparse symbolic reductions, rank-profile APIs, solver
  APIs, and certificate APIs remain future or adjacent work, not current RNS8
  contracts.
