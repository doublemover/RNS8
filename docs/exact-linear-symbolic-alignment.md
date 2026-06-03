# Exact Linear Algebra And Symbolic Computation Alignment Notes

Date: 2026-06-03

This note refines the computational-algebra alignment for exact linear algebra
and symbolic computation. It is a targeting document. It does not make RNS8 a
general exact-linear-algebra package, a finite-field BLAS/LAPACK replacement,
or a symbolic computer algebra system.

RNS8 remains an exact integer GEMM and RNS matrix arithmetic library for AMD
GPUs. Its direct product center is persistent residue storage plus
per-modulus `int8 x int8 -> int32` GEMM, bounded `i64/u64`, exact-wide RNS
output, finite-u8 GEMM, and explicit reconstruction/export semantics.

## Research Procedure

This pass used six read-only exploration commander lanes:

- exact dense linear algebra;
- multimodular and rational linear algebra;
- finite-field BLAS;
- symbolic computation kernels;
- structured and polynomial matrices;
- libraries, artifacts, and GPU translation.

Each commander managed its own `gpt-5.3-codex-spark` citation-chain subagents
through `codex exec`, saved raw details under
`temp/exact-symbolic-alignment/`, deduped locally, and returned a compact
synthesis. Some Spark attempts hit context limits or failed to emit durable
output; their incomplete traces were excluded unless the commander could
reconcile the claim against a primary paper or official artifact.

The lead reconciled commander reports against the current RNS8 research spec,
PGWQ, prior computational-algebra alignment, FHE/lattice alignment, and
primary or official sources.

## Executive Findings

- Dense modular GEMM is a credible exact-linear-algebra kernel. It sits under
  finite-field BLAS, blocked elimination, rank, determinant, solve, nullspace,
  characteristic polynomial, and some symbolic dense-reduction workloads.
- RNS8 maps directly to exact dense GEMM and small explicit finite-u8 GEMM. It
  maps only adjacently to full exact-linear-algebra workflows because those
  workflows also need PLUQ/CUP or PLE decomposition, triangular solve, echelon
  form, CRT/CRA controllers, rational reconstruction, and certificate phases.
- Word-size prime fields, extension fields such as `GF(2^e)`, bit-packed
  `GF(2)`, and full FFLAS-style finite-field BLAS remain outside current
  finite-u8 semantics.
- Multimodular and rational linear algebra strongly support treating
  reconstruction as a controller-backed backend: deterministic CRT, early
  terminated CRA, rational reconstruction, p-adic or Dixon solving, and
  fault-tolerant reconstruction are distinct modes.
- Symbolic computation is mostly workload inspiration. Dense F4 over finite
  fields and FGLM multiplication-matrix phases can become adjacent dense
  linear-algebra scenarios, but F5 signature control, sparse symbolic
  reductions, resultants/subresultants, NTT polynomial multiplication, and
  CAS-wide workflows are not dense-GEMM evidence.
- Structured and polynomial matrices need declared structure and domain
  metadata. Toeplitz, Hankel, Sylvester, Cauchy, Popov, Hermite, Smith, and
  polynomial-matrix labels cannot be inferred from dense storage.
- External exact-algebra and CAS libraries are oracle, reference, workload, or
  comparison sources. They are not required production dependencies.
- CUDA exact-algebra artifacts such as CUMODP and Linac are design inputs and
  translation studies until HIP-native kernels are compiled, exactly compared,
  and timed on AMD targets.

## Source-Ranked Evidence Map

### Dense Exact Linear Algebra

| Source | Type | RNS8 implication |
|---|---|---|
| Dense Linear Algebra over Word-Size Prime Fields, Dumas, Giorgi, and Pernet, ACM TOMS 2008 / arXiv `cs/0601133`: https://arxiv.org/abs/cs/0601133 | Primary paper | Dense finite-field GEMM is the closest external match for RNS8's GEMM-centered finite-algebra story, but the paper targets word-size prime fields rather than byte-sized RNS planes. |
| FFLAS-FFPACK docs: https://linbox-team.github.io/fflas-ffpack/ | Official artifact | `fgemm`, `ftrsm`, rank, determinant, solve, nullspace, and characteristic/minimal polynomial validate GEMM as substrate, not as the whole RNS8 product. |
| IML: https://cs.uwaterloo.ca/~astorjoh/iml.html | Official artifact | Dense integer solve, nullspace, certified solve, mod-p rank, determinant, and inverse are useful adjacent workload definitions. |
| Fast Computation of the Rank Profile Matrix and the Generalized Bruhat Decomposition, Dumas, Pernet, and Sultan, arXiv `1601.01798`: https://arxiv.org/abs/1601.01798 | Primary paper | PLUQ/CUP-style rank-profile algorithms are exact-LA phases around GEMM; RNS8 does not currently expose factorization. |
| Elimination-based certificates for triangular equivalence and rank profiles, Dumas, Kaltofen, Lucas, and Pernet, DOI `10.1016/j.jsc.2019.07.013`: https://doi.org/10.1016/j.jsc.2019.07.013 | Primary paper | Certificate phases should be scenario and verification metadata, not new default correctness semantics. |
| Efficient Computation of the Characteristic Polynomial, Dumas, Pernet, and Wan, arXiv `cs/0501074`: https://arxiv.org/abs/cs/0501074 | Primary paper | Characteristic/minimal polynomial scenarios must distinguish dense elimination/Krylov/CRT phases. |
| Wiedemann, Solving Sparse Linear Equations over Finite Fields, DOI `10.1109/TIT.1986.1057137` | Primary paper | Sparse black-box exact LA is matvec/Krylov dominated and should not be ranked as dense GEMM. |
| Freivalds, A probabilistic algorithm for verifying matrix products, DBLP record: https://dblp.org/rec/conf/mfcs/Freivalds79 | Bibliographic primary locator | Product verification is useful for explicit research/debug metadata, not default exact API behavior. |

### Multimodular And Rational Linear Algebra

| Source | Type | RNS8 implication |
|---|---|---|
| Generic design of Chinese remaindering schemes, Dumas, Gautier, and Roch, arXiv `1005.0830`: https://arxiv.org/abs/1005.0830 | Primary paper | CRT/CRA should be modeled as residue computation, controller, and builder. |
| P-adic reconstruction of rational numbers, Wang, Guy, and Davenport, DOI `10.1145/1089292.1089293` | Primary paper | Rational reconstruction needs explicit numerator/denominator bounds and failure states. |
| Dixon, Exact solution of linear equations using p-adic expansions, DOI `10.1007/BF01459082` | Primary paper | Dixon/p-adic solve is a solver mode, not a reinterpretation of bounded CRT GEMM. |
| FLINT `fmpq_mat`, `fmpz_mat`, `fmpz`, and `padic` docs: https://flintlib.org/doc/fmpq_mat.html and https://flintlib.org/doc/fmpz_mat.html | Official artifact | CPU oracle for rational/integer solve, determinant, rank, CRT, p-adic contexts, and deterministic/probabilistic mode separation. |
| NTL docs: https://libntl.org/doc/ZZ.cpp.html and https://libntl.org/doc/mat_ZZ.cpp.html | Official artifact | Strong precedent for bound-driven rational reconstruction and explicit solve/determinant algorithms. |
| LinBox CRA and Dixon solver docs: https://linalg.org/linbox-html/struct_lin_box_1_1_chinese_remainder_sequential.html | Official artifact | Termination, bad-prime handling, certification, and method selection are controller concerns. |
| Maple `IntegerLinearSolve`: https://www.maplesoft.com/support/help/Maple/view.aspx?path=LinearAlgebra%2FModular%2FIntegerLinearSolve | Official artifact | Multiple-prime CRA solve can be fast but may be probabilistic; RNS8 must name such modes explicitly if studied. |
| Abbott, Fault-Tolerant Modular Reconstruction of Rational Numbers, arXiv `1303.2965`: https://arxiv.org/abs/1303.2965 | Primary paper | Fault-tolerant reconstruction is optional resilience research, not a production fast path unless bad residues are modeled. |

### Finite-Field BLAS And Small Characteristic

| Source | Type | RNS8 implication |
|---|---|---|
| Efficient dot product over word-size finite fields, Dumas, arXiv `cs/0404008`: https://arxiv.org/abs/cs/0404008 | Primary paper | Delayed reduction and representation choices should inform reducer scheduling, but word-size fields are not finite-u8. |
| Simultaneous Modular Reduction and Kronecker Substitution, Dumas, Fousse, and Salvy, arXiv `0809.0063`: https://arxiv.org/abs/0809.0063 | Primary paper | Packing and simultaneous reduction are useful finite-u8 research analogues, not direct GPU evidence. |
| Givaro: https://linalg.org/linbox/givaro/ | Official artifact | Optional finite-field and finite-ring semantic oracle. |
| FLINT `nmod_mat` and `fmpz_mod_mat`: https://flintlib.org/doc/nmod_mat.html and https://flintlib.org/doc/fmpz_mod_mat.html | Official artifact | CPU reference for modular matrices over prime and composite moduli. |
| M4RI, Efficient Multiplication of Dense Matrices over `GF(2)`, arXiv `0811.1714`: https://arxiv.org/abs/0811.1714 | Primary paper / artifact | Bit-packed `GF(2)` is semantically adjacent but not RNS8's int8 performance model. |
| M4RIE, arXiv `1111.6900`: https://arxiv.org/abs/1111.6900 | Primary paper / artifact | `GF(2^e)` extension-field linear algebra is not `Z/2^eZ`; current RNS8 finite-u8 does not cover it. |
| FiniteFieldSolve, DOI `10.1016/j.cpc.2024.109171` | System paper | Modern exact solve workloads use finite fields and rational reconstruction; RNS8 should label solve phases without claiming full solver coverage. |

### Symbolic Computation Kernels

| Source | Type | RNS8 implication |
|---|---|---|
| Faugere, A new efficient algorithm for computing Groebner bases (F4), DOI `10.1016/S0022-4049(99)00005-5` | Primary paper | F4 uses symbolic preprocessing and linear algebra; only dense finite-field matrix phases are adjacent to RNS8. |
| Faugere, A new efficient algorithm for computing Groebner bases without reduction to zero (F5), DOI `10.1145/780506.780516` | Primary paper | F5 is a signature/controller algorithm; it is not itself a GEMM workload. |
| Faugere, Gianni, Lazard, and Mora, Efficient Computation of Zero-dimensional Groebner Bases by Change of Ordering, DOI `10.1006/jsco.1993.1051` | Primary paper | FGLM gives adjacent multiplication-matrix/order-conversion scenarios, not default GEMM evidence. |
| Magma Groebner handbook: https://magma.maths.usyd.edu.au/magma/handbook/text/62 | Official artifact | Some CAS modes distinguish dense F4 and even GPU/CUDA dense finite-field work, but this is not AMD/RNS8 evidence. |
| Maple Groebner algorithms: https://www.maplesoft.com/support/help/Maple/view.aspx?path=Groebner%2FBasis_algorithms | Official artifact | Production CAS systems use F4-style methods; domain orchestration remains outside RNS8. |
| Singular manual: https://www.singular.uni-kl.de/Manual/4-4/index.htm | Official artifact | Supports symbolic phase labels such as Groebner and FGLM; not a dense GEMM product target. |
| FLINT `fmpz_poly`: https://flintlib.org/doc/fmpz_poly.html | Official artifact | Resultants, GCDs, and subresultant algorithms are polynomial workflows, usually not dense GEMM. |
| Wolfram resultant and subresultant docs: https://reference.wolfram.com/language/ref/Resultant | Official artifact | Sylvester determinant is one possible lowering, but CAS resultants are broader polynomial algorithms. |

### Structured And Polynomial Matrices

| Source | Type | RNS8 implication |
|---|---|---|
| Giorgi, Jeannerod, and Villard, On the Complexity of Polynomial Matrix Computations, DOI `10.1145/860854.860889` | Primary paper | Polynomial-matrix algorithms can lower to matrix multiplication only after polynomial-domain semantics are explicit. |
| Neiger, Rosenkilde, and Solomatov, Computing Popov and Hermite Forms of Rectangular Polynomial Matrices, DOI `10.1145/3208976.3208988` | Primary paper | Popov/Hermite forms are scenario vocabulary, not current backend functionality. |
| Kaltofen, Krishnamoorthy, and Saunders, Fast Parallel Computation of Hermite and Smith Forms of Polynomial Matrices, DOI `10.1137/0608057` | Primary paper | Parallel/probabilistic polynomial-matrix methods need explicit determinism metadata if studied. |
| Kailath, Kung, and Morf, Displacement Ranks of Matrices and Linear Equations, DOI `10.1016/0022-247X(79)90124-0` | Primary paper | Toeplitz/Hankel/Cauchy-style workloads require declared structure metadata. |
| Borodin and Moenck, Fast Modular Transforms, DOI `10.1016/S0022-0000(74)80029-2` | Primary paper | Product/remainder trees, multipoint evaluation, and interpolation are tree phases, not dense GEMM by default. |
| Brent and Kung, Fast Algorithms for Manipulating Formal Power Series, DOI `10.1145/322092.322099` | Primary paper | Modular composition can produce BSGS matrix scenarios but remains polynomial-domain work. |
| FLINT polynomial matrix docs: https://flintlib.org/doc/fmpz_poly_mat.html | Official artifact | Optional CPU oracle for polynomial matrices; no RNS8 GPU performance implication. |

### Libraries, Artifacts, And GPU Translation

| Source | Role for RNS8 |
|---|---|
| LinBox / IML | Exact-LA workload taxonomy, solve/rank/determinant/nullspace/certificate references. |
| FFLAS-FFPACK / Givaro | Finite-field BLAS and arithmetic references; not evidence that finite-u8 covers word-size fields. |
| FLINT / NTL / Boost.Multiprecision | CPU exact, modular, CRT, rational, polynomial, and finite-field oracle sources. |
| Sage / Singular / Nemo / Magma / Maple / Wolfram | CAS semantic or phase-classification oracles where licensing and availability permit. |
| M4RI / M4RIE | Small-characteristic references and explicit mismatch guards for bit-packed `GF(2)` and extension fields. |
| CUMODP: https://www.cumodp.org/ | CUDA modular dense-matrix/polynomial design input; not AMD evidence. |
| Linac, arXiv `2605.25863`: https://arxiv.org/abs/2605.25863 | Recent CUDA finite-field elimination artifact; useful horizon source, not RNS8 evidence. |
| AMD HIP porting guide: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html | Translation checklist for CUDA assumptions, wave behavior, memory layout, and target-specific validation. |

## Workload Fit Map

### Direct Fit

- Dense exact integer GEMM with bounded `i64/u64` output.
- Exact-wide integer GEMM with persistent RNS output or explicit limb export.
- Small finite-ring GEMM over `Z/qZ` for `2 <= q <= 256`.
- Small prime-field GEMM over `GF(p)` for prime `p <= 251`.
- CRT/Garner/MRS reconstruction for explicit integer outputs.
- Residue-current chained GEMM where no intermediate export is required.

### Adjacent Fit

- Finite-field BLAS workloads where GEMM is the hot kernel but factorization,
  triangular solve, echelon, rank-profile, determinant, or solve phases remain
  outside current RNS8.
- Exact integer or rational solve workflows using multimodular or p-adic
  methods, when RNS8 is only the dense modular GEMM/reconstruction accelerator.
- Characteristic/minimal polynomial workflows where dense linear algebra is
  measured separately from Krylov, CRT, and verification.
- F4 dense finite-field matrix phases and FGLM multiplication-matrix phases,
  when matrix density and domain metadata are explicit.
- Polynomial-matrix and structured-matrix lowerings where dense subproblems
  are extracted and labeled separately from polynomial-domain work.

### Mismatch Or Non-Goal

- Full exact-linear-algebra package functionality: factorization, TRSM,
  echelon, determinant, rank, solve, nullspace, certificates, and black-box
  workflows as public RNS8 promises.
- Word-size prime fields as current finite-u8 coverage.
- `GF(2^e)` extension fields, bit-packed `GF(2)`, and table-driven small
  characteristic algorithms as current RNS8 performance claims.
- Sparse Wiedemann/block-Wiedemann workflows as dense GEMM benchmarks.
- F5 signature control, sparse F4 symbolic reductions, Groebner basis
  orchestration, resultants/subresultants, and CAS-wide workflows as current
  dense GEMM targets.
- NTT/FFT polynomial multiplication and GPU polynomial libraries as covered by
  RNS8 dense GEMM performance evidence.
- CUDA exact-algebra artifacts as AMD HIP evidence.

## RNS8 Targeting Recommendations

### Keep The Product Claim Kernel-Centered

RNS8 should present exact linear algebra and symbolic computation as workload
sources. The credible product claim is a dense exact integer/RNS GEMM backend
with explicit finite-u8 and reconstruction semantics, not a full algebra layer.

### Make Algebra Phases First-Class Scenario Labels

Benchmarks and review reports should distinguish:

- dense modular GEMM;
- PLUQ/CUP/PLE or rank-profile factorization;
- triangular solve and echelon updates;
- CRT/CRA build and reconstruction controller;
- rational reconstruction and p-adic/Dixon solve;
- certificates and verification;
- symbolic preprocessing and sparse reduction;
- product/remainder tree and polynomial-domain phases.

### Keep Determinism Explicit

Default exact APIs stay deterministic. CRA early termination, Freivalds,
Las Vegas Smith/Popov/Hermite routines, probabilistic determinant/rank modes,
and fault-tolerant reconstruction are useful research or diagnostic modes only
when metadata records probability, seed, modulus set, repetition count,
failure semantics, and verification context.

### Preserve Finite Semantics

Finite-u8 evidence should stay tied to its explicit contracts:

- `RNS8_FINITE_RING_U8`: `Z/qZ` for `2 <= q <= 256`;
- `RNS8_FINITE_FIELD_U8`: prime fields `GF(p)` for `p <= 251`;
- unsupported today: word-size prime fields, extension fields, bit-packed
  `GF(2)`, and full finite-field BLAS.

### Treat CAS And GPU Artifacts As Roles

External sources should be recorded by role: CPU exact oracle, CAS semantic
oracle, finite-field reference, algorithm reference, benchmark comparison,
CUDA translation study, or explicit non-goal. Optional library discovery must
not become production-backend dependency or evidence of GPU correctness.

## PGWQ Alignment

The PGWQ already contains the right architectural hooks. This pass mainly
sharpens scenario and phase language:

- Add exact-LA phase labels for rank-profile, PLUQ/CUP/PLE, triangular solve,
  echelon, determinant, inverse, solve, nullspace, characteristic/minimal
  polynomial, certificate, and Freivalds verification.
- Add multimodular/rational labels for deterministic CRT, early-terminated
  CRA, p-adic/Dixon solve, rational reconstruction, and fault-tolerant
  reconstruction.
- Add symbolic labels for F4 sparse, F4 dense finite-field, F5 signature
  control, FGLM, resultants/subresultants, Sylvester determinant, NTT
  polynomial multiplication, and CAS modular reconstruction.
- Add structured/polynomial labels for polynomial matrix multiplication,
  Popov, Hermite, Smith, Toeplitz, Hankel, Cauchy, displacement-rank,
  product tree, remainder tree, interpolation, and modular composition.
- Track symbolic preprocessing, controller time, certificate time, tree setup,
  and reconstruction/export separately from raw GEMM time.
- Keep all of these labels as scenario/evidence language unless a future
  implementation adds explicit APIs and tests for the corresponding algebra
  phase.

## Spec Alignment

No spec mismatch was found in this pass.

The current spec already says:

- finite-u8 field semantics are prime-field only;
- `Z/2^eZ` is not `GF(2^e)`;
- rational reconstruction is an optional separate export surface;
- default exact APIs do not use probabilistic early termination;
- external exact-algebra libraries are optional comparison dependencies.

No `docs/RNS8_RESEARCH_SPEC.md` change is needed for this alignment pass.

## Unresolved Research Gaps

- No HIP-native exact-linear-algebra or symbolic-computation artifact was found
  that directly validates RNS8's byte-sized RNS/INT8 GEMM design on AMD GPUs.
- No commander built or ran optional CPU libraries as live differential
  oracles.
- No CUDA artifact was cloned, hipified, compiled, or benchmarked.
- Word-size prime fields, extension fields, modular factorization/TRSM,
  rank-profile/echelon APIs, determinant/solve/nullspace APIs, polynomial
  storage, NTT domains, and sparse matvec remain future or adjacent work.
