# Computational Algebra Alignment Notes

This note aligns RNS8 with computational-algebra practice. It is a targeting
document, not a claim that RNS8 is a general computer algebra system. RNS8
remains an exact integer GEMM and RNS arithmetic library whose core production
primitive is persistent residue storage plus per-modulus `int8 x int8 -> int32`
GEMM on AMD GPUs.

## Research Procedure

This note was produced from a multi-pass source review covering exact linear
algebra, finite-field and modular BLAS, CRT/RNS reconstruction, polynomial and
number-theoretic algebra, algebraic matrix algorithms, and GPU translation
artifacts. Claims below are reconciled against current RNS8 docs and primary or
official sources. Raw notes stayed under ignored `temp/` paths.

## Executive Findings

- Dense modular GEMM is a real computational-algebra kernel. FFLAS-FFPACK,
  LinBox, IML, FLINT, and related systems use modular matrix kernels beneath
  rank, determinant, solve, nullspace, characteristic polynomial, and rational
  reconstruction workflows.
- RNS8 maps directly to exact dense GEMM and to explicit small finite rings or
  prime fields that fit `uint8_t` storage. It does not currently map directly
  to word-size prime fields, extension fields such as `GF(2^e)`, or
  FFLAS-style finite-field BLAS beyond GEMM.
- CRT/Garner/MRS reconstruction is not a minor output detail. For exact algebra
  workloads, reconstruction is a backend/controller problem with constants,
  state, prefix policy, status handling, and optional rational reconstruction
  surfaces.
- Polynomial and number-theoretic algebra are mostly architectural inspiration
  for RNS8, not direct product fit. NTT/FFT, product trees, remainder trees,
  interpolation, PRS/subresultants, and polynomial matrix algorithms should
  appear as scenario labels and lowering vocabulary, not as evidence that the
  current GEMM core replaces NTT kernels.
- Strassen, Winograd, rectangular tensor algorithms, sparse black-box methods,
  and structured matrix paths should stay workload-backed research lanes. They
  must not displace tuned classical dense GEMM unless end-to-end evidence
  includes extra additions, temporary storage, prefix inflation, pack,
  reduction, export, and verification overhead.
- External computational-algebra libraries are oracles, references, and
  benchmark context. They should not become required correctness backends or
  production dependencies.
- CUDA computational-algebra artifacts such as Linac and CUMODP are useful
  design and risk inputs. They are not AMD evidence until HIP-native code is
  compiled, exactly compared, and measured on the target device family.

## Source-Ranked Evidence Map

### Exact Linear Algebra And Modular BLAS

| Source | Type | RNS8 implication |
|---|---|---|
| Dense Linear Algebra over Word-Size Prime Fields, Dumas, Giorgi, and Pernet, ACM TOMS 2008 / arXiv `cs/0601133`: https://arxiv.org/abs/cs/0601133 | Primary paper | Dense finite-field GEMM is the closest external match for RNS8's GEMM-centered direction, but the target is word-size prime fields rather than byte-sized RNS planes. |
| FFLAS-FFPACK official docs: https://linbox-team.github.io/fflas-ffpack/ | Official artifact | `fgemm`, triangular solve, rank, determinant, nullspace, characteristic polynomial, and minimal polynomial validate GEMM as a substrate, not as the whole product. |
| LinBox official docs: https://linalg.org/linbox/linbox/ | Official artifact | Exact algebra users need solve, rank, determinant, Smith/Frobenius form, rational reconstruction, sparse/structured support, and black-box workflows. |
| Efficient Computation of the Characteristic Polynomial, Dumas, Pernet, and Wan, arXiv `cs/0501074`: https://arxiv.org/abs/cs/0501074 | Primary paper | Dense small-field and integer characteristic-polynomial algorithms use early termination and Chinese remaindering, supporting explicit deterministic/probabilistic separation. |
| Integer Matrix Library (IML): https://cs.uwaterloo.ca/~astorjoh/iml.html | Official artifact | Rational solve, nullspace, certified solve, mod-p rank/determinant/inverse are useful workload targets beyond GEMM. |
| FiniteFieldSolve, 2024, DOI `10.1016/j.cpc.2024.109171` | Primary/system paper | Modern computational science still uses exact finite-field/rational solving, so RNS8 should model solve workloads instead of only square GEMM. |

### Finite Fields And Small-Characteristic Algebra

| Source | Type | RNS8 implication |
|---|---|---|
| Givaro official repo: https://github.com/linbox-team/givaro | Official artifact | Useful finite-field and finite-ring semantic oracle for CPU comparison and test design. |
| Efficient dot product over word-size finite fields, Dumas, arXiv `cs/0404008`: https://arxiv.org/abs/cs/0404008 | Primary paper | Delayed modular reduction and representation choices are central to finite-field performance. |
| Efficient Multiplication of Dense Matrices over `GF(2)`, Albrecht, Bard, and Hart, DOI `10.1145/1644001.1644010`: https://arxiv.org/abs/0811.1714 | Primary paper / official repo | Bit-packed `GF(2)` algorithms are a different performance model from RNS8's `mod 2` int8 GEMM. |
| M4RIE, Albrecht, DOI `10.1145/2442829.2442838`: https://arxiv.org/abs/1111.6900 | Primary paper / official repo | `GF(2^e)` extension-field linear algebra is not `Z/2^eZ`; current RNS8 finite-u8 does not implement extension fields. |
| FLINT docs: https://flintlib.org/doc/ | Official artifact | `nmod_mat`, finite-field matrices, integer matrices, and CRT routines are CPU oracle and comparison sources. |
| NTL official docs: https://www.libntl.shoup.net/ | Official artifact | Portable integer, finite-field, polynomial, vector, and matrix reference semantics. |

### CRT, RNS, And Reconstruction

| Source | Type | RNS8 implication |
|---|---|---|
| The Residue Number System, Garner, DOI `10.1109/TEC.1959.5219515` | Primary paper | Mixed-radix/Garner reconstruction supports RNS8's bounded export strategy. |
| Generic design of Chinese remaindering schemes, Dumas, Gautier, and Roch, arXiv `1005.0830`: https://arxiv.org/abs/1005.0830 | Primary paper | Reconstruction should be modeled as residue computation plus controller plus builder, not as a single opaque export step. |
| FLINT `fmpz` and `fmpz_mat` CRT docs: https://flintlib.org/doc/fmpz.html and https://flintlib.org/doc/fmpz_mat.html | Official artifact | Precomputed multi-CRT and multimodular matrix routines are practical CPU references for repeated reconstruction. |
| CoCoALib numeric theory docs: https://cocoa.altervista.org/cocoalib/doc/html/NumTheory.html | Official artifact | Stateful CRT and rational reconstruction should remain explicit APIs, not hidden semantics. |
| P-adic reconstruction of rational numbers, Wang, Guy, and Davenport, DOI `10.1145/1089292.1089293` | Primary paper | Rational reconstruction is important for algebra consumers but separate from bounded integer export. |
| Chinese Remaindering with Errors, Goldreich, Ron, and Sudan, ePrint `1999/002`: https://eprint.iacr.org/1999/002 | Primary paper | Redundant/check residues can inform diagnostic modes, but not replace deterministic exactness. |

### Polynomial And Number-Theoretic Algebra

| Source | Type | RNS8 implication |
|---|---|---|
| The Fast Fourier Transform in a Finite Field, Pollard, DOI `10.1090/S0025-5718-1971-0301966-0` | Primary paper | Finite-field FFT/NTT is central to polynomial multiplication, distinct from dense GEMM. |
| Accelerating NTT for Bootstrappable Homomorphic Encryption on GPUs: https://arxiv.org/abs/2012.01968 | Primary GPU paper | GPU NTT bottlenecks are transform layout, twiddle/root handling, modular reduction, and memory movement. |
| Accelerating Polynomial Multiplication for Homomorphic Encryption on GPUs: https://arxiv.org/abs/2209.01290 | Primary GPU paper | Polynomial acceleration is NTT/Hadamard/reduction dominated, not dense-GEMM dominated. |
| NTTSuite: https://arxiv.org/abs/2405.11353 | Benchmark paper | Polynomial benchmark corpora should classify NTT algorithm, batch, device, and layout. |
| NTL `ZZ_pX` docs: https://libntl.org/doc/ZZ_pX.cpp.html | Official artifact | NTL polynomial arithmetic uses FFT/CRT and modular-composition routines, reinforcing transform and reconstruction themes. |
| FLINT `fmpz_poly` docs: https://flintlib.org/doc/fmpz_poly.html | Official artifact | Resultants, GCDs, and polynomial exact algorithms are multimodular/euclidean workloads. |
| Polynomial Matrix Computations, Giorgi, Jeannerod, and Villard, DOI `10.1145/860854.860889` | Primary paper | Polynomial-matrix work can lower to matrix multiplication, but only after polynomial-domain structure is explicit. |
| Fast Modular Transforms, Borodin and Moenck, DOI `10.1016/S0022-0000(74)80029-2` | Primary paper | Product/remainder trees, multipoint evaluation, and interpolation deserve scenario labels separate from dense GEMM. |

### Algebraic Matrix Algorithms And Structures

| Source | Type | RNS8 implication |
|---|---|---|
| Gaussian Elimination is Not Optimal, Strassen, DOI `10.1007/BF02165411` | Primary paper | Fast matrix multiplication is legitimate research, but extra additions and temporaries must be measured in RNS8's exact pipeline. |
| On multiplication of `2 x 2` matrices, Winograd, DOI `10.1016/0024-3795(71)90009-7` | Primary paper | Reduced multiplication count can increase additive traffic, bounds, and prefix requirements. |
| Exploiting Fast Matrix Multiplication within Level 3 BLAS, Higham, DOI `10.1145/98267.98290` | Primary paper | Practical thresholds matter more than asymptotic appeal. |
| Solving Sparse Linear Equations over Finite Fields, Wiedemann, DOI `10.1109/TIT.1986.1057137` | Primary paper | Sparse/black-box algebra is matvec/Krylov dominated and should not be ranked as dense GEMM. |
| Displacement ranks of matrices and linear equations, Kailath, Kung, and Morf, DOI `10.1016/0022-247X(79)90124-0` | Primary paper | Structured matrices need explicit metadata; structure cannot be inferred from dense storage. |
| FLINT applications page: https://flintlib.org/applications.html | Official artifact | Integer bit-width, determinant, rank, and polynomial workloads supply useful scenario shapes. |

### Libraries, Artifacts, And GPU Translation

| Source | Role for RNS8 |
|---|---|
| FFLAS-FFPACK / Givaro | Algorithm reference and optional CPU comparison for finite fields and rings. |
| FLINT / NTL / Boost.Multiprecision | CPU oracle and differential comparison for integer, modular, finite-field, and polynomial edge cases. |
| LinBox / IML | Exact-linear-algebra workload definitions for solve, rank, determinant, nullspace, and certificates. |
| Sage / Nemo / AbstractAlgebra / Magma | CAS-level semantic oracle where licensing/runtime constraints permit offline comparison. |
| M4RI / M4RIE | Specialized small-characteristic references; useful to avoid overclaiming `mod 2` and `mod 2^e`. |
| Linac / CUMODP | CUDA finite-field and modular-arithmetic artifacts; design inputs only until HIP-native evidence exists. |
| AMD HIP porting docs: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html | Translation risk checklist for warp-size assumptions, CUDA-only libraries, and target-specific validation. |

## Workload Fit Map

### Direct Fit

- Dense exact integer GEMM with bounded `i64/u64` output.
- Exact-wide integer GEMM where output remains RNS or is explicitly exported as
  fixed-width limbs.
- Small explicit finite-ring GEMM over `Z/qZ` for `2 <= q <= 256`.
- Small explicit prime-field GEMM over `GF(p)` for prime `p <= 251`.
- Batched dense GEMM, repeated-B, repeated-A/B, and residue-current chains.
- CRT/Garner/MRS reconstruction for integer outputs.

### Adjacent Fit

- Modular rank, determinant, solve, nullspace, and characteristic/minimal
  polynomial routines that reduce to dense finite-field GEMM plus
  factorization, triangular solve, CRT/CRA, rational reconstruction, and
  certificates.
- Polynomial matrix multiplication and modular-composition lowerings where the
  lowered matrix kernels use compatible moduli and dense shapes.
- Product-tree, remainder-tree, and batched reconstruction workloads where
  RNS8 can benchmark layout and grouped scheduling without claiming a
  polynomial backend.
- Freivalds verification and CRA early termination as explicit research or
  validation modes with recorded probability metadata.

### Mismatch Or Non-Goal

- Word-size prime fields as a current direct API target. They need a separate
  finite backend or explicit multimodular lowering.
- `GF(2^e)` extension fields. `Z/2^eZ` ring arithmetic is not extension-field
  arithmetic.
- Bit-packed `GF(2)` and table-driven `GF(2^e)` kernels as current performance
  claims.
- NTT/FFT polynomial multiplication, product/remainder trees, interpolation,
  subresultants, and resultants as default dense-GEMM workloads.
- Sparse black-box Wiedemann/Lanczos workloads as dense GEMM benchmarks.
- Strassen/Winograd/tensor algorithms as production strategy without
  end-to-end RNS8 evidence.
- CUDA exact-algebra artifacts as AMD evidence.

## RNS8 Targeting Recommendations

### Keep The Product Statement Narrow

RNS8 should describe itself as exact integer GEMM and RNS matrix arithmetic for
AMD GPUs. Computational algebra is a source of workloads and algorithmic
pressure, not a reason to rebrand RNS8 as a CAS, FHE runtime, polynomial
library, or full exact-linear-algebra package.

### Treat GEMM As A Kernel, Not The Whole Algebra Layer

The most credible computational-algebra opportunity is to become a high-quality
modular dense GEMM backend. Rank, determinant, solve, nullspace, and rational
reconstruction require additional algebra phases. PGWQ scenario labels should
therefore record the algebra family and phase, not only `M/N/K`.

### Make Reconstruction Backend-Visible

RNS8 should keep optimizing reconstruction as a first-class backend:

- prefix-9 and prefix-20 Garner/MRS kernels;
- exact-wide limb-count specializations;
- residue-current no-export outputs;
- batched CRT for many small or export-heavy workloads;
- product-tree/balanced CRT only where measured workloads justify setup cost;
- optional rational reconstruction as an explicit computational-algebra export.

### Keep Finite Semantics Explicit

Finite-u8 paths are useful for small computational-algebra experiments, but the
docs and benchmarks must distinguish:

- `Z/qZ` rings, including composite moduli;
- `GF(p)` prime fields for prime `p <= 251`;
- unsupported `GF(2^e)` extension fields;
- unsupported word-size prime fields;
- optional external CPU/CAS oracles.

### Expand Scenario Metadata

Scenario and evidence records should eventually include:

- `algebra_family`: dense-gemm, finite-blas, crt-export, rational-reconstruct,
  rank, determinant, solve, nullspace, polynomial-matrix, ntt-pressure,
  product-tree, remainder-tree, modular-compose, structured, black-box-sparse;
- `structure_id`: none, diagonal, triangular, banded, Toeplitz, Hankel,
  block-sparse, low-rank, Gram, Sylvester, polynomial-matrix;
- `domain`: bounded-integer, exact-wide, wrap64, finite-ring, prime-field,
  extension-field-unsupported, polynomial-domain-unsupported;
- `field_or_ring_metadata`: modulus, characteristic, extension degree,
  irreducible polynomial when applicable, and currentness/domain state;
- `shape_signature`: square, rectangular, GEMV/skinny, rank-k, block update,
  many-small, repeated-B, repeated-A/B;
- `reconstruction_profile`: prefix, limb count, product-tree setup, status
  mode, rational reconstruction mode;
- `determinism_mode`: deterministic exact, debug verification, probabilistic
  research, or unsupported.

### Keep External Artifacts In Their Lane

FFLAS-FFPACK, LinBox, FLINT, NTL, Givaro, Sage, Nemo, Magma, M4RI, and M4RIE
are useful for semantic comparisons, workload shapes, and source citation. They
should remain optional. Linac, CUMODP, and CUDA NTT systems should be studied
for ideas, but every borrowed idea has to be translated into HIP, compiled,
exactly compared, and measured on the intended AMD target before it becomes
RNS8 evidence.

## Exact Linear Algebra And Symbolic Follow-Up

A second alignment pass focused specifically on exact linear algebra and
symbolic computation is recorded in
[exact-linear-symbolic-alignment.md](exact-linear-symbolic-alignment.md). Its
main correction is precision, not a product-direction change:

- PLUQ/CUP/PLE rank-profile algorithms, triangular solve, echelon recovery,
  determinant, inverse, solve, nullspace, characteristic/minimal polynomial,
  and certificate workflows are adjacent exact-LA phases around dense modular
  GEMM, not current RNS8 public functionality.
- Multimodular and rational workflows should be described through explicit
  controller modes: deterministic CRT, early-terminated CRA, rational
  reconstruction, p-adic/Dixon solve, and fault-tolerant reconstruction.
- Symbolic computation should be classified by phase. Dense F4 finite-field
  matrices and FGLM multiplication matrices can be adjacent dense-LA scenarios;
  sparse F4, F5 signature control, resultants/subresultants, NTT polynomial
  multiplication, and CAS-wide workflows are not dense-GEMM evidence.
- Structured and polynomial-matrix workloads need declared structure and
  polynomial-domain metadata before any dense-GEMM lowering claim.

## Computer Algebra Systems Follow-Up

A CAS-focused follow-up is recorded in
[computer-algebra-systems-alignment.md](computer-algebra-systems-alignment.md).
It keeps the same product boundary and sharpens how external CAS evidence
should be used:

- CAS systems are semantic, orchestration, oracle, and workload ecosystems.
  They can provide domain/coercion vocabulary, phase labels, benchmark shapes,
  and external comparison outputs; they should not define RNS8 product scope.
- RNS8 scenario metadata should record domain family, parent/domain id,
  coefficient ring, finite modulus, prime/composite status, extension degree,
  coercion/export policy, exactness mode, source role, and oracle role.
- AUTO backend selection is not algebraic coercion. RNS8 should continue to
  reject ambiguous finite-ring, prime-field, exact-wide, wrap64, and external
  CAS semantics unless a caller supplies an explicit contract.
- CUDA CAS artifacts such as CUMODP, Linac, and Magma dense-F4 notes are
  translation studies until HIP-native AMD evidence exists.

## PGWQ Alignment

The current PGWQ already has the right architecture hooks: adaptive prefix
minimization, reconstruction backend, residue-channel fusion, grouped
scheduling, epilogue DSL, plan-level lowering, scenario corpus, evidence
database, and finite data specialization.

Recommended refinements are targeted:

- Add computational-algebra scenario families to the scenario corpus.
- Add finite-field semantic distinctions to finite-u8 and plan metadata.
- Add rational reconstruction and product-tree CRT as explicit reconstruction
  research surfaces, not hidden export behavior.
- Add external library roles as oracle/reference/comparison/non-goal.
- Add polynomial-domain vocabulary to plan-level lowering while marking it
  unsupported until a real polynomial backend exists.
- Add prefix-inflation accounting for Strassen/Winograd-style experiments.
- Keep sparse/black-box and structured matrix paths out of dense baseline
  ranking unless the workload explicitly declares that structure.

## Spec Alignment

The existing RNS8 spec is mostly aligned. The useful clarifications are:

- `RNS8_FINITE_FIELD_U8` is a prime-field `GF(p)` contract for explicit prime
  `p <= 251`; it is not an extension-field contract.
- `RNS8_FINITE_RING_U8` with `q = 2^e` is `Z/2^eZ`, not `GF(2^e)`.
- Rational reconstruction is an optional computational-algebra export surface
  if added; it must not define or reinterpret bounded integer GEMM, exact-wide
  integer export, finite-u8 output, or strict wrap64 semantics.
- Redundant residues and error-detecting CRT are research/diagnostic modes,
  not replacements for deterministic exactness.

## Unresolved Research Gaps

- No primary source surfaced an AMD HIP implementation that directly matches
  RNS8's byte-sized RNS / INT8 GEMM design. RNS8 remains novel there.
- No Linux ROCm or Instinct computational-algebra evidence was gathered in this
  documentation pass.
- No optional libraries were built or used as live differential oracles.
- No CUDA artifact was cloned, hipified, or tested.
- Word-size-prime finite fields, extension fields, polynomial storage, NTT
  domains, sparse SpMV, modular triangular solve, rank-profile, and determinant
  APIs remain future or adjacent work, not current RNS8 contracts.
