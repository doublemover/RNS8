# FHE And Lattice Alignment Notes

Date: 2026-06-03

This note aligns RNS8's performance roadmap with current lattice-crypto and
homomorphic-encryption practice. It is a targeting document, not a claim that
RNS8 is an FHE library or that its arithmetic kernels establish cryptographic
security.

## Research Procedure

The alignment pass combined a source sweep across RNS arithmetic and schemes,
GPU FHE kernels, exact-GEMM bridging, library and parameter corpora, AMD
microarchitecture, and correctness/security/evidence practice. Final
conclusions below are reconciled against the current README, research spec,
roadmap status, PGWQ, and primary or official sources.

## Executive Findings

- FHE implementations are deeply RNS-oriented, but their hottest arithmetic is
  usually polynomial arithmetic over large NTT-friendly coefficient moduli:
  NTT/INTT, base extension, key switching, rotations/automorphisms, rescale or
  modulus switching, bootstrapping DFTs, and coefficientwise products.
- RNS8's persistent residue storage, explicit semantic contracts,
  reconstruction/lazy-export roadmap, grouped scheduler, epilogue DSL,
  finite-modulus specialization, and evidence database are well aligned with
  the shape of modern FHE systems.
- RNS8's current small-modulus `int8 x int8 -> int32` exact-GEMM core is not a
  native replacement for FHE coefficient-prime NTT arithmetic. CKKS/BFV/BGV
  libraries typically operate over many 30- to 60-bit NTT primes, not only
  byte-sized pairwise-coprime moduli.
- Dense GEMM is an adjacent and conditional opportunity for FHE workloads. It
  can matter for plaintext pre/post processing, batched modular dot products,
  linear-transform lowering, key-switch digit aggregation, and ML workloads
  expressed as encrypted linear layers, but it should not be treated as the
  default bottleneck without scenario evidence.
- The PGWQ should stay focused on exact integer GEMM while adding an
  FHE-derived scenario corpus and clear cautions about what maps directly,
  what maps only through decomposition, and what does not map.

## Source-Ranked Evidence Map

Primary and official sources used for this alignment:

- A Full RNS Variant of Approximate Homomorphic Encryption, Cheon, Han, Kim,
  Kim, and Song, ePrint 2018/931:
  https://eprint.iacr.org/2018/931
- A Full RNS Variant of FV-like Somewhat Homomorphic Encryption Schemes,
  Bajard, Eynard, Hasan, and Zucca, ePrint 2016/510:
  https://eprint.iacr.org/2016/510
- An Improved RNS Variant of the BFV Homomorphic Encryption Scheme, Halevi,
  Polyakov, and Shoup, ePrint 2018/117:
  https://eprint.iacr.org/2018/117
- Implementation and Performance Evaluation of RNS Variants of the BFV
  Homomorphic Encryption Scheme, Al Badawi, Polyakov, Aung, Veeravalli, and
  Rohloff, IEEE TETC 2021:
  https://digitalcommons.njit.edu/fac_pubs/4213/
- Approximate Homomorphic Encryption with Reduced Approximation Error, Kim,
  Papadimitriou, and Polyakov, ePrint 2020/1118:
  https://eprint.iacr.org/2020/1118
- A Full RNS Implementation of RSA, Imbert and Bajard, IEEE TC 2004:
  https://www.lirmm.fr/~imbert/pdfs/rsa_rns_ieeetc_2004.pdf
- New RNS Barrett Algorithms, Garg and Xiao, arXiv 2016:
  https://arxiv.org/abs/1602.01551
- Fast sign detection and partial reconstruction work:
  https://userpages.cs.umbc.edu/phatak/645/supl/phatak-jpdc-2016-rns-sd.pdf
- OpenFHE documentation, repository, and key-switch docs:
  https://openfhe-development.readthedocs.io/en/latest/
  https://github.com/openfheorg/openfhe-development
  https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/modules/pke/pke_keyswitch.html
- Microsoft SEAL repository:
  https://github.com/microsoft/SEAL
- Lattigo repository and package docs:
  https://github.com/tuneinsight/lattigo
  https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/schemes/ckks
  https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/schemes/bgv
- HElib CKKS security guidance:
  https://ibm.github.io/fhe-toolkit-linux/html/helib/md__opt__i_b_m__f_h_e-distro__h_elib__c_k_k_s-security.html
- HEonGPU documentation and paper:
  https://heongpu.readthedocs.io/en/latest/
  https://heongpu.readthedocs.io/en/latest/bootstrapping.html
  https://eprint.iacr.org/2024/1543
- PhantomFHE repository and paper:
  https://github.com/encryptorion-lab/phantom-fhe
  https://eprint.iacr.org/2023/049
- FIDESlib paper and repository:
  https://arxiv.org/abs/2507.04775
  https://github.com/CAPS-UMU/FIDESlib
- Homomorphic Encryption on GPU and GPU key-switching papers:
  https://eprint.iacr.org/2022/1222
  https://eprint.iacr.org/2025/124
- Accelerating Number Theoretic Transformations for Bootstrappable
  Homomorphic Encryption on GPUs:
  https://arxiv.org/abs/2012.01968
- Accelerating Polynomial Multiplication for Homomorphic Encryption on GPUs:
  https://arxiv.org/abs/2209.01290
- NTTSuite: Number Theoretic Transform Benchmarks for Accelerating Encrypted
  Computation:
  https://arxiv.org/abs/2405.11353
- TensorFHE, Cheddar, Gazelle, and faster homomorphic linear transformations:
  https://arxiv.org/abs/2212.14191
  https://arxiv.org/abs/2407.13055
  https://arxiv.org/abs/1801.05507
  https://link.springer.com/chapter/10.1007/978-3-319-96884-1_4
- cuHE, cuFHE, and SoK FHE accelerators:
  https://github.com/vernamlab/cuHE
  https://github.com/vernamlab/cuFHE
  https://arxiv.org/abs/2212.01713
- Homomorphic Encryption Standard v1.1 and security guidelines:
  https://homomorphicencryption.org/wp-content/uploads/2024/08/Homomorphic-Encryption-Standard-v1.1.pdf
  https://homomorphicencryption.org/security-guidelines/
- ROCm Composable Kernel, rocWMMA, hipBLASLt, HIP, RGA, and architecture docs:
  https://rocm.docs.amd.com/projects/composable_kernel/
  https://rocm.docs.amd.com/projects/rocWMMA/en/latest/
  https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
  https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html
  https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
  https://clang.llvm.org/docs/AMDGPUBuiltinReference.html
  https://gpuopen.com/rga/

Recent or secondary research should be treated as horizon scanning until
independently reproduced on target hardware:

- FHECore: Rethinking GPU Microarchitecture for Fully Homomorphic Encryption:
  https://arxiv.org/abs/2602.22229
- Dataflow-Oriented Classification and Performance Analysis of
  GPU-Accelerated Homomorphic Encryption:
  https://arxiv.org/abs/2603.16692
- Sparse Fully Homomorphic Encrypted DNN work using FIDESlib:
  https://arxiv.org/abs/2604.11659

## What Maps Cleanly To RNS8

### Persistent Residue Domains

Modern CKKS, BFV, and BGV implementations avoid repeated conversion out of RNS
where possible. OpenFHE explicitly describes efficient RNS algorithms, and
Lattigo describes optimized arithmetic for power-of-two cyclotomic rings with
RLWE primitives and scheme implementations. This supports RNS8's decision to
make residue-current matrix storage persistent instead of treating RNS as a
temporary BLAS wrapper.

The mapping is still loose. FHE wants persistent polynomial tower storage with
scheme, ring dimension, coefficient/evaluation form, modulus-chain level,
scale, plaintext modulus, Q/P special-prime basis, key-material identity, and
rotation/automorphism state. RNS8's matrix residue storage is an architectural
analogue, not an FHE ciphertext storage format.

RNS8 action:

- Keep `residue-current` and future `next-op` metadata high priority.
- Treat lazy export as a core architecture item, not a convenience feature.
- Add scenario labels for chained residue-domain work even when final native
  export is delayed.
- Use `ntt-current`, `coefficient-current`, `tower-current`,
  `key-material-current`, and `modulus-chain-current` as research vocabulary
  before exposing any public state.

### Base Extension, Reconstruction, And Partial Conversion

RNS CKKS and BFV work spends significant effort on basis switching, modulus
raising, modulus dropping, scaling, and rounding. That validates the PGWQ
focus on reconstruction backends, prefix-specialized CRT/Garner paths,
partial reconstruction, output precision tiering, and proof-aware prefix
selection.

RNS8 action:

- Keep CRT/Garner/MRS export work near the top of the queue.
- Compare full reconstruction, partial sign/range reconstruction, and lazy
  residue output as separate backend decisions.
- Add FHE-inspired scenario phases for `ModUp`, `ModDown`, `BaseExtend`,
  `Rescale`, `GadgetDecompose`, `ExternalProduct`, and key switching.
- Do not collapse export, status, base conversion, and reconstruction into one
  undifferentiated benchmark phase.

### Scheduler And Memory Residency

GPU FHE systems are memory-resident and often memory-bandwidth constrained.
HEonGPU highlights GPU-resident bootstrapping and memory pressure from Galois
keys; FIDESlib emphasizes complete GPU CKKS primitives and OpenFHE
interoperability; PhantomFHE is CUDA-specific and warns about GPU
compatibility. These observations support RNS8's persistent grouped scheduler,
workspace arena, prepack cache, multi-stream overlap, and toolchain-matrix
work.

RNS8 action:

- Make scheduler experiments include many small dependent tasks, not only one
  large square GEMM.
- Add FHE-shaped task tables for many NTTs, many primes, key switching,
  rotations, bootstrapping stages, and repeated key-material reads.
- Keep repeated-A/B cache identity strict and explicit.
- Keep Windows `gfx1100`, Linux ROCm Radeon, and Instinct evidence separate.

### Epilogue Fusion And Reducer Specialization

FHE polynomial arithmetic often alternates transform, modular multiplication,
rescale/mod-switch, key-switch decomposition, and layout changes. GPU NTT
papers emphasize modular reduction variants, twiddle generation, shared-memory
traffic, and fusing adjacent operations such as Hadamard products with NTT
stages.

RNS8 action:

- Continue the shared epilogue DSL for centered reduction, finite canonical
  output, range/status flags, and CRT fragments.
- Add FHE-inspired epilogue vocabulary to internal planning research:
  `Ntt`, `Intt`, `Butterfly`, `Hadamard`, `Barrett`, `Montgomery`,
  `LazyReduce`, `BaseExtend`, `ModDrop`, `Rescale`, `KeySwitchDigit`,
  `ExternalProduct`, and `Automorphism`.
- Treat these as workload-modeling names unless and until RNS8 implements a
  real FHE polynomial backend.

## What Does Not Map Directly

### Small RNS8 Moduli Are Not FHE Coefficient Moduli

RNS8's default ladder uses small pairwise-coprime byte-sized moduli to recover
exact integer GEMM outputs from many `int8 x int8 -> int32` GEMMs. FHE
coefficient modulus chains normally use much larger NTT-friendly primes and
ring dimensions chosen for RLWE security, noise budget, transform support, and
scheme depth. The Homomorphic Encryption Standard frames parameter selection
around ring dimension, ciphertext modulus, error distribution, and secret
distribution, not just arithmetic throughput.

RNS8 implication:

- Do not claim that RNS8's current modulus ladder is an FHE parameter set.
- Treat FHE alignment as workload shape guidance and arithmetic-kernel
  inspiration, not as cryptographic compatibility.
- If future FHE-polynomial work is added, it needs separate coefficient-prime,
  NTT-root, security-level, scale/error, and scheme-parameter metadata.

### Dense GEMM Is Not The Default FHE Kernel

The strongest GPU FHE evidence points toward NTT/INTT, key switching,
rotations, bootstrapping, and coefficientwise modular arithmetic. Dense GEMM
can appear after algorithmic lowering, especially in encrypted inference and
linear transforms, but FHE schemes usually express SIMD slot operations through
rotations and plaintext/ciphertext products rather than conventional dense
matrix multiplication over native plaintext arrays.

RNS8 implication:

- The PGWQ should not imply that faster dense exact GEMM automatically means
  faster CKKS/BFV/BGV.
- Add FHE-derived scenarios that explicitly distinguish dense-GEMM-like
  lowering from NTT/key-switch dominated paths.
- Use TensorFHE, Cheddar, Gazelle, and homomorphic-linear-transform work as
  evidence that matrix-engine and linear-algebra ideas can matter, not as
  evidence that current RNS8 dense GEMM already accelerates FHE primitives.
- Prioritize shape evidence before moving FHE-motivated items up the queue.

### RNS8 Correctness Is Not FHE Security

RNS8 can prove exact arithmetic against CPU references. It does not prove
IND-CPA, RLWE security, noise-budget safety, parameter adequacy, key security,
decryption safety, side-channel resistance, or fault tolerance for an FHE
scheme. CKKS is approximate arithmetic; exact integer success does not imply
CKKS precision, scale, rescale, or decryption accuracy.

RNS8 implication:

- Keep cryptographic caveats visible in any FHE/lattice-facing docs.
- Do not reuse arithmetic status flags as security status.
- Avoid using "noise budget" for benchmark variance. Use "timing variance" or
  "thermal variance"; reserve "noise budget" for FHE ciphertext correctness.
- If secret-dependent FHE kernels ever enter scope, constant-time and
  side-channel review becomes a separate security project.

## FHE-Derived Scenario Corpus

The PGWQ scenario corpus should add FHE/lattice-inspired cases as workload
proxies. These should not become proof gates; they are ranking tools for the
existing queue.

### Parameter Fixtures

- SEAL-style ring dimensions and maximum coefficient-modulus bit budgets:
  `N=4096/8192/16384/32768` and representative `109/218/438/881` total
  coefficient-modulus-bit buckets.
- SEAL CKKS chain example: `N=8192` with `{60,40,40,60}` coefficient-modulus
  bits and CKKS slot count `N/2`.
- OpenFHE BFVrns-style fixtures: plaintext modulus such as `65537`, depth 2,
  and small rotation sets such as `{1,2,-1,-2}`.
- OpenFHE CKKS-style fixtures: `scaleModSize=50`, small batch-size examples,
  auto ring dimension, and CKKS `N/2` slot cap.
- Lattigo security buckets: `LogN=12..15`, example CKKS chains
  `[45,40,40,40,40]` and `[35,60,60]`, BGV fixtures with explicit
  `Q/P/LogQ/LogP`, and `PlaintextModulus`.
- HEonGPU/FIDESlib/PhantomFHE GPU corpus fixtures: operation coverage,
  bootstrapping stages, serialization/object-size surfaces, key-material
  residency, and CUDA hardware notes.

### Scenario Families

- NTT/INTT pressure proxy: many independent power-of-two polynomial transforms
  at ring dimensions such as 4096, 8192, 16384, and 32768 with several modulus
  channels.
- Key-switch digit aggregation proxy: repeated decomposition digits, same key
  material reused many times, large read-only B-like operand, and many small
  modular dot products.
- Rotation/automorphism proxy: permutation-heavy data movement followed by
  modular multiply-add work.
- Bootstrapping linear-transform proxy: ModRaise, CoeffToSlot, approximate
  modular reduction or EvalMod, SlotToCoeff, high Galois-key memory pressure,
  and strict separation between arithmetic time and key/material movement.
- CKKS rescale/mod-drop proxy: chained residue-domain operations where only a
  suffix or prefix of modulus channels remains current.
- BFV/BGV exact-integer proxy: modular arithmetic over explicit plaintext
  modulus plus coefficient modulus chain metadata, without claiming current
  finite-u8 paths cover BFV.
- Encrypted-inference linear-layer proxy: repeated plaintext matrix, many
  encrypted vectors or slots, and clear labels for whether the lowering is
  dense GEMM, diagonal/rotation method, MVM/convolution, or coefficientwise
  batched arithmetic.
- Library-interoperability proxy: shapes and metadata inspired by OpenFHE,
  SEAL, Lattigo, HElib, HEonGPU, PhantomFHE, cuHE/cuFHE, and FIDESlib.

RNS8 benchmark metadata should record scenario family, source inspiration,
ring dimension or polynomial degree, `N` or `LogN`, slot count,
coefficient-modulus count, Q/P tower metadata, decomposition digit count,
ciphertext component count, evaluation-key count, transform/current-domain
state, reuse profile, output-domain requirement, and evidence scope.

## CUDA-To-AMD Translation

Most recent GPU FHE systems are CUDA or NVIDIA first. Their architecture
lessons are useful, but their performance numbers are not AMD evidence.

RNS8 translation checklist:

- Keep Windows `gfx1100`, Linux Radeon, and Instinct CDNA evidence separate.
- Do not hardcode CUDA warp assumptions; audit wave32 versus wave64 behavior.
- Use runtime feature queries and target-specific autotune keys.
- Recheck lane masks, launch bounds, LDS use, global-memory coalescing,
  prefetch strategy, and register pressure on RDNA and CDNA separately.
- Treat hipBLASLt as a raw GEMM baseline surrounded by RNS8-owned pack/reduce
  work, not as an FHE-ready fused backend.
- Prefer CK, rocWMMA, direct HIP, and eventually AMDGPU builtins for fused
  residue or FHE-inspired modular kernels when measured bottlenecks justify
  that control.
- Feed HIP event timing, RGA/ISA summaries, VGPR/SGPR/LDS use, occupancy, and
  target id into the evidence database before promoting any platform claim.

## PGWQ Alignment Recommendations

Keep the existing PGWQ ordering, but interpret the following items through the
FHE/lattice lens:

- Adaptive Prefix Minimization: strong RNS8-native win, but do not confuse
  byte-modulus prefix deletion with FHE modulus-chain security selection.
- Reconstruction Backend And Lazy Export: directly aligned with FHE-style
  stay-in-RNS execution, basis conversion, partial reconstruction, and delayed
  host export.
- Residue-Channel Fusion: relevant to multi-modulus FHE channels, key-switch
  decomposition channels, and NTT-stage fusion, but current RNS8 byte-sized
  channels are a different arithmetic domain from large coefficient primes.
- Persistent Grouped Scheduler: highly aligned with key-switch, rotation, and
  bootstrapping workloads where many dependent modular tasks and key-material
  reads dominate.
- Shared Epilogue DSL: add future vocabulary for NTT-adjacent operations,
  base extension, rescale/mod-drop, Barrett/Montgomery/lazy reducers, and
  key-switch digit processing.
- End-To-End Layout Search: include NTT-current, coefficient-major,
  modulus-major, digit-major, key-major, tower-major, Q/P basis, and
  CRT/export-friendly layouts as research-axis names.
- RNS-Native Chains And Next-Op API: add `ntt-current`,
  `coefficient-current`, `tower-current`, and key-material-current as future
  modeling states, but do not expose them as API states until real polynomial
  kernels exist.
- Plan-Level Algebraic Lowering: the best place to prevent overclaiming dense
  GEMM. The IR should know whether a workload is dense GEMM, diagonal
  method, NTT, INTT, base conversion, key switch, rotation, relinearization,
  bootstrap, or coefficientwise arithmetic.
- Scenario Benchmark Corpus: add FHE/lattice proxies before using FHE claims
  to reorder implementation work.
- Toolchain Matrix: prioritize Linux ROCm and Instinct validation separately
  from Windows `gfx1100`, because production FHE GPU libraries and cluster
  evidence are mostly data-center oriented.

## Recommended Targeting Position

RNS8 should describe itself as an exact integer GEMM and RNS arithmetic system
whose architecture is informed by FHE/lattice systems, not as an FHE runtime.
The strongest near-term targeting is:

1. Use FHE/lattice papers to improve RNS8's residue-domain architecture:
   persistent storage, lazy export, base-conversion thinking, scheduler
   grouping, and scenario design.
2. Use GPU FHE papers to keep attention on memory traffic, transform-like
   data movement, key/reused-operand residency, and fusion of adjacent modular
   stages.
3. Use FHE libraries to derive realistic scenario metadata and reuse patterns.
4. Keep dense exact GEMM claims scoped to exact GEMM unless a workload-specific
   FHE lowering proves dense GEMM is the dominant operation.
5. Keep cryptographic security, noise budget, and parameter selection out of
   RNS8 performance claims unless a future FHE-specific project adds those
   contracts explicitly.
