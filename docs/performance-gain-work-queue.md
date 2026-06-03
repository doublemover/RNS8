# Performance Gain Work Queue

This queue is ordered by expected Windows RX 7900 XTX / `gfx1100`
end-to-end performance value, not ease. Use it to drive implementation slices
from this point forward. Keep evidence claims local to the measured platform:
Windows `gfx1100` evidence does not imply Linux ROCm, Radeon Linux, or
Instinct CDNA readiness.

The central optimization question is no longer only "which backend computes a
single GEMM fastest?" RNS8 has to ask more structural questions:

- How many residue or slice GEMMs can be avoided?
- How many pack, launch, scratch, status, export, and D2H materializations can
  be removed?
- Can the output stay in the domain the next operation needs?
- Can reconstruction be partial, delayed, fused, or skipped?
- Can many small or irregular exact tasks become one persistent grouped
  workload?

## Ground Rules

- Every performance slice needs same-contract CPU/direct-HIP baseline, release
  build, fixed seed, at least 3 warmups, at least 9 repeats, schema validation,
  selected-kernel metadata, and exact CPU differential before promotion.
- Do not promote discovery captures, smoke captures, or Windows evidence into
  Linux or Instinct claims.
- Every new kernel or layout must update `selected_kernel`, `epilogue_mode`,
  `workspace_mode`, `isa_evidence`, autotune key fields, docs, benchmark schema
  fixtures, and stale-cache rejection.

## Current Evidence Snapshot

- `hip-vector-alu-int64` is a real bounded i64/u64 runtime backend and remains
  the reviewed Windows `gfx1100` bounded-u64 leader at 64, 128, 512, and 1024.
  It is bounded-only and must not be generalized into exact-wide, finite, or
  wrap64 semantics.
- Bounded i64 has prior reviewed Windows `gfx1100` winners split by shape:
  rocWMMA wins 512 with `rocwmma_i8_i32_signed_hot_residue_v1`; hipBLASLt wins
  1024 with `hipblaslt_int8_i32_scratch_reduce_baseline_v1`. The latest
  post-fix 512/1024 validation snapshot in
  [performance-wins.md](performance-wins.md) kept 512 on direct HIP and found a
  narrow CK 1024 win, so rerun target shapes before installing durable cache
  policy.
- Adaptive bounded i64 at 1024 has a reviewed rocWMMA winner:
  `rocwmma_i8_i32_signed_tiled_hot_residue_v1`. Tiny adaptive cases and
  bounded-u64 adaptive cases remain blocked by vector/direct baselines.
- Finite-u8 has reviewed Windows `gfx1100` winners across ring 251, ring 255,
  and field 251. rocWMMA wins most 64/128/512 groups, CK wins 1024 ring cases,
  and hipBLASLt wins the 1024 field-251 group.
- Exact-wide reviewed Windows `gfx1100` winners are CK for signed 1024 and
  unsigned 128/512/1024. Other exact-wide reviewed shapes stay on direct HIP.
- Direct HIP `direct_hip_wrap64_byte_gemm36_tiled_2d_v3` remains the measured
  strict wrap64 GPU path. The internal rocWMMA wrap64 candidate matches
  checksums in candidate-inclusive release review but loses to direct HIP at
  every 64/128/512/1024 shape.
- rocWMMA has a narrow runtime reusable B cache for non-tiled RNS plans with
  `K <= 65536`. hipBLASLt now has workspace-local repeated-A and repeated-B
  prepack caches for fixed-prefix single-K-block RNS work. Neither is a broad
  production prepack cache. Current event-valid reuse wins and setup
  break-even points are tracked in [performance-wins.md](performance-wins.md).

## FHE/Lattice Alignment Notes

FHE and lattice-crypto systems are strong workload inspiration for this queue,
but they do not turn RNS8 into an FHE library. Modern CKKS/BFV/BGV
implementations are RNS-heavy and GPU FHE systems are dominated by NTT/INTT,
base conversion, key switching, rotations/automorphisms, rescale or modulus
switching, bootstrapping linear transforms, coefficientwise modular products,
memory residency, and key-material movement. Dense exact GEMM is an adjacent
and conditional opportunity, not the default FHE hot path.

RNS8-specific implications:

- Do not claim that the byte-sized RNS8 modulus ladder is an FHE coefficient
  modulus chain. FHE parameter sets require separate ring-dimension,
  NTT-prime, scale/error, security, and scheme metadata.
- Use FHE/lattice papers to sharpen RNS8 architecture: persistent residue
  domains, lazy export, base-conversion thinking, grouped scheduling,
  reducer/epilogue fusion, and scenario benchmarks.
- Track external-scenario metadata explicitly: `scheme_adapter`,
  `evidence_scope`, `semantic_contract`, `backend_family`, `wave_size`,
  `rocm_or_hip_sdk`, and `validated_on_real_target`.
- Treat CUDA FHE papers and libraries as design input until reproduced on AMD.
  Porting requires wave32/wave64, lane-mask, launch-bound, LDS/register,
  memory-coalescing, target-id, and HIP/ROCm-version audits.
- Add FHE-derived scenario proxies for NTT/INTT pressure, key-switch digit
  aggregation, rotation-heavy linear transforms, CKKS rescale/mod-drop chains,
  BFV/BGV explicit-modulus arithmetic, bootstrapping stages, and
  encrypted-inference linear layers.
- Keep cryptographic security, RLWE parameter selection, noise growth,
  side-channel behavior, and decryption safety out of RNS8 performance claims
  unless a future FHE-specific project adds those contracts explicitly.

See `docs/fhe-lattice-alignment.md` for the source-ranked research synthesis
behind these notes.

## Computational Algebra Alignment Notes

- RNS8 maps most directly to exact dense GEMM and to small explicit finite
  rings or prime fields. It is not a general computer algebra system, a
  polynomial/NTT library, or a full finite-field BLAS/LAPACK replacement.
- Dense modular GEMM is a real kernel beneath exact rank, determinant, solve,
  nullspace, characteristic/minimal polynomial, and rational-reconstruction
  workflows. Those workflows also need factorization, triangular solve,
  CRT/CRA, verification, and certificate phases that RNS8 does not yet expose.
- Exact-linear-algebra phase labels should be precise: PLUQ/CUP/PLE
  rank-profile work, echelon recovery, determinant, inverse, solve, nullspace,
  characteristic/minimal polynomial, Freivalds verification, and certificate
  workflows are adjacent phases around GEMM unless RNS8 implements them
  explicitly.
- Symbolic-computation labels need even stronger boundaries. Dense F4
  finite-field matrices and FGLM multiplication-matrix phases can be adjacent
  dense-LA scenarios; sparse F4, F5 signature control, resultants,
  subresultants, NTT polynomial multiplication, and CAS-wide workflows are not
  dense-GEMM evidence.
- Finite-u8 should distinguish `Z/qZ` rings from prime fields `GF(p)`.
  Extension fields such as `GF(2^e)` and word-size prime fields are not current
  RNS8 contracts.
- Polynomial workloads should be scenario and lowering vocabulary first:
  NTT/FFT, product trees, remainder trees, interpolation, modular composition,
  subresultants, Sylvester matrices, and polynomial matrices do not imply that
  the current dense GEMM core replaces polynomial kernels.
- External libraries such as FFLAS-FFPACK, Givaro, LinBox, FLINT, NTL, Sage,
  Nemo, Magma, M4RI, and M4RIE are oracle/reference/comparison sources, not
  required production backends.
- CUDA exact-algebra artifacts such as Linac and CUMODP are design inputs and
  port-risk studies until HIP-native kernels produce target-specific RNS8
  correctness and timing evidence.
- CAS domain and coercion models should be used as scenario metadata, not
  inherited semantics. Track parent/domain family, coefficient ring, finite
  modulus, prime/composite status, extension degree, exactness mode,
  coercion/export policy, and oracle role explicitly. AUTO backend selection is
  not algebraic coercion.
- Treat CAS systems as workload and oracle ecosystems. Sage, Magma, Maple,
  Wolfram, Singular, Macaulay2, GAP, Oscar/Nemo/Hecke, PARI/GP, and Normaliz
  can classify phases and provide external comparison outputs, but RNS8 should
  not imply CAS-wide correctness, implicit coercion, symbolic orchestration, or
  package-runtime behavior from GEMM evidence.
- Do not use phrases such as "CAS-correct", "certified exact LA", or "secure
  probabilistic verification" for raw GEMM evidence. Certificates,
  Freivalds-style checks, bad-prime handling, CRA early termination, and
  redundant residues are scenario or research metadata unless implemented as
  explicit APIs.

## Ordered Work Items

### 1. Adaptive Prefix Minimization

This is the largest RNS8-native win because it deletes whole residue GEMMs
instead of only making them faster.

Technical direction:

- Move beyond fixed-prefix and coarse per-tile schedules toward the minimum
  exact prefix per tile, tile group, row/column norm class, or repeated
  workload profile.
- Use row absolute sums, column absolute sums, tile max bounds, finite
  histograms, zero density, and caller-provided bounds as plan inputs when the
  scan cost can amortize.
- Add plan metadata that records why a tile group selected a prefix and lets
  benchmark review compare "planes computed" against the fixed-prefix
  baseline.
- Treat modulus order as a performance object. Cheap reducers, high
  contribution moduli, sign/range helper channels, and CRT locality can justify
  a different execution order than the default ladder order, as long as the
  public mathematical prefix remains explicit.

RNS8-specific notes:

- The existing direct-HIP adaptive correctness path already proves tile-local
  prefix scheduling. The next step is to make adaptive selection shape-aware,
  data-profile-aware, and accelerator-friendly.
- Bounded-u64 reviewed release matrices show vector-ALU remains difficult to
  beat. Adaptive prefix work should especially target bounded-i64 and
  exact-wide-ish distributions where plane deletion can beat native vector ALU.
- Exact-wide export should reuse prefix and bound analysis to avoid exporting or
  checking limbs the contract cannot reach.

Likely first slices:

- Add a benchmark workload family for adaptive real distributions, not only the
  default 65x65x64 and 1024x1024x1024 cases.
  Implemented as `rns8-bench --input-profile adaptive-bands` plus
  `tools/benchmark_sweep.py --include-adaptive-workloads`, which adds
  profile-driven 256, 512x1024 rectangular, and 1024 adaptive cases while
  preserving the old uniform default-adaptive cases.
- Add row/column/tile-bound scan timing as explicit benchmark phases.
  Implemented for current per-tile bounded captures as `tile_bound_scan` host
  timing, with schema-required raw timing, summary, phase-order, and
  phase-availability metadata.
- Collapse uniform per-tile schedules back to fixed-prefix dispatch without
  duplicating the fixed-prefix implementation.
  Implemented for no-op per-tile captures where every tile still requires the
  existing full bounded prefix; uniform reduced-prefix schedules remain
  materialized until their tiled dispatch path is independently validated as a
  net win.

Relation to existing queue:

- Deepens "Adaptive Scheduling", "Small Shapes", "AUTO And Cache
  Architecture", and "Exact-Wide".

### 2. Reconstruction Backend And Lazy Export

Treat reconstruction as an algorithmic backend, not merely an export step.
Once GEMM accelerates, CRT/Garner/status/D2H traffic can dominate.

Technical direction:

- Compare direct CRT/Garner, mixed-radix reconstruction, prefix-specialized
  fixed code, partial sign/range reconstruction, and limb-count-specialized
  exact-wide export.
- Compare reconstruction controllers, not only kernels: prefix-specialized
  Garner/MRS, balanced or product-tree batched CRT, residue-current no-export,
  optional rational reconstruction, and diagnostic check-residue modes should
  have separate identities.
- For FHE-derived scenarios, model `ModUp`, `ModDown`, base extension,
  rescale, level drop, Q/P tower movement, and partial sign/range conversion
  as explicit conversion phases rather than generic export.
- Move CRT constants into constant memory, LDS, or compact device tables where
  access patterns justify it.
- Search output layouts for reconstruction, not only for GEMM: residue-major,
  tile-major, cell-major residue vectors, and CRT-friendly staging.
- Add lazy residue output as a first-class plan result when the next operation
  is another RNS operation.
- Batch reconstruction for many small GEMMs when launch overhead dominates.

RNS8-specific notes:

- Direct HIP already performs bounded i64/u64 GPU CRT export with device status
  reporting. This gives a correctness baseline for alternative reconstruction
  kernels.
- Exact-wide reviewed CK winners should be profiled export-first before CK GEMM
  variants are expanded further.
- Lazy export must not blur semantics: bounded native output, exact-wide limbs,
  finite canonical bytes, strict wrap64 low64 output, and residue-current
  output remain distinct plan outcomes.
- Rational reconstruction is useful for computational-algebra consumers, but it
  is an explicit future export surface, not a reinterpretation of bounded,
  exact-wide, finite-u8, or wrap64 outputs.
- Product-tree or balanced CRT should be benchmarked for exact-wide,
  many-output, and many-small workloads, but prefix-9 bounded export should not
  assume it beats prefix-specialized Garner until measured.

Likely first slices:

- Prefix-9 and prefix-20 bounded export specializations.
  Implemented for Direct-HIP fixed-prefix host exports as compile-time
  fixed-prefix CRT reconstruction kernels for prefix 9 and prefix 20, with the
  generic runtime-prefix export kernel retained as fallback for all other
  prefixes. Exact-wide Direct-HIP limb export also dispatches the prefix-20
  fixed-prefix reconstruction kernel.
- Exact-wide 1/2/4/8/16/32 limb export variants with compact D2H staging.
  Implemented benchmark coverage for the limb-count variants via
  `rns8-bench --exact-wide-limbs` and
  `tools/benchmark_sweep.py --include-exact-wide-limb-variants`; this measures
  the exact-wide export path at requested limb widths. Direct-HIP export stages
  and copies `rows * cols * limb_count` limbs, and full-width device exports
  now elide range-status memset/D2H traffic when overflow is structurally
  impossible: signed limb counts 4..32 and unsigned limb counts 3..32.
  Prefix-20 Direct-HIP signed and unsigned export kernels also dispatch
  compile-time fixed limb-count variants for 1/2/4/8/16/32 limbs, with the
  runtime limb-count kernel retained for other widths.
- A residue-current output mode for chained RNS GEMM benchmarks. Implemented as
  exact-wide benchmark/tooling coverage via `rns8-bench --residue-chain-length`
  and `tools/benchmark_sweep.py --residue-chain-length`: measured repeats keep
  intermediate outputs resident in RNS form, report zero per-repeat
  `crt_export`, and run one final untimed host limb export only for checksum
  evidence. This is not yet a public API output-domain mode.
- A batched CRT/reconstruction report that separates kernel time, status
  handling, compact copy time, constants placement, prefix grouping, limb count,
  and tree setup cost.

Relation to existing queue:

- Deepens "Export/Status Overhead", "Exact-Wide", "AUTO And Cache
  Architecture", and "RNS-Native Chains And Next-Op API".

### 3. Residue-Channel Fusion

The current mental model still tends toward one residue plane as one GEMM-like
unit. Breaking that abstraction can remove repeated source loads, pack work,
launches, LDS traffic, and stores.

Technical direction:

- Pack multiple residues from one native load.
- Interleave two or more residues in adjacent INT8 lanes or fragment groups
  where backend mapping allows it.
- Let one CTA own the same C tile for multiple moduli when VGPR/LDS pressure is
  tolerable.
- Pair moduli with compatible reducer families so one epilogue can emit
  multiple residue planes.
- Share A transforms across multiple B residue planes for repeated-B workloads.

RNS8-specific notes:

- This is not simply grouped GEMM. It changes the representation and epilogue
  surface so a single scheduled unit can carry multiple channels.
- Moduli 256, 255, and 251 are the obvious first family because reducers are
  specialized and finite-u8 uses the same constants heavily.
- Exact-wide and bounded prefix-9 both benefit if the first few high-value
  channels are fused.

Likely first slices:

- Multi-modulus pack kernels that load native A/B once and emit two or three
  centered residue planes. Implemented first for direct-HIP fixed-prefix RNS
  packs: bounded prefix 9 and exact-wide prefix 20 now use native-load-once
  pack kernels for signed and unsigned inputs, with the generic per-plane pack
  kernel retained for all other prefixes. This improves the existing pack phase
  without changing persistent RNS matrix semantics.
- A direct-HIP small-shape fused multi-plane baseline before CK/rocWMMA
  variants.
- Autotune key extension for residue group identity and layout.

Relation to existing queue:

- Deepens "Direct-HIP RNS Fallback", "CK Path", "rocWMMA Path", "Finite-u8",
  and "Reusable B Prepack".

### 4. Fused Pack+GEMM

For one-shot, small, and transient-A workloads, global packed intermediates can
cost more than they save.

Technical direction:

- Load native bounded inputs, canonical finite bytes, or resident RNS source
  tiles directly into wave/CTA-local transform code, then feed matrix fragments
  without writing a global packed buffer.
- Materialize only the reused operand for repeated-B or repeated-A workloads.
- Keep transient A in registers or LDS while B uses a reusable backend-native
  layout.
- Add finite-u8 paths that center canonical bytes in registers for moduli
  251/255/256. Direct-HIP finite-u8 pack, resident GEMM reduction, and
  export now dispatch fixed-modulus kernels for 251, 255, and 256; the full
  fused pack+GEMM transient-input path remains open.

RNS8-specific notes:

- hipBLASLt and rocWMMA repeated-B evidence says B reuse matters. Fused
  pack+GEMM should focus first on transient A with reusable B.
- Small 64/128 shapes often lose to setup. Removing pack buffers and launches
  is more useful than expanding heavy matrix-engine kernels there.

Likely first slices:

- Direct-HIP one-shot bounded prefix-9 fused load/compute path.
  Implemented for direct-HIP global bounded i64/u64 one-shot calls whose plan
  resolves to prefix 9: A and B are copied as native int64/uint64 device
  buffers, the grouped prefix-9 direct-HIP kernel centers native inputs inside
  the CTA tile load path, and C is materialized directly as resident RNS
  residues for the existing CRT export path. Per-tile/adaptive plans,
  wider-prefix stress cases, persistent matrix APIs, and non-direct-HIP
  backends keep the established resident pack/GEMM/export route. This is not
  yet a reviewed speedup claim.
- rocWMMA transient-A fused pack against reusable B for non-tiled RNS.
- Benchmark split between one-shot and persistent reuse so wins are not hidden.
  Implemented as `rns8-bench --oneshot` for bounded i64/u64 CPU and
  direct-HIP global-bound captures plus `tools/benchmark_sweep.py
  --include-oneshot` / `--oneshot-only`. One-shot captures use the public
  one-shot API per repeat, report zero external `pack` and `crt_export`
  phases because those costs are folded into the measured API call, emit a
  distinct `benchmark_execution_mode`, and keep direct-HIP one-shot captures
  out of autotune promotion. This is a measurement surface only until release
  captures compare one-shot end-to-end time against persistent direct-HIP,
  vector-ALU, CPU, and accelerator baselines.

Relation to existing queue:

- Deepens "Small Shapes", "Reusable B Prepack", "hipBLASLt Path", and
  "Host/Transfer".

### 5. End-To-End Layout Search

Layout search must optimize the whole pipeline, not just GEMM.

Technical direction:

- Search residue-major, cell-major, tile-major, prefix-major, and interleaved
  residue layouts.
- For FHE/lattice proxies, include polynomial-tower layouts: coefficient-major,
  NTT-domain, modulus-major, digit-major, key-switch-key-major, Q/P-basis,
  chain-level, and automorphism-friendly permutation layouts.
- Search B cache layouts: backend-native, residue-swizzled, K-blocked,
  modulus-interleaved, and repeated-B column-major panels.
- Search output layouts for CRT/export: residue planes, cell-major residue
  vectors, compact D2H staging, and limb-major exact-wide export.
- Include compressed residue blocks where `int8_t` planes waste bandwidth due
  to padding or metadata rather than arithmetic.

RNS8-specific notes:

- `rns_i8_modulus_major_v2`, `rns_i8_tile_swizzled_b_v1`,
  `finite_u8_centered_plane_v2`, `wrap64_byte_limb_gemm36_v2`, and
  `rns_i4_packed_v0` should be treated as one layout family map, not isolated
  experiments.
- A layout that slows raw GEMM slightly can still win if it removes export or
  pack traffic.

Likely first slices:

- A layout matrix document section in the benchmark reports.
- Separate pack A, pack B, raw GEMM, residue store, CRT/export, D2H, and
  end-to-end rows for each layout.
- First compare current modulus-major, rocWMMA B cache, and a CRT-friendly
  output staging layout on 512/1024 bounded i64 and exact-wide.

Relation to existing queue:

- Deepens "Reusable B Prepack", "Finite-u8", "Exact-Wide", "Wrap64 Direct-HIP
  v3", and "INT4/IU4".

### 6. Persistent Grouped Scheduler

HIP Graphs reduce launch overhead, but RNS8 also needs a device-resident
grouped scheduler for irregular residue/tile/prefix work.

Technical direction:

- Build grouped plans for same shape/many moduli, same B/many A, many small
  independent GEMMs, tile groups with the same prefix, and mixed exact workloads.
- Include FHE-shaped task tables for many NTTs, many coefficient primes,
  base-conversion fragments, key-switch/relinearization fragments, rotation
  batches, bootstrapping stages, and repeated key-material reads.
- Use persistent blocks or grouped backend dispatch to pull pack/GEMM/reduce/
  export work from a device-side task list.
- Explore Stream-K and split-K ideas for small M/N with large K, prefix-heavy
  schedules, and wave quantization tails.
- Combine repeated-B caches with split-K so B stays hot while work is balanced.

RNS8-specific notes:

- The current adaptive scheduler groups selected prefixes. The next step is to
  make that grouping visible to matrix-engine backends and grouped batch APIs.
- Many exact tasks with different semantic outputs should be grouped only when
  the plan metadata stays explicit.

Likely first slices:

- `rns8_grouped_plan` benchmark-only API shape for several independent GEMMs.
- Device task table for prefix/tile groups.
- Persistent or grouped direct-HIP implementation before accelerator-specific
  versions.

Relation to existing queue:

- Deepens "HIP Graphs And Launch Batching", "Adaptive Scheduling",
  "Large-Shape Release Matrix", and "Small Shapes".

### 7. Shared Epilogue DSL

Reducers, status, CRT fragments, finite canonical export, and stores should not
be reinvented separately in direct HIP, CK, rocWMMA, and hipBLASLt-adjacent
paths.

Technical direction:

- Define internal epilogue nodes such as `CenteredMod256`, `CenteredMod255`,
  `CenteredMod251`, `BarrettModP`, `ResidueStore`, `RangeFlag`,
  `NativeStore`, `CRTPrefix9`, `ExactWideLimbStore`, and `CanonicalU8Store`.
- Add research vocabulary for FHE/lattice modular pipelines: `Butterfly`,
  `PointwiseMul`, `LazyReduce`, `Montgomery`, `BaseExtend`, `ModDrop`,
  `Rescale`, `KeySwitchDigit`, `ExternalProduct`, and `Automorphism`.
- Compose nodes into backend-specific epilogues without duplicating arithmetic
  policy.
- Keep backend limitations explicit: hipBLASLt may need external post kernels,
  while CK/rocWMMA/direct HIP can fuse more.

RNS8-specific notes:

- This is an internal architecture cleanup with performance implications. It
  should reduce drift between finite direct-HIP reducers and CK/rocWMMA
  reducers.
- The DSL should make status behavior, reduction family, and output layout part
  of the selected epilogue identity.

Likely first slices:

- Shared reducer metadata and device helper functions for 256/255/251.
- Refactor direct-HIP and rocWMMA finite reducers to use the same reducer node
  definitions.
- Add epilogue family strings that describe composed behavior.

Relation to existing queue:

- Deepens "CK Path", "rocWMMA Path", "Finite-u8", "Export/Status Overhead",
  and "Wrap64 Matrix Engine Redesign".

### 8. Generated Kernel Search

Hand-picked variants will bottleneck. RNS8 needs a generated kernel zoo with
boring variants included.

Technical direction:

- Generate combinations of M/N/K tile, waves per CTA, LDS staging depth,
  K-block cap, A/B vector width, pack layout, residue count per CTA, epilogue
  reducer, store vectorization, split-K mode, and persistent mode.
- Compile with constants baked in for modulus, prefix length, signedness,
  finite modulus, status mode, layout, and tile sizes.
- Prune build-time variant sets with modes such as minimal, `gfx1100_full`, and
  research.
- Use microprobes to rank candidates during exploration, then run normal
  release review for candidates that survive.

RNS8-specific notes:

- CK is already a template-heavy model. RNS8 should generate both CK aliases
  and direct HIP/rocWMMA variants so one backend does not define the search
  space.
- Generated variant names must flow into `selected_kernel`, autotune keys, and
  docs.

Likely first slices:

- Python generator for CK/rocWMMA parameter aliases under ignored generated
  output.
- Static selected-kernel registry for generated variants.
- Build-time kernel-set option to avoid making normal builds huge.

Relation to existing queue:

- Deepens "CK Path", "rocWMMA Path", "AMDGPU Builtins", "INT4/IU4", and
  "AUTO And Cache Architecture".

### 9. Roofline And Evidence Database

RNS8 needs to identify the current bottleneck before optimizing kernels.

Technical direction:

- Build a RNS8-specific roofline model for GEMM arithmetic intensity, pack
  bandwidth, B prepack amortization, residue store bandwidth, CRT/export
  bandwidth, host/device transfer, launch overhead, and API scheduling.
- Add FHE/lattice proxy bottleneck classes: twiddle/root traffic,
  NTT/INTT pass count, base-conversion traffic, key-material reads,
  automorphism permutation traffic, bootstrapping stage traffic, and
  evaluation-key residency.
- Store benchmark knowledge as a durable corpus: target, HIP/ROCm version,
  backend, semantic, shape, selected kernel, layout, timing summaries, ISA hash,
  VGPR/SGPR/LDS, workspace bytes, correctness hash, and thermal/power metadata
  when available.
- Classify shapes as compute-bound, pack-bound, launch-bound, export-bound,
  transfer-bound, or status-bound.

RNS8-specific notes:

- Current benchmark schema and review reports already capture much of the
  identity material. The missing piece is an analysis layer that tells the
  next implementer what to optimize first.
- Exact-wide and finite-u8 should be modeled separately from bounded i64/u64;
  their export and modulus costs differ.
- Computational-algebra scenarios need phase labels in the evidence database:
  dense GEMM, modular factorization, triangular solve, CRT/CRA combine,
  rational reconstruction, verification, certificate generation, polynomial
  transform, product/remainder tree, and black-box sparse matvec.
- Exact-LA and symbolic scenarios should also record controller and
  preprocessing time: rank-profile selection, symbolic preprocessing, sparse
  reduction, tree setup, p-adic lifting, certificate generation, and
  reconstruction/export can dominate even when the dense modular GEMM phase is
  fast.
- External CAS or exact-algebra libraries should be recorded by role: CPU
  oracle, CAS semantic oracle, finite-field reference, algorithm reference,
  benchmark comparison, workload source, CUDA translation study, or non-goal.
  Record licensing and dependency status separately so optional or proprietary
  oracles never become implied production dependencies.
- CAS-oriented evidence rows should include domain/coercion metadata:
  `domain_family`, `parent_domain_id`, `coercion_policy`,
  `coefficient_ring`, `exactness_mode`, `finite_modulus`,
  `prime_or_composite`, `extension_degree`, `phase_label`, `oracle_role`,
  `dense_kernel_extracted`, and `reconstruction_mode`.

Likely first slices:

- Extend review Markdown with bottleneck classification derived from existing
  timing phases.
- Add an ignored evidence database builder that ingests schema v4 captures and
  review reports.
- Add optional RGA/ISA resource summaries to that database.

Relation to existing queue:

- Deepens "Instrumentation", "Host/Transfer", "Large-Shape Release Matrix",
  and "AUTO And Cache Architecture".

### 10. Finite Data Specialization

Finite-u8 should specialize not only by modulus, but also by data distribution.

Technical direction:

- Add histogram-guided plans for binary, sparse-small, uniform field, ring 256,
  ring 255, and prime-field workloads.
- Evaluate bitset/popcount style paths for binary or nearly binary finite
  matrices.
- Push pseudo-Mersenne, fold, low-byte, and reciprocal reducers through CK and
  rocWMMA epilogues.
- Keep finite layouts separate from bounded CRT layouts; finite paths use one
  explicit modulus and prefix-zero storage.

RNS8-specific notes:

- Current finite release evidence shows different winners by modulus and shape.
  AUTO should eventually include finite data profile classes, not only modulus
  and shape.
- Ring 251, ring 255, ring 256, field 251, and representative generic
  composite/prime moduli need separate workload buckets.

Likely first slices:

- Add finite histogram capture for benchmark-generated inputs.
- Add CK/rocWMMA reducers for 251/255/256.
- Add finite workload suites for binary, sparse, and full-uniform inputs.

Relation to existing queue:

- Deepens "Finite-u8", "CK Path", "rocWMMA Path", "End-To-End Layout Search",
  and "AUTO And Cache Architecture".

### 11. RNS-Native Chains And Next-Op API

Exporting after every GEMM wastes work when the next operation can consume RNS.

Technical direction:

- Add plan hints or explicit output modes for residue-current output,
  native-current output, final export, repeated-B reuse, many same-shape calls,
  validation-only calls, and RNS-chain continuation.
- For FHE/lattice-derived research scenarios, model `ntt-current`,
  `coefficient-current`, `tower-current`, key-material-current, and
  modulus-chain-current states as planning vocabulary before exposing any
  public API state.
- Keep `rns8_matrix` capable of representing currentness honestly:
  residue-current, native-current, finite-current, wrap-byte-current, and stale
  host/native states must not be inferred from type alone.
- Allow `RNS GEMM -> RNS GEMM -> export` without reconstructing the
  intermediate result.
- Add fast device-to-device conversion kernels where AUTO needs to move between
  vector-native and RNS domains.

RNS8-specific notes:

- Vector-produced bounded C currently cannot feed RNS GEMM without conversion
  or rejection. That should become an explicit planning choice instead of an
  accidental dead end.
- Lazy residue output pairs naturally with reconstruction backend work and
  expression-level fusion.

Likely first slices:

- Benchmark-only `output_mode=residue` path for bounded RNS chains.
  Implemented for bounded i64/u64 `rns8-bench --residue-chain-length` with
  explicit RNS backends, global chain-safe bounds, square RNS chain shapes, and
  zero-valued measured `crt_export`; the benchmark exports once after measured
  repeats only for checksum generation. AUTO/vector-ALU chains stay rejected at
  the benchmark CLI until mixed-backend chain lowering selects explicit
  conversion points; per-tile bounded chains stay rejected until adaptive chain
  bounds exist.
- Device native-to-RNS bounded conversion kernels.
  Implemented as internal HIP device-to-device `native_i64_to_rns_kernel` and
  `native_u64_to_rns_kernel` wrappers that reuse the centered direct-HIP pack
  kernels with device-native sources. AUTO/direct-HIP bounded matrices can now
  lazily materialize stale RNS residues from current native device storage
  before an RNS-backend GEMM consumes them. This is an internal transition
  primitive, not a public output-domain API or vector autotune promotion policy.
- Plan metadata exposing next-op hints and chosen output domain.
  Implemented in `rns8_get_plan_packing_info` with explicit input/output
  domains, host/device output currentness, and next-operation flags for final
  export, RNS GEMM continuation, native GEMM continuation, native-to-RNS
  conversion eligibility, and reusable B prepack availability.

Relation to existing queue:

- Deepens "Native Vector-ALU Production Backend", "Reconstruction Backend",
  "AUTO And Cache Architecture", and "Expression-Level Fusion".

### 12. Plan-Level Algebraic Lowering

AUTO cannot stay a bag of local backend heuristics. RNS8 needs an internal
operation representation before lowering to a backend.

Technical direction:

- Represent operations such as `MatMul`, `Export`, `ResidueAdd`, `FiniteAdd`,
  `NativeToRns`, `RnsToNative`, `RankUpdate`, and `Batch` with semantic,
  bounds, reuse, layout, and desired-output metadata.
- Add research-only operation labels for FHE/lattice scenario modeling:
  `Ntt`, `Intt`, `Hadamard`, `BaseExtend`, `ModUp`, `ModDown`, `Rescale`,
  `GadgetDecompose`, `ExternalProduct`, `KeySwitch`, `Automorphism`,
  `Relinearize`, and `Bootstrap`. These labels should prevent dense-GEMM
  overclaiming; they are not public FHE APIs.
- Include computational-algebra vocabulary in the internal IR even before those
  operations become public APIs: `Rank`, `Determinant`, `Solve`, `Nullspace`,
  `RankProfile`, `PLUQ`, `CUP`, `PLE`, `TriangularSolve`, `Echelon`,
  `CharPoly`, `MinPoly`, `Certificate`, `CRABuild`, `DixonSolve`,
  `PadicLift`, `RationalReconstruct`, `FreivaldsCheck`, `Ntt`, `Intt`,
  `PointwiseMul`, `ProductTree`, `RemainderTree`, `Interpolate`,
  `ModularCompose`, `GroebnerF4Sparse`, `GroebnerF4DenseFiniteField`,
  `F5SignatureReduction`, `FGLM`, `SubresultantPRS`, `ResultantSylvesterDet`,
  `SylvesterMat`, `Popov`, `Hermite`, `Smith`, and `PolyMatMul`.
- Lower `Export(MatMul(A,B))` differently from `MatMul(A,B)` whose result feeds
  another RNS GEMM.
- Lower repeated-B workloads toward prepack and persistent scheduling.
- Lower finite and bounded small-value workloads toward specialized finite,
  direct INT32, vector, or RNS paths based on the explicit semantic contract.

RNS8-specific notes:

- Semantics remain explicit. The IR should not infer signedness, exact-wide,
  finite, or wraparound behavior from C++ types.
- The first version can be internal to plan creation and benchmark tooling
  without changing the public ABI.
- Polynomial and structured-matrix terms are classification and scenario
  vocabulary until a real backend exists. The IR should prevent overclaiming by
  distinguishing dense GEMM lowerings from NTT, product-tree, black-box sparse,
  or structured-matrix workloads.

Likely first slices:

- Internal plan-lowering description object populated from current descriptors.
  Implemented as the internal `PlanLoweringDescription` helper, derived from
  current plan backend, packing, and schedule metadata without adding another
  public ABI surface.
- Debug/inspect output that explains selected operation lowering.
  Implemented in `rns8-inspect` for autotune exact-hit plans as text and JSON
  `plan_lowering` output covering operation, semantic contract, backend family,
  input/output domain, desired output, schedule, packing, reuse, conversion, and
  lowering path.
- AUTO selection that considers desired output domain and reuse hints.
  Implemented for the current final/native output-domain case by allowing
  reviewed `hip-vector-alu-int64` bounded i64/u64 autotune entries to be
  selected by AUTO. Reuse-hint promotion remains intentionally constrained by
  the existing prepacked-reuse-not-autotune-promotable policy until setup-cost
  and end-to-end wins are promoted as production cache behavior.

Relation to existing queue:

- Deepens "AUTO And Cache Architecture", "Native Vector-ALU Production
  Backend", "RNS-Native Chains", and "CPU/GPU Hybrid AUTO".

### 13. Expression-Level Fusion

RNS8 should eventually fuse exact algebra around GEMM, not only GEMM itself.

Technical direction:

- Support internal expressions such as `C = A*B + D`, `C = A*B - A*E`,
  `C = alpha*A*B + beta*C`, finite residue add/sub, and validation/status
  updates in one output path.
- Fuse residue add/sub and finite reduction into GEMM epilogues where backend
  mapping allows it.
- Avoid reconstructing intermediate exact integers for algebra that is valid in
  the residue domain.

RNS8-specific notes:

- This belongs behind explicit semantics and should not change current GEMM
  public APIs until the internal lowering is stable.
- Finite-u8 and RNS chains are the best early targets because their algebra can
  stay residue-domain.

Likely first slices:

- Internal residue add/sub kernels for RNS-current matrices.
- Fused finite `A*B + D mod q` benchmark path.
- Epilogue DSL node for add/sub before store.

Relation to existing queue:

- Deepens "Shared Epilogue DSL", "Finite Data Specialization", and
  "RNS-Native Chains".

### 14. Shape-Specialized Paths

Not every workload should be forced through square GEMM architecture.

Technical direction:

- Add separate paths for GEMV, batched GEMV, tall-skinny GEMM, wide-skinny
  GEMM, tiny square GEMM, symmetric/Gram products, triangular products,
  banded/diagonal products, and submatrix views.
- Consider direct vector-ALU or wave reductions for true GEMV instead of
  matrix-engine setup.
- Batch many GEMVs with the same B into GEMM when that reformulation wins.
- Use direct INT32 as a middle lane when bounded inputs and output bounds fit.

RNS8-specific notes:

- The release matrix is currently square-heavy. Shape families can reorder
  backend winners and expose wins hidden by square-only review.
- Structured shortcuts are useful only when workload structure is explicit or
  obvious from API shape, not inferred unsafely.

Likely first slices:

- Add shape-family benchmark cases: GEMV, skinny, many small, symmetric/Gram,
  and submatrix view.
- Add direct-HIP/vector GEMV baseline.
- Add view lowering choice: direct strided load vs copy-to-packed.

Relation to existing queue:

- Deepens "Small Shapes", "Large-Shape Release Matrix", "Direct-HIP RNS
  Fallback", and "FP8/Ozaki, Strassen, Sparsity".

### 15. CPU/GPU Hybrid AUTO

For tiny or export-bound cases, GPU arithmetic is not automatically the best
end-to-end path.

Technical direction:

- Let AUTO consider CPU direct bounded, CPU pack plus GPU GEMM, GPU residues
  plus CPU export, GPU GEMM plus CPU CRT, and fully GPU paths.
- Separate one-shot from persistent reuse: the hybrid winner can differ when
  A/B are already resident.
- Keep CPU/reference timing comparable to GPU captures without fabricating GPU
  target metadata.

RNS8-specific notes:

- Reviewed bounded-u64 shows vector-ALU dominates accelerators through 1024.
  Tiny cases may reasonably stay CPU or vector.
- Wrap64 CPU is faster than direct HIP at 64 in the release baseline, while
  direct HIP wins larger shapes. AUTO needs that kind of shape split.

Likely first slices:

- Add CPU-direct tiny bounded one-shot cases to review reports.
- Add hybrid export experiments for exact-wide and bounded prefix-9.
- Add selector explanation for CPU/hybrid choices in inspect output.

Relation to existing queue:

- Deepens "Small Shapes", "Native Vector-ALU Production Backend",
  "Reconstruction Backend", and "Host/Transfer".

### 16. Device Plan Cache And Workspace Arena

Repeated workloads should not repeatedly marshal schedules, constants, and
temporary allocations.

Technical direction:

- Add device-resident plan tables containing pointers, strides, prefix
  schedules, reducer constants, layout descriptors, and output modes for
  grouped/persistent kernels.
- Replace per-path scratch management with workspace lifetime classes:
  transient tile, persistent prepack, export staging, status, autotune probe,
  and grouped scheduler state.
- Partition arenas by stream, device, semantic, shape family, and reuse profile
  where needed.

RNS8-specific notes:

- Current matrix/workspace APIs already expose storage and packing info. The
  arena should build on that rather than invent backend-private hidden state.
- hipBLASLt and rocWMMA repeated-B caches are early examples of workspace-local
  reuse that should become a consistent policy.

Likely first slices:

- Add workspace arena accounting to benchmark metadata.
- Move repeated-B cache allocation into named workspace lifetime classes.
- Add device schedule buffer reuse for matching adaptive schedule fingerprints.

Relation to existing queue:

- Deepens "Reusable B Prepack", "HIP Graphs And Launch Batching",
  "Host/Transfer", and "Persistent Grouped Scheduler".

### 17. Verification-Cost Reduction

Full exact CPU differentials are required for promotion, but exploratory search
can use cheaper oracles to iterate faster.

Technical direction:

- Use residue-domain comparison before CRT when testing layout or epilogue
  variants.
- Use checksum or sampled cell oracles during generated-kernel exploration.
- Use metamorphic identities such as `(A+B)C == AC+BC` in the residue domain.
- Compare candidates against the simple direct-HIP correctness path before
  running full CPU exact output over every cell.

RNS8-specific notes:

- This is development-loop acceleration, not relaxed correctness. It should
  keep the existing promotion standards untouched.
- Wrap64 candidate work already uses checksum and sampled CPU-oracle cells as
  exploratory evidence; that pattern can be generalized.

Likely first slices:

- Benchmark/test helper for residue-domain candidate-vs-direct-HIP comparison.
- Checksum metadata for generated-kernel sweeps.
- Metamorphic finite-u8 and RNS-current tests.

Relation to existing queue:

- Deepens "Generated Kernel Search", "Wrap64 Matrix Engine Redesign",
  "Instrumentation", and "Large-Shape Release Matrix".

### 18. Error-Detecting Exact Fast Path

Some exact workloads may usually need fewer residues than worst-case bounds.
RNS8 can exploit that only when fallback is deterministic and semantics stay
exact.

Technical direction:

- Compute likely-needed prefix first, verify with redundant residue or bound
  metadata, then compute additional residue planes only when needed.
- Use redundant channels for sign, range, overflow, or error detection when
  they are cheaper than full reconstruction.
- Schedule high-value or cheap residue planes first so verification can overlap
  later work.

RNS8-specific notes:

- This is distinct from probabilistic early termination. Default exact APIs
  remain deterministic.
- Adaptive prefix minimization and residue-plane priority scheduling are the
  natural entry points.

Likely first slices:

- Deterministic speculative prefix benchmark mode for bounded-i64 distributions.
- Redundant small-modulus status channel experiment.
- Device-side partial verification for already-computed prefix groups.

Relation to existing queue:

- Deepens "Adaptive Prefix Minimization", "Reconstruction Backend", and
  "Persistent Grouped Scheduler".

### 19. Result Cache And Incremental GEMM

Repeated or slowly changing workloads should not always pay full GEMM cost.

Technical direction:

- Add optional fingerprints for repeated operands, prepack metadata, and
  result reuse. Avoid expensive full-content hashes unless the workload
  explicitly amortizes them.
- Add delta paths for changed rows, changed columns, block updates, and
  low-rank updates.
- Add `rank-k` and batched outer-product update paths that choose direct
  HIP/vector for small k and group into GEMM when k grows.

RNS8-specific notes:

- Source version is necessary but not sufficient for memoization. Cheap
  checksums or user-managed stable IDs may be useful for repeated workloads.
- Incremental paths are workload-specific and should stay out of default GEMM
  until benchmark suites prove they matter.

Likely first slices:

- Repeated-operand fingerprint metadata in benchmark-only mode.
- Changed-row direct-HIP bounded update prototype.
- RNS-current rank-k update benchmark.

Relation to existing queue:

- Deepens "Reusable A And A/B Caches", "RNS-Native Chains", and
  "Shape-Specialized Paths".

### 20. Lane/LDS/Store/Prefetch Microarchitecture Audits

The matrix-engine path needs target-specific microarchitecture facts, not only
API-level backend calls.

Technical direction:

- Build fragment-map harnesses for rocWMMA/AMDGPU target ids that record lane
  id, fragment element id, matrix coordinate, and store order.
- Audit LDS bank conflicts for A tiles, B tiles, accumulator staging, epilogue
  staging, and packed residue stores.
- Search store paths: vectorized residue stores, interleaved C stores, compact
  no-padding stores, bit-packed status, and export-staging layouts.
- Explore AMD analogues for async copy/prefetch/ping-pong pipelines: rotating
  LDS buffers, global-to-LDS overlap, MMA-to-epilogue overlap, and K-pipeline
  depth.

RNS8-specific notes:

- rocWMMA residue emission depends on stable fragment mapping. Capture that as
  an artifact before writing fragile lane-owned store kernels.
- RGA and ISA capture should feed the evidence database rather than remain
  one-off temp files.

Likely first slices:

- `fragment_map_gfx1100.json` generated under ignored `temp/`.
- LDS pattern microbenchmarks for A/B loads and C stores.
- Store-path variants for residue and exact-wide export staging.

Relation to existing queue:

- Deepens "rocWMMA Path", "AMDGPU Builtins", "Instrumentation", "Store/Export
  Overhead", and "Generated Kernel Search".

### 21. Toolchain Matrix

Toolchain versions and compile flags are optimization variables.

Technical direction:

- Compare HIP SDK/ROCm versions, clang flags, CK versions, rocWMMA versions,
  Windows versus Linux, and target ids such as `gfx1100`, `gfx1101/gfx1102`,
  `gfx1200/gfx1201`, `gfx942`, and `gfx950` where real hosts exist.
- Keep generated kernel and autotune databases versioned by compiler,
  accelerator library, target id, runtime, and driver.
- Track whether a backend regression is algorithmic, compiler-generated, or
  library-version-specific.

RNS8-specific notes:

- Windows `gfx1100` is the local bring-up path. Linux ROCm and Instinct remain
  separate validation targets and cannot inherit Windows evidence.
- CK and rocWMMA move quickly; a tuned parameter set can become stale when the
  pinned dependency changes.
- Computational-algebra libraries should be tracked by role rather than
  dependency status: FLINT/NTL/Boost as CPU exact oracles, FFLAS-FFPACK/Givaro
  as finite-field references, LinBox/IML as workload and certificate sources,
  Sage/Magma/Maple/Wolfram/Singular/Macaulay2/GAP/Oscar/Nemo/Hecke/PARI/GP as
  CAS semantic oracles or phase classifiers where available, Normaliz and GAP
  packages as workload sources, and M4RI/M4RIE as small-characteristic
  non-goals unless a real extension-field backend appears.
- CUDA artifacts such as Linac, CUMODP, and GPU NTT systems are translation
  studies. Review their lane assumptions, CUDA library dependencies,
  warp-size assumptions, memory layouts, and NVIDIA-specific instructions
  before any HIP experiment is scoped.
- Proprietary or differently licensed CAS artifacts should stay external:
  compare against their outputs or documented phases when useful, but do not
  make them hidden runtime, build, or correctness dependencies.

Likely first slices:

- Toolchain comparison report format using existing benchmark metadata.
- Pin generated-kernel ISA hashes to compiler/library identity.
- Add Linux ROCm placeholders that require real host captures before any claim.
- Add a computational-algebra oracle matrix to the evidence database so future
  reports separate optional CPU/CAS comparison from RNS8 production evidence.

Relation to existing queue:

- Deepens "Large-Shape Release Matrix", "AUTO And Cache Architecture",
  "Linux ROCm Port", and "Instinct Production" roadmap work.

### 22. Scenario Benchmark Corpus

Square synthetic matrices are not enough. RNS8 needs workload classes that
exercise actual performance risks.

Technical direction:

- Add canonical scenarios: one-shot small, one-shot large, repeated-B,
  repeated-A/B, many small independent GEMMs, exact-wide export-heavy,
  finite fixed modulus, RNS-chain no export, adaptive bounded real
  distributions, GEMV/skinny, wrap64 carry-heavy, and large exploratory shapes.
- Add FHE/lattice-derived proxies: NTT/INTT pressure, key-switch digit
  aggregation, rotation/automorphism-heavy linear transforms, CKKS
  rescale/mod-drop chains, BFV/BGV explicit-modulus arithmetic, bootstrapping
  stages, and encrypted-inference linear-layer lowerings labeled by whether
  they are dense GEMM, diagonal/rotation, MVM/convolution, or coefficientwise
  arithmetic.
- Add parameter fixtures inspired by SEAL, OpenFHE, Lattigo, HElib, HEonGPU,
  PhantomFHE, cuHE/cuFHE, and FIDESlib: `N`/`LogN`, slot count,
  coefficient-modulus chain, Q/P towers, plaintext modulus, scale bits,
  decomposition digit count, ciphertext component count, and evaluation-key
  count.
- Add computational-algebra scenarios: dense finite-field BLAS, modular rank,
  determinant, solve, nullspace, rational reconstruction, Freivalds-verified
  product, PLUQ/CUP/PLE rank profile, echelon recovery, block LU/TRSM-like
  update, rectangular rank-k, characteristic/minimal polynomial, p-adic/Dixon
  solve, early-terminated CRA, polynomial matrix multiplication,
  modular-composition BSGS, F4 dense finite-field matrix phases, F4 sparse
  reduction, F5 signature control, FGLM multiplication-matrix conversion, NTT
  pressure, batched NTT, product/remainder tree, subresultant/PRS, Sylvester
  determinant, structured band/triangular/diagonal, Toeplitz/Hankel/Cauchy,
  low-displacement-rank, and black-box Wiedemann/Lanczos matvec.
- Add failure-mode scenarios: max bounds, negative centered residues,
  modulus-edge residues, overflow-near accumulators, stale cache layouts,
  padded dimensions, non-contiguous strides, K-block boundaries, and mismatched
  finite moduli.
- Report winners by scenario family, not only by shape.

RNS8-specific notes:

- The current 64/128/512/1024 release matrices are valuable but not sufficient
  to rank next work across chained RNS, repeated-B, exact-wide export-heavy, and
  finite distribution-sensitive workloads.
- Scenario labels should flow into review reports and the evidence database so
  future queue ordering is driven by real workload classes.
- Scenario metadata should include `algebra_family`, `structure_id`,
  `shape_signature`, `bound_profile`, `prefix_budget`, `density`, `reuse_mode`,
  `fast_mm_level`, `determinism_mode`, field/ring metadata, and reconstruction
  profile where applicable. Exact-LA/symbolic metadata should additionally
  include `phase_id`, `symbolic_precompute`, `controller_mode`,
  `certificate_mode`, `structure_declared`, and whether the reported timing is
  raw dense GEMM, dense-LA phase, reconstruction, or whole symbolic workflow.
  CAS-oriented metadata should further include `source_role`, `cas_system`,
  `parent_domain_id`, `coercion_policy`, `coefficient_ring`,
  `prime_or_composite`, `extension_degree`, `exactness_mode`, `oracle_role`,
  `dense_kernel_extracted`, and `artifact_lineage` for CUDA or external
  comparison artifacts.
- CAS scenario labels should include full-workflow non-goals as well as direct
  dense-kernel extraction: `SageMatrixBenchmark`, `PARIGPLewisWester`,
  `F4DenseFiniteFieldPhase`, `F4SparseReduction`, `F5SignatureControl`,
  `FGLMMultiplicationMatrix`, `CUMODPPolynomial`, `LinacFiniteFieldElim`,
  `GBLAF4DenseBlock`, `MagmaDenseF4`, `NormalizConeMatrix`,
  `GAPPackageMatrix`, and `CUDATranslationStudy`.

Likely first slices:

- Extend `tools/benchmark_sweep.py` scenario definitions without changing raw
  capture schema first.
- Add scenario tables to review Markdown.
- Include repeated-B hipBLASLt/rocWMMA and RNS-chain lazy-export cases.
- Include FHE/lattice proxy metadata: ring dimension or polynomial degree,
  coefficient-modulus count, decomposition digit count, transform/current
  domain, key-material reuse profile, evidence scope, and output-domain
  requirement.
- Add a computational-algebra scenario table before promoting any dense-GEMM
  claim into rank/determinant/solve/polynomial wording.

Relation to existing queue:

- Deepens every item in this queue because it gives each optimization a
  workload context.

### 23. Native Vector-ALU Production Backend

Status: runtime backend implemented in
[src/backend_vector_alu](../src/backend_vector_alu). The benchmark harness in
[benchmarks/hip_vector_alu_baseline_kernels.hip](../benchmarks/hip_vector_alu_baseline_kernels.hip)
still exists for same-contract release comparisons.

Technical direction:

- Keep the backend semantic-specific: bounded i64/u64 only; no exact-wide,
  finite, wrap64, or CRT fallback.
- Preserve compact native device storage, native currentness, byte counts, and
  source-version accounting.
- Make native-to-RNS conversion an explicit plan choice when a vector-produced C
  needs to feed an RNS operation.
- Keep bounded-u64 AUTO honest: reviewed Windows `gfx1100` evidence says the
  vector backend is the leader at 64/128/512/1024.

Likely first slices:

- Add device-to-device native i64/u64 to RNS conversion kernels.
- Add selector explanations for vector vs RNS vs CPU choices.
- Add shape-family vector baselines for skinny/GEMV scenarios.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "RNS-Native Chains", and "Plan-Level Algebraic
  Lowering".

### 24. Reusable B Prepack And Tile-Swizzled Layout

Status: rocWMMA has a narrow non-tiled RNS B cache with
`rns_i8_tile_swizzled_b_v1` identity and `prepack-v2` keying. hipBLASLt has
workspace-local repeated-A and repeated-B prepack paths for fixed-prefix
single-K-block RNS work. `production_prepack_cache_available` remains `0`.

Details: the first hipBLASLt 1024 slice removes repeated heuristic selection
from identical hot dispatches by caching the selected matmul algorithm in
process-local memory keyed by device, padded shape, scratch leading dimension,
and workspace size. Windows `gfx1100` smoke captures improved the saved 1024
bounded-i64 sample from 26.6 ms end-to-end / 12.4 ms host RNS GEMM to 14.6 ms
then 12.8 ms end-to-end / 6.3 ms then 5.6 ms host RNS GEMM, but those are
five-repeat smoke captures with visible component-timing noise, not
promotion-grade evidence.
The next hipBLASLt repeated-B slice adds a workspace-local B prepack cache for
single-K-block fixed-prefix RNS GEMM. On Windows `gfx1100`, 1024 bounded-i64
release repeated-B captures reported host RNS GEMM at 5.1 ms and 4.7 ms with B
reuse, while the paired repeated-A run was 9.2 ms. The hipBLASLt transpose-pack
event dropped to 0.35-0.38 ms for repeated-B versus 2.1 ms for repeated-A.
This is still workspace-local repeated-workload evidence, not a durable
production cache or full 64/128/512/1024/2048 promotion sweep.
A matching A-cache slice now caches stable repeated-A operands in the same
workspace-local style. On Windows `gfx1100`, 1024 bounded-i64 repeated-A
captures reported host RNS GEMM at 6.6 ms and 7.5 ms versus the earlier 9.2 ms
paired repeated-A capture; a repeated-A/B capture reported 5.0 ms host RNS GEMM
with zero per-repeat pack phase. These are repeated-workload tuning captures,
not production-cache promotion evidence.

Technical direction:

- Expand B reuse from narrow runtime cache slices into a layout-family policy.
- Keep cache identity tied to backend, target id, selected kernel, semantic,
  prefix schedule hash, tile shape, K-block, operand role, source version,
  finite modulus, device id, matrix layout, and operand layout.
- Support repeated-B first, then A-only and A/B caches after B cache behavior is
  stable across finite, exact-wide, fixed-prefix, and adaptive workloads.
- Pair reusable B with fused transient-A pack and persistent/grouped scheduling.

Likely first slices:

- Benchmark repeated-B scenarios for hipBLASLt, rocWMMA, CK, and direct HIP
  with the same review vocabulary.
- Add finite and exact-wide repeated-B cache experiments.
- Add schedule-aware cache keys for adaptive/tiled plans.

Relation to new architecture work:

- Feeds "Fused Pack+GEMM", "End-To-End Layout Search", "Persistent Grouped
  Scheduler", and "Workspace Arena".

### 25. Bounded-i64 Winner Tuning

Status: reviewed Windows `gfx1100` winners are rocWMMA at 512 and hipBLASLt at
1024. The first hipBLASLt slices removed repeated heuristic selection and added
workspace-local repeated-A and repeated-B prepack evidence. The heuristic cache
is intentionally non-durable and does not replace reviewed autotune-cache
identity, library-version rejection, stale-kernel rejection, timing-split,
split-K, finite-u8, or durable cache work. The fixed-prefix RNS path caches A
and B transposed hipBLASLt operands only when source version, device, shape,
prefix, and byte-size identity match; it is skipped for finite-u8,
adaptive/tiled plans, and split-K.

Technical direction:

- Tune winners, not losers. For 512 rocWMMA, focus on B layout, A transient
  pack, residue epilogue, store path, and launch count.
- For 1024 hipBLASLt, focus on repeated-A/B prepack, scratch/reduce behavior,
  heuristic replay, and external reducer locality.
- Re-run 64/128/512/1024 plus 2048 before assuming the split persists.
- Treat bounded i64 as the first production proving ground for residue-channel
  fusion, layout search, and generated variants.

Likely first slices:

- rocWMMA 512 A-pack/B-layout/store variants.
- hipBLASLt 1024 repeated-A/B release matrix.
- 2048 bounded-i64 exploratory release matrix.

Relation to new architecture work:

- Feeds "Residue-Channel Fusion", "Fused Pack+GEMM", "End-To-End Layout
  Search", and "Generated Kernel Search".

### 26. Large-Shape Release Matrix

Technical direction:

- Run 2048/4096/8192 exploratory release matrices within caps for bounded,
  finite, exact-wide, and wrap64 where feasible.
- Keep large-shape reports separate from promotion if required baselines cannot
  finish within run caps.
- Use large shapes to detect when launch overhead stops dominating and raw
  matrix-engine throughput, export, memory traffic, or K scheduling becomes the
  real limiter.

Likely first slices:

- 2048 bounded-i64/u64 and finite-u8 first.
- Exact-wide 2048 export-heavy profile.
- Wrap64 direct-HIP v3 2048 exploratory run if runtime is tolerable.

Relation to new architecture work:

- Feeds "Roofline And Evidence Database", "Toolchain Matrix", and "Scenario
  Benchmark Corpus".

### 27. hipBLASLt Path

Status: heuristic lookup is cached in process-local memory for matching
device/shape/workspace. Fixed-prefix RNS repeated-A and repeated-B can reuse
workspace-local transposed operands when identity matches. This is not a public
prepack cache.

Technical direction:

- Treat hipBLASLt as a black-box INT8 matmul primitive surrounded by RNS8-owned
  generated pack/reduce/export pipeline.
- Prewarm and replay selected algorithms at plan/workspace creation.
- Separate A pack, B pack, heuristic, matmul, scratch, reduce, export, and D2H
  phases wherever possible.
- Specialize external reduce kernels for 256/255/251 and prefix-9 bounded
  paths.
- Use HIP Graphs or grouped host dispatch for repeated hipBLASLt workflows.

Likely first slices:

- Release repeated-A/B bounded-i64 1024 matrix with current workspace-local
  cache.
- A/B prepack support for finite-u8 and exact-wide where layout matches.
- External reducer specialization for 251/255/256 and prefix-9.

Relation to new architecture work:

- Feeds "Reusable B Prepack", "Shared Epilogue DSL", "Fused Pack+GEMM", and
  "Host API Batching".

### 28. CK Path

Technical direction:

- Generate a broader CK alias set: block size, M/N/K per block, CShuffle tile,
  vector widths, M01, K-block cap, reducer family, store vectorization, and
  split-K/Stream-K variants.
- Replace branch/while style centered modulo with shared reducer nodes.
- Avoid `temp_c` copy/add paths when padded or K-split cases can use a cleaner
  output epilogue.
- Dispatch by shape, semantic, finite modulus, prefix schedule, and scenario
  family.

Likely first slices:

- CK finite 1024 ring winner tuning.
- Exact-wide CK export-first profiling before GEMM variants.
- CK generated alias matrix for 512/1024 bounded i64.

Relation to new architecture work:

- Feeds "Generated Kernel Search", "Shared Epilogue DSL", "Finite Data
  Specialization", and "Toolchain Matrix".

### 29. rocWMMA Path

Technical direction:

- Reduce shared-memory round-trip after `store_matrix_sync`; use fragment-map
  evidence before lane-owned residue emission.
- Specialize 256/255/251 reductions in the epilogue with shared reducer nodes.
- Expand B swizzle, transient-A pack, store-path, K-block, and split-K variants.
- For adaptive bounded, group tile entries by prefix, shape, and resource
  behavior to avoid per-entry overhead and tail effects.

Likely first slices:

- Fragment-map harness for `gfx1100`.
- rocWMMA 512 bounded-i64 store/reducer variants.
- Adaptive bounded-i64 1024 prefix-group scheduler variants.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits", "Residue-Channel Fusion",
  "Generated Kernel Search", and "Persistent Grouped Scheduler".

### 30. Finite-u8

Status: direct HIP has fixed-modulus pack, GEMM reduction, and export kernels
for 251/255/256. Reviewed accelerator winners already exist for ring 251,
ring 255, and field 251 at 64/128/512/1024, but broader finite specialization
remains open.

Technical direction:

- Push finite reducer specialization into CK and rocWMMA epilogues.
- Add `finite_u8_centered_plane_v2` with layout selected by backend and
  distribution.
- Benchmark ring 251, ring 255, ring 256, field 251, generic prime, and
  generic composite cases.
- Include finite modulus and finite data profile in plan/autotune identity.
- Keep finite semantics explicit: `RNS8_FINITE_RING_U8` is `Z/qZ`, while
  `RNS8_FINITE_FIELD_U8` is a prime-field `GF(p)` contract for `p <= 251`.
  `GF(2^e)` extension fields and word-size prime fields require separate
  future semantics or lowering.
- Treat FFLAS-FFPACK/Givaro as algorithm references and optional CPU
  comparisons, not as evidence that current finite-u8 covers all computational
  finite-field BLAS workloads.

Likely first slices:

- Direct-HIP fixed-modulus 251/255/256 finite-u8 pack/GEMM/export.
  Implemented: canonical byte ingress, centered-residue GEMM reduction, and
  canonical byte export use compile-time modulus kernels for the hot finite
  moduli, while generic runtime-modulus kernels remain the fallback for other
  ring/field moduli.
- CK/rocWMMA 251/255/256 reducer variants.
- Finite histogram-guided workload suite.
- 1024 finite winner retuning for CK ring and hipBLASLt field.

Relation to new architecture work:

- Feeds "Finite Data Specialization", "Shared Epilogue DSL", and
  "End-To-End Layout Search".

### 31. Exact-Wide

Status: CK has reviewed Windows `gfx1100` wins for exact-wide signed 1024 and
unsigned 128/512/1024. Exact-wide signed 64/128/512 and unsigned 64 remain on
direct HIP.

Technical direction:

- Optimize export first: limb loop coalescing, status traffic, compact D2H,
  fixed limb-count specialization, and output precision tiering.
- Keep signed and unsigned export functions and metadata separate.
- Benchmark whether GEMM or limb export dominates before expanding CK/rocWMMA
  matrix kernels.
- Use lazy RNS output for exact-wide chains where no immediate limb export is
  required.

Likely first slices:

- 1/2/4/8/16/32 limb export variants. Implemented for Direct-HIP prefix-20
  signed and unsigned exports with compile-time fixed limb-count kernels,
  plus existing benchmark sweep coverage for those widths.
- Prefix-specialized exact-wide export kernels. Implemented for Direct-HIP
  prefix-20 reconstruction; nonstandard prefixes still use the generic export
  kernel.
- Exact-wide RNS-chain scenario benchmark.

Relation to new architecture work:

- Feeds "Reconstruction Backend", "RNS-Native Chains", "Roofline", and
  "CK Path".

### 32. Direct-HIP RNS Fallback

Technical direction:

- Keep direct HIP as the inspectable correctness/performance fallback and a
  comparison target for accelerator variants.
- Specialize prefix-9 bounded kernels.
- Try one thread computing multiple neighboring columns to reuse A.
- Batch multiple moduli per launch for small shapes.
- Explore fused pack+GEMM for one-shot/small direct-HIP workloads.

Likely first slices:

- Prefix-9 bounded direct-HIP kernel. Implemented first as
  `direct_hip_prefix9_grouped_rns_gemm_v1`, which batches the nine default
  bounded RNS planes into one `grid.z` grouped launch per K block while keeping
  the existing tiled math and centered reducers.
- Multi-modulus launch batching for 64/128. Implemented first for fixed-prefix
  direct-HIP RNS plans: prefix 9 bounded and prefix 20 exact-wide now share the
  grouped launch path, with exact-wide plans reporting
  `direct_hip_prefix20_grouped_rns_gemm_v1`. This has build, correctness,
  schema, and event-smoke evidence, but still needs release-sweep performance
  review before it becomes a durable speedup claim.
- Direct-HIP fused pack+GEMM small-shape baseline.

Relation to new architecture work:

- Feeds "Fused Pack+GEMM", "Residue-Channel Fusion", and
  "Verification-Cost Reduction".

### 33. Export/Status Overhead

Technical direction:

- Avoid status memset/D2H when plan bounds or output mode make status traffic
  unnecessary.
- Use compact contiguous D2H staging and separate padded host copy only when
  needed.
- Specialize bounded export for prefix 9 and prefix 20.
- Make status mode semantic: impossible, deferred, sampled/debug, or exact
  depending on plan and benchmark mode.

Likely first slices:

- Prefix-9 bounded export specialization.
- Exact-wide compact D2H staging.
- Status phase timing split across accelerator exports.

Relation to new architecture work:

- Feeds "Reconstruction Backend", "Shared Epilogue DSL", "CPU/GPU Hybrid
  AUTO", and "Roofline".

### 34. Adaptive Scheduling

Technical direction:

- Compress schedule entries by selected-prefix group, tile extent, shape, and
  resource behavior.
- Collapse uniform tile bounds to fixed-prefix paths.
- Reuse device schedule buffers when fingerprints match.
- Tune tile size 64/128/256/512 by shape and backend.
- Add priority scheduling for cheap/high-value residue planes when speculative
  exact fallback is in play.

Likely first slices:

- Device schedule buffer reuse.
- Prefix-grouped accelerator dispatch for adaptive bounded.
- Adaptive tile-size scenario matrix.

Relation to new architecture work:

- Feeds "Adaptive Prefix Minimization", "Persistent Grouped Scheduler", and
  "Error-Detecting Exact Fast Path".

### 35. Small Shapes

Technical direction:

- Prefer low-overhead direct/vector/CPU/hybrid paths where matrix-engine pack
  work cannot amortize.
- Use HIP Graph capture, host API batching, or persistent grouped scheduler for
  repeated fixed small shapes.
- Keep one-shot and persistent-reuse benchmarks separate.
- Consider fused pack+GEMM and multi-modulus batching before adding heavier
  accelerator variants.

Likely first slices:

- 64/128 one-shot vs persistent scenario matrix.
- CPU/direct/vector/accelerator selector explanation for tiny cases.
- HIP Graph repeated-small benchmark mode.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "Fused Pack+GEMM", "Shape-Specialized Paths",
  and "Persistent Grouped Scheduler".

### 36. Wrap64 Direct-HIP v3

Status: direct HIP v3 remains the measured strict wrap64 GPU path.

Technical direction:

- Optimize the baseline before another matrix-engine candidate.
- Vectorize byte-limb load/store through `uint64_t` where layout permits.
- Reduce repeated byte extraction in packed-cell accumulation.
- Try 32-bit diagonal accumulators where safe, widening at carry boundaries.
- Increase tile K or compute multiple output cells per thread if register
  pressure allows.

Likely first slices:

- Byte extraction micro-optimization.
- Vectorized load/store layout experiment.
- Multi-output-cell direct-HIP variant.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits", "Shape-Specialized Paths", and
  "Scenario Benchmark Corpus".

### 37. Wrap64 Matrix Engine Redesign

Status: the internal rocWMMA candidate has strong correctness evidence but
loses structurally to direct HIP v3 at every reviewed 64/128/512/1024 shape.

Technical direction:

- Do not lightly iterate on the current candidate.
- Reduce the number of WMMA passes for the 36 byte-pair products or radically
  reduce high-bit correction cost.
- Try grouped diagonals, reused fragments, nibble/byte hybrid layouts, and
  carry-friendly output staging.
- Consider CRT/Ozaki-like slice hybrids only as research paths until they show
  a real structural advantage.

Likely first slices:

- Diagonal grouping design note and microbenchmark.
- Nibble/byte high-bit correction experiment.
- Compare candidate variants against direct-HIP v3 in wrap64 scenario corpus.

Relation to new architecture work:

- Feeds "Generated Kernel Search", "End-To-End Layout Search", "INT4/IU4",
  and "Lane/LDS/Store/Prefetch Audits".

### 38. Reusable A And A/B Caches

Technical direction:

- Add A cache only after B cache behavior is stable.
- Full A/B cache identity must include both source versions and plan
  fingerprints.
- Benchmark setup amortization across repeat counts, not only one repeated
  GEMM.
- Use cache hints from next-op/reuse metadata rather than hidden backend
  inference.

Likely first slices:

- A-only repeated benchmark mode using current cache metadata vocabulary.
- Full A/B repeated bounded-i64 and finite-u8 scenarios.
- Cache-key mismatch diagnostics for both operands.

Relation to new architecture work:

- Feeds "Result Cache And Incremental GEMM", "RNS-Native Chains", and
  "Workspace Arena".

### 39. HIP Graphs And Launch Batching

Technical direction:

- Use graph capture for repeated fixed-shape workflows: pack, per-prefix GEMM,
  export.
- Add FHE/lattice proxy graphs for repeated key switching, rotation batches,
  bootstrapping stage sequences, and fixed-shape NTT/base-conversion pipelines.
- Use host API batching for many enqueued GEMMs/exports where graph capture is
  too rigid.
- Preserve status/error behavior during graph replay.
- Combine with persistent grouped scheduler for small shapes and adaptive
  prefix groups.

Likely first slices:

- Benchmark-only begin/enqueue/end batch API.
- HIP Graph replay for repeated 64/128 bounded and finite cases.
- Compare graph replay against device persistent queue for many small tasks.

Relation to new architecture work:

- Feeds "Persistent Grouped Scheduler", "Small Shapes", and "Device Plan
  Cache".

### 40. Instrumentation

Status: `codex/gfx1100-signal-forge` adds the first GFX1100 evidence-tooling
lane.

- CK and rocWMMA event captures now have a deep scope,
  `accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export`,
  with aggregate pack/matmul/copy/add labels plus zero-based per-prefix labels.
  This is attribution evidence only; it is not a speedup claim.
- Native vector-ALU captures use
  `vector_alu_default_stream_native_int64_operation_groups` and expose pack A,
  pack B, status memset, native kernel, status D2H, output D2H, and aggregate
  pack/GEMM/export labels.
- Use `tools/gpu_event_report.py` to validate a capture and rank event phases by
  median/share before deciding where the optimizer should spend time.
- Use `tools/gpu_isa_report.py --target gfx1100 --object <hip-object>` or
  `--build-tree <build-dir>` to write LLVM objdump ISA summaries under
  `temp/isa-reports/`. The report records symbols, WMMA/MFMA counts, global
  stores, LDS mentions, waits, and VGPR/SGPR/occupancy when available. RGA CLI
  reporting remains optional.
- Keep all captures, dumps, and reports in ignored `temp/`; do not promote
  instrumentation output into autotune cache entries or performance claims.

Technical direction:

- Add per-kernel/per-prefix/per-tile event hooks for CK and rocWMMA.
- For FHE/lattice proxies, add per-transform, per-prime, per-key-switch,
  per-rotation, and per-bootstrapping-stage timing labels before comparing
  dense-GEMM-adjacent work against NTT/key-switch-dominated work.
- Feed RGA/disassembly output, VGPR/SGPR/LDS, waits, stores, and occupancy into
  the evidence database.
- Keep counters debug/probe-only and out of normal benchmark hot paths.
- Add thermal/power/run-order metadata where available for consumer GPU runs.

Likely first slices:

- CK/rocWMMA per-kernel timing split.
- RGA summary parser into benchmark review reports.
- Interleaved baseline ordering in sweeps to reduce thermal noise.

Relation to new architecture work:

- Feeds "Roofline And Evidence Database", "Generated Kernel Search", and
  "Toolchain Matrix".

### 41. Host/Transfer

Technical direction:

- Use pinned staging pools, mapped host memory, async D2H export, and compact
  export buffers only when benchmark metadata reflects them.
- Separate host API overhead from device work.
- Cache workspace allocations and avoid first-use allocation inside measured
  repeats.
- Overlap pack next, GEMM current, and export previous with streams when
  workspace partitioning is clear.

Likely first slices:

- Pinned staging benchmark option with metadata.
- Async exact-wide export/D2H overlap experiment.
- Multi-stream repeated-B pipeline scenario.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "Reconstruction Backend", "Workspace Arena",
  and "Roofline".

### 42. AUTO And Cache Architecture

Technical direction:

- Keep exact-shape keys mature, then add shape-family and scenario-family
  recommendations.
- Consider semantic, shape, prefix schedule, K-block, tile size, finite
  modulus, target id, HIP/accelerator version, data profile, reuse profile,
  output mode, and next-op hints.
- Reject stale kernel names, stale epilogue names, and mismatched toolchains.
- Keep raw benchmark cache writes separate from reviewed cache installation.

Likely first slices:

- Selector explanation output for reviewed cache hits and fallbacks.
- Data-profile and reuse-profile fields in planning metadata.
- Shape-family recommendation layer after exact-shape cache hits.

Relation to new architecture work:

- Feeds "Plan-Level Algebraic Lowering", "Toolchain Matrix", "Scenario
  Benchmark Corpus", and "CPU/GPU Hybrid AUTO".

### 43. AMDGPU Builtins

Technical direction:

- Keep builtins as a microkernel lab after CK/rocWMMA identify a concrete
  bottleneck.
- Start with one semantic, one shape family, one target id, and one instruction
  family.
- Use direct lane mapping and store-path control to answer questions rocWMMA
  cannot.

Likely first slices:

- `gfx1100` builtin residue-store microkernel if rocWMMA store path remains
  blocked.
- Builtin prototype for one finite-u8 or bounded-i64 shape.
- Compare against generated rocWMMA/CK variants before expanding.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits" and "Generated Kernel Search".

### 44. INT4/IU4

Technical direction:

- Treat as packed layout research, not production work.
- Evaluate bit-sliced, nibble-sliced, byte-sliced, and mixed 4x8
  decompositions.
- Account for centered residue range, pack/unpack cost, K-block safety,
  reducer cost, and memory traffic.
- Retire per semantic/target when packed layout plus epilogue cannot beat tuned
  INT8.

Likely first slices:

- `rns_i4_packed_v0` finite or bounded narrow-distribution experiment.
- Nibble-unpack overhead microbenchmark.
- Compare against residue-channel fusion before expanding.

Relation to new architecture work:

- Feeds "End-To-End Layout Search", "Wrap64 Matrix Engine Redesign", and
  "Generated Kernel Search".

### 45. FP8/Ozaki, Strassen, Sparsity

Technical direction:

- Keep FP8/Ozaki and CRT/Ozaki hybrids research-only and out of default exact
  APIs.
- Use Ozaki/slice thinking for exact-wide and wrap64 only when it may reduce
  the number of low-precision GEMMs or high-bit correction passes.
- Treat Strassen and structured sparsity as workload-backed experiments, not
  default dense GEMM work.
- Account for Strassen/Winograd prefix inflation: added/subtracted
  intermediates can widen bounds, increase selected modulus count, add pack and
  temporary traffic, and erase the multiplication-count win.
- Add sparse/structured-zero exact paths only for explicit workload structure:
  zero rows/cols, block sparse, diagonal/banded, triangular, or Gram products.

Likely first slices:

- Hybrid slice/RNS design sketch for wrap64 and exact-wide.
- Structured-zero scenario benchmark.
- Symmetric/Gram direct-HIP prototype if workload appears.

Relation to new architecture work:

- Feeds "Shape-Specialized Paths", "Wrap64 Matrix Engine Redesign", and
  "Scenario Benchmark Corpus".

### 46. Multi-GPU

Technical direction:

- Linux-only later. Split by modulus groups first, not K, unless a real Linux
  profile says otherwise.
- Account for reconstruction, scheduling, peer transfer, and output-domain
  costs.
- Treat multi-GPU as a production-platform project, not a Windows `gfx1100`
  local speedup.

Likely first slices:

- Linux ROCm multi-GPU capability and topology inspection.
- Modulus-group scheduling design for bounded prefix-9 and exact-wide.
- Reconstruction placement comparison: per-GPU partial vs final gathered.

Relation to new architecture work:

- Feeds "Persistent Grouped Scheduler", "Reconstruction Backend", and
  "Toolchain Matrix".

## Best Next Batches

### Batch A: Measurement And Scenario Foundation

- Build the roofline/evidence database analysis layer from current schema v4
  captures and review reports.
- Add scenario benchmark families for repeated-B, exact-wide export-heavy,
  finite distributions, RNS chains, small one-shot, many-small, skinny/GEMV,
  wrap64 carry-heavy, and large exploratory shapes.
- Add computational-algebra scenarios for finite-field BLAS, modular
  rank/determinant/solve/nullspace, rational reconstruction, polynomial
  matrix/modular-composition lowerings, PLUQ/CUP/PLE rank profiles,
  p-adic/Dixon solve, early-terminated CRA, F4 dense finite-field matrices,
  FGLM multiplication-matrix conversion, NTT/product-tree pressure, rectangular
  rank-k, and structured matrix families.
- Add FHE/lattice-derived NTT/key-switch/rotation/bootstrap proxies to the
  same scenario benchmark family, with evidence scope and output-domain
  metadata recorded separately from dense-GEMM claims.
- Add per-kernel CK/rocWMMA event timing and RGA resource summaries.

### Batch B: Immediate Shape Wins

- Finish bounded-i64 winner tuning for rocWMMA 512 and hipBLASLt 1024.
- Optimize exact-wide export before broadening exact-wide GEMM variants.
- Expand finite-u8 CK/rocWMMA reducer specialization for 251/255/256.
- Optimize direct-HIP wrap64 v3 before another matrix-engine candidate.

### Batch C: Representation Wins

- Implement multi-modulus pack and residue-channel fusion experiments.
- Add fused pack+GEMM for one-shot/small bounded and finite workloads.
- Compare end-to-end layouts across RNS, finite, exact-wide, and wrap64.
- Add polynomial-tower and Q/P-basis layout sketches for FHE/lattice proxy
  scenarios without treating them as public RNS8 storage formats.

### Batch D: Scheduler And Reuse Wins

- Expand repeated-B cache work across rocWMMA, hipBLASLt, CK, finite, and
  exact-wide.
- Add persistent/grouped scheduler experiments for adaptive prefix groups and
  many small GEMMs.
- Add tower/key-material reuse scenario labels so FHE/lattice-inspired reuse
  does not get collapsed into ordinary A/B matrix reuse.
- Add HIP Graph and host batching modes for repeated fixed-shape workflows.

### Batch E: Research And Platform Work

- Keep AMDGPU builtins, INT4/IU4, Ozaki hybrids, Strassen, sparsity, and
  multi-GPU behind explicit research/platform work.
- Run Linux ROCm and Instinct work only on real supported hosts with separate
  evidence from Windows `gfx1100`.
