# Performance Gain Detailed Backlog And Research Notes

This document preserves the long-form research backlog that used to live in
[performance-gain-work-queue.md](performance-gain-work-queue.md). Use the work
queue as the ranked execution control panel; use this file for archived
evidence snapshots, alignment notes, ordered work items, and batch-level
planning context.

Imported maximum-performance roadmap triage moved to
[performance-gain-imported-research-notes.md](performance-gain-imported-research-notes.md).
This backlog file stays focused on archived evidence snapshots, alignment
notes, ordered work items, and batch-level planning context.

## Current Evidence Snapshot

- `hip-vector-alu-int64` is a real bounded i64/u64 runtime backend, but the
  current-v2 bounded-u64 refresh no longer treats it as the universal leader:
  the reviewed local winners split across CPU, Direct HIP, CK, and hipBLASLt by
  shape. It remains bounded-only and must not be generalized into exact-wide,
  finite, or wrap64 semantics.
- Bounded i64 has current Windows `gfx1100` v2 release-review evidence for 512,
  1024, and 2048. The June 4, 2026 seed `20260604` sweep kept 512 on Direct HIP
  `direct_hip_tiled_active_prefix_rns_gemm_v2` at 1851 us and selected hipBLASLt
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` at 1024 with a
  4174 us median, 1.09x faster than Direct HIP and 8.13x faster than vector ALU.
  The 2048 large-shape slice selected CK
  `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` at 14220 us,
  1.57x faster than Direct HIP. Bounded-u64 2048 selected rocWMMA
  `rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2` at 15128 us, 1.22x
  faster than Direct HIP. Both 2048 non-reuse entries are installed in the
  local reviewed cache.
- Current CK and rocWMMA RNS selected kernels have v2 common-modulus reducer
  identities after the shared epilogues gained explicit 256/255/251 reduction.
  The 512/1024 bounded-i64 v2 release captures are now complete and did not
  promote CK or rocWMMA. Older CK/rocWMMA v1 bounded/exact-wide timings remain
  historical for any shape or path that has not been rerun with the matching v2
  selected-kernel identity.
- The active hipBLASLt source path now uses
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`; previous
  `hipblaslt_int8_i32_scratch_reduce_baseline_v1` timings are historical and
  should not be mixed into new autotune cache evidence.
- Adaptive bounded now has current-v2 `adaptive-bands` Windows `gfx1100`
  release-review evidence. Direct HIP wins the reviewed bounded-i64
  256x256x512, bounded-i64 1024x1024x1024, and bounded-u64 512x1024x512
  groups at 1848 us, 4937 us, and 4224 us median end-to-end respectively; CK
  and rocWMMA current-v2 tiled paths lose to Direct HIP and no adaptive cache
  entry is promoted. The older 1024 bounded-i64 rocWMMA
  `rocwmma_i8_i32_signed_tiled_hot_residue_v1` result remains historical only.
- Finite-u8 has current Windows `gfx1100` v2 release-review winners for 64, 128,
  512, 1024, hot-modulus 2048 across ring 251/255/256 and field 251, generic
  ring 127/253 at 2048, and generic field 127 at 2048.
  Current installed local entries include ring-251 128/1024/2048 rocWMMA,
  ring-251 4096 hipBLASLt,
  generic ring-127 2048 rocWMMA, generic ring-253 2048 rocWMMA,
  ring-255 1024 CK, ring-255 2048 hipBLASLt, ring-255 4096 CK,
  ring-256 128/512/2048 rocWMMA, ring-256 1024/4096 hipBLASLt,
  field-127 512/2048 CK, field-251 512 rocWMMA, field-251 1024 CK,
  and field-251 2048/4096 hipBLASLt. Ring-255 64 is deliberately not promoted
  because CPU reference is faster; ring-256 2048 hipBLASLt is not promoted
  because it loses to Direct HIP and lacks required events.
- Exact-wide has current Windows `gfx1100` v2 release-review winners for 64,
  512, 1024, 2048, and 4096. Eight event-valid entries are installed in the
  local default cache: unsigned 64 hipBLASLt, signed 512 rocWMMA, signed 1024
  hipBLASLt, unsigned 1024 CK, and signed/unsigned 2048/4096 hipBLASLt. Signed
  64, signed 128, unsigned 128, and unsigned 512 stay on Direct HIP. Broader
  limb-count variants remain open, while 2048/4096 evidence points the next
  execution work toward export specialization and lazy residue-current
  workflows.
- Direct HIP `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` is the
  measured strict wrap64 GPU path for reviewed 64/128/512/1024/2048/4096 local
  `gfx1100` shapes. The internal rocWMMA wrap64 candidate matches
  checksums in candidate-inclusive release review but loses to direct HIP at
  every 64/128/512/1024 shape; strict wrap64 remains a Direct-HIP correctness
  path rather than an AUTO cache entry.
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
- Use row/column summaries inside exact per-tile bound discovery. Implemented
  for bounded i64/u64 benchmark scans: the prepass marks nonzero A rows and B
  columns, skips whole tile scans when either side is all zero, and skips known
  zero rows/columns inside mixed tiles while keeping the existing exact
  output-cell maximum for every nonzero product tile. Release evidence under
  `temp/perf-work-queue/tile-bound-zero-shortcut/` shows a 1.35x
  `tile_bound_scan` speedup on the 512 bounded-u64 adaptive-band case with
  unchanged tile-bound hash and adaptive schedule metadata; the bounded-i64
  sibling capture is schema-valid and event-valid after the same scanner change.
- Collapse uniform per-tile schedules back to fixed-prefix dispatch without
  duplicating the fixed-prefix implementation.
  Implemented for no-op per-tile captures where every tile still requires the
  existing full bounded prefix; uniform reduced-prefix schedules remain
  materialized until their tiled dispatch path is independently validated as a
  net win.
- Compact direct-HIP active-prefix scheduled launches. Implemented for
  per-tile direct-HIP RNS GEMM as
  `direct_hip_tiled_active_prefix_rns_gemm_v2`: the workspace stores a compact
  per-modulus active-entry schedule and the GEMM dispatch uses it to avoid tile
  blocks for residue planes that a tile did not select. The public row-major
  device schedule is no longer uploaded for nonzero adaptive schedules because
  the backend validates against the host schedule and launches from the compact
  active schedule.
- Compact Direct-HIP zero-output schedules. Implemented as
  `direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3`: zero-output public
  schedule entries are omitted from the active GEMM schedule and handled by the
  event-visible `direct_hip_zero_output_tile_memset` operation before nonzero
  scheduled GEMM work. Mixed zero/nonzero schedules still upload the public
  device schedule for tile extents; uniform all-zero schedules use a contiguous
  selected-plane memset and report zero GEMM schedule workspace bytes. Smoke
  evidence lives under
  `temp/perf-work-queue/direct-hip-zero-active-schedule/` and validates schema
  v4 plus required GPU events for a 256x256x512 bounded-u64 adaptive-bands
  capture.
- Direct-HIP zero row/column product schedules. Implemented for explicit
  trusted per-tile proof masks as
  `direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1` and the
  combined
  `direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1`. The
  public descriptor carries copied zero A-row and B-column masks, schedule info
  reports proof counts and covered output products, Direct-HIP workspaces and
  scheduled exports upload the masks, and stale/mismatched captures are rejected
  by schema, release-review keying, result comparison, and autotune-key checks.
  Release-build C++ tests cover the plan contract and a real `gfx1100`
  scheduled GEMM/export CPU comparison; smoke captures under
  `temp/perf-zero-rowcol-smoke/` validate schema v4 plus required GPU events for
  bounded i64/u64 65x65x64 adaptive-band combined zero-tile/row-column cases.

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
  fixed-prefix reconstruction kernel. Direct-HIP scheduled per-tile bounded
  i64/u64 export now dispatches compile-time CRT reconstruction variants for
  selected prefixes 1 through 9 inside the adaptive export kernel, retaining the
  runtime-prefix reconstruction helper only as the fallback for unusual
  selected prefixes.
- Exact-wide 1/2/3/4/8/16/32 limb export variants with compact D2H staging.
  Implemented benchmark coverage for the limb-count variants via
  `rns8-bench --exact-wide-limbs` and
  `tools/benchmark_sweep.py --include-exact-wide-limb-variants`; this measures
  the exact-wide export path at requested limb widths. Direct-HIP export stages
  and copies `rows * cols * limb_count` limbs, and full-width device exports
  now elide range-status memset/D2H traffic when overflow is structurally
  impossible: signed and unsigned limb counts 3..32.
  Prefix-20 Direct-HIP signed and unsigned export kernels also dispatch
  compile-time fixed limb-count variants for 1/2/3/4/8/16/32 limbs. The
  3-limb variant is especially important for exact-wide captures because it is
  the compact full-width 192-bit device reconstruction output and can avoid both
  status traffic and a fourth all-zero or sign-extension output limb. The runtime
  limb-count kernel is retained for other widths. Windows `gfx1100` release
  captures under `temp/perf-work-queue/exact-wide-3limb-export-current/` and
  `temp/perf-work-queue/exact-wide-3limb-export-rerun/` are schema-valid and
  event-valid, and direct-HIP GPU differential tests now compare limb 3 against
  CPU for signed and unsigned exports. Do not promote this as a stable speedup
  yet: the first unsigned 512 pass favored 3 limbs on host export median
  (1676 us versus 7220 us for 4 limbs), but the reverse-order rerun favored
  4 limbs on host export median (3030 us versus 3513 us for 3 limbs).
- A residue-current output mode for chained RNS GEMM benchmarks. Implemented as
  exact-wide benchmark/tooling coverage via `rns8-bench --residue-chain-length`
  and `tools/benchmark_sweep.py --residue-chain-length`: measured repeats keep
  intermediate outputs resident in RNS form, report zero per-repeat
  `crt_export`, and run one final untimed host limb export only for checksum
  evidence. The June 4, 2026 chain-event pass made supported chain captures
  GPU-event-visible for per-repeat pack and chained `rns_gemm` work while
  keeping export phases absent from `gpu_event_phase_order`; the validation
  helper also accepts `tools/gpu_event_report.py --require-events` as the strict
  event gate. The June 5, 2026 Direct-HIP release-mode pass under
  `temp/perf-work-queue/exact-wide-rns-chain-direct-current/` captured
  exact-wide signed 128x128x128 chain-length-3 residue-current output with
  schema-v4 validation and required GPU events. The per-repeat repack chain
  measured 6102 us median end-to-end with zero per-repeat CRT export; the
  explicit reusable-B chain measured 1201 us median end-to-end after 11718 us
  setup. The matching independent host-export captures measured 2517 us median
  for four-limb output and 1538 us for three-limb output, so the lazy chain is a
  useful workload signal but not a same-output cache promotion. This is not yet
  a public API output-domain mode.
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
- Benchmark-only residue-channel fusion experiments are now exposed through
  `rns8-bench --residue-channel-fusion` for explicit Direct-HIP, global-bound,
  fixed-requested prefix-9 bounded captures. Schema validation requires
  `fusion_mode=residue_channel_width3_experimental_benchmark_only`,
  `pack_layout=native_i8_row_major_residue_channel_width3`,
  `residue_group_width=3`, and the generated reducer identity. AUTO and public
  one-shot calls never route to this experiment.
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
  export now dispatch fixed-modulus kernels for 251, 255, and 256.
  Direct-HIP public finite-u8 one-shot ring/field calls now also use a native
  transient-input GEMM path that copies canonical `uint8_t` A/B buffers,
  centers them inside the GEMM tile load, skips resident A/B pack kernels, and
  materializes C through the existing finite export path. This is not yet a
  reviewed speedup claim.
- CK and rocWMMA finite-u8 reducer specialization was tested for the same
  fixed-modulus family. The 251/255 pseudo-Mersenne reducer variants were not
  promoted because focused Windows `gfx1100` measurements made them worse or
  too noisy relative to the existing reviewed v1 accelerator identities.
  Modulus 256 now has explicit CK/rocWMMA v2 selected-kernel identities with a
  shared mask reducer, but the June 4, 2026 release/event sweep at 512x512x512
  still favored direct-HIP end-to-end, so no reviewed cache entry was written.
  Current CK/rocWMMA plans now report fixed-modulus 251/255/256 v2 identities
  and route those cases through shared common-modulus reducer helpers; the old
  generic accelerator identities are rejected for those explicit moduli by
  schema/cache tooling. Fresh release captures are still required before any
  historical 251/255 winner timing can be transferred to the new v2 identities.

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
  backends keep the established resident pack/GEMM/export route. A follow-up
  large-shape bounded i64/u64 specialization now routes public one-shot
  `m/n/k >= 512` Direct-HIP calls to
  `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2`, where each
  worker computes two neighboring output columns and reuses the centered A tile
  value across both accumulators. Smaller bounded one-shot shapes remain on
  `direct_hip_prefix9_native_input_grouped_rns_gemm_v1` because release
  evidence was noisy or not favorable at 64/128. Windows `gfx1100` bounded-u64
  release captures under
  `temp/oneshot-colpair-before/` and
  `temp/oneshot-colpair-release-gated/` show the routed bounded-u64 512 case
  improving average end-to-end time by 1.09x and median end-to-end time by
  1.21x against the prior v1 one-shot kernel, with schema-valid and
  event-valid final captures. The June 5, 2026 bounded-i64 route under
  `temp/perf-work-queue/direct-hip-i64-oneshot-colpair/` improves the 512
  public one-shot median from 9368 us to 3048 us, with matching checksum and a
  schema/event-valid final capture. The same-shape persistent resident
  Direct-HIP capture still measured faster at 2126 us, so this is a one-shot
  implementation win rather than a resident-workflow routing change.
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
- Direct-HIP one-shot finite-u8 fused load/compute path.
  Implemented for public finite ring/field `uint8_t` one-shot calls. The
  benchmark/schema surface reports `rns8_finite_u8_public_oneshot`, exact
  native-input event phases, zero external pack/export timings, and a persistent
  direct-HIP finite baseline prerequisite before any speedup claim.
- Direct-HIP transient uniform-small A+B bounded prefix-9 path. Implemented for
  explicit fixed-prefix global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --prefix-policy fixed-requested
  --transient-uniform-small-inputs` captures.
  The benchmark copies A and B as row-major `int8_t` device buffers every
  repeat, dispatches
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1`,
  writes resident prefix-9 RNS residues, and uses the normal CRT export path.
  This is a benchmark/runtime path only; it does not change public matrix
  semantics or default AUTO routing. The benchmark/schema surface reports
  `rns8_bench_uniform_small_i8_ab_transient_path`,
  `transient_uniform_small_i8_ab_inputs`,
  `uniform_small_i8_ab_transient_residue_then_crt_export`, and explicit
  `bounded_uniform_small_i8_a_h2d`,
  `bounded_uniform_small_i8_b_h2d`, and
  `bounded_uniform_small_i8_ab_transient_gemm_kernel_group` GPU event phases.
  Windows `gfx1100` release captures under
  `temp/perf-work-queue/uniform-small-transient-i8/` used release binaries,
  3 warmups, 9 measured repeats, fixed requested prefix 9, and seed 123. All
  transient, persistent, and one-shot captures are schema-valid and event-valid,
  and checksums match within each same-shape trio.

  | Semantics | N | Persistent median end-to-end us | Transient median end-to-end us | One-shot median end-to-end us | Transient vs persistent | Transient vs one-shot |
  |---|---:|---:|---:|---:|---:|---:|
  | bounded-i64 | 64 | 930 | 1026 | 1108 | 0.91x | 1.08x |
  | bounded-i64 | 128 | 1533 | 1060 | 3336 | 1.45x | 3.15x |
  | bounded-i64 | 512 | 3701 | 1844 | 5130 | 2.01x | 2.78x |
  | bounded-u64 | 64 | 1379 | 1270 | 901 | 1.09x | 0.71x |
  | bounded-u64 | 128 | 1733 | 1169 | 1153 | 1.48x | 0.99x |
  | bounded-u64 | 512 | 2617 | 2056 | 4120 | 1.27x | 2.00x |

  Decision: keep as an explicit measured small/fixed-prefix candidate and use it
  as the direct-HIP fused pack+GEMM comparison surface. It beats same-contract
  persistent direct-HIP in five of six measured rows, but it is not a blanket
  default-routing promotion because bounded-i64 64 regressed against persistent
  and bounded-u64 64/128 remains better or roughly tied with public one-shot.
- Direct-HIP transient-A plus reusable-B finite-u8 path.
  Implemented for `rns8-bench --backend hip-direct --semantics finite-u8-*`
  with `--reuse-packed-b`: B is packed once into persistent finite storage,
  A is copied as canonical native `uint8_t` per repeat, and the Direct-HIP GEMM
  centers A inside the tile load path while consuming resident centered B. The
  benchmark/schema surface reports `transient_native_a_resident_b_reuse`,
  `rns8_bench_native_a_reuse_b_path`, `finite_native_a_gemm_kernel`, and a
  zero-valued `finite_pack_kernel`. Windows `gfx1100` release captures under
  `temp/finite-native-a-reuse-b-release-r33/` show setup-amortized wins at 33
  repeats for ring-255 512/1024 and field-251 512/1024, while 128-size cases
  stay experimental because setup and Windows timing variance can erase the
  per-repeat pack savings.
- Direct-HIP transient-A plus reusable-B bounded prefix-9 path.
  Implemented for global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --reuse-packed-b` captures whose Direct-HIP
  plan resolves to fixed prefix 9. Adaptive-band captures generally use the
  native-A/resident-RNS-B route: B is packed once into resident RNS storage,
  A is copied as native `int64_t`/`uint64_t` per repeat, and the grouped
  prefix-9 GEMM centers A inside the tile load while consuming resident centered
  B. The benchmark/schema surface reports `transient_native_a_resident_b_reuse`,
  `rns8_bench_native_a_reuse_b_path`, a distinct
  `bounded_native_a_reuse_b_gemm_kernel_group`, and zero `pack_kernel`.
  Large bounded-u64 adaptive-band reuse-B captures with `m/n/k >= 512` now route
  to `direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2` and
  emit `bounded_native_a_colpair_reuse_b_gemm_kernel_group`, reusing each
  centered native A tile value across two neighboring output columns while B
  stays resident in RNS storage. A Windows `gfx1100` release smoke under
  `temp/perf-work-queue/direct-hip-u64-reuse-b-colpair/` is schema-valid and
  event-valid; at 512 with 33 repeats it measured 3218.94 us for same-build
  non-reuse direct HIP versus 2842.46 us setup-inclusive per repeat for the
  colpair reuse-B route, a 1.13x setup-amortized win. The 5-repeat smoke still
  loses setup-inclusively, so this remains a many-repeat explicit reuse path,
  not an AUTO/default-routing claim.
  The default uniform-small benchmark profile now takes the faster specialized
  route: A and B are represented as single row-major `int8_t` planes because
  all prefix-9 centered residues are identical for the generated `[-16,16]`
  signed and `[0,16]` unsigned values. B is copied once during reuse setup,
  A is copied per repeat, and v2
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2`
  lets each thread compute two neighboring output columns so it can reuse the
  same A tile value across both accumulators while fanning the same A/B planes
  across the prefix-9 RNS output planes in one grouped launch. That surface
  reports
  `transient_uniform_small_i8_a_resident_i8_b_reuse`,
  `rns8_bench_uniform_small_i8_ab_reuse_b_path`, the
  `uniform_small_i8_ab_resident_b_residue_then_crt_export` epilogue, the
  `bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group` event phase,
  and zero `pack_kernel`.
  Windows `gfx1100` release captures under
  `temp/uniform-small-i8-ab-colpair-release/` and the bounded-u64 1024 reruns
  under `temp/uniform-small-i8-ab-colpair-u64-1024-rerun/` used release
  binaries, 3 warmups, and 9 measured repeats. They are schema-valid and
  event-valid. Against the prior v1 single-column implementation in
  `temp/uniform-small-i8-ab-colpair-before/`, the v2 first-pass release matrix
  improved per-repeat end-to-end time by 2.10x for bounded i64 512, 2.54x for
  bounded i64 1024, and 1.43x for bounded u64 512; bounded u64 1024 stayed
  export-noise sensitive but v2 reduced the GEMM phase in that first pass by
  1.29x and beat the same-backend non-reuse baseline setup-inclusively in three
  focused reruns. Keep this as an explicit reuse-path implementation win, not an
  AUTO/default-routing claim, until workload-level reuse policy decides when
  setup and reuse metadata should drive backend selection.
- Direct-HIP reusable-A bounded prefix-9 uniform-small path. Implemented for
  global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --reuse-packed-a --prefix-policy
  fixed-requested --max-prefix 9` captures whose Direct-HIP plan resolves to
  fixed prefix 9. The path copies uniform-small A once during prepack setup,
  copies uniform-small B per repeat, and reuses the same colpair grouped kernel
  launch under the explicit
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1`
  selected-kernel id. The benchmark/schema surface reports
  `transient_uniform_small_i8_b_resident_i8_a_reuse`,
  `rns8_bench_uniform_small_i8_ab_reuse_a_path`, the
  `uniform_small_i8_ab_resident_a_residue_then_crt_export` epilogue, the
  `bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group` event phase,
  and zero `pack_kernel`. Windows `gfx1100` release captures under
  `temp/perf-work-queue/` used release binaries, 3 warmups, and 33 measured
  repeats. Against a clean `a75b0a2` same-contract repeated-A baseline,
  setup-inclusive speedups were 3.04x and 1.32x for bounded i64 512/1024, and
  1.33x and 1.30x for bounded u64 512/1024. Keep this as an explicit
  fixed-prefix reuse-path implementation win, not an AUTO/default-routing claim.
  Large bounded-u64 adaptive-band repeated-A captures with `m/n/k >= 512` now
  have a matching native-B/resident-RNS-A colpair route:
  `direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1`.
  A is packed once into resident RNS storage, B is copied as native `uint64_t`
  per repeat, and the grouped prefix-9 GEMM centers B inside the tile load while
  consuming resident centered A. The benchmark/schema surface reports
  `transient_native_b_resident_a_reuse`, `rns8_bench_native_b_reuse_a_path`, the
  `resident_a_native_b_centered_residue_then_crt_export` epilogue, the
  `bounded_native_b_colpair_reuse_a_gemm_kernel_group` event phase, and zero
  `pack_kernel`. Windows `gfx1100` release captures under
  `temp/perf-work-queue/direct-hip-u64-reuse-a-colpair/` used release binaries,
  3 warmups, and 33 measured repeats. They are schema-valid and event-valid. At
  512 the same-build non-reuse Direct-HIP baseline averaged 6475.15 us per
  repeat while the native-B reuse-A route averaged 4633.30 us setup-inclusive
  per repeat, a 1.40x setup-amortized win. At 1024 the corresponding numbers
  were 10613.20 us versus 9892.63 us, a 1.07x setup-amortized win. Event traces
  show the native-B colpair GEMM phase itself is slower than the normal grouped
  Direct-HIP GEMM, so keep this as an explicit repeated-A path whose current win
  comes from reduced pack/export cost, not as an AUTO/default-routing claim.

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

### 7. Generated Reducer/Epilogue Registry

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
- The generated registry should make status behavior, reduction family, and output layout part
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
  identity material. The first analysis layer now tells the next implementer
  which semantic/scenario/backend groups consume the most measured bottleneck
  time; remaining work is to feed it broader current capture corpora and let it
  steer release A/B work.
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
  timing phases. Implemented in `tools/benchmark_sweep.py`: review JSON now
  stores a per-candidate bottleneck classification from pack, GEMM, export, and
  unattributed end-to-end overhead medians, and `review_report.md` prints that
  classification beside promotion blockers and primary loss phase.
- Add an ignored evidence database builder that ingests schema v4 captures and
  review reports. Implemented as `tools/evidence_database.py`: it validates
  capture JSON with the existing schema, optionally joins
  `tools/benchmark_sweep.py` review reports and scenario manifests, estimates
  coarse per-capture work/traffic/arithmetic-intensity metrics, classifies the
  dominant bottleneck from host phases and GPU event categories, and writes
  ignored `evidence_database.json`, `evidence_rows.csv`, and
  `evidence_summary.md` outputs under `temp/` by default. The first database
  schema records target/device/toolchain identity, backend/kernel, semantic,
  shape, finite modulus, prefix/selected-prefix, exact-wide limb count,
  pack/reuse mode, output domain, timing medians, event bottleneck category,
  estimated GOP/s, pack/export bandwidth estimates, review promotion blockers,
  and scenario-family metadata when available.
- Add optional RGA/ISA resource summaries to that database. Implemented as
  `tools/evidence_database.py --isa-report <file-or-dir>`: the database ingests
  `tools/gpu_isa_report.py` `*-isa-summary.json` outputs, matches them to
  captures by normalized backend and GPU target, records report paths, symbol
  counts, WMMA/MFMA/global-store/LDS/wait/instruction totals, VGPR, SGPR,
  occupancy when available, RGA status, and emits compact ISA resource tables in
  `evidence_summary.md`. The reports remain temp-only evidence inputs.
- Add a corpus-level roofline priority surface. Implemented in
  `tools/evidence_database.py`: each row now carries a conservative
  `roofline_target` and `optimization_hint`, and `evidence_summary.md` includes
  both `GPU Roofline Priority` and global `Roofline Priority` tables grouped by
  target/scenario/semantic/target-id and ranked by total measured bottleneck
  time. The loader also has an opt-in `--skip-invalid` mode for broad ignored
  temp corpora, recording stale rejected captures in the output instead of
  blocking all valid captures. This makes the database an execution control
  surface for deciding the next A/B run without turning analysis output into an
  autotune or promotion claim.

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
- Generated epilogue node for add/sub before store.

Relation to existing queue:

- Deepens "Generated Reducer/Epilogue Registry", "Finite Data Specialization", and
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
  Partially implemented for the native vector-ALU backend as a conservative
  long-K N=1 specialization: `n == 1`, `k >= 4096` bounded
  i64/u64 plans now select `hip_vector_alu_i64_gemv_n1_exact_192b_v1` or
  `hip_vector_alu_u64_gemv_n1_exact_192b_v1`. The kernel parallel-reduces K
  across a 256-thread block while preserving the existing software 192-bit
  exact accumulator and native output/export contract. The active route now
  covers all long-K N=1 vector captures (`n == 1`, `k >= 4096`), not only the
  original 1x1 dot-product gate; multi-row correctness is covered by the HIP
  differential suite, and a 128x1x4096 bounded-i64 runtime smoke under
  `temp/vector-gemv-n1-tall-smoke/` is schema-valid and event-valid with the
  GEMV selected-kernel id. Windows `gfx1100` release captures under
  `temp/perf-work-queue/vector-gemv-n1/` compare against detached pre-change
  commit `96781eb`; at 1x1x65536, bounded-u64 improved average end-to-end by
  2.22x and median end-to-end by 3.39x, while bounded-i64 improved average
  end-to-end by 4.44x and median end-to-end by 7.41x. A broader 1024x1x1024
  smoke remains below the long-K threshold because pack/copy overhead dominated
  and the reduction kernel was not a setup-inclusive win there.
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

- Current-v2 bounded-u64 evidence splits small and medium choices across CPU,
  Direct HIP, CK, and hipBLASLt. Vector should be routed only where current
  same-contract evidence shows an end-to-end win.
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
  capture schema first. Implemented as `tools/benchmark_sweep.py --scenario`
  with named scenario families for `adaptive-bands`, `repeated-b`,
  `exact-wide-export`, `finite-distributions`, `rns-chain`, `small-oneshot`,
  `finite-generic-moduli`, `many-small`, `skinny-gemv`,
  `computational-algebra-proxies`, `fhe-lattice-proxies`, `wrap64-carry`,
  `large-exploratory`, `layout-search`, and `all`.
  Scenario mode reuses the normal schema v4 captures and release-review logic,
  but writes a separate `scenario_manifest.json` plus `scenario_manifest.md`
  under the sweep output root. The manifest records scenario family, item name,
  shape, backend, pack/reuse mode, output domain, evidence scope, rationale,
  command, capture path, and optional proxy metadata so workload labels do not
  mutate dense-GEMM capture schema or autotune contract keys.
- Expand large-shape scenario coverage beyond bounded 2048. Implemented in
  `large-exploratory`: bounded i64/u64, exact-wide signed/unsigned, finite-u8
  ring/field, and strict wrap64 probes now cover 2048 plus tolerable 4096
  shapes. These entries are explicitly exploratory and are meant to classify
  launch/export-bound versus throughput-bound behavior before backend-specific
  tuning.
- Add generic finite-u8 modulus coverage without overstating accelerator
  support. Implemented in `finite-generic-moduli`: non-hot prime 127 and
  composite 253 ring cases, field-prime 127, and 2048 generic probes are
  separated from the specialized 251/255/256 accelerator paths. The June 5,
  2026 generic 2048 release refresh added CPU-backed rocWMMA cache entries for
  ring 127 and ring 253 plus a CK cache entry for field 127; broader generic
  modulus-family and 4096 coverage remains intentionally narrow.
- Expand many-small scenario coverage beyond the first tiny proxy. Implemented
  in `many-small`: bounded i64/u64 square jobs, skinny N=1 bounded-u64 jobs,
  exact-wide signed jobs, finite-u8 ring jobs, and public one-shot baselines now
  share pre-grouped baseline metadata. These are not grouped-dispatch speedup
  claims; they are the control surface for proving whether batching 64/128 and
  skinny exact jobs is worth implementing. The first full release sweep is now
  schema-valid after the small public Direct-HIP resident-fallback one-shot
  contract was encoded. It produced no cache promotions and should be used as
  the independent-call baseline for grouped dispatcher work.
- Add scenario tables to review Markdown. Implemented as the separate scenario
  manifest Markdown table beside `review_report.md`; raw review groups stay
  contract-keyed so scenario labels cannot accidentally make incompatible
  captures comparable.
- Include repeated-B hipBLASLt/rocWMMA and RNS-chain lazy-export cases.
  Implemented in the scenario corpus: `repeated-b` emits bounded-i64 512/1024
  `--reuse-packed-b` captures across the current GPU backend set, while
  `rns-chain` emits signed/unsigned bounded and exact-wide residue-current
  chains at 128/256 with three- and four-GEMM depths. Final checksum export is
  kept outside the measured repeat loop so lazy-export timing is not mixed with
  host-output timing.
- Add explicit end-to-end layout-search coverage. Implemented in
  `layout-search`: bounded fixed-prefix RNS final-export and RNS-next-op
  captures, exact-wide prefix-20 limb-export and RNS-next-op captures,
  finite-u8 hot ring/field layout captures, and strict wrap64 byte-limb layout
  captures share `workflow_name=end_to_end_layout_search` metadata so layout
  comparisons do not get inferred from unrelated backend timings.
- Include FHE/lattice proxy metadata: ring dimension or polynomial degree,
  coefficient-modulus count, decomposition digit count, transform/current
  domain, key-material reuse profile, evidence scope, and output-domain
  requirement. Implemented in `tools/benchmark_sweep.py` as
  `fhe-lattice-proxies`; the metadata is carried into
  `tools/evidence_database.py` rows, CSV columns, and Markdown summary tables.
- Add a computational-algebra scenario table before promoting any dense-GEMM
  claim into rank/determinant/solve/polynomial wording. Implemented in
  `tools/benchmark_sweep.py` as `computational-algebra-proxies` with dense
  finite-field BLAS, rank-k update, F4 dense phase, FGLM multiplication-matrix,
  and CRT/rational-reconstruction export labels.

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
- Keep bounded-u64 AUTO honest: the June 4, 2026 current-v2 refresh replaced
  the old vector-leadership assumption with shape-specific CPU, Direct-HIP, CK,
  and hipBLASLt winners. Vector routing needs current same-contract evidence
  for each promoted family.

Likely first slices:

- Add device-to-device native i64/u64 to RNS conversion kernels. Implemented in
  the current HIP direct/native currentness path as
  `hip_direct_native_i64_to_rns_device` and
  `hip_direct_native_u64_to_rns_device`, with
  `ensure_bounded_native_residues_current_for_rns_plan` materializing RNS
  device residues when a native-current bounded vector matrix feeds an RNS
  plan.
- Add selector explanations for vector vs RNS vs CPU choices.
- Add shape-family vector baselines for skinny/GEMV scenarios. Implemented in
  the scenario corpus as `skinny-gemv`, which emits N=1 bounded i64/u64
  scenarios and includes the canonical runtime `hip-vector-alu-int64` backend.
  `tools/test_benchmark_sweep.py` now pins that scenario to the runtime vector
  backend and N=1 shapes.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "RNS-Native Chains", and "Plan-Level Algebraic
  Lowering".

### 24. Reusable B Prepack And Tile-Swizzled Layout

Status: rocWMMA has a narrow non-tiled RNS B cache with
`rns_i8_tile_swizzled_b_v1` identity and `prepack-v2` keying. hipBLASLt has
workspace-local repeated-A and repeated-B prepack paths for fixed-prefix
single-K-block RNS work. Direct-HIP now has benchmark-only finite-u8 and
bounded prefix-9 transient-A/resident-B reuse paths. `production_prepack_cache_available`
remains `0`.

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

Status: current Windows `gfx1100` v2 release-review winners are Direct HIP at
512 and hipBLASLt at 1024. The June 4, 2026 bounded-i64 sweep wrote and locally
installed one reviewed 1024 hipBLASLt v2 cache entry; no 512 accelerator entry
was promotable. The first hipBLASLt slices removed repeated heuristic selection
and added workspace-local repeated-A and repeated-B prepack evidence. The
heuristic cache is intentionally non-durable and does not replace reviewed
autotune-cache identity, library-version rejection, stale-kernel rejection,
timing-split, split-K, finite-u8, or durable cache work. The fixed-prefix RNS
path caches A and B transposed hipBLASLt operands only when source version,
device, shape, prefix, and byte-size identity match; it is skipped for finite-u8,
adaptive/tiled plans, and split-K.

Technical direction:

- Tune winners, not losers. For 512, current evidence points back to Direct HIP;
  focus on pack/export cost, launch count, and one-shot active-prefix behavior
  before spending more time on CK/rocWMMA variants that already lost the v2
  release review.
- For 1024 hipBLASLt, focus on repeated-A/B prepack, scratch/reduce behavior,
  heuristic replay, and external reducer locality.
- Re-run 64/128 plus 2048 before assuming the current 512/1024 split persists
  across the surrounding shape range.
- Treat bounded i64 as the first production proving ground for residue-channel
  fusion, layout search, and generated variants.

Likely first slices:

- Direct-HIP 512 pack/export and launch-count tuning.
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
- Wrap64 direct-HIP v4 2048 exploratory run if runtime is tolerable.

Relation to new architecture work:

- Feeds "Roofline And Evidence Database", "Toolchain Matrix", and "Scenario
  Benchmark Corpus".

### 27. hipBLASLt Path

Status: heuristic lookup is cached in process-local memory for matching
device/shape/workspace. Fixed-prefix RNS repeated-A and repeated-B can reuse
workspace-local transposed operands when identity matches. This is not a public
prepack cache. The active hipBLASLt kernel identity is
`hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`: the separate
INT32 scratch reducer dispatches fixed-modulus kernels for 256, 255, and 251
and keeps the generic reducer for other ladder moduli.

Technical direction:

- Treat hipBLASLt as a black-box INT8 matmul primitive surrounded by RNS8-owned
  generated pack/reduce/export pipeline.
- Prewarm and replay selected algorithms at plan/workspace creation.
- Separate A pack, B pack, heuristic, matmul, scratch, reduce, export, and D2H
  phases wherever possible.
- Specialize external reduce kernels for 256/255/251 and prefix-9 bounded
  paths. Implemented for hipBLASLt scratch reduction in
  `src/backend_hipblaslt/hipblaslt_kernels.hip`; the source metadata,
  schema gates, stale-autotune checks, and fixtures now use
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`.
  Windows `gfx1100` smoke evidence under
  `temp/hipblaslt-reducer-v2-smoke/` validates schema and GPU events for
  bounded-i64 512/1024; the 1024 r9 capture reported
  `hipblaslt_i32_to_residue_reduce` median 80.38 us. This is local smoke
  evidence. The later June 4, 2026 current-v2 release review selected this
  kernel at 1024 with 4174 us median end-to-end and installed the reviewed local
  default cache entry for that exact bounded-i64 plan key.
- Use HIP Graphs or grouped host dispatch for repeated hipBLASLt workflows.

Likely first slices:

- Release repeated-A/B bounded-i64 1024 matrix with current workspace-local
  cache.
- A/B prepack support for finite-u8 and exact-wide where layout matches.
- External reducer specialization for 251/255/256 and prefix-9. Implemented
  for the current hipBLASLt scratch-reduce path as the v2 selected kernel above.

Relation to new architecture work:

- Feeds "Reusable B Prepack", "Generated Reducer/Epilogue Registry", "Fused Pack+GEMM", and
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

- Feeds "Generated Kernel Search", "Generated Reducer/Epilogue Registry", "Finite Data
  Specialization", and "Toolchain Matrix".

### 29. rocWMMA Path

Technical direction:

- Reduce shared-memory round-trip after `store_matrix_sync`; use fragment-map
  evidence before lane-owned residue emission.
- Specialize 256/255/251 reductions in the epilogue with shared reducer nodes.
  Implemented in the shared CK/rocWMMA RNS epilogue reducers and reflected in
  v2 selected-kernel/schema/cache identities; release-reviewed speedups remain
  open.
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
for 251/255/256. CK and rocWMMA expose common-modulus 251/255/256 selected
kernel identities backed by shared reducer helpers. The June 4-5, 2026
current-v2 release reviews closed the 64/128/512/1024 and hot-modulus 2048
promotion questions for ring 251, ring 255, ring 256, and field 251, and the
June 5 generic 2048 and field refreshes added CPU-backed entries for ring 127,
ring 253, field 127, and field 251 at 512. Sixteen event-valid accelerator
entries beat both CPU and Direct HIP where required and were installed in the
local default cache. Ring-255 64 was not promoted
because CPU reference was faster than the accelerator path, and ring-256 2048
hipBLASLt was not promoted because it lost to Direct HIP and lacked required
events.

Technical direction:

- Push finite reducer specialization into CK and rocWMMA epilogues.
- Add `finite_u8_centered_plane_v2` with layout selected by backend and
  distribution.
- Extend the reviewed matrix to generic prime/composite and larger 4096-class
  cases before assuming the current explicit-modulus split generalizes.
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
  Implemented: API planning, schema validation, cache installation, and
  backend-info tests now require explicit v2 selected-kernel identities for
  those moduli. Release-reviewed speedups remain open.
- Finite histogram-guided workload suite.
- 1024 finite winner retuning for CK ring and hipBLASLt field.

Relation to new architecture work:

- Feeds "Finite Data Specialization", "Generated Reducer/Epilogue Registry", and
  "End-To-End Layout Search".

### 31. Exact-Wide

Status: the June 4-5, 2026 current-v2 release reviews covered exact-wide signed
and unsigned 64/128/512/1024 plus the large-shape signed/unsigned 2048 slice.
Unsigned 64 promotes hipBLASLt, signed 512 promotes rocWMMA, signed 1024
promotes hipBLASLt, unsigned 1024 promotes CK, and signed/unsigned 2048 both
promote hipBLASLt. Signed 64, signed 128, unsigned 128, and unsigned 512 stay
on Direct HIP. The six promoted exact-wide entries are event-valid and installed
in the local default cache. The 2048 winners are export-bound after GEMM
acceleration, so export specialization and lazy residue-current workflows are
the next exact-wide execution targets.

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
- Direct-HIP fused pack+GEMM small-shape baseline. Implemented as the explicit
  `--transient-uniform-small-inputs` bounded prefix-9 benchmark/runtime path
  described in "Fused Pack+GEMM"; it provides schema-valid, event-valid
  same-shape evidence against persistent direct-HIP and public one-shot for
  64/128/512 on local Windows `gfx1100`.

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

- Prefix-9 bounded export specialization. Implemented for fixed-prefix
  Direct-HIP bounded export, and now also used by scheduled per-tile adaptive
  bounded export through selected-prefix 1..9 compile-time CRT dispatch.
- Exact-wide compact D2H staging. Implemented through the shared Direct-HIP
  compact export copier for exact-wide signed and unsigned limb exports. Large
  padded signed exact-wide exports now stay on the direct compact/pitched D2H
  path by default because repeated Windows `gfx1100` release captures under
  `temp/perf-work-queue/exact-wide-export-staging/` showed forced pinned staging
  losing for signed 512x512x512 limb-4 export (`crt_export` median 3084.39 us
  and 3497.97 us in two forced r33 runs versus 1009.40 us and 2282.29 us in the
  matching disabled r33 runs). `RNS8_HIP_PINNED_EXPORT_STAGING=1` can still force
  the path for experiments, and unsigned exact-wide limb-3 keeps the padded
  default because the same matrix showed a host-export median win
  (774.88 us forced versus 1084.08 us disabled).
- Status phase timing split across accelerator exports. Implemented in the
  benchmark/schema contract: bounded exports report `crt_export_status_memset`,
  `crt_export_kernel`, `crt_export_status_d2h`, and `crt_export_d2h`, while
  exact-wide exports report `exact_wide_export_status_memset`,
  `exact_wide_export_kernel`, `exact_wide_export_status_d2h`, and
  `exact_wide_export_d2h` across Direct-HIP, CK, rocWMMA, and hipBLASLt captures
  that use Direct-HIP export. Full-width exact-wide captures whose limb count
  covers the backend reconstruction width require zero-valued status phases and
  reject stale nonzero status timing.

Relation to new architecture work:

- Feeds "Reconstruction Backend", "Generated Reducer/Epilogue Registry", "CPU/GPU Hybrid
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

- Direct-HIP device schedule upload minimization. Implemented for scheduled
  GEMM workspaces: nonzero adaptive schedules upload only the compact active
  schedule, mixed zero/nonzero schedules retain the public device schedule for
  zero-tile extents, and uniform all-zero schedules report zero schedule
  workspace bytes.
- Broader device schedule buffer reuse across matching fingerprints.
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

- 64/128 one-shot vs persistent scenario matrix. Started with the fixed-prefix
  uniform-small bounded direct-HIP matrix under
  `temp/perf-work-queue/uniform-small-transient-i8/`, covering persistent,
  transient-native A+B, and public one-shot at 64/128/512 for bounded i64/u64.
  This is enough to keep the new transient path honest for small-shape work, but
  the broader CPU/vector/accelerator small-shape selector matrix remains open.
- CPU/direct/vector/accelerator selector explanation for tiny cases.
- HIP Graph repeated-small benchmark mode.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "Fused Pack+GEMM", "Shape-Specialized Paths",
  and "Persistent Grouped Scheduler".

### 36. Wrap64 Direct-HIP v4

Status: direct HIP v4 is the measured strict wrap64 GPU path for the local
Windows `gfx1100` 64/128/512/1024/2048 validation matrix. Paired release
captures under `temp/perf-work-queue/wrap64-v4/` showed median end-to-end
speedups over v3 of 1.07x, 1.17x, 1.02x, and 5.60x for default
64/128/512/1024 captures, and 1.22x, 4.67x, 1.07x, and 6.74x for
reuse-packed-input captures. The June 5, 2026
`large-release-validation` 2048 follow-up under
`temp/perf-work-queue/large-release-validation-2048-wrap64-current/` is now
release-reviewed with CPU byte-limb and Direct-HIP v4 baselines: Direct HIP
measured 58331 us median end-to-end versus 13423400 us for the CPU reference,
with required wrap64 GPU events and no missing required baselines.

Technical direction:

- Keep optimizing the direct-HIP baseline before another matrix-engine
  candidate. v4 uses direct unsigned byte products in the scalar direct-HIP
  kernel, dispatches a safe uint32 low-diagonal accumulator for `K <= 4096`,
  widens at carry propagation, and keeps scalar pack/export kernels for 64-like
  small shapes where vectorized compact pack/export lost end-to-end.
- Use vectorized byte-limb load/store through `uint64_t` where layout and shape
  evidence permits.
- Increase tile K or compute multiple output cells per thread if register
  pressure allows.

Likely first slices:

- Multi-output-cell direct-HIP variant. Implemented as the opt-in
  `direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5` experiment behind
  `RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT=1` for large `K <= 4096` wrap64 shapes.
  It computes two adjacent output cells per thread while reusing the staged A
  byte-limb cell, reports a distinct `wrap64_byte_gemm36_colpair_2d_kernel`
  GPU event phase, and has CPU differential coverage for a 256x257x5 odd-column
  tail. Windows `gfx1100` release captures under
  `temp/perf-work-queue/wrap64-colpair-experiment-current/` are schema/event
  valid, but do not promote the variant: at 512, default v4 beat colpair on
  both GEMM median (904.7 us vs 2501.2 us) and end-to-end median (2339 us vs
  7750 us); at 1024, colpair improved GEMM median only narrowly (4466.0 us vs
  4838.8 us) while losing end-to-end median (9775 us vs 9657 us). Keep default
  routing on v4 and treat colpair as an inspectable tuning candidate, not a
  performance win.
- 2048 v4 release run. Completed for CPU/direct-HIP release review; remaining
  follow-up is an ISA/resource report and any measured v4 microarchitecture
  tuning against this baseline.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits", "Shape-Specialized Paths", and
  "Scenario Benchmark Corpus".

### 37. Wrap64 Matrix Engine Redesign

Status: the internal rocWMMA candidate has strong correctness evidence but
loses structurally to direct HIP v4 at every reviewed 64/128/512/1024 shape.

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
- Compare candidate variants against direct-HIP v4 in wrap64 scenario corpus.

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

Status: the June 2026 GFX1100 evidence-tooling pass added the first dedicated
event and ISA reporting lane.

- Benchmark captures now carry helper-lane objects for `plan_packing`,
  `plan_lowering`, `requested_next_op`, `output_policy`, `auto_selector`,
  `target_variant`, and `device_allocation`. These fields make pack layout,
  lowering path, next-operation intent, status/export policy, AUTO fallback
  reasoning, target namespace, and post-warmup allocation behavior visible to
  `tools/result_compare.py` and `tools/benchmark_sweep.py`.
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
  reporting remains optional. Add `--capture <capture.json>` to link the ISA
  summary to a validated benchmark contract.
- Use `tools/gpu_counter_report.py <capture.json> --counter <csv-or-json>
  --isa-summary <isa-summary.json>` to assemble temp-only counter/ISA
  explanations under `temp/gpu-counter-reports/`.
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

- Compact contiguous export D2H fast path. Implemented for Direct-HIP bounded,
  finite-u8, exact-wide, and wrap64 exports: contiguous host outputs now use a
  linear `hipMemcpy`, while padded leading dimensions keep the existing
  `hipMemcpy2D` path. Windows `gfx1100` release smokes under
  `temp/compact-export-release/` validated schema/events across bounded i64,
  finite-u8, exact-wide, and wrap64; finite-u8 512 median export D2H improved
  from the prior 66.2 us r33 capture to 49.2 us, but end-to-end averages remain
  too noisy for a headline claim.
- Pinned host export staging for large padded Direct-HIP outputs. Implemented
  as an internal reusable thread-local HIP pinned host buffer for bounded,
  finite-u8, and selected exact-wide Direct-HIP exports whose compact output
  copy is at least 64 KiB and whose caller destination has padding. The backend
  copies compact device export buffers into pinned staging, scatters into the
  caller's requested leading dimension on the host, preserves the existing
  required `crt_export_d2h` / `finite_export_d2h` /
  `exact_wide_export_d2h` GPU event labels, and emits an ignored diagnostic
  timing sample named `export_host_staging_copy` when backend timing is enabled.
  `rns8-bench` now records `direct_hip_export_staging_policy`,
  `direct_hip_pinned_export_staging_threshold_bytes`, and the benchmark output
  destination layout in `timing_metadata`, so capture metadata distinguishes
  padded-only default, disabled, and forced staging runs. The benchmark also
  exposes `--output-ld-padding`; captures now report `output_logical_ld`,
  `output_ld_padding`, `benchmark_output_logical_ld`, and
  `benchmark_output_ld_padding`, and schema/sweep review keys treat padded and
  contiguous destinations as different contracts.
  `RNS8_HIP_PINNED_EXPORT_STAGING=0` disables the staging path, while
  `RNS8_HIP_PINNED_EXPORT_STAGING=1` forces it for contiguous-output A/B
  measurements and for force-only semantics. A 512x512 contiguous-output
  Windows `gfx1100` smoke did not support default-on staging, so contiguous
  benchmark outputs keep the existing linear D2H path by default. Focused
  128x128 Windows `gfx1100` smokes under
  `temp/perf-work-queue/padded-output-ld/` validated default padded, disabled
  padded, and forced contiguous captures with matching logical checksums and
  required GPU events. Exact-wide signed padded staging is now force-only after
  local 512x512x512 limb-4 r33 captures showed it losing to disabled staging on
  host export medians, while exact-wide unsigned limb-3 keeps default padded
  staging because the same validation showed a host-export median win. This is
  an implemented transfer path with semantic-specific default policy, not a
  blanket headline speedup claim.
- Wrap64 padded export staging follow-up. Implemented the reusable compact
  export copier as a shared Direct-HIP internal utility so the wrap64 byte-limb
  export path can exercise the pinned staging pool and host scatter path without
  duplicating transfer code. Wrap64 does not enable padded pinned staging by
  default: `RNS8_HIP_PINNED_EXPORT_STAGING=1` can force the path for experiments,
  but default wrap64 export stays on the direct compact/pitched D2H copy until a
  real win exists. Windows `gfx1100` captures under
  `temp/perf-work-queue/wrap64-padded-export-staging/` validated schema/events
  for default, disabled, and forced staging at 128 and 512 with matching
  checksums inside each shape. The 512x512 forced capture lost badly versus the
  default policy (`wrap64_export_d2h` median 244.8 us versus 66.4 us, end-to-end
  average 7177.78 us versus 3029.56 us), so this is correctness/tooling
  coverage and an explicit deprioritization of default wrap64 pinned export
  staging, not a promoted speedup.
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

### 43. Reuse Contract Ledger And Persistent Matrix Policy

Reuse wins are real, but they are not same-contract replacements for one-shot
calls. The queue needs a durable contract ledger before AUTO can reason about
them.

Technical direction:

- Define a `reuse_contract` record for benchmark captures and reviewed cache
  candidates: operand role, source matrix id, source version, setup operation,
  setup cost, measured repeat count, steady-state break-even repeat count,
  output domain, next-op hint, target id, selected backend, selected kernel,
  and invalidation reason.
- Keep public prepack cache, benchmark reuse modes, resident matrix reuse, and
  AUTO eligibility as separate mechanisms. They may share metadata, but they
  must not silently share promotion policy.
- Add stale-source proofs: mutated input, changed descriptor, changed semantic
  contract, changed modulus/prefix schedule, changed target id, changed HIP SDK,
  changed backend version, and changed workspace fingerprint must reject reuse.
- Track setup cost in both absolute microseconds and amortized per repeat.
  Every report should show both setup-inclusive and steady-state speedups.

Likely first slices:

- Add `tools/reuse_contract_report.py` over reviewed captures and installed
  cache entries.
- Extend `tools/result_compare.py` review output with reuse break-even rows.
- Add schema fixtures rejecting reuse captures without source-version material
  or repeat-count metadata.

Promotion gate:

- A reuse path can only influence AUTO after the ledger proves a concrete
  workload family and repeat-count threshold, with exact CPU final-output
  comparison and stale-source rejection.

### 44. Persistent Resident Matrix Lifetime Implementation

Persistent resident matrices are the RNS8 representation story, but benchmark
paths still hide too much lifetime behavior inside one process call.

Technical direction:

- Make resident A/B/C matrix lifetimes explicit in benchmark metadata and
  future public API design: creation, pack/import, currentness, source version,
  workspace binding, output currentness, and release/reset.
- Separate resident RNS, resident native, resident finite, and resident byte
  limb states. Do not infer semantic meaning from `int64_t`, `uint64_t`, or
  `uint8_t` host types.
- Add tests that mutate resident inputs after prepack/import and prove old
  workspaces or prepack handles reject stale source versions.
- Treat persistent output as a first-class next-op input. A matrix that is
  device-current RNS output should feed an RNS consumer without host export
  unless the caller requests native/final output.

Likely first slices:

- Add benchmark scenario pairs for create-once/reuse-many versus fresh-per-call
  resident matrices.
- Add `resident_lifetime` metadata to captures: matrix roles, source versions,
  current storage state, output domain, and workspace identity.
- Add public C API design notes for persistent output-domain handles without
  changing ABI in this slice.

Promotion gate:

- Resident lifetime changes can route only after source-version, workspace, and
  output-domain mismatches fail deterministically across CPU and HIP contexts.

### 45. Device Grouped Dispatcher For Many-Small Workloads

Host API batching proves that repeated host orchestration matters, but it does
not remove device launch and scheduling overhead enough for most small
bounded/finite cases.

Current status:

- Branch-local `rns8-bench --grouped-dispatch N` now executes a
  benchmark-owned same-shape persistent task group instead of reporting
  metadata-only unsupported status. It reuses the existing resident task
  machinery with one shared plan, one A/B/C matrix triplet and workspace per
  task, aggregate pack/GEMM/export timing, per-task output checksums folded
  into the capture checksum, schema-v4 `benchmark_grouped_dispatch_evidence`
  metadata, and `many_small_grouped_report.py` rows with
  `capture_status=executed`.
- Tiny Windows `gfx1100` Direct-HIP smokes for bounded-i64, finite-u8, and
  exact-wide signed validate schema and required GPU events. These smokes prove
  the evidence path, not a performance win.
- `tools/many_small_grouped_report.py` now compares grouped-dispatch per-task
  medians against the fastest independent baseline and the same-backend
  host-batch baseline for the normalized same-output contract, and emits JSON
  plus Markdown summaries. The first release-count exact-wide signed 64
  group32 capture was a benchmark candidate win at 991.94 us per task; the
  current async exact-wide export-slab follow-up improves that to 792.66 us
  per task while still beating both independent Direct HIP and Direct-HIP
  hostbatch32.
- The current one-kernel grouped export follow-up keeps the same exact-wide
  signed 64 group32 checksum and required events while replacing the per-task
  export-kernel loop with one grouped export launch and one compact D2H. The
  signed capture reports 795.19 us per task, still 4.88x faster than
  independent Direct HIP and 2.39x faster than hostbatch32, but effectively
  flat versus the 792.66 us async slab median. The export phase itself improves
  materially: aggregate `crt_export` average drops from 5670 us to 1212 us.
  The exact-wide unsigned 64 group32 twin is schema/event-valid at 703.34 us
  per task, but it remains historical smoke evidence superseded by the focused
  unsigned baseline closeout under
  `temp/perf-work-queue/many-small-grouped-unsigned-current/`.
- The grouped pack+export follow-up under
  `temp/perf-work-queue/many-small-grouped-pack-current/` now groups the A and
  B pack stages for the same benchmark-owned exact-wide 64 group32 path: compact
  native slabs are copied once per operand per measured repeat, one grouped pack
  kernel runs per operand, GEMM still loops through resident tasks host-side,
  and the one-kernel grouped export path handles contiguous output. The signed
  capture reports 228.06 us per task, 17.01x faster than independent Direct HIP,
  8.34x faster than hostbatch32, and 3.49x faster than the previous
  grouped-export-only capture. The aggregate host pack average drops from
  16873 us to 778 us, and GPU event pack average drops from 12815 us to 461 us.
  The exact-wide unsigned 64 group32 twin is schema/event-valid at 249.63 us per
  task, but it remains historical smoke evidence superseded by the focused
  unsigned baseline closeout.
- The grouped pack+GEMM+export follow-up under
  `temp/perf-work-queue/many-small-grouped-gemm-current/` adds a same-shape
  task-prefix Direct-HIP grouped GEMM for the benchmark-owned exact-wide 64
  group32 path. The strategy is
  `device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h`: compact
  A/B slabs are copied once per measured repeat, grouped pack kernels produce
  per-task resident RNS matrices, one grouped task-prefix GEMM kernel group
  covers all resident tasks, and one grouped exact-wide export kernel plus one
  compact D2H handles output. The signed capture reports 66.47 us per task,
  58.37x faster than independent Direct HIP, 28.63x faster than hostbatch32,
  and 3.43x faster than the grouped pack+export capture. Event median
  `rns_gemm` drops from 4701.94 us to 168.70 us. The exact-wide unsigned 64
  group32 twin is schema/event-valid at 63.56 us per task; the focused
  unsigned rerun under
  `temp/perf-work-queue/many-small-grouped-unsigned-current/` adds the missing
  independent and hostbatch32 baselines and classifies the same strategy as a
  candidate win at 79.09 us per task, 18.70x faster than independent Direct
  HIP and 13.48x faster than Direct-HIP hostbatch32.
- Grouped-dispatch captures now emit a schema-v4 `task_descriptor_contract`
  nested under `grouped_dispatch`. The contract records
  `same_shape_resident_task_triplets_v1`, `single_same_shape_bucket`, task
  count, shape/tile/prefix key, semantics, output domain, source-version
  policy, one-workspace-per-task shared-plan policy, combined per-task checksum
  policy, fail-fast per-task operation status, and whether the executed path
  used a host resident task loop or device pointer tables plus compact slabs.
  `tools/many_small_grouped_report.py` surfaces that descriptor policy and
  blocks candidate wins when the contract is invalid. A rebuilt
  `windows-release` exact-wide unsigned grouped smoke under
  `temp/perf-work-queue/grouped-descriptor-contract-smoke/` is schema-valid and
  has required Direct-HIP GPU events.
- The bounded grouped pack+GEMM follow-up adds the generic private Direct-HIP
  grouped RNS helper for bounded i64/u64 and routes benchmark
  `--grouped-dispatch` bounded host-batch lanes through
  `device_grouped_pack_gemm_host_exports`. Focused release controls under
  `temp/perf-work-queue/grouped-bounded-release-controls/` cover CPU and
  Direct-HIP independent baselines, Direct-HIP hostbatch32, and Direct-HIP
  grouped rows for bounded-i64/u64 64 group32. The report classifies both
  grouped rows as candidate wins: i64 at 544.56 us per task, 1.97x faster than
  the best independent CPU baseline and 2.19x faster than hostbatch32; u64 at
  532.47 us per task, 1.46x faster than the best independent CPU baseline and
  2.11x faster than hostbatch32. Required GPU events pass, and that historical
  row showed the remaining bounded export bottleneck before the grouped export
  closeout below.
- The bounded grouped export closeout changes the bounded grouped strategy to
  `device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h`, adding
  grouped i64/u64 CRT export kernels that write one compact device output slab
  followed by one compact D2H. Focused release controls under
  `temp/perf-queue-grouped-bounded-export-release/` classify both bounded rows
  as stronger candidate wins: i64 at 53.625 us per task, 19.97x faster than
  the best independent CPU baseline and 23.32x faster than hostbatch32; u64 at
  53.25 us per task, 10.95x faster than the best independent CPU baseline and
  22.95x faster than hostbatch32. Required GPU events and same-task-count
  checksum parity pass.
- The broader grouped matrix closeout under
  `temp/perf-queue-grouped-broader-release/` makes the grouped-dispatch
  scenario corpus self-contained for the current bounded/finite grouped rows
  by adding missing same-task-count host-batch controls and the open
  bounded-i64 128 and bounded-u64 skinny grouped cases. The release report has
  33 schema-v4 captures, 19 GPU captures with required events, and five
  `many_small_grouped_report.py` candidate wins with no missing baselines:
  bounded-i64 64 group32 at 57.41 us per task, bounded-i64 128 group64 at
  55.88 us per task, bounded-u64 64 group32 at 59.38 us per task,
  bounded-u64 128x1x1024 group128 at 100.76 us per task, and finite-ring u8
  mod251 64 group32 at 33.53 us per task. The new finite grouped path uses
  `device_grouped_pack_gemm_and_finite_export_kernel_batched_d2h`, so pack,
  finite GEMM, finite export, and compact output D2H are all grouped rather
  than per-task public API calls.
- The exact-wide 128 grouped control closeout under
  `temp/perf-queue-grouped-exact128-release/` adds same-task-count exact-wide
  signed and unsigned 128 host-batch controls plus grouped-dispatch rows. The
  focused release report has 20 schema-v4 captures, 12 GPU captures with
  required events, and four `many_small_grouped_report.py` candidate wins with
  no missing baselines: exact-wide signed 64 group32 at 68.12 us per task,
  exact-wide signed 128 group32 at 184.97 us per task, exact-wide unsigned 64
  group32 at 75.53 us per task, and exact-wide unsigned 128 group32 at
  155.88 us per task. The 128 rows keep the same grouped pack, same-shape
  grouped Direct-HIP RNS GEMM, grouped exact-wide export kernel, and compact
  output D2H contract as the 64 rows.
- The descriptor-backed execution helper closeout moved bounded, finite, and
  exact-wide benchmark grouped lanes onto internal Direct-HIP helpers that
  validate the grouped descriptor/resource aggregate before pack, GEMM, and
  export. Debug smokes under `temp/grouped-execution-helper-smoke/` and
  `temp/grouped-execution-helper-smoke-unsigned/` produced seven schema-v4
  captures with required Direct-HIP GPU events across bounded i64, finite-ring
  u8, exact-wide signed, and exact-wide unsigned grouped paths. This is
  structural cleanup evidence, not a new release performance claim.
- The same-shape bucket-plan closeout moves grouped descriptor construction
  into `hip_direct_build_same_shape_grouped_bucket_plan`, with benchmark lanes
  now holding a backend-built bucket plan and extracting its checked
  single-bucket descriptor for pack/GEMM/export. The focused debug smoke under
  `temp/grouped-bucket-plan-smoke/` produced seven schema-v4 captures with
  required Direct-HIP events across the same bounded, finite, exact-wide
  signed, and exact-wide unsigned grouped paths. This closes the benchmark-local
  descriptor-construction gap, but not the public/generic multi-bucket ABI.
- The remaining performance work is now the broader dispatcher and contract
  work: public or mechanically routed task descriptors, descriptor/lifetime
  validation beyond benchmark metadata, release-size comparisons against
  fastest independent calls, and durable workload-family wins beyond
  benchmark-owned same-shape dispatch.

Technical direction:

- Generalize the benchmark-owned same-shape grouped task GEMM into a descriptor
  dispatcher that accepts an array of same-shape or bucketed task descriptors
  and launches one grouped device workload where possible.
- Start with same semantic, same shape, same prefix/modulus schedule, and same
  output policy. Mixed semantics and mixed shapes come later through buckets,
  not one fully generic dispatcher.
- Keep per-task exact checksums and per-task status. A single failed task must
  not mask successful neighboring tasks.
- The benchmark-side descriptor contract is now explicit, internally routed
  through descriptor-backed grouped phase helpers, and constructed through a
  backend-owned same-shape bucket-plan builder. The next implementation step is
  turning that bucket plan into public or generic device-readable compact
  records: A/B/C storage offsets, leading dimensions, source versions, selected
  prefix/modulus, output policy, and checksum/export mode.

Likely first slices:

- Public or generic descriptor dispatcher for the benchmark-owned exact-wide,
  bounded, and finite paths now that internal descriptor-backed phase execution
  and same-shape bucket-plan construction have replaced per-lane pointer/slab
  assembly.
- Bucketed same-semantic grouped descriptors so same-shape task batches stop
  being the only executable grouped form.

Promotion gate:

- Grouped execution remains benchmark-only until it beats fastest independent
  calls setup-inclusively across at least one durable workload family, with
  required GPU events and exact per-task CPU comparisons.

### 46. Exact-Wide Final-Output Chain Matrix And RNS Output API Draft

Residue-current chains are promising because they remove intermediate CRT
export. They still need same-final-output proof and a clean API story.

Technical direction:

- Compare three contracts separately: independent calls with final output after
  each GEMM, RNS-chain with one final export, and RNS-chain that stays resident
  for a following RNS operation.
- Final-output chain captures now exist in branch-local benchmark tooling:
  `rns8-bench --residue-chain-final-export` measures the final logical export
  inside each repeat, schema v4 records `residue_chain_final_host_export`, and
  the `rns-chain-final-output` scenario family emits bounded and exact-wide
  candidates. Bounded and exact-wide independent export/repack controls now
  also exist through `--residue-chain-independent-final-export` and
  `residue_chain_independent_final_host_export`; focused bounded-i64 128 and
  bounded-u64 256 reports show resident chains beating same-backend independent
  export/repack by 1.84x and 1.96x respectively, while focused exact-wide
  signed 128 and unsigned 256 reports show 9.80x and 10.80x same-backend wins.
  The next work is broader exact-wide chain matrices and API lifetime semantics,
  not basic capture plumbing.
- Draft a public API model for residue-current output handles: explicit output
  domain, semantic contract, prefix/modulus schedule, currentness, and allowed
  consumers.
- Include reusable-B and persistent A/B/C variants because chain wins can come
  from either skipped export or skipped setup.

Likely first slices:

- Exact-wide signed/unsigned 128 and 512 chain-length 2/3/5 release matrix.
- Broader exact-wide same-output reports beyond the focused signed 128 and
  unsigned 256 controls: independent final-output calls with explicit limb
  import/repack versus chain plus final export.
- ABI-neutral design doc for future residue-current output and lazy export.

Promotion gate:

- No public output-domain API or AUTO routing until the final requested output
  is exact, measured, and compared against independent-call baselines.

### 47. Export-Bound Exact-Wide Optimization And Limb Variants

Large exact-wide accelerator wins are now export-bound. The next wins are
likely reconstruction, status handling, compact D2H, and limb-count policy.

Technical direction:

- Treat exact-wide export as a backend with selected-kernel identity, not a
  passive copy step.
- Compare limb widths only within the caller-requested contract. Three-limb
  signed output is valid for full prefix-20 range, but it does not replace a
  caller's four-limb ABI request.
- Search fixed limb-count kernels for 1/2/3/4/8/16/32 limbs and runtime fallback
  for all other supported widths.
- Split export phases: reconstruction kernel, status memset, status D2H,
  compact output D2H, padded scatter, host staging copy, and checksum/export
  verification.
- Keep export constants target-visible: constant memory, compact device tables,
  LDS staging, or inline fixed constants must be reflected in ISA reports.

Likely first slices:

- Release A/B for exact-wide signed/unsigned 512/1024/2048 with 3/4/8 limbs.
- Prefix-20 fixed constants placement experiment for Direct-HIP export.
- Compact contiguous and padded-output export matrix with `output_policy`
  grouping.

Promotion gate:

- Promote only a setup-inclusive export-path win for the exact requested limb
  contract, with exact CPU limb comparison and required export GPU events.

### 48. CRT/Reconstruction Fusion And GPU Export Kernel Zoo

The current export path reconstructs after GEMM. Some workloads can fuse
reduction, sign/range checks, reconstruction, and output packing.

Technical direction:

- Add named reconstruction controller families: fixed-prefix Garner, mixed
  radix, product-tree CRT, balanced CRT, sign-only/range-only partial
  reconstruction, fused reducer-to-CRT, and residue-current no-export.
- Keep bounded, exact-wide, finite-u8, and wrap64 separated. CRT fusion is not
  meaningful for strict wrap64 byte-limb low64 export, and finite canonical
  export has different status semantics.
- Explore fusing `i32 -> centered residue -> CRT contribution` for Direct-HIP
  and accelerator epilogues when global residue stores dominate.
- Add check-residue or redundant-residue research variants only with explicit
  verification metadata; they are not default exact APIs.

Likely first slices:

- Direct-HIP prefix-9 bounded fused reducer/export prototype for one-shot.
- Exact-wide prefix-20 product-tree CRT benchmark kernel.
- Export kernel zoo schema fixtures rejecting stale generic reconstruction
  identities.

Promotion gate:

- Every reconstruction variant needs selected-kernel, epilogue, workspace,
  constants, status policy, and ISA evidence before it can enter a reviewed
  A/B comparison.

### 49. Budgeted 4096 Release Gate

Status: completed as a validation lane and archived in
[performance-gain-completed-work.md](performance-gain-completed-work.md). The
4096 rows are no longer only release-gate evidence for supported non-reuse
contracts: the promotion-ledger closeout installed eight eligible bounded,
finite hot-modulus, and exact-wide 4096 cache entries. Strict wrap64 remains a
Direct-HIP correctness path, not an AUTO cache entry, and repeated-B remains a
reuse workload contract.

4096 shapes are useful for throughput classification, but they can consume a
lot of local time and CPU reference budget.

Technical direction:

- Add a budgeted runner for 4096 release scenarios: chunked CPU reference,
  resume, max-new-captures, timeout, memory cap, and reviewed summary output.
- Split 4096 into classification tiers: GPU-only exploratory, CPU-backed
  release, and installed-cache-eligible release. Only the last tier can promote.
- Prefer exact-wide and finite 4096 only when 2048 evidence suggests the
  backend is not purely launch-bound.
- Keep repeated-B and chain variants as workload-contract evidence unless the
  setup/lifetime ledger is complete.

Likely first slices:

- `large-release-validation-4096-budgeted` scenario with disabled-by-default
  execution. Implemented and run: bounded i64/u64 4096 now have CPU,
  Direct HIP, runtime vector ALU, hipBLASLt, CK, and rocWMMA release captures;
  finite hot 4096 now has CPU, Direct HIP, hipBLASLt, CK, and rocWMMA release
  captures for ring 251/255/256 and field 251; exact-wide signed and unsigned
  4096 have CPU, Direct HIP, hipBLASLt, CK, and rocWMMA release captures;
  strict wrap64 4096 has byte-limb and Direct-HIP release captures. Follow-up
  one-pass reference captures first completed
  exact-wide signed `cpu-reference` and strict wrap64 `wrap64-byte-limb` with
  full-output checksums, then release-reference reruns completed both required
  rows with 3 warmups and 9 measured repeats. Exact-wide signed `cpu-reference`
  recorded 113755000 us median end-to-end with checksum
  `5508849193854467465`; strict wrap64 `wrap64-byte-limb` recorded
  102905000 us with checksum `13518998852724169131`. The exact-wide unsigned
  budgeted group recorded matching checksum `9643325300233475427`, with
  hipBLASLt fastest at 162382 us, Direct HIP at 614116 us, and CPU reference
  at 105462000 us.
- CPU reference chunking report that records chunk size, seed, checksum, and
  wall time. Superseded by the completed long-timeout 3/9 release-reference
  rows for the current 4096 gate; keep chunking as a future cost-reduction
  tool, not as a remaining blocker for this validation lane.
- Per-capture timeout enforcement. Implemented as
  `benchmark_sweep.py --capture-timeout-seconds`, which records timed-out
  captures as `.failed.json` with explicit timeout metadata.
- Review blocker for 4096 groups that lack required baselines. Implemented in
  `tools/release_gate_report.py` schema v2 as grouped release-gate readiness
  output with required baselines, missing valid baselines, failed timeout
  records, unattempted baselines, release-capture readiness, and blocker
  counts. The current combined release-reference 4096 report has 44 completed
  captures plus 2 historical failed timeout rows across 9 groups; exact-wide
  signed, exact-wide unsigned, strict wrap64, bounded, and finite hot groups
  now have complete required release-review baselines. Historical
  timeout rows remain visible but no longer count as active group blockers once
  the required backend has a valid capture.

Promotion gate:

- No 4096 public claim without CPU/direct-HIP baselines, required GPU events,
  fixed seed, release build, and complete target/toolchain metadata.

### 50. hipBLASLt Bounded-i64 1024 A/B Lane

The current bounded-i64 1024 hipBLASLt win is narrow. It is valuable enough to
protect, but not wide enough to stop tuning.

Implementation status, 2026-06-06:

- `tools/bounded_i64_1024_review.py` is the focused disposition path. It joins
  setup-inclusive 1024 bounded-i64 captures with target validation, variance,
  counter/resource evidence, and promotion-ledger rows, then reports
  `keep cache`, `replace cache`, `keep experimental`, `drop/deprioritize`, or
  `unsupported accelerator`.
- `benchmarks/scenarios/hipblaslt_bounded_i64_1024_ab.json` now covers the
  required comparator set plus stable-A, stable-B, and stable-A+B reuse roles.
  This is future CDNA validation infrastructure, not CDNA evidence.

Technical direction:

- Compare current specialized reducer against scratch layout, pack layout,
  reduction kernel, stream/event, and workspace reuse variants.
- Include Direct-HIP current baseline and hipBLASLt A/B/A+B reuse variants in
  the same release review, but keep reuse contract comparisons separate.
- Track whether hipBLASLt wins from matmul throughput, pack savings, reducer,
  or export timing. Do not optimize the wrong phase.
- Add regression guard captures so stale hipBLASLt cache entries fail when
  selected-kernel or event labels change.

Likely first slices:

- Focused 1024 bounded-i64 release A/B matrix with current v2, reducer
  variants, prepacked variants, and Direct-HIP baseline.
- ISA/counter report for hipBLASLt pack/reduce phases where available.
- Review report that flags narrow wins under a configurable margin.

Promotion gate:

- Keep or replace the cache entry only when exact correctness, required events,
  and setup-inclusive median continue to beat Direct HIP.

### 51. Direct-HIP Resident Matrix Redesign After Colpair Rejection

The resident colpair attempt taught the right lesson: a narrower GEMM median is
irrelevant if end-to-end timing regresses.

Technical direction:

- Redesign from the resident dataflow backward: resident RNS layout, schedule
  upload, tile shape, selected-prefix groups, export interaction, and workspace
  reuse.
- Avoid routing changes until the candidate is stable across repeated reruns
  and not just one favorable GEMM phase.
- Use occupancy/resource reports before writing more variants: register
  pressure, LDS, scratch, global stores, coalescing, and wait states should
  explain the previous outliers.
- Keep one-shot native-input colpair separate from resident RNS input kernels.

Likely first slices:

- Resident Direct-HIP audit report for current v2 versus rejected colpair.
- New resident tile/layout prototype with explicit selected-kernel identity.
- Rerun fixed-prefix 9 and selected-prefix 512/1024 side by side.

Promotion gate:

- Candidate must beat current resident Direct-HIP median end-to-end, not only
  raw GEMM, and must preserve exact CPU comparison plus required events.

### 52. Finite Generic Modulus Family Map

Generic finite-u8 now has real wins, but the map is incomplete and still easy
to over-generalize.

Technical direction:

- Classify moduli by field/ring, prime/composite, hot/specialized/generic,
  reducer structure, and signed-centered representation cost.
- Use same shapes across moduli to identify backend families rather than one-off
  cache entries.
- Keep modulus-specific selected-kernel names for hot paths and generic names
  for true generic paths. Schema must reject stale hot/generic confusion.
- Include CPU and Direct-HIP baselines for every promoted generic modulus.

Likely first slices:

- 128/512/1024/2048 release map for moduli 127, 251, 253, 255, 256 and one
  additional prime/composite pair.
- Report table grouped by modulus class and backend winner.
- Generic reducer ISA gate for divide-free hot paths where expected.

Promotion gate:

- Promote only exact modulus/semantic/shape keys with required events and no
  implied family-wide claim.

### 53. Modulus-Set Search And Residue-Count Autotuning

RNS8 currently treats the default modulus ladder as a correctness and
performance object. Future tuning should search it deliberately, not casually.

Technical direction:

- Search modulus sets by range product, reducer cost, CRT constant cost,
  residue-channel fusion friendliness, accelerator compatibility, export cost,
  and target-specific instruction behavior.
- Separate public default ladder changes from experimental benchmark ladders.
  Any public ladder change touches spec, schema, cache keys, default-prefix
  tables, and exact reference tests.
- Add residue-count autotuning for bounded and exact-wide: minimum range prefix,
  safety margin, redundant/check residue, export limb width, and selected
  backend capability.
- Include FHE/lattice-inspired NTT-friendly primes as workload proxies only
  unless RNS8 explicitly implements NTT/key-switch operations.

Likely first slices:

- `--modulus-set experimental:<name>` benchmark-only capture metadata.
- Offline search tool that emits candidate ladders with product bits, reducer
  constants, and expected prefix counts.
- Schema fixtures proving non-default ladders cannot be mistaken for default
  captures or cache entries.

Promotion gate:

- No default modulus change without CPU/GPU differential tests, prefix product
  table updates, stale-cache invalidation, and release comparisons for every
  affected semantic.

### 54. Adaptive Prefix Grouped Scheduler

Adaptive prefix schedules can delete work but add scheduling and launch
overhead. Grouping is the next structural test.

Technical direction:

- Group work by selected prefix, tile extent, modulus plane, and zero-mask
  class. Avoid one launch per tiny group.
- Encode group descriptors compactly in workspace memory and report group count,
  active tile count, zero tile count, and selected prefix histogram.
- Compare grouped adaptive execution against current compact active-prefix
  scheduling and fixed-prefix fallback.
- Keep accelerator backends separate until Direct-HIP proves the scheduler
  shape.

Likely first slices:

- Direct-HIP adaptive grouped benchmark path for bounded-u64 adaptive-bands.
- Per-group event labels for pack, grouped GEMM, zero memset, and export.
- Result comparison key for schedule strategy and group descriptor identity.

Promotion gate:

- Grouping must beat current adaptive Direct-HIP setup-inclusively and preserve
  exact per-tile CPU reference behavior.

### 55. Streaming Pack/Compute/Export Overlap

Once resident and reuse policies are explicit, repeated workflows can pipeline
pack, compute, and export instead of serializing every phase.

Technical direction:

- Use separate HIP streams for pack-next, GEMM-current, and export-previous
  only when buffers and status storage are partitioned safely.
- Require explicit event dependencies: pack completion before GEMM, GEMM before
  export, status before host read, and final stream synchronization before
  checksum.
- Start with benchmark-only double buffering, then triple buffering if export
  remains visible.
- Include pinned and compact transfer policy in the overlap contract.

Likely first slices:

- Repeated-B Direct-HIP bounded 512/1024 overlap benchmark.
- Exact-wide 128/512 chain plus final export overlap scenario.
- Event report that shows overlap by wall-clock end-to-end shrinking more than
  the sum of per-stream event medians.

Promotion gate:

- Keep disabled unless serial and overlapped paths produce identical statuses,
  checksums, errors, and currentness under stress and failure cases.

### 56. Tile-Shape Autotuning

Tile shape interacts with occupancy, memory coalescing, matrix-engine fragment
shape, export locality, and schedule overhead.

Technical direction:

- Generate tile M/N/K variants for Direct-HIP, CK, rocWMMA, hipBLASLt wrappers,
  and wrap64 separately. One tile policy will not fit all backends.
- Encode tile identity in selected-kernel and autotune keys. A capture with a
  tuned tile must not look like the default kernel.
- Join tile sweeps with resource reports: VGPR, SGPR, LDS, scratch, occupancy,
  wait instructions, global stores, and event-phase shifts.
- Include rectangular and skinny shapes; square-only tile tuning can harm real
  workloads.

Likely first slices:

- Direct-HIP bounded 512/1024 tile search over 64/128/256 M/N and K-block
  variants.
- Finite-u8 2048 tile sweep for the current Direct-HIP baseline and one
  accelerator wrapper.
- `tools/tile_shape_report.py` grouping captures by tile identity and resource
  limiter.

Promotion gate:

- Promote per target/backend/semantic/shape family only after release A/B with
  stable events and stale-kernel schema rejection.

### 57. Workspace Arena Implementation Lane

Allocation counters now reveal whether measured repeats allocate. The next step
is an arena that prevents avoidable allocation by design.

Technical direction:

- Build a plan/workspace-owned device arena for scratch, schedules, status,
  compact exports, and temporary packed buffers.
- Make suballocation deterministic and source-versioned. Arena reuse must fail
  on mismatched plan, target, backend, semantic, shape, prefix/modulus, and
  output policy.
- Keep stream safety explicit: either one arena per stream pipeline lane or
  event-guarded reuse.
- Report arena size, high-water mark, suballocation count, and measured-repeat
  allocation delta.

Likely first slices:

- Direct-HIP bounded/exact-wide workspace arena for benchmark persistent paths.
- CTest that proves measured repeats do not allocate after warmup.
- Schema fields for arena high-water mark and allocation-free repeat proof.

Promotion gate:

- Promote only when allocation counters prove zero measured-repeat allocation
  and stale workspace tests fail cleanly.

### 58. HIP Graph Replay Expansion Beyond Direct-HIP RNS Chains

HIP Graphs are attractive for repeated fixed-shape workflows, but they freeze a
lot of state. The current branch now has a narrow benchmark-only Direct-HIP
resident RNS chain graph replay lane. Broader graph work still needs strict
identity and error equivalence before routing.

Technical direction:

- Preserve the current narrow contract first: fixed Direct-HIP plan, fixed
  matrix descriptors, fixed output policy, explicit nonblocking HIP stream, and
  resident RNS GEMM launches only.
- Treat pack, export, finite-u8, wrap64, and mixed-backend graph capture as
  follow-on work, not implicit coverage from the Direct-HIP chain lane.
- Include status/error behavior in the graph contract. A graph replay must not
  hide range errors, stale currentness, or failed HIP calls.
- Compare graph replay against serial ordinary calls, host API batching, and
  streaming overlap for the same workload.
- Keep graph capture benchmark-only until ABI implications are clear.

Likely first slices:

- Release-size Direct-HIP repeated bounded 128/512 fixed-shape graph benchmark
  against the same non-graph residue-current chain.
- Finite-u8 128 graph benchmark where CPU/Direct-HIP setup dominates.
- Extended graph identity schema object with plan/workspace/source/output
  hashes once the narrow lane survives release review.

Promotion gate:

- No expansion or routing until the narrow graph lane matches ordinary calls for
  success, range errors, stale inputs, setup accounting, and cleanup across
  repeated runs.

### 59. Shape-Family AUTO Shadow Mode

The reviewed cache is exact-shape by design. A shape-family selector should be
observable long before it routes.

Technical direction:

- Use `tools/shape_family_shadow_report.py` to propose a family recommendation
  from reviewed exact cache entries and list blockers such as exact-shape cache
  ownership, missing same-family evidence, advisory-only routing, or family
  policy not being mechanically enforced.
- Use conservative family boundaries: semantic, backend, target namespace,
  finite modulus class, prefix policy, output domain, reuse contract, and shape
  bucket.
- Keep the real selector on exact reviewed keys while shadow reports accumulate
  evidence.

Likely first slices:

- Extend the report from cache-entry-only queries to reviewed capture matrices
  with CPU/direct baselines, variance margin, and release representative counts.
- `rns8-inspect --selector-shadow` JSON/text mode.
- Benchmark metadata `auto_selector.shadow_recommendation`.
- Report how many current captures would have matched the shadow policy and why
  they are blocked.

Promotion gate:

- Shape-family routing only after blockers are mechanical and every family
  recommendation has release-reviewed representatives plus margin policy.

### 60. Promotion Ledger Adoption And Cache-Install Gate

The installed reviewed cache is now important enough to need its own audit
surface. The ledger and installer can now work together; the remaining
performance-control work is making that path routine for every cache
replacement and stale-entry review.

Implementation status, 2026-06-06:

- `tools/promotion_ledger.py` records target-validation groups, target/cache
  eligibility, target class/family, variance state, stale invalidation reasons,
  and cache coverage summaries.
- `tools/install_autotune_cache.py` records add/replace history and installed
  cache coverage summaries. It can require target-validation ledger gates, and
  CDNA target ids require matching target-validation proof when installed
  through the promotion-ledger path.

Technical direction:

- Use `tools/promotion_ledger.py` to join installed cache entries to evidence summaries,
  reviewed captures, target/toolchain metadata, selected kernel, epilogue,
  workspace, speedup, and promotion blockers cleared.
- Use `tools/install_autotune_cache.py --promotion-ledger <ledger.json>` to
  reject source cache entries that are missing from the ledger or still carry
  promotion blockers; add `--require-variance-gate` for narrow/noisy lanes that
  require `tools/perf_variance_report.py` evidence.
- Record replacement history: old key, new key, reason, date, commit, target,
  and validation command family.
- Add stale invalidation reasons: target id mismatch, HIP SDK mismatch,
  selected-kernel mismatch, epilogue mismatch, workspace mismatch, schema
  mismatch, evidence missing, or margin below threshold.

Likely first slices:

- Use the new installer ledger gate on every reviewed-cache install/update
  before accepting the cache diff.
- Add CI/self-test fixture for stale cache entry reporting beyond the current
  installer and Starfoundry report smokes.
- Docs table summarizing installed local cache coverage by semantic.

Promotion gate:

- Treat ledger consistency as release hygiene before installing or replacing
  reviewed cache entries.

### 61. Counter-Driven Occupancy/Resource Audit Batch

Event timing tells which phase is large. Resource and counter data should
explain whether the kernel is limited by occupancy, memory, stores, LDS, waits,
or instruction mix.

Implementation status, 2026-06-06:

- `tools/gpu_counter_report.py` keeps the original per-capture report, accepts
  optional per-capture attachment manifests, and emits batch reports grouped by
  target, roofline target, and semantic. The report joins event medians,
  profiler counters, ISA summaries, resource signals, bottleneck
  classification, and estimated work metrics.
- Batch rows summarize VGPR, SGPR, LDS, scratch, occupancy, memory-pressure,
  wait/stall, store, matrix-instruction, and work-intensity signals while
  keeping missing counter, partial ISA, and missing event evidence visible.
- Fixture coverage proves present counters, missing counters, partial ISA data,
  ISA attachment, resource summaries, multi-capture batch output, and
  per-capture attachment manifests without requiring CDNA hardware. Real CDNA
  profiler exports remain pending.

Technical direction:

- Extend counter reports to join event summaries, ISA summaries, RGA/LLVM
  resources, rocprofiler counters, VGPR/SGPR/LDS/scratch, occupancy, wait
  instructions, global stores, and roofline groups.
- Start with top bottleneck groups from `tools/evidence_database.py` instead of
  hand-picked kernels.
- Record run-order, warmup/repeat count, target id, clocks/power/thermal data
  where available, and profiler overhead caveats.
- Keep counters explanation-only. Exact correctness and timing remain the
  release gates.

Likely first slices:

- Batch audit for exact-wide 2048 export-bound captures.
- Batch audit for Direct-HIP resident colpair rejection versus current v2.
- Batch audit for finite-u8 2048 accelerator winners and non-winners.

Research anchors:

- AMD HIP occupancy/resource guidance identifies VGPR, SGPR, LDS, warp slots,
  workgroup size, coalescing, divergence, and compute/memory overlap as tuning
  factors:
  <https://rocmdocs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
- ROCm Compute Profiler examples show resource-limiter interpretation must be
  paired with actual wavefront occupancy:
  <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.0/tutorial/profiling-by-example.html>

Promotion gate:

- Counter findings can prioritize kernel work but cannot install cache entries
  or replace release timings.

### 62. Linux/RDNA/CDNA Validation Matrix And Target Report Gate

Windows `gfx1100` is the local bring-up target. It is not Linux ROCm, RDNA4, or
Instinct evidence.

Implementation status, 2026-06-06:

- `tools/target_validation_report.py` now emits per OS/target/toolchain groups
  with build, CTest, smoke, release-capture, profiler, accelerator-status, and
  cache-eligibility phases. Windows, Linux RDNA, and Linux CDNA fixture coverage
  prevents accidental cross-target promotion.
- This closes the repository-side readiness infrastructure only. Public Linux,
  RDNA4, or CDNA claims still require real supported-host runs.

Technical direction:

- Add target-family validation templates for `gfx11xx`, `gfx12xx`,
  `gfx9x/gfx94x`, and unknown targets.
- Separate build success, CTest success, smoke success, release capture success,
  profiler success, and installed-cache eligibility.
- Record HIP/ROCm version, target id, accelerator library version, driver
  version, OS, GPU name, memory size, and clock/power caveats.
- Do not reuse Windows cache entries on Linux targets without exact reviewed
  target evidence.
- Use `tools/target_validation_report.py` to summarize target evidence without
  cross-target inference.

Likely first slices:

- `docs/platform-validation-matrix.md` or a section in release checklist.
- Target-validation reports over captures and dependency reports from real
  supported hosts.
- Linux ROCm direct-HIP CPU/differential checklist before accelerators.

Promotion gate:

- Public target claims require real supported host evidence for that target
  family, not cross-target inference.

### 63. Verification Amortization And Real FHE/Lattice Workload Suite

RNS8 should learn from real FHE/lattice workloads without claiming to implement
or validate full cryptosystems.

Technical direction:

- Add workload proxies for CKKS/BFV/BGV-like NTT batches, key switching,
  relinearization, rotations, ModUp, ModDown, rescale, base extension, level
  drop, bootstrapping stages, tower reuse, and Q/P basis movement.
- Keep dense GEMM evidence separate from NTT/key-switch evidence. A matrix
  multiplication win is not an FHE operation win unless the workload lowering
  and output-domain contract say so.
- Add `workload_family`, `scheme_proxy`, `ring_degree`, `q_count`, `p_count`,
  `tower_layout`, `ntt_state`, `key_material_reuse`, `rotation_count`,
  `bootstrapping_stage`, and `verification_policy` metadata.
- Use verification amortization only as tooling: repeated captures may reuse
  CPU/reference structure, but promoted captures still need exact CPU
  differential coverage for the final requested output.
- Include library-inspired parameter templates, not library compatibility
  claims. RNS8 should not say it supports SEAL, OpenFHE, Lattigo, HElib, or
  HEonGPU workloads until it imports or reproduces their actual contracts.

Likely first slices:

- `fhe-lattice-proxy` scenario family with NTT/key-switch/rotation/bootstrap
  stage labels and no promoted claims.
- `tools/fhe_workload_report.py` that groups captures by tower count,
  transform count, key reuse, and output domain.
- Verification-amortization report that records exactly what was reused and
  which final exact comparisons still ran.

Research anchors:

- OpenFHE documents key switching as the operation used for ciphertext
  automorphisms/rotations and relinearization, with RNS/BV/HYBRID variants:
  <https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/modules/pke/pke_keyswitch.html>
- Lattigo's CKKS bootstrapping package exposes bootstrapping, CoeffsToSlots,
  EvalMod, ModUp, ScaleDown, SlotsToCoeffs, and evaluation-key size surfaces:
  <https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping>
- HEonGPU describes CKKS bootstrapping stages as Mod Raise, Coeff to Slot,
  Approximate Modular Reduction, and Slot to Coeff, and calls out GPU memory
  pressure from Galois keys:
  <https://heongpu.readthedocs.io/en/latest/bootstrapping.html>
- Microsoft SEAL's repository examples include CKKS basics and rotation
  examples, useful as workload-shape inspiration but not as RNS8 compatibility
  proof:
  <https://github.com/microsoft/SEAL>

Promotion gate:

- Keep FHE/lattice captures as proxy workload evidence until RNS8 implements a
  real operation contract with exact reference checks and library-compatible
  semantics.

### 64. AMDGPU Builtins

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

### 65. INT4/IU4

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

### 66. FP8/Ozaki, Strassen, Sparsity

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

### 67. Multi-GPU

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

### 68. Strict Wrap64 Direct-HIP v4 Carry/Byte-Limb Tuning

The strict `mod 2^64` path has real Direct-HIP v4 evidence and real open
microarchitecture questions. The next work should tune the byte-limb path that
already wins, not restart from the losing rocWMMA candidate.

Technical direction:

- Compare carry propagation, byte-limb packing, export policy, u32 accumulator
  tiling, store shape, and kernel resource variants.
- Keep CPU byte-limb reference, Direct-HIP v4 baseline, event timing, and ISA
  reports in the same release review.
- Include 512/1024/2048 and 4096 exploratory shapes so tuning does not regress
  the reviewed large path.

Promotion gate:

- Promote only same-contract Direct-HIP end-to-end wins with required wrap64 GPU
  events; matrix-engine wrap64 paths remain experimental until they beat v4.

### 69. CPU Small-Shape Optimized Fallback And Selector Thresholds

The many-small review made CPU a real performance winner for selected tiny
contracts. That should become explicit selector policy instead of being treated
as only correctness overhead.

Technical direction:

- Microbenchmark bounded-i64 32, bounded-u64 64, finite-u8 64, and nearby
  thresholds with cache-local CPU paths, optional host vectorization, and
  thread-count policy.
- Compare CPU, Direct HIP, runtime vector ALU, hipBLASLt, CK, and rocWMMA in the
  same release groups.
- Make selector explanations say when CPU is chosen for performance.

Promotion gate:

- Route to CPU only when release evidence beats GPU alternatives for the exact
  semantic, shape, modulus, and output contract.

### 70. Release Variance And Performance Regression Gate

Several candidate wins are narrow or export-noisy. The queue needs a gate that
decides when a small speedup is real enough to route.

Technical direction:

- Add multi-run release review for narrow cache candidates, reuse rows, and
  rejected-but-close Direct-HIP variants.
- Use `tools/perf_variance_report.py` to record within-capture p95/median
  spread, run-to-run median spread for same-contract backend/kernel reruns,
  required speedup margin, and promotion blockers for single-run, noisy, or
  under-sampled evidence.
- Keep recording median, minimum, variance, outliers, run order,
  thermal/clock caveats where available, and phase-shift evidence from GPU
  events in the underlying release artifacts.
- Add stale-cache regression guards for installed entries whose margin is inside
  measured noise.

Promotion gate:

- Do not promote a row when the speedup margin is within observed variance or
  when reruns move the bottleneck in a way the implementation does not explain.

### 71. 8192 GPU-Only Throughput Scout

The sweep tool already exposes 8192 exploratory shapes. Those should be used
only to classify throughput behavior after 4096, not to create claims.

Technical direction:

- Run budgeted GPU-only 8192 scouts for bounded, finite-u8, exact-wide, and
  wrap64 where memory and runtime allow.
- Require schema validity, GPU events, target/toolchain metadata, and explicit
  non-promotional review labels.
- Compare 8192/4096 scaling against the same commit and backend set.

Promotion gate:

- Keep 8192 as throughput classification until a budgeted CPU/reference method
  exists for same-contract release review.

### 72. Vector/Native-Output-To-RNS Fused Producer-Consumer Path

The vector/native-to-RNS bridge is exposed, but extra materialization can erase
the vector win. The performance work is making the bridge cheap enough for a
chain.

Technical direction:

- Fuse or otherwise minimize native-output-to-RNS conversion between a
  vector/native producer and Direct-HIP RNS consumer.
- Measure conversion, reusable-B setup, chain GEMM, final export, and exact
  final-output checks in one release group.
- Keep output-domain/currentness metadata explicit so the bridge is not a hidden
  semantic conversion.

Promotion gate:

- Promote only when the complete producer-consumer chain beats native host
  export plus repack and passes exact final-output comparison.

### 73. Finite-u8 Data-Distribution Release Matrix

Finite-u8 evidence is currently strong by modulus but thin by input
distribution. Reducers and CPU/GPU cutoffs may change when data are binary,
sparse, or low-magnitude.

Technical direction:

- Add release groups for binary, sparse, low-Hamming, low-centered, and
  full-uniform finite-ring/field inputs.
- Cover hot moduli and generic prime/composite moduli at 128/512/1024/2048.
- Record distribution metadata in review groups so selector policy cannot mix
  incompatible workloads.

Promotion gate:

- Promote distribution-specific routes only with explicit distribution metadata
  and same-contract CPU/direct/accelerator baselines.

### 74. Split-K And K-Block Large-Shape Variants

Large bounded and exact-wide shapes may be limited by K-block policy,
accumulator caps, schedule upload, or store pressure rather than backend family.

Technical direction:

- Generate split-K, tile-K, K-block, and accumulator-policy variants for
  bounded, exact-wide, finite-u8, and wrap64 where the semantics allow them.
- Keep accumulator-safety metadata, selected-kernel identity, autotune keys,
  event phases, and ISA/resource reports tied to each variant.
- Start from 2048/4096 cases where launch overhead is less dominant.

Promotion gate:

- Promote only per semantic/backend/target when the selected kernel and cache
  key encode the K-block contract and correctness remains exact.

### 75. Result Cache And Incremental GEMM Research Lane

Repeated exact workloads may reuse partial products or results, but that is a
workload contract, not a default GEMM behavior.

Technical direction:

- Prototype source identity, versioning, dirty-region metadata, partial
  recompute, result lifetime, and invalidation reporting.
- Compare repeated workloads against ordinary recompute including cache
  management cost.
- Keep final exact CPU comparisons for the requested output.

Promotion gate:

- Keep this research-only until caller-visible mutation/version contracts make
  reuse exact and auditable.

### 76. Multi-GPU Sharding And Device-Concurrency Platform Lane

Multi-GPU is a real future performance direction, but it is Linux/Instinct-scale
platform work rather than local Windows `gfx1100` optimization.

Technical direction:

- Inspect Linux ROCm multi-GPU topology, peer access, memory limits, and
  profiler availability.
- Prototype modulus-group sharding before K sharding unless profiling says
  otherwise.
- Compare reconstruction placement: per-device partial reconstruction versus
  final gathered reconstruction.

Promotion gate:

- Claim only target/host combinations that pass real supported-host build,
  CTest, smoke, release capture, and profiling gates.

### 77. Layout Implementation Search After Scenario Surface

The `layout-search` scenario family is closed as a surface. The active work is
now real layout implementation and same-contract A/B evidence.

Technical direction:

- Implement residue-plane interleave, leading-dimension policy, packed-residue,
  output-current, finite-u8, exact-wide prefix-20, and wrap64 byte-layout
  variants.
- Attribute pack, GEMM, reduction/export, status, and D2H effects separately.
- Keep conversion/setup cost in the measured contract so a layout does not win
  by moving work out of frame.

Promotion gate:

- Keep only layouts that beat the current layout end-to-end under release review
  with event attribution and exact CPU comparison.

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
- Use promotion-ledger and reuse-contract reports before converting more
  workload-specific wins into cache or selector policy.
- Add counter-driven occupancy/resource audit batches for the current
  export-bound exact-wide captures and rejected Direct-HIP resident colpair
  captures.
- Add release variance/regression gates for narrow wins before they become
  durable cache or selector entries.

### Batch B: Immediate Shape Wins

- Finish bounded-i64 winner tuning for Direct HIP 512 and hipBLASLt 1024; the
  current v2 release review installed the 1024 hipBLASLt cache entry and left
  512 on Direct HIP.
- Continue exact-wide export tuning before broadening exact-wide GEMM variants;
  the current 512/1024 v2 matrix is reviewed and installed, but 64/128, 2048,
  limb-count variants, and chain/lazy-export workloads remain open.
- Move exact-wide export-bound work to the front of this batch: limb-count
  variants, prefix-20 constants placement, compact/padded D2H, status elision,
  and same-output lazy-chain final export comparisons.
- Extend finite-u8 CK/rocWMMA reducer specialization beyond the now-reviewed
  64/128/512/1024 ring-251/ring-255/ring-256/field-251 matrix into 2048,
  generic prime, and generic composite cases.
- Add the hipBLASLt bounded-i64 1024 A/B lane because the installed win is
  narrow and should be protected by current direct-HIP and event baselines.
- Continue direct-HIP wrap64 v4 follow-up tuning before another matrix-engine
  candidate.
- Add CPU small-shape fallback thresholds for the many-small contracts where the
  release review already shows CPU wins.
- Add split-K/K-block large-shape variants after the 2048/4096 matrices identify
  real throughput bottlenecks.

### Batch C: Representation Wins

- Implement multi-modulus pack and residue-channel fusion experiments.
- Add fused pack+GEMM for one-shot/small bounded and finite workloads.
- Compare end-to-end layouts across RNS, finite, exact-wide, and wrap64.
- Add modulus-set search and residue-count autotuning as benchmark-only
  experimental ladders with explicit non-default metadata.
- Add CRT/reconstruction fusion and export kernel zoo variants with
  selected-kernel, epilogue, workspace, constants, and status-policy metadata.
- Turn the closed layout-search scenario surface into actual layout
  implementation A/Bs.
- Make vector/native-output-to-RNS producer-consumer chains cheap enough to beat
  host export plus repack before considering selector policy.
- Add polynomial-tower and Q/P-basis layout sketches for FHE/lattice proxy
  scenarios without treating them as public RNS8 storage formats.

### Batch D: Scheduler And Reuse Wins

- Expand repeated-B cache work across rocWMMA, hipBLASLt, CK, finite, and
  exact-wide.
- Build the reuse contract ledger and persistent resident matrix lifetime
  policy before using reuse evidence for selector behavior.
- Add persistent/grouped scheduler experiments for adaptive prefix groups and
  many small GEMMs.
- Add a device grouped dispatcher after the host-batch evidence gate, starting
  with exact-wide signed 64 and bounded 64/128 same-shape buckets.
- Add workspace arenas, streaming pack/compute/export overlap, HIP Graph replay,
  and tile-shape autotuning as separate benchmark lanes so launch, allocation,
  and resource wins are not conflated.
- Add tower/key-material reuse scenario labels so FHE/lattice-inspired reuse
  does not get collapsed into ordinary A/B matrix reuse.
- Add HIP Graph and host batching modes for repeated fixed-shape workflows.

### Batch E: Research And Platform Work

- Keep AMDGPU builtins, INT4/IU4, Ozaki hybrids, Strassen, sparsity, and
  multi-GPU behind explicit research/platform work.
- Keep 8192 runs GPU-only and non-promotional until budgeted CPU/reference
  release review is possible.
- Run Linux ROCm and Instinct work only on real supported hosts with separate
  evidence from Windows `gfx1100`.
- Keep the real FHE/lattice workload suite as proxy workload evidence until
  RNS8 implements actual operation contracts with exact reference checks.
- Keep verification amortization tooling-only unless every promoted capture
  still has exact CPU differential coverage for the final requested output.
