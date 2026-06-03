# RNS8 Roadmap Status

Status date: 2026-06-03

This document records live implementation status against the current roadmap.
It is not a substitute for `docs/RNS8_RESEARCH_SPEC.md`; when status and spec
disagree, the spec remains the target and this file identifies the gap.

## Implemented And Verified

- Phase 0 host foundation: C ABI, CMake targets, CPU reference, tests, tools,
  dependency checker, and benchmark/result comparison shell.
- Phase 1 Windows direct HIP bring-up on Radeon RX 7900 XTX / `gfx1100`:
  explicit hipcc integration, device inspection, residue conversion,
  one-modulus ring GEMM, K-block splitting, and CPU differential tests.
- Device-resident direct HIP RNS matrices: HIP matrices own device residue
  buffers, upload buffers, export buffers, and status buffers; `rns8_gemm_rns`
  consumes device residues directly instead of copying host residues in the hot
  GEMM path. Internal allocation counters prove repeated same-shape persistent
  pack/GEMM/export reuses warmed matrix-owned buffers without additional
  direct-HIP allocation or free calls.
- Direct HIP fused INT32-to-centered-residue reduction: the correctness kernel
  stages 16x16 output tiles with shared A/B residue tiles, reduces each K block
  to the centered residue in the kernel, and does not write full INT32 output
  matrices to global memory. Centered-range correction uses source-level mask
  arithmetic instead of source-level `if` branches.
- Bounded i64/u64 GPU export: direct HIP reconstructs bounded i64/u64 output on
  device with a fixed three-limb Garner path for prefixes up to 20, reports
  range errors through device status, handles the full signed output range
  including `INT64_MIN`, and copies compact output into the caller's host
  layout.
- Fixed 9-modulus bounded i64/u64 GEMM: CPU and direct HIP public one-shot
  bounded APIs pass CPU differential tests, including full-width boundary and
  K-block cases. The CPU/reference source now names the Phase 2 fixed
  9-modulus milestone explicitly and locks default prefix-9 scheduling,
  K-split edge behavior around 65536, signed cancellation, unsigned full-width
  padded output, and signed/unsigned per-tile schedule parity.
- Public one-shot GEMM lifetime: bounded i64/u64, strict wrap64, and finite
  ring/field one-shot APIs share one internal resident owner for plan, A/B/C
  matrix, and workspace construction before routing through the corresponding
  persistent pack/GEMM/export path.
- Persistent RNS behavior: public matrix/workspace APIs exercise persistent A/B/C
  storage and verify device pointer stability through pack, GEMM, and export.
  Workspaces are tagged with backend, shape, prefix, semantics, bound kind,
  bound value, tile geometry, selected-prefix schedule metadata, and an
  internal schedule fingerprint, and reject same-shape reuse across bounded,
  per-tile bounded, exact-wide, wrap64, and different per-tile schedule
  contracts. Per-tile bounded matrices must carry matching plan tile geometry
  before GEMM/export dispatch. Successful bounded RNS GEMM stamps C with an
  input-derived source version, and rejected dispatch leaves that version
  untouched.
- Plan schedule inspection: bounded and wrap64 plans expose output tile grid,
  required prefix, selected prefix, and prefix-group metadata through public ABI
  queries. Global bounded plans still use one fixed selected prefix for every
  tile. CPU reference and direct HIP per-tile bounded plans copy tile bounds
  into the plan, select variable exact prefixes, report adaptive prefix/skip
  metadata, execute only selected per-tile prefixes, and export with
  tile-local bounds.
- Exact-wide RNS output: exact-wide signed and unsigned semantics accept
  `RNS8_BOUND_NONE`, compute persistent RNS output, and reject bounded-looking
  CRT metadata. CPU and direct HIP RNS output are checked against
  Boost.Multiprecision residue oracles. CPU little-endian limb export is
  implemented for fixed-width signed two's-complement and unsigned magnitude
  output, with `ld` interpreted as an element stride and `limb_count` as the
  per-element width in `[1, 32]`. Direct HIP exports signed and unsigned
  exact-wide limbs from device-resident RNS matrices without synchronizing host
  residue storage and reports range errors when the requested fixed width is too
  small while preserving destination storage. Direct HIP differential coverage
  now includes max-width 32-limb padded exports for centered signed negatives
  and high-bit unsigned magnitudes. Stale nonzero bounds, tile-bound metadata,
  stale-prefix matrices, bounded matrices, wrap64 byte-limb matrices, and
  signed/unsigned cross-export calls are rejected for exact-wide descriptors
  and exports.
- Strict wraparound byte-limb backend: CPU one-shot and persistent `mod 2^64`
  GEMM use byte-limb matrix storage and the Comba reference, match
  Boost.Multiprecision low-64-bit results, and keep RNS/CRT APIs fenced off
  from wrap descriptors. The CPU reference also includes an exhaustively tested
  signed-INT8 correction helper for reconstructing unsigned byte products as
  signed byte product plus explicit high-bit correction terms when future
  accelerator paths expose only signed INT8 products, plus a 36-byte-pair
  decomposition oracle over the low eight Comba diagonals that matches Boost
  low-64 results and the current Comba reference.
- Public direct-HIP strict wrap64 byte-limb correctness path: HIP_DIRECT wrap
  matrices own device byte-limb buffers, pack/GEMM/export consume those buffers
  without RNS residue allocation, public one-shot and persistent APIs match the
  CPU byte-limb reference, padded host export layouts are tested, and repeated
  same-shape export reuses the matrix-owned export buffer. The GEMM kernel is
  now an inspectable tiled byte-limb correctness path that sums the same
  low eight byte-product diagonals with device-side signed-INT8 correction for
  the 36 byte-product pairs that can affect the low 64 bits and then carries
  into the low 64 bits; it is not an optimized matrix-engine byte-GEMM
  accelerator path.
- Finite ring/field `uint8_t` APIs: CPU reference and direct HIP now implement
  explicit-modulus finite GEMM for both one-shot and persistent resident matrix
  paths. The public finite one-shot calls hard-cut through resident finite
  matrices/workspaces and no longer keep a separate ad hoc HIP one-shot GEMM
  route. `RNS8_FINITE_RING_U8` accepts moduli in `[2, 256]`;
  `RNS8_FINITE_FIELD_U8` requires prime moduli `<= 251`. The finite path uses
  `RNS8_BOUND_NONE`, `bound = 0`, `max_prefix = 0`, no tile bounds,
  one-plane prefix-zero centered-residue storage for the requested modulus,
  K-split INT8xINT8->INT32 ring GEMM with fused centered reduction, and
  canonical `uint8_t` export. CPU and direct-HIP tests cover composite, prime,
  modulus-256, padded layout, K-split, cross-modulus rejection, finite one-shot
  resident-kernel timing-label guards, and same-shape resident HIP allocation
  reuse cases.
- Direct-HIP per-tile bounded adaptive correctness path: HIP_DIRECT bounded
  plans with `RNS8_BOUND_PER_TILE_MAX_ABS` or
  `RNS8_BOUND_PER_TILE_MAX_UNSIGNED` use grouped direct HIP tile launches for
  selected prefixes and tile-local device CRT export. Tests compare signed and
  unsigned output against the CPU reference, cover tile-local range errors,
  prove skipped high-prefix residue planes remain untouched, keep matrices
  device-resident through GEMM/export, and reject malformed private tiled
  schedules before GEMM launch or export/status buffer allocation.
- Benchmark schema v4: benchmark captures include an explicit integer
  `"schema_version": 4`, command line, live git commit, compiler/HIP/device
  metadata, raw timings, summaries, configured HIP toolchain metadata, explicit
  null values for unavailable timing sources, direct-HIP GPU event timing arrays
  when complete, exact `hipEventElapsedTime` source/scope validation for
  direct-HIP event captures, mandatory declared GPU event phase order with exact
  timing-key matching, `gpu_event_phase_order: null` plus explicit unavailable
  metadata when event timing is not applicable, strict wrap64 CPU and direct-HIP
  byte-limb benchmark metadata, fixed-prefix schedule metadata, measured
  schedule-info query timing, explicit
  phase-availability metadata for fused or not-applicable reduction, direct-HIP
  per-tile adaptive bounded capture metadata, hipBLASLt/CK/rocWMMA accelerator
  selected-kernel metadata, exact-wide signed/unsigned benchmark contracts with
  fixed-width limb export epilogues and exact-wide HIP export event labels,
  schema validation tooling, CTest coverage for schema self-tests, current
  fixtures, exact-wide benchmark smoke, and same-contract result comparison,
  and comparison-tool support for current schema v4 plus capture-specific GPU
  event phase orders. Adaptive captures are evidence for the selected
  correctness path only; they are not optimized matrix-engine performance
  claims.
- Benchmark review tooling v3: `tools/benchmark_sweep.py` emits review reports
  with `schema_version = 3`, per-phase medians, speedups versus direct-HIP and
  vector-ALU baselines where applicable, promotion blockers, winner rationale,
  selected kernel, target id, HIP SDK and accelerator library versions,
  compiler/git/seed/warmup/repeat metadata, event source/status, workspace
  bytes, and explicit cache-write status. The sweep self-test verifies eligible,
  finite cache promotion, not-requested, and written cache states. Production
  promotion requires `--review-mode release`; smoke reports and raw benchmark
  captures cannot write production cache entries directly.
  `--include-exact-wide --release-matrix` generates exact-wide signed/unsigned
  release command matrices across the promotable square shapes and exact-wide
  backend set. Exact-wide groups require same-contract CPU and direct-HIP
  baselines, with no vector-ALU baseline. finite-u8 groups can now write
  reviewed cache entries scoped by the explicit finite modulus in the plan
  autotune key.
- Result comparison now separates same-contract semantic fields from backend
  evidence and GPU compatibility. CPU/reference or wrap64 byte-limb baselines
  can compare against GPU captures without inventing a target id, while GPU-vs-GPU
  comparisons still require matching compiler, configured target, HIP toolchain,
  device target, runtime, and driver fields.
- Accelerator benchmark event hooks: explicit hipBLASLt captures expose
  hipBLASLt pack, INT8-to-INT32 matmul, and residue-reduction operation groups.
  Explicit CK and rocWMMA captures expose the shared `rns_gemm_kernel_group`
  operation-group hook plus direct-HIP pack/export labels when event capture is
  complete. These are HIP event timings for backend operation groups, not
  per-kernel/per-tile/per-prefix scheduler telemetry.
- hipBLASLt baseline accelerator: opt-in Windows `gfx1100` correctness backend
  under `RNS8_ENABLE_HIPBLASLT=ON`, using hipBLASLt INT8-to-INT32 GEMM,
  padded INT32 scratch, and separate centered-residue reduction. CPU/direct-HIP
  differentials cover bounded, exact-wide RNS output, finite u8, K splits, tile
  tails, and adaptive-schedule rejection. Captures remain baseline evidence and
  report `performance_validated=false`.
- CK fused accelerator: opt-in Windows `gfx1100` correctness backend under
  `RNS8_ENABLE_CK=ON`, using repo-local CK headers plus RNS8-owned HIP
  pack/output kernels. It supports fixed-prefix bounded, adaptive per-tile
  bounded, exact-wide RNS output, and finite u8 with fused centered-residue
  output. CPU/direct-HIP differentials, benchmark schema fixtures, and ISA
  evidence are present. The CK preset generates the RNS8 WMMA no-divide
  block-map include overlay from the pinned CK header during configure, tracks
  that generated header as a CK HIP object dependency, and fails fast if CK's
  expected upstream or patched `MakeDefaultBlock2CTileMap` block is absent.
  Reviewed CK finite-u8 winners for selected shapes have been installed into
  the default local AUTO cache, and CK bounded, exact-wide, and finite-ring
  AUTO cache-hit paths are covered by hermetic fake-default-cache smokes; other
  CK release-smoke winners remain shape-scoped evidence until reviewed and
  installed.
- rocWMMA fused accelerator: opt-in Windows `gfx1100` correctness backend under
  `RNS8_ENABLE_ROCWMMA=ON`, using repo-local rocWMMA headers and RNS8-owned
  HIP kernels. It supports fixed-prefix bounded, adaptive per-tile bounded,
  exact-wide RNS output, and finite u8 with signed INT8 WMMA and fused
  centered-residue output. CPU/direct-HIP differentials, benchmark schema
  fixtures, and an ISA gate requiring `v_wmma` with no scalar
  divide/remainder/reciprocal mnemonics or unintended INT32 global stores are
  present. Reviewed rocWMMA bounded, adaptive bounded, and finite-u8 winners
  for selected shapes have been installed into the default local AUTO cache;
  hermetic fake-default-cache smokes cover bounded, exact-wide, and finite-field
  default-path AUTO selection.
  Additional rocWMMA release-smoke winners remain shape-scoped evidence until
  reviewed and installed.
- Platform readiness reporting: dependency checker reports host readiness gates,
  Windows HIP/RDNA3 gates, Linux ROCm gates as not applicable on Windows, and
  optional accelerator components as candidate evidence only. Linux presets keep
  active offload targets separate from RDNA/CDNA coverage metadata, and shallow
  hipBLASLt/CK/rocWMMA probes report headers, libraries, tools, and CMake module
  evidence without enabling accelerator backends. AMDGPU builtin readiness is a
  separate not-ready gate until real target-specific exact kernels exist.
  Readiness JSON separates implemented correctness backend families from
  candidate accelerator evidence through `readiness.correctness_backend_validation`;
  dependency discovery validates no correctness backend by itself, and
  accelerator component/probe records are explicitly marked
  `candidate_accelerator_evidence_only`.
  Opt-in Python and CMake accelerator probe modes record compile/link/runtime
  evidence under `temp/` or probe-only build directories while keeping probe
  evidence separate from backend enablement. Readiness output also separates
  correctness-backend validation from candidate accelerator evidence and reports
  hard-cut self-check metadata so discovery cannot be read as enabled backend
  validation. CTest configure-negative cases pin that
  `RNS8_ENABLE_AMDGPU_BUILTINS` and non-enabled accelerator configurations fail
  fast until real correctness backends exist.
  Report-level `hard_cut_self_checks` keep accelerator evidence, backend
  enablement, and Windows/Linux/Instinct validation boundaries
  machine-readable.

## Requirement Audit

1. Direct HIP residency and lifetime: implemented for persistent RNS and wrap64
   matrices. Matrix-owned residue, byte-limb, upload, export, and status
   buffers have explicit ownership and teardown. Repeated same-shape persistent
   pack/GEMM/export is now allocation-observed after warmup. Workspace reuse is
   contract-checked by semantics, bound kind, bound value, tile geometry,
   selected-prefix schedule metadata, and a copied-schedule fingerprint, not
   only by shape.
2. Direct HIP fused INT32-to-centered-residue reduction: implemented in the
   direct correctness kernel. The path writes centered `int8_t` residues and
   does not materialize full INT32 output matrices in global memory. GEMM
   reduction now uses exact small-modulus reciprocal metadata validated by the
   host launch boundary. The Windows HIP CTest gate extracts the compiled
   `gfx1100` code object and checks that the direct HIP RNS GEMM kernels contain
   reciprocal multiply-high instructions and no divide/remainder/rcp mnemonics.
   Architecture-tuned matrix kernels and broader ISA validation remain future
   performance work.
3. GPU bounded i64/u64 CRT/export: implemented for direct HIP with device-side
   Garner reconstruction, range-error status reporting, signed `INT64_MIN`
   handling, unsigned `UINT64_MAX` handling, per-tile bounds, and CPU
   differential tests. CPU Boost.Multiprecision CRT remains the reference path.
4. Benchmarking: schema v4 covers fixed seeds, command line, live git commit,
   compiler and HIP toolchain metadata, GPU identity, backend/shape/semantics,
   prefix and schedule metadata, warmups/repeats, raw timings, medians/p95s,
   HIP event timings where complete, comparison tooling, and raw captures under
   ignored `temp/`. Fused reduction is represented as explicitly not separately
   timed instead of synthesized from GEMM time. No performance claims are made.
5. Phase 2/3 fixed 9-modulus bounded GEMM and persistent matrix behavior:
   implemented as named status milestones with CPU and Windows HIP tests,
   including a literal `RNS8_DEFAULT_BOUNDED_PREFIX == 9` contract check. The
   CPU/reference milestone is source-covered by Boost.Multiprecision exact
   differentials for the 65535/65536/65537 K edge, signed cancellation,
   unsigned full-width padded output, and per-tile selected-prefix parity.
6. Exact-wide CPU output/export semantics: implemented separately from bounded
   i64/u64 and wrap64 semantics. Signed export uses fixed-width
   two's-complement limbs over the centered exact integer, using the same
   `x >= ceil(P / 2)` negative threshold as centered residue packing; unsigned
   export uses fixed-width magnitude limbs over the canonical nonnegative
   integer. The ABI treats `ld` as an element stride, stores `limb_count`
   contiguous limbs per element for `limb_count` in `[1, 32]`, and reports
   range errors instead of truncating while preserving destination storage.
   Invalid widths, invalid strides, null export handles, stale nonzero bounds,
   non-none bound kinds, tile-bound metadata, stale prefixes, bounded matrix
   handles, wrap64 byte-limb matrix handles, and cross-semantic export attempts
   are rejected at the API boundary. Unit coverage pins one-limb signed
   boundaries, 32-limb sign extension, signed high-bit negative export,
   unsigned overflow rejection, two-limb unsigned success including high-bit
   magnitude cases, padded export, descriptor rejection, and wrong
   export-function rejection.
7. Strict `mod 2^64`: implemented only through byte-limb storage for CPU and
   direct HIP. Odd-modulus CRT remains fenced off from wraparound descriptors.
   The direct-HIP tiled byte-limb path is a correctness kernel, not an
   optimized matrix-engine accelerator.
7a. Finite ring/field `uint8_t`: implemented as explicit one-shot and
    persistent resident CPU/direct-HIP APIs. Finite matrices own one
    prefix-zero centered-residue plane and stamp the explicit pack modulus;
    resident GEMM/export reject mismatched modulus state. Finite calls do not
    use CRT prefixes, bounded export, exact-wide export, or wrap64 byte-limb
    routing.
8. Linux ROCm and Instinct: represented by presets, readiness gates, target
   coverage metadata, and docs. Validation remains `NOT_APPLICABLE` on this
   Windows host and requires a real supported Linux ROCm host; Windows evidence
   is not a substitute for Linux Radeon or Instinct CDNA validation. Readiness
   output keeps `windows_evidence_validates_linux_rocm=false` and
   `windows_evidence_validates_instinct=false`.
9. hipBLASLt baseline: implemented as an opt-in Windows `gfx1100` correctness
   backend under `RNS8_ENABLE_HIPBLASLT=ON`. It uses padded transposed INT8
   pack buffers, hipBLASLt INT8-to-INT32 GEMM, padded INT32 scratch, and a
   separate HIP centered-residue reduction. CPU/direct-HIP differentials cover
   bounded, exact-wide RNS output, finite u8, K splits, tail padding, and
   adaptive-schedule rejection. It remains baseline-only, with
   `performance_validated=false`.
10. CK fused accelerator: implemented as an opt-in Windows `gfx1100`
    correctness backend under `RNS8_ENABLE_CK=ON`. It has compiled kernels,
    CPU/direct-HIP differentials, schema fixtures, and ISA evidence for fixed
    bounded, adaptive bounded, exact-wide RNS output, and finite u8.
11. rocWMMA fused accelerator: implemented as an opt-in Windows `gfx1100`
    correctness backend under `RNS8_ENABLE_ROCWMMA=ON`. It has compiled
    kernels, CPU/direct-HIP differentials, schema fixtures, and ISA evidence
    for fixed bounded, adaptive bounded, exact-wide RNS output, and finite u8.
    AMDGPU builtins remain fail-fast.

## Backend Promotion Status

| Backend | Dependency readiness | Correctness enabled | Exact differential | ISA evidence | Release-reviewed evidence | AUTO production promotion |
|---|---|---|---|---|---|---|
| CPU reference | required | yes | yes | not applicable | baseline only | fallback/reference |
| Direct HIP | required on Windows HIP builds | yes | yes | reciprocal/no-divide gates for current correctness kernels | baseline for RNS and finite-u8; strict wrap64 release-reviewed baseline at 1828 us for 64, 2090 us for 128, 7757 us for 512, and 39359 us for 1024 | GPU fallback |
| hipBLASLt | optional discovered dependency plus opt-in preset | yes when `RNS8_ENABLE_HIPBLASLT=ON` | yes in opt-in preset | library baseline evidence | bounded i64 1024 release-reviewed fastest accelerator at 8326 us; bounded u64 blocked by vector baseline; finite field-251 1024 release-reviewed fastest accelerator at 2327 us | selector wired; bounded 1024 and finite field-251 1024 AUTO cache-hit smokes validated on Windows `gfx1100`; default local cache installed from reviewed entries |
| CK | repo-local CK headers plus opt-in preset | yes when `RNS8_ENABLE_CK=ON` | yes in opt-in preset | `v_wmma` gate with no scalar divide/rcp and no unintended INT32 stores in matched CK WMMA symbols | bounded i64 512 release-reviewed candidate at 2408 us but not fastest; closest bounded u64 candidate at 512/1024 but still slower than vector; finite ring 1024 release-reviewed fastest accelerator at 1428 us for modulus 251 and 1354 us for modulus 255; exact-wide signed 1024 and unsigned 128/512/1024 release-reviewed fastest accelerator | selector wired; bounded CK, finite ring-251 1024, and exact-wide CK AUTO cache-hit paths tested from the default local cache on Windows `gfx1100` |
| rocWMMA | repo-local rocWMMA headers plus opt-in preset | yes when `RNS8_ENABLE_ROCWMMA=ON` | yes in opt-in preset | `v_wmma` gate with no scalar divide/rcp and no unintended INT32 stores | bounded i64 512 release-reviewed fastest accelerator at 2399 us; adaptive bounded i64 1024 release-reviewed fastest accelerator at 5095 us; bounded u64 blocked by vector baseline; finite 64/128/512 groups release-reviewed fastest accelerator | selector wired; bounded 512, adaptive 1024, and finite ring-251 512 AUTO cache-hit smokes validated, with default local cache exact-hit inspected on Windows `gfx1100` |
| AMDGPU builtins | no discovery-only readiness path | no | no | no | none | fail-fast |
| Wrap64 matrix-engine | current direct-HIP byte-limb path is baseline only | no accelerator candidate | no accelerator candidate | no matrix-engine evidence | direct-HIP v3 remains measured release GPU path at 1828 us for 64x64x64, 2090 us for 128x128x128, 7757 us for 512x512x512, and 39359 us for 1024x1024x1024 | not durable |

Durable AUTO promotion is not the same as a temp smoke cache entry. A shape
enters production selection only after `--review-mode release` captures with complete
same-contract baselines are reviewed, the generated cache entry is accepted for
the production cache, and `rns8-inspect` reports the exact validated hit plus
runtime target/version-matched selection rationale for that plan key. The
runtime target/version rejection path exists, AUTO plan dispatch from a
validated reviewed entry is wired for bounded, adaptive bounded, exact RNS, and
finite HIP-resident accelerator candidates, and
`tools/install_autotune_cache.py` validates and merges reviewed cache files
into an explicit or default cache path, and `--replace-existing` intentionally
discards stale or non-reviewed destination entries instead of preserving them.
The default Windows cache at `%LOCALAPPDATA%\rns8-gemm\autotune.json` has been
populated from `temp\reviewed-autotune-production-candidate.json` with 19
reviewed entries. The remaining gap is broader production coverage plus a
wrap64 matrix-engine candidate, not selector dispatch or default-cache
installation tooling for reviewed finite, exact-wide, or bounded plan keys.

## Not Yet Implemented

- Optimized matrix-engine HIP kernels and broader instruction-level validation
  beyond the current no-divide reciprocal GEMM gate. The direct HIP kernels are
  correctness bring-up kernels, not performance evidence, even though their GEMM
  reduction now uses validated exact reciprocal metadata.
- Production performance gates for the fixed 9-modulus bounded milestone. The
  current fixed-prefix CPU and direct-HIP paths are correctness-grade and
  unoptimized unless a reviewed benchmark capture says otherwise.
- AMDGPU builtin accelerator backends. They remain feature-detected future
  paths and are not correctness requirements.
- Broader durable production AUTO coverage from reviewed release cache entries.
  Selector dispatch is wired for HIP-resident accelerator candidates and has
  bounded, exact-wide, and finite-u8 fake-default-cache integration tests for
  the relevant hipBLASLt, CK, and rocWMMA presets.
  `tools/install_autotune_cache.py` validates and merges reviewed release cache
  files, and the default local Windows cache has been populated from the
  current bounded-i64, adaptive bounded, finite-u8, and exact-wide reviewed
  caches. A bounded-i64 release matrix for 64, 128, 512, and 1024 produced temp reviewed cache
  entries for rocWMMA at 512 and hipBLASLt at 1024, exact `rns8-inspect` hits
  on Windows `gfx1100`, and schema-valid `rns8-bench --backend auto` cache-hit
  smokes with `backend_selected=wmma` or `backend_selected=hipblaslt` and
  `backend_metadata.performance_validated=true`. A bounded-u64 release matrix for the same shapes
  produced no cache entries because `hip-vector-alu-int64` was fastest in all
  four same-contract groups. The adaptive bounded release matrix produced one
  temp reviewed rocWMMA entry for bounded i64 1024, with an exact
  `rns8-inspect` hit and schema-valid AUTO cache-hit smoke; bounded-u64
  adaptive groups remained blocked. The finite-u8 release matrix has
  60 captures, 12 complete same-contract groups, and 12 temp reviewed cache
  entries: rocWMMA for all 64/128/512 groups, CK for the 1024 ring groups, and
  hipBLASLt for the 1024 field-251 group. Representative exact `rns8-inspect`
  hits and AUTO cache-hit smokes are validated on Windows `gfx1100`. The
  installed default-cache selector path was revalidated with schema-valid
  captures:
  `temp\default-cache-auto-rocwmma-bounded-i64.json`,
  `temp\default-cache-auto-ck-finite-ring.json`, and
  `temp\default-cache-auto-hipblaslt-finite-field.json`; each reports
  `backend_requested=auto`, `backend_metadata.performance_validated=true`, and
  selected-backend HIP event timing. Current release-smoke reviews can still
  produce temp-only candidate winners for additional shapes, and those smoke
  artifacts remain evidence only until reviewed and installed. Raw
  `rns8-bench --write-autotune-cache`
  writes are always rejected; the reviewed promotion path is
  `tools\benchmark_sweep.py --review-mode release --write-autotune-cache`.
  The exact-wide signed/unsigned release matrix has 40 captures across eight
  same-contract groups for shapes 64, 128, 512, and 1024. It wrote four temp
  reviewed CK entries to `temp\reviewed-autotune-exact-wide-full.json`:
  exact-wide signed 1024 at 19686 us, exact-wide unsigned 128 at 2995 us,
  exact-wide unsigned 512 at 6753 us, and exact-wide unsigned 1024 at 15393 us
  median end-to-end. Signed 64/128/512 and unsigned 64 stayed blocked by the
  direct-HIP baseline. The merged default cache has exact `rns8-inspect` hits
  for those four CK keys, and schema-valid AUTO captures under
  `temp\default-cache-auto-exact-wide-reviewed` select `backend_selected=ck`,
  report `comparison_baseline.status=reviewed_release_same_contract_baseline`,
  `backend_metadata.performance_validated=true`, and exact-wide export event
  phases. Accelerator-gated CTest definitions also include exact-wide signed
  fake-default-cache AUTO hit smokes for hipBLASLt, CK, and rocWMMA presets and
  finite-u8 fake-default-cache AUTO hit smokes for hipBLASLt finite-field, CK
  finite-ring, and rocWMMA finite-field paths; those smokes have been executed
  in the Windows `gfx1100` release presets.
- Optimized strict `mod 2^64` GPU byte GEMMs, accelerator integration of the
  signed-INT8 correction algebra, and broader production-host/device validation
  beyond the current Windows direct-HIP CPU differentials.
  A reviewed release wrap64 baseline now covers 64, 128, 512, and 1024 square
  shapes with CPU byte-limb reference and direct HIP at three warmups, nine
  repeats, and seed `20260602`. Direct HIP
  `direct_hip_wrap64_byte_gemm36_tiled_2d_v3` measures 1828 us, 2090 us,
  7757 us, and 39359 us median end-to-end at those shapes versus CPU byte-limb
  medians of 710 us, 5845 us, 576082 us, and 4729230 us. The review path still
  keeps current CPU/direct-HIP wrap64 baselines non-promotable because they are
  not accelerator backends. The matrix-engine candidate remains open.
- Packed low-bit matrix-engine pipeline work: persistent packed layout versions
  beyond the current correctness layouts, B prepack/tile swizzle caches,
  repeated-A/B amortization sweeps, IU4/INT4 experiments, and FP8/Ozaki
  research-mode experiments with explicit verification metadata. The benchmark
  and sweep tools now expose `--reuse-packed-inputs` so those sweeps can record
  one-time `prepack_setup_us` separately from repeated GEMM/export timings, but
  durable packed-layout and prepack-cache production paths are still unshipped.
- Optimized finite-field algorithms beyond the explicit-modulus
  correctness-grade CPU/direct-HIP finite path.
- Linux ROCm direct HIP parity, Linux hipBLASLt baseline, Linux CK validation,
  Instinct CDNA validation, profiling, power runs, and cluster reproducibility
  notes. These require a real Linux ROCm host with supported hardware.
- Architecture hot kernels, production-grade release sweeps with at least three
  warmups and nine repeats for promotable shapes, durable fastest-accelerator
  autotune promotion, deeper accelerator per-kernel/per-tile HIP event timing
  hooks, and production performance gate evaluation.
- Multi-GPU modulus split experiments.

## Latest Evidence

This section records lead-run integration evidence from prior checkpoints. It
does not promote ignored `temp/` captures or historical schema versions into the
current tracked schema contract.

- `python tools\windows_dev.py cmake --build --preset windows-debug`: passed
  after the reciprocal-reduction and ISA-gate work; the wrapper loads the
  Visual Studio developer environment automatically from a plain PowerShell
  shell.
- `temp\autotune_bench.json` plus `temp\autotune_cache_smoke.json`: fixed-seed
  strict wrap64 direct-HIP smoke for the side-band autotune cache writer.
  `tools\benchmark_schema.py` accepted the emitted benchmark JSON, and
  `rns8-inspect --autotune-key` reported an exact unvalidated cache hit from an
  isolated `RNS8_AUTOTUNE_CACHE_PATH`.
- `ctest --preset windows-debug --output-on-failure`: 167/167 passed on the
  Windows HIP debug build. The private mismatched-modulus/wrong-reciprocal
  metadata smoke now runs in HIP builds, and the HIP-only
  `hip_direct_kernel_isa_check` extracts the `gfx1100` code object and verifies
  the direct RNS GEMM kernels contain `v_mul_hi_u32` with no divide/remainder/rcp
  mnemonics.
- `build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke`: CPU reference
  verification and direct HIP pack, ring, bounded GEMM, adaptive bounded GEMM,
  finite `uint8_t`, and wrap64 smoke passed, including resident finite CPU/HIP
  parity.
- Recent pushed hard-cut commits through `c25b481` add persistent finite
  `uint8_t` CPU/direct-HIP support and exact reciprocal metadata validation for
  direct HIP RNS GEMM reduction.
- Recent pushed hard-cut commits through `6fbccd9` add device-current direct-HIP
  RNS input/export enforcement, matrix-owned bounded export schedule/bounds
  metadata, tiled wrap64 compact-cell staging, removal of retired host-residue
  export synchronization, removal of unused HIP workspace scratch state,
  exact-wide fixed-width ABI/readiness hardening, and CRT/prefix-9 CPU coverage.
- Benchmark schema coverage ran inside the Windows CTest pass:
  `benchmark_schema_self_test`, `benchmark_schema_current_fixtures`, and
  `benchmark_result_compare_same_contract` all passed.
- Verified post-`8bb336c` implementation patch tightens direct-HIP private
  tiled schedule rejection and selected-prefix grouping, adds larger padded
  wrap64 resident tile-tail parity, strengthens exact-wide fixed-width limb
  stride coverage, and records accelerator enablement plus exact-wide platform
  validation policy in dependency readiness output after the Windows HIP build,
  CTest pass, HIP smoke, `git diff --check`, and benchmark schema self-test
  above.
- `python tools\check_dependencies.py`: host readiness passed on Windows
  `gfx1100`; accelerator enable flags remain fail-fast/evidence-only and Linux
  ROCm/Instinct exact-wide validation remains not applicable on this host.
  Current dependency output also separates
  `readiness.correctness_backend_validation` from candidate accelerator
  evidence and emits `hard_cut_self_checks` for report consistency.
- Verified post-`de1c251` exact-wide/readiness integration patch with an
  incremental Windows HIP build, targeted CTest for direct-HIP exact-wide
  max-width padded export and stale bounded metadata rejection, and dependency
  readiness output. The dependency report now exposes explicit false
  correctness-backend validation fields and explicit false Windows-to-Linux /
  Windows-to-Instinct validation claims.
- Verified bounded direct-HIP tiled-kernel hardening with an incremental Windows
  HIP build that recompiled `hip_direct_kernels.hip` and targeted CTest for the
  shared-memory tiled-tail ring GEMM, per-tile workspace/matrix residency
  contract, mixed-prefix K-split resident reuse, and prefix-9 K-block boundary
  coverage.
- `python tools\benchmark_schema.py tests\fixtures\benchmark_schema\v4_wrap64_hip.json
  tests\fixtures\benchmark_schema\v4_bounded_u64_adaptive_hip.json
  tests\fixtures\benchmark_schema\v4_bounded_i64_adaptive_hip.json`: current
  tracked v4 direct-HIP fixtures validated.
- `python tools\result_compare.py
  tests\fixtures\benchmark_schema\v4_bounded_u64_adaptive_hip.json
  tests\fixtures\benchmark_schema\v4_bounded_u64_adaptive_hip.json`: same-contract
  comparison passed with comparable direct-HIP GPU event phase order.
- `ctest --test-dir build/cpu-debug --output-on-failure`: prior CPU-only pass
  reported 58/58 passed; HIP smoke tests skipped in CPU-only build.
- The CPU and Windows HIP test passes include plan schedule inspection coverage
  for fixed-prefix bounded tile groups, CPU per-tile adaptive bounded groups,
  copied per-tile bound lifetime, wrap64 prefix-zero byte-limb scheduling, and
  tile-size validation.
- The Windows HIP test pass includes persistent direct-HIP allocation-reuse
  coverage: after warmup, repeated persistent pack/GEMM/export leaves
  allocation/free counters and device/upload/export/status buffer pointers
  unchanged and does not create a C upload buffer. The bounded signed K-split
  case covers padded `lda/ldb/ldc` and a repeated same-shape direct-HIP
  persistent path.
- The CPU and Windows HIP test pass includes hard-cut descriptor and workspace
  guards for stale exact-wide bounds, stray tile-bound storage on global
  descriptors, oversized matrix-owned storage, invalid exact-wide limb widths,
  and same-shape workspaces from the wrong semantic or bound-kind contract.
- The CPU test pass includes literal `RNS8_DEFAULT_BOUNDED_PREFIX == 9`
  contract coverage, bounded signed/unsigned range errors for too-small but
  otherwise valid global and per-tile bounds, padded bounded K-edge output
  preservation around 65535/65536/65537, full-width signed and unsigned bounded
  padded output checks, and padded exact-wide signed and
  unsigned limb export sentinel checks.
- The CPU test pass includes bounded signed and unsigned one-shot GEMMs over
  2x2 output tile grids whose tiles use selected prefixes 1, 2, 3, and 4 and
  export against tile-local bounds.
- The Windows HIP test pass includes prefix-20 bounded signed and unsigned GPU
  export checks against the CPU reference, including `INT64_MIN` and
  `UINT64_MAX` boundary outputs.
- The Windows HIP test pass includes direct-HIP per-tile bounded signed and
  unsigned output comparisons against the CPU reference, tile-local range-error
  checks, padded host export sentinels, schedule parity checks, and skipped
  high-prefix residue plane checks.
- The Windows HIP test pass includes a direct HIP one-modulus centered
  correction boundary case that compares negative, threshold, and near-zero
  residues against the CPU ring-GEMM reference.
- The Windows HIP test pass includes
  `private HIP wrap64 byte-limb GEMM matches CPU reference`,
  `direct HIP public wrap64 byte-limb path matches CPU reference`,
  `direct HIP public wrap64 tiled byte-limb path matches CPU for random padded
  layouts`, and `direct HIP wrap64 rejects CRT-style descriptors`, covering the
  low-level kernel smoke, public HIP_DIRECT one-shot/persistent byte-limb APIs,
  padded host layouts, full-width random/boundary values, and CRT metadata
  rejection against the CPU reference. Additional carry-heavy tiled wrap64
  cases compare CPU and direct-HIP output against the byte-diagonal oracle, and
  repeated export proves matrix-owned HIP export-buffer reuse while preserving
  padded host sentinels.
- The Windows HIP test pass also includes signed and unsigned exact-wide RNS
  differential checks against CPU residues plus direct HIP exact-wide limb
  export checks for padded host layouts, range errors, and signed
  two's-complement sign extension.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json`:
  detected AMD Radeon RX 7900 XTX / `gfx1100`.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend wrap64-byte-limb
  --json`: reported the CPU wrap64 byte-limb reference backend.
- `python tools\check_dependencies.py`: host readiness and Windows RDNA3 direct
  HIP gates passed; Linux ROCm/Instinct gates reported not applicable on this
  Windows host. Accelerator dependency discovery remains candidate evidence
  until an explicit backend preset is built and tested. CK, rocWMMA, and
  hipBLASLt are not promoted to correctness requirements or correctness-backend
  validation by discovery alone; AMDGPU builtins remain not ready.
- `python tools\check_dependencies.py --accelerator-probes --json`: host
  readiness stayed true while accelerator gates stayed `ok=false`. CK and
  rocWMMA probes did not run because headers were not discovered. hipBLASLt was
  candidate evidence through AMD's `roc::hipblaslt` CMake target,
  `libhipblaslt.dll.a` import archive, and `libhipblaslt.dll` runtime; no
  separate MSVC `hipblaslt.lib` is required. The hipBLASLt tiny host API probe
  auto-loaded the Visual Studio developer environment, linked the import
  archive, and ran successfully while remaining
  `candidate_accelerator_evidence_only`. The real hipBLASLt baseline is
  validated by `windows-hipblaslt-debug`, not this probe. AMDGPU builtin probes
  reported `NOT_RUN_NO_CORRECTNESS_KERNEL`; backend enablement remained
  disabled.
- `python tools\windows_dev.py cmake --preset windows-msvc-hip-accelerator-probe`:
  configured successfully, reported hipBLASLt
  imported-target/header/import-archive/DLL evidence with
  `msvc_link_probe=passed`, CK/rocWMMA not discovered, AMDGPU builtin evidence
  disabled until target-specific exact kernels exist, and accelerator backend
  enablement disabled.
- `python tools\windows_dev.py cmake --build --preset windows-accelerator-probe --target rns8-inspect`:
  built the direct-HIP inspection binary from the probe preset while keeping all
  accelerator backend enablement flags disabled.
- Benchmark captures are kept under `temp/`:
  `rns8-cpu-bounded-i64.json`, `rns8-cpu-bounded-u64.json`,
  `rns8-hip-bounded-u64.json`, `rns8-hip-bounded-u64-repeat.json`,
  `rns8-hip-bounded-u64-event-smoke.json`, and
  `rns8-hip-bounded-u64-schedule-smoke.json`. Historical schema v3 smoke
  captures in ignored `temp/` include `rns8-v3-cpu-bounded-i64.json`,
  `rns8-v3-hip-bounded-u64*.json`, `rns8-v3-wrap-u64.json`, and
  `rns8-v3-hip-wrap-u64.json`; they are not current tracked fixtures.
- `temp\rns8-hip-bounded-u64-event-smoke.json`: historical schema v2 event
  smoke evidence for `gfx1100`, live `git_commit`, `gpu_event_timing=true`,
  and nonnegative direct-HIP event arrays for `pack`, `rns_gemm`, and
  `crt_export`.
- `temp\rns8-hip-bounded-u64-schedule-smoke.json`: historical schema v2
  schedule smoke evidence with `--tile-m 64 --tile-n 64`, fixed selected prefix
  metadata, required prefix metadata, one prefix group, and
  `adaptive_execution_applied=false`.
- `python tools\result_compare.py --json temp\rns8-hip-bounded-u64.json
  temp\rns8-hip-bounded-u64-repeat.json`: same-contract comparison passed,
  including matching fixed-prefix schedule metadata and matching GPU event
  timing metadata. Captures are raw evidence only and do not establish a
  performance claim.
- Historical checkpoint command:
  `python tools\benchmark_schema.py temp\rns8-v3-cpu-bounded-i64.json
  temp\rns8-v3-hip-bounded-u64.json
  temp\rns8-v3-hip-bounded-u64-repeat.json
  temp\rns8-v3-hip-bounded-u64-repeat2.json temp\rns8-v3-wrap-u64.json
  temp\rns8-v3-hip-wrap-u64.json`: runtime captures validated as schema v3 at
  that checkpoint, including measured `scheduling` timing and explicit
  reduction availability metadata. Current tooling accepts schema v4 only.
- `temp\rns8-v4-hip-toolchain-smoke.json`: schema v4 direct-HIP bounded
  capture validated with configured HIP toolchain metadata, parsed HIP SDK root
  version `7.1`, hipcc version text from `hipcc --version`, exact
  `hipEventElapsedTime` source identity, and default bounded direct-HIP event
  source scope.
- Historical checkpoint command:
  `python tools\result_compare.py --json
  temp\rns8-v3-hip-bounded-u64-repeat.json
  temp\rns8-v3-hip-bounded-u64-repeat2.json`: same-contract schema v3
  comparison passed with comparable direct-HIP GPU event phase order. Captures
  are raw historical evidence only and do not establish a performance claim.
- `temp\rns8-v4-hip-bounded-u64-adaptive.json` and
  `temp\rns8-v4-hip-bounded-u64-adaptive-repeat.json`: direct-HIP per-tile
  adaptive bounded captures validated as schema v4, with exact seeded-input
  tile-bound metadata, `selected_kernel=direct_hip_tiled_rns_gemm_v1`,
  `adaptive_execution_applied=true`, complete HIP event timing in
  `direct_hip_bounded_adaptive_default_stream_backend_operation_groups`, and
  same-contract `tools\result_compare.py --json` comparison including matching
  tile-bound hash. Captures are raw evidence only and do not establish a
  performance claim.
- `temp\rns8-wrap-u64-bench.json` and
  `temp\rns8-wrap-u64-bench-repeat.json`: fixed-seed strict wrap64 CPU
  byte-limb captures with `prefix=0`, `bound_kind=none`,
  `packed_layout_version=byte_limb_v1`, nullable GPU event timing, and successful
  `tools\result_compare.py --json` contract comparison. Captures are raw
  evidence only and do not establish a performance claim.
- `temp\rns8-hip-wrap-u64-event-smoke.json` and
  `temp\rns8-hip-wrap-u64-event-smoke-repeat.json`: fixed-seed strict wrap64
  direct-HIP byte-limb captures with `prefix=0`, `bound_kind=none`,
  `packed_layout_version=byte_limb_v1`, `gpu_event_timing=true`, and
  wrap64-specific event phases from the byte-GEMM36 direct-HIP correctness kernel. Current
  schema v4 HIP wrap64 captures report
  `selected_kernel=direct_hip_wrap64_byte_gemm36_tiled_2d_v3`,
  `wrap64_byte_gemm36_tiled_2d_kernel`, `wrap64_export_kernel`, and
  `wrap64_export_d2h` event phases. Captures are raw evidence only and do not
  establish a performance claim.
- `temp\rns8-v4-hip-wrap-u64-byte-gemm36.json` and
  `temp\rns8-v4-hip-wrap-u64-byte-gemm36-repeat.json`: fixed-seed strict
  wrap64 direct-HIP byte-GEMM36 correctness captures validated as historical
  schema v4 evidence before the tiled byte-limb kernel rename.
  `tools\result_compare.py --json` reported the same selected kernel, event
  source scope, GPU event phase order, shape, seed, and semantic contract.
  Captures are raw evidence only and do not establish a performance claim.
- `python tools\test_benchmark_schema.py`: current benchmark schema fixture
  self-test passed, including malformed raw timing length, GPU event summary,
  invalid schedule metadata, wrap64 prefix, event-nullability, rejected retired
  schema versions, reduction-availability, and v4 per-tile adaptive contract
  checks.
- Current `tools\benchmark_schema.py` accepts schema v4 only. Older v1/v2/v3
  captures listed above are historical evidence and are not accepted by current
  tracked fixtures or current comparison tooling.
