# Performance Gain Work Queue

This queue is ordered by expected `gfx1100` end-to-end performance win, not ease.
Use it to drive implementation slices from this point forward. Keep evidence
claims local to the measured platform: Windows RX 7900 XTX / `gfx1100` evidence
does not imply Linux ROCm or Instinct readiness.

## Ground Rules

- Every performance slice needs same-contract CPU/direct-HIP baseline, release
  build, fixed seed, at least 3 warmups, at least 9 repeats, schema validation,
  selected-kernel metadata, and exact CPU differential before promotion.
- Do not promote discovery captures, smoke captures, or Windows evidence into
  Linux or Instinct claims.
- Every new kernel or layout must update `selected_kernel`, `epilogue_mode`,
  `workspace_mode`, `isa_evidence`, autotune key fields, docs, benchmark schema
  fixtures, and stale-cache rejection.

## Ordered Work Items

### 1. Native Vector-ALU Production Backend

Status: runtime backend implemented in
[src/backend_vector_alu](../src/backend_vector_alu). The benchmark harness in
[benchmarks/hip_vector_alu_baseline_kernels.hip](../benchmarks/hip_vector_alu_baseline_kernels.hip)
still exists for same-contract release comparisons.

- Add a real semantic-specific backend, not a generic accelerator:
  `bounded-i64/u64` only; no exact-wide, finite, wrap64, or CRT fallback.
  Implemented as public `RNS8_BACKEND_HIP_VECTOR_ALU_INT64`.
- Add native device storage to `rns8_matrix`: `hip_native_i64`,
  `hip_native_u64`, byte counts, currentness flags, and source version. Existing
  RNS matrices cannot feed vector kernels because they store residues, not
  original integers.
- Update `rns8_pack_i64/u64`: when HIP/AUTO and bounded semantics, populate
  compact native storage in addition to RNS, or lazily allocate native storage
  only when the selected plan needs it. Lazy is memory-friendlier; eager is
  simpler and avoids host-source loss.
- Add native export path: after vector GEMM, C can be `native_current=true` and
  `device_residues_current=false`; `rns8_export_i64/u64` must copy native C
  directly for vector-backed bounded plans.
- Guard misuse: vector-produced C cannot be consumed by RNS GEMM unless
  converted to residues. Either add native-to-RNS conversion or reject with a
  clear invalid-argument state.
- Promote only after vector path beats matrix-engine candidates in reviewed
  release groups and cache selection can choose it deterministically.

### 2. Reusable B Prepack And Tile-Swizzled Layout

Current rocWMMA B-cache is narrow and lives through `rns8_create_prepack_cache`
and `rns8_gemm_rns_prepacked_b` in [src/core/api.cpp](../src/core/api.cpp) and
[src/backend_wmma/wmma_backend_kernels.hip](../src/backend_wmma/wmma_backend_kernels.hip).

- Turn `rocwmma_b_colmajor_i8_n16_kblock65536_v1` into a measured
  `rns_i8_tile_swizzled_b_v1`.
- Key by backend, target id, kernel variant, semantic, prefix schedule hash,
  tile shape, K-block, operand role, source version, finite modulus, and layout
  version.
- Support non-tiled RNS first, then exact-wide, then finite. Adaptive/tiled
  schedules need schedule-aware cache keys and tile-local selected-prefix
  compatibility.
- Extend benchmark modes to distinguish one-shot, repeated-A, repeated-B,
  repeated-A/B, production prepack cache, and transient residency.
- Do not set `production_prepack_cache_available=1` until invalidation, role
  mismatch, stale matrix, device mismatch, and source-version rejection are all
  tested.

### 3. Bounded-i64 Winner Tuning

Current winners are rocWMMA at 512 and hipBLASLt at 1024. Tune winners, not
losers.

- For 512 rocWMMA: attack B layout, A transient pack, residue reduction, and
  launch count.
- For 1024 hipBLASLt: attack pack/transposed layout, heuristic overhead, INT32
  scratch, and separate reduce kernel.
- Add kernel variants to autotune keys, not hidden compile-time switches.
- Run 64/128/512/1024 plus 2048 before assuming the 512/1024 split persists.

### 4. Large-Shape Release Matrix

Before deep kernel work, run 2048/4096/8192 exploratory release matrices within
caps.

- Existing evidence only covers 64/128/512/1024 for most semantics.
- Large shapes may flip winners because launch and pack overhead amortize
  differently.
- Add `--include-exploratory-large` result grouping, but keep cache promotion
  blocked until complete baselines finish.

### 5. hipBLASLt Path

See [src/backend_hipblaslt/hipblaslt_backend.cpp](../src/backend_hipblaslt/hipblaslt_backend.cpp).

- Cache `hipblasLtMatmulAlgoGetHeuristic` per shape/layout/toolchain instead
  of querying inside hot dispatch.
- Prepack B for repeated-B hipBLASLt: current path packs A and B into transient
  workspace every dispatch.
- Separate timings: A pack, B pack, heuristic, matmul, reduce, and scratch
  memset if any.
- Investigate whether scratch/reduce can be narrowed: no custom epilogue is
  likely, but reduce kernels can specialize moduli and avoid generic reciprocal
  where possible.
- Ensure hipBLASLt cache entries reject wrong library version and stale kernel
  names.

### 6. CK Path

See [src/backend_ck/ck_backend_kernels.hip](../src/backend_ck/ck_backend_kernels.hip).

- Current `DeviceGemmWmma_CShuffle` parameters are one point in a large space.
  Generate multiple aliases: block size, M/N/K per block, CShuffle tile,
  transfer vector widths, M01, and K-block cap.
- Current `CenteredModulo` has branch/while style reduction. Replace it with
  branchless reciprocal two-subtract or specialized reducers for 256/255/251.
- Avoid temporary output copy/add when not needed; padded or K-split cases
  currently use `temp_c` plus copy/add kernels.
- Add per-variant ISA gates: require `v_wmma`, reject divide/rcp, reject
  unintended INT32 global stores.
- Dispatch by shape and semantic through autotune, not hard-coded "CK is good".

### 7. rocWMMA Path

See [src/backend_wmma/wmma_backend_kernels.hip](../src/backend_wmma/wmma_backend_kernels.hip).

- Reduce shared-memory round-trip after `store_matrix_sync`; consider direct
  lane-owned residue emission if rocWMMA fragment mapping permits stable
  indexing.
- Specialize mod 256/255/251 reductions in the rocWMMA epilogue.
- Add B tile swizzle and A pack variants; measure `pack_a`, `pack_b`, and
  `rns_gemm_kernel_group` separately.
- For adaptive bounded, group tile entries by prefix and shape to avoid
  per-entry overhead and wasted padded work.
- Try K-block variants below 65536 for occupancy/register pressure, not only
  overflow safety.

### 8. Finite-u8

Current finite reducers specialize direct HIP for 251/255/256, but broader
finite specialization remains open.

- Push reducer specialization into CK/rocWMMA epilogues, not only direct HIP.
- Add `finite_u8_centered_plane_v2`: canonical finite input converted to
  centered signed matrix-engine operands with layout chosen for the winner
  backend.
- Benchmark separate matrices for ring 251, ring 255, ring 256, field 251, plus
  representative non-specialized prime/composite moduli.
- Cache keys must include finite modulus. Never infer finite behavior from
  `uint8_t`.

### 9. Exact-Wide

CK already wins signed 1024 and unsigned 128/512/1024.

- Optimize exact-wide export first: limb loop coalescing, status traffic, D2H
  copy packing, and fixed limb-count specialization.
- Add exact-wide limb-count variants: 1, 2, 4, 8, 16, 32. Do not run generic
  32-limb export when the benchmark asks for 1 or 2 limbs.
- Keep signed/unsigned export functions separate; cross-export rejection is
  correctness-critical.
- For CK/rocWMMA exact-wide, measure whether GEMM or limb export is dominant
  before touching matrix kernels.

### 10. Direct-HIP RNS Fallback

Current direct HIP uses 16x16 scalar output tiles in
[src/backend_hip_direct/hip_direct_kernels.hip](../src/backend_hip_direct/hip_direct_kernels.hip).

- Specialize prefix-9 bounded kernels.
- Try one thread computing 2 or 4 neighboring columns to reuse A.
- Try larger K tile than 64 only if LDS pressure and occupancy remain healthy.
- Batch multiple moduli per launch for small shapes.
- Keep no-divide reciprocal ISA gate; add occupancy/register reporting.

### 11. Export/Status Overhead

Export can dominate once GEMM accelerates.

- Avoid status memset/D2H when range is statically impossible from plan bounds.
- Specialize bounded export for prefix 9 and prefix 20.
- Use compact contiguous D2H output staging; keep padded host layout copy
  separate only when needed.
- Add event labels for status memset, kernel, status D2H, and output D2H across
  all accelerator exports.

### 12. Adaptive Scheduling

- Compress schedule entries by selected-prefix group and tile extent.
- For uniform tile bounds, collapse to fixed-prefix path.
- Avoid copying schedule to device per workspace when schedule fingerprint
  matches an existing device buffer.
- Tune tile size 64/128/256/512 by shape; do not assume 128 default is best.

### 13. Small Shapes

64/128 often lose to overhead.

- Add low-overhead direct/vector path selected by shape.
- Avoid matrix-engine pack work when matrix-engine arithmetic cannot amortize
  setup.
- Consider HIP Graph capture for repeated small-shape calls.
- Use one-shot-vs-persistent benchmarks separately; small one-shot and
  persistent reuse have different winners.

### 14. Wrap64 Direct-HIP v3

Current direct HIP beats the rocWMMA candidate. Optimize the baseline first.

- Vectorize byte-limb load/store through `uint64_t` where layout permits.
- Reduce repeated byte extraction in
  `rns8_wrap64_accumulate_byte_gemm36_from_packed_cells`.
- Try diagonal accumulators in 32-bit where safe, widening only at carry
  boundaries.
- Increase tile K or compute multiple output cells per thread if register
  pressure allows.
- Add event timings for pack, byte-GEMM36 kernel, and export separately; current
  labels already exist.

### 15. Wrap64 Matrix Engine Redesign

Do not iterate lightly on the current candidate; it loses structurally.

- Goal is fewer WMMA passes for the 36 byte-pair products or a radically
  cheaper high-bit correction.
- Try grouped diagonals: compute low diagonals with packed signed lanes, then
  correction terms with reused fragments.
- Consider representing bytes as two 4-bit halves only if correction algebra
  and packing overhead beat direct HIP.
- Promotion requires direct-HIP v3 baseline, CPU byte-limb oracle, checksum
  parity, ISA evidence, and reviewed release win.

### 16. Reusable A And A/B Caches

- Lower priority than B, but useful for batched workloads.
- Add `RNS8_OPERAND_A` support only after B cache is production-safe.
- Full A/B cache must validate both source versions and reject mismatched plan
  fingerprints.
- Benchmark `prepack_setup_us` amortization over repeat counts, not just one
  repeated GEMM.

### 17. HIP Graphs And Launch Batching

- Use for repeated fixed-shape workflows: pack, per-prefix GEMM, export.
- Graph capture should be opt-in or benchmarked separately; graph replay must
  preserve status/error behavior.
- Biggest likely value: small shapes, many prefixes, adaptive per-tile launch
  groups.

### 18. Instrumentation

- Add per-kernel/per-prefix event hooks for CK and rocWMMA; operation-group
  timing is not enough.
- Add RGA/disassembly capture under `temp/`, checking `v_wmma`, LDS use,
  stores, waits, VGPR/SGPR, and occupancy.
- Keep counters debug/probe-only; do not poison normal benchmark paths.

### 19. Host/Transfer

- Use pinned host buffers for benchmark H2D/D2H only if reflected in metadata.
- Separate API overhead from device work; do not compare host timing to HIP
  event timing as substitutes.
- Cache workspace allocations and avoid first-use allocation inside measured
  repeats.

### 20. AUTO And Cache Architecture

- Every promoted winner needs exact plan key identity: backend, kernel,
  semantic, shape, prefix schedule, K-block, tile size, finite modulus, target
  id, HIP/accelerator version.
- Reject stale kernel names and stale epilogue names, as current cache code
  already starts doing.
- Add shape-family recommendations only after exact-shape keys are mature.

### 21. AMDGPU Builtins

- Use only after a measured CK/rocWMMA bottleneck is identified.
- Start with one target-specific exact kernel and a fail-fast CMake flag.
- Admit only with exact differential tests and ISA proof; no discovery-only
  readiness.

### 22. INT4/IU4

- Treat as experimental layout work, not immediate production.
- Must account for centered residue range, pack/unpack cost, K-block safety, and
  exact reference comparison.
- Retire if packed layout plus epilogue does not beat tuned INT8 on reviewed
  same-contract captures.

### 23. FP8/Ozaki, Strassen, Sparsity

- Research-only until exact verification metadata exists.
- Useful only if a real workload distribution makes them plausible.
- Keep outside default exact APIs.

### 24. Multi-GPU

- Not a single-`gfx1100` win.
- Linux-only later: split by modulus groups, prove at least target speedup at
  8192, and handle reconstruction/scheduling overhead.

## Best First Batch

1. Promote native vector-ALU into a real bounded-u64/i64 backend with native
   matrix storage.
2. Expand rocWMMA B prepack into a production repeated-B path and benchmark it
   hard.
3. Sweep CK/rocWMMA/hipBLASLt variants for bounded-i64 512/1024 and finite
   1024.
4. Run exploratory 2048/4096/8192 release matrices.
5. Optimize direct-HIP wrap64 v3 before touching another wrap64 matrix-engine
   candidate.
