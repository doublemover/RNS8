# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

The commands and JSON contracts below are capture mechanics, not performance
baselines. A timing capture becomes comparison evidence only after the current
schema validator accepts it, the semantic contract matches the comparison
target, and a reviewed baseline exists for the same backend family, target, and
shape.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend wrap64-byte-limb --semantics wrap-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 5 --seed 7
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics wrap-u64 --m 4 --n 4 --k 8 --warmups 1 --repeats 2 --seed 11
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --bound-mode per-tile --require-adaptive-execution --m 65 --n 65 --k 64 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 7
build\windows-msvc-hip-debug\rns8-bench.exe --backend auto --semantics bounded-i64 --m 8 --n 8 --k 8 --warmups 1 --repeats 1 --seed 23
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics exact-wide-signed --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics exact-wide-unsigned --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics finite-u8-ring --modulus 255 --m 64 --n 64 --k 64 --warmups 1 --repeats 3 --seed 1
```

The benchmark reports:

- stable `schema_version` metadata,
- requested and selected backend,
- selected kernel reported by the plan backend metadata API,
- `backend_metadata` from `rns8_get_plan_backend_info`, including selected
  kernel, accelerator/correctness/matrix-engine booleans, compiled/exact/perf
  validation booleans, accelerator library/version, capability status,
  epilogue mode, workspace mode, workspace byte requirement, ISA evidence, and
  autotune key,
- semantic contract,
- bound mode plus per-tile bound source/order/min/max/hash metadata when the
  capture uses `RNS8_BOUND_PER_TILE_*`,
- matrix shape,
- layout, K-block size, tile size, epilogue type, and packed layout version
  when exposed,
- schedule metadata from `rns8_get_plan_schedule_info`, including tile grid,
  required prefix, selected prefix, prefix group count, and adaptive
  prefix/skip flags,
- fixed seed,
- warmup and repeat counts,
- prefix count,
- command line,
- git commit resolved from the configured source checkout at benchmark runtime,
  with the CMake configure-time value used only when git is
  unavailable,
- compiler version,
- configured AMDGPU target list,
- configured HIP toolchain metadata, including HIP enablement, HIP SDK/ROCm
  root, hipcc path, hipcc version captured from `hipcc --version`, and parsed
  SDK/ROCm root version when available,
- HIP device identity and runtime metadata when using the direct HIP backend,
- clock/power settings when available; currently `null`,
- structured comparison-baseline status. Current unreviewed captures use
  `comparison_baseline.status: "required_not_recorded"` and
  `speedup_claimed: false`, with explicit prerequisite baseline names for the
  same semantic contract. Bounded captures require at least
  `same_contract_cpu_reference` and
  `same_contract_direct_hip_vector_alu_int64`; accelerator captures also name
  the same-contract direct-HIP correctness baseline. Strict wrap64 captures
  require the CPU byte-limb reference and direct-HIP byte-GEMM36 baseline.
  finite-u8 captures require CPU reference and direct-HIP finite-u8 baselines.
  `derived_tops_equivalent` remains `null` until a release-reviewed
  same-contract baseline is attached. `performance_validated=true` captures
  must use `comparison_baseline.status:
  "reviewed_release_same_contract_baseline"`,
- timing source, timing caveat, and structured timing metadata,
- explicit GPU event timing availability metadata and direct-HIP event timing
  arrays when backend hooks collect a complete repeat,
- one-time planning and matrix allocation time,
- one-time schedule metadata query time,
- average packing time,
- average persistent RNS GEMM time,
- average per-modulus GEMM estimate for RNS captures,
- average CRT export time,
- average end-to-end time for the measured phases,
- raw per-repeat timing arrays plus average, median, and p95 summaries.

Raw benchmark captures do not write production autotune cache entries. The
review path is `tools/benchmark_sweep.py --review-mode release
--write-autotune-cache`, which first validates schema, groups captures by
same-contract semantics/shape/layout/target/toolchain/input seed, requires the
matching CPU/GPU baselines, requires at least three warmups and nine measured
repeats for every capture in the same-contract group, and writes only fastest
reviewed accelerator winners. Cache entries are keyed by
`backend_metadata.autotune_key` and store backend, target, HIP SDK or
accelerator library version, shape, semantic contract, finite modulus when
present, layout, prefix schedule hash, K-block, tile size, epilogue, selected
kernel, workspace bytes, reviewed median timings, and validation status.
finite-u8 plan keys include the explicit finite modulus, so reviewed finite
cache entries are shape-and-modulus scoped. Unreviewed raw captures are not
performance validation claims, and the default smoke review mode is never a
production promotion path.
`tools/install_autotune_cache.py` is the deterministic install surface for
reviewed cache files: it validates schema-v1 reviewed-release entries, checks
identity fields against the autotune key including finite modulus, merges by
key, and writes either an explicit destination or the default cache path.

## Windows `gfx1100` release-smoke snapshot

The first release-smoke review run on Windows `gfx1100` used release opt-in
hipBLASLt, CK, and rocWMMA builds plus fixed seed `20260602`, one warmup, and
one measured repeat for the full release matrices. Raw captures and temp cache
outputs live under `temp/benchmark-sweeps/windows-gfx1100-release-*` and
`temp/accelerator-release-smoke/`; they are intentionally not tracked. Under
the current review tooling those captures are diagnostic only because they do
not satisfy the production release threshold of at least three warmups and nine
measured repeats.

Release-smoke bounded global captures covered CPU reference, direct HIP,
`hip-vector-alu-int64`, hipBLASLt, CK, and rocWMMA for bounded i64/u64 square
shapes 64, 128, 512, and 1024. The smoke review identified two candidate temp
winners that must be rerun under `--review-mode release` before production
cache promotion: bounded i64 512 selected rocWMMA
`rocwmma_i8_i32_signed_hot_residue_v1` at 2513 us end-to-end, and bounded i64
1024 selected CK `ck_wmma_cshuffle_i8_i32_centered_epilogue_v1` at 7838 us
end-to-end. Bounded u64 produced no accelerator candidate winners because
direct-HIP or vector-ALU baselines were faster for the reviewed shapes.

Release-smoke adaptive bounded captures covered the default 65x65x64 and
1024x1024x1024 per-tile schedules with CPU, direct HIP, vector-ALU, CK, and
rocWMMA. Only the 65x65x64 adaptive cases produced smoke candidate winners,
both selecting rocWMMA `rocwmma_i8_i32_signed_tiled_hot_residue_v1`: 1152 us
for bounded i64 and 1238 us for bounded u64. The 1024 adaptive cases remained
blocked by direct-HIP/vector baselines.

Release-smoke finite-u8 captures covered ring moduli 251 and 255 plus field
modulus 251 for square shapes 64, 128, 512, and 1024. Ring modulus 251 selected
CK for 64, 128, and 1024, and rocWMMA for 512. Ring modulus 255 selected
rocWMMA for 64, 128, and 512, and hipBLASLt for 1024. Field modulus 251
selected CK for 64 and 128, and rocWMMA for 512 and 1024.

Release-smoke wrap64 baseline captures kept
`direct_hip_wrap64_byte_gemm36_tiled_2d_v3` as the measured GPU path for strict
`mod 2^64`. The production-threshold release baseline is recorded below. The
internal rocWMMA wrap64 byte-GEMM36 candidate has expanded Windows `gfx1100`
private correctness differentials and ISA smoke evidence, but no wrap64
accelerator promotion was made because it has not been integrated as a public
backend or beaten direct HIP in reviewed release captures. AMDGPU builtins
remain fail-fast because the release-smoke reviews did not identify a shape
requiring a builtin kernel with exact differentials, ISA evidence, and better
timings than CK/rocWMMA.

## Windows `gfx1100` release-reviewed bounded-i64 matrix

A release-mode review on June 3, 2026 covered bounded i64 square shapes 64,
128, 512, and 1024 with CPU reference, direct HIP, `hip-vector-alu-int64`,
hipBLASLt, CK, and rocWMMA captures from release builds. Every capture used
three warmups, nine measured repeats, and seed `20260602`. The report produced
24 captures, four same-contract review groups, no missing required baselines,
and two temp reviewed cache entries in
`temp\reviewed-autotune-bounded-i64-full.json`.

The 64 and 128 groups were complete but not promotable because the vector-ALU
baseline stayed fastest: 397 us at 64x64x64 and 506 us at 128x128x128 median
end-to-end. At 512x512x512, rocWMMA
`rocwmma_i8_i32_signed_hot_residue_v1` was the fastest promotable accelerator
at 2399 us, followed by CK at 2408 us, vector-ALU at 3217 us, direct HIP at
4263 us, hipBLASLt at 6270 us, and CPU reference at 1542970 us. At
1024x1024x1024, hipBLASLt
`hipblaslt_int8_i32_scratch_reduce_baseline_v1` was fastest at 8326 us,
followed by direct HIP at 11195 us, vector-ALU at 11327 us, rocWMMA at
11565 us, CK at 18109 us, and CPU reference at 15657400 us.

`rns8-inspect` reports exact validated hits for both promoted keys on runtime
target `gfx1100`: the WMMA 512 entry uses runtime version
`repo-local release/rocm-rel-7.1` and the hipBLASLt 1024 entry uses
`hipBLASLt 100100`. With `RNS8_AUTOTUNE_CACHE_PATH` set to the temp full cache,
schema-valid AUTO smokes emit `backend_requested: "auto"`,
`backend_selected: "wmma"` for 512 and `backend_selected: "hipblaslt"` for
1024, `backend_metadata.performance_validated: true`, and
`comparison_baseline.status: "reviewed_release_same_contract_baseline"`. This
is reviewed Windows `gfx1100` release evidence and temp cache proof; it is not
yet a durable installed production cache policy.

## Windows `gfx1100` release-reviewed bounded-u64 matrix

A release-mode review on June 3, 2026 covered bounded u64 square shapes 64,
128, 512, and 1024 with CPU reference, direct HIP, `hip-vector-alu-int64`,
hipBLASLt, CK, and rocWMMA captures from release builds. Every capture used
three warmups, nine measured repeats, and seed `20260602`. The report produced
24 captures, four same-contract review groups, no missing required baselines,
and zero promotable cache entries. The requested cache path
`temp\reviewed-autotune-bounded-u64-full.json` was not written because the
cache-write status was `no_promotable_entries`.

Every reviewed shape was blocked by the vector-ALU baseline. Median end-to-end
leaders were `hip-vector-alu-int64` at 361 us for 64x64x64, 452 us for
128x128x128, 1653 us for 512x512x512, and 5649 us for 1024x1024x1024. The
closest accelerator candidates were WMMA at 1160 us for 64, WMMA at 1228 us
for 128, CK at 2347 us for 512, and CK at 5707 us for 1024. Since no
accelerator beat the required same-contract vector baseline, this matrix is
reviewed release evidence for keeping bounded-u64 AUTO on the correctness
fallback rather than promoting an accelerator cache entry.

## Windows `gfx1100` release-reviewed adaptive bounded matrix

A release-mode review on June 3, 2026 covered the default adaptive per-tile
bounded cases: 65x65x64 with 64x64 tiles and 1024x1024x1024 with 128x128
tiles, for both bounded i64 and bounded u64. The matrix used CPU reference,
direct HIP, `hip-vector-alu-int64`, CK, and rocWMMA captures from release
builds. Every capture used three warmups, nine measured repeats, and seed
`20260602`. The report produced 20 captures, four same-contract review groups,
no missing required baselines, and one temp reviewed cache entry in
`temp\reviewed-autotune-adaptive-bounded-full.json`.

The promoted entry is bounded i64 1024x1024x1024 with adaptive skip active:
rocWMMA `rocwmma_i8_i32_signed_tiled_hot_residue_v1` measured 5095 us median
end-to-end, followed by direct HIP at 6469 us, CK at 6854 us, vector-ALU at
13310 us, and CPU reference at 3774230 us. The cache entry records workspace
262144 bytes, runtime version `repo-local release/rocm-rel-7.1`, and validation
status `reviewed_release_same_contract_fastest_windows_gfx1100`. `rns8-inspect`
reports an exact validated hit for this key, and a matching adaptive
`rns8-bench --backend auto` smoke emits `backend_selected: "wmma"`,
`backend_metadata.performance_validated: true`, selected kernel
`rocwmma_i8_i32_signed_tiled_hot_residue_v1`, and schema-valid
`comparison_baseline.status: "reviewed_release_same_contract_baseline"`.

The remaining adaptive groups were complete but blocked. Bounded i64 65x65x64
stayed on vector-ALU at 402 us versus WMMA at 1041 us. Bounded u64
1024x1024x1024 stayed on vector-ALU at 6658 us, with direct HIP at 7225 us and
WMMA at 7253 us. Bounded u64 65x65x64 stayed on vector-ALU at 626 us, with CPU
reference at 1094 us and WMMA at 1238 us. No bounded-u64 adaptive accelerator
entry is promotable from this release review.

## Windows `gfx1100` release-reviewed finite-u8 matrix

A release-mode review on June 3, 2026 covered finite-u8 ring moduli 251 and
255 plus finite-u8 field modulus 251 for square shapes 64, 128, 512, and 1024.
The matrix used CPU reference, direct HIP, hipBLASLt, CK, and rocWMMA captures
from release builds, three warmups, nine measured repeats, and seed `20260602`.
It produced 60 captures, 12 same-contract review groups, no missing required
baselines, and 12 temp reviewed cache entries in
`temp\reviewed-autotune-finite-full-plan-keyed.json`.

The promoted entries are scoped by explicit `finite_modulus` in the plan
autotune key. rocWMMA
`rocwmma_i8_i32_signed_finite_u8_hot_residue_v1` won the 64, 128, and 512
groups for field-251, ring-251, and ring-255. Median end-to-end timings were
835/862/1087 us for field-251, 900/863/1049 us for ring-251, and
852/851/1083 us for ring-255 at 64/128/512 respectively. CK
`ck_wmma_cshuffle_finite_u8_centered_epilogue_v1` won the 1024 ring groups at
1428 us for modulus 251 and 1354 us for modulus 255. hipBLASLt
`hipblaslt_int8_i32_scratch_reduce_baseline_v1` won the 1024 field-251 group
at 2327 us. `rns8-inspect` reports exact validated hits for representative
hipBLASLt, CK, and rocWMMA keys on runtime target `gfx1100`, with runtime
versions `hipBLASLt 100100` and `repo-local release/rocm-rel-7.1`.
Schema-valid AUTO smokes select `backend_selected=hipblaslt`,
`backend_selected=ck`, and `backend_selected=wmma` for those representative
keys, with `backend_metadata.performance_validated: true` and reviewed-release
comparison metadata.

## Windows `gfx1100` release-reviewed wrap64 baseline

A release-mode review on June 3, 2026 covered strict wrap64 64x64x64,
128x128x128, 512x512x512, and 1024x1024x1024 with CPU byte-limb reference and
direct HIP captures from release builds. Every capture used three warmups, nine
measured repeats, and seed `20260602`. The report produced eight captures, four
same-contract review groups, no missing required baselines, and no cache entries
because no public wrap64 accelerator backend exists.

Direct HIP `direct_hip_wrap64_byte_gemm36_tiled_2d_v3` remains the measured
production GPU correctness path at 1828 us for 64, 2090 us for 128, 7757 us for
512, and 39359 us for 1024 median end-to-end. The CPU
`cpu_wrap64_byte_limb_reference_v1` measured 710 us, 5845 us, 576082 us, and
4729230 us at those shapes while consuming persistent byte-limb storage and
using exact unsigned `uint64_t` wraparound arithmetic for the low-64 product.
Any future wrap64 matrix-engine candidate must beat the direct-HIP v3 release
baseline with exact byte-limb differentials and ISA evidence before it can
displace the current path.

The internal rocWMMA wrap64 byte-GEMM36 candidate can now be captured with
`rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64` or added to
wrap64 sweeps with `--include-wrap64-wmma-candidate`. Candidate captures use a
fixed 16x16 WMMA schedule, report `backend_selected: "wmma"` and
`selected_kernel: "rocwmma_wrap64_byte_gemm36_candidate_v0"`, expose the
`wrap64_wmma_candidate_gemm36_kernel_group` HIP event phase, and remain
`performance_validated: false`. Sweep promotion keeps an explicit
`internal_candidate_not_public_backend` blocker until this path becomes a real
public backend with reviewed release evidence. Current private correctness
coverage checks single-cell K tails, exact 16x16x16 tiles, padded carry-heavy
tails, ragged two-tile output, and the `k=32768` accepted / `k=32769` rejected
candidate boundary against direct HIP and the CPU byte-pair oracle. It also
checks release-shaped 64x64x64 and 128x128x128 full-output differentials
against direct HIP and the CPU byte-pair oracle. The benchmark smoke
additionally validates same-seed 64x64x64 CPU byte-limb, direct-HIP, and
rocWMMA-candidate captures through matching `checksum_u64` values; this is
release-shape smoke evidence, not reviewed release promotion or 512/1024 output
comparison.

Bounded i64/u64 captures use persistent RNS matrices, a nonzero CRT prefix, and
`epilogue_type: "crt_export"`. Strict wrap captures use byte-limb storage with
either the CPU byte-limb reference backend or the direct-HIP tiled byte-limb
correctness path: `semantics: "wrap_u64_mod_2_64"`, `bound_kind: "none"`, `bound: 0`,
`prefix: 0`, `packed_layout_version: "byte_limb_v1"`, and `epilogue_type:
"low64_wrap_export"`. Wrap captures use the current host timing keys
`rns_gemm` and `crt_export`; their phase notes identify these as
`rns8_gemm_wrap_u64` and `rns8_export_wrap_u64`.
`per_modulus_gemm_estimate_applicable` is `false` for wrap captures.
Exact-wide captures use persistent RNS matrices with `RNS8_BOUND_NONE`, a
nonzero max-prefix RNS ladder, `semantics: "exact_wide_signed"` or
`"exact_wide_unsigned"`, `bound_kind: "none"`, `bound: 0`,
`packed_layout_version: null`, and `epilogue_type:
"exact_wide_signed_limb_export"` or `"exact_wide_unsigned_limb_export"`. They
export fixed-width little-endian `uint64_t` limbs and cannot be normalized into
bounded i64/u64 or strict wrap64 timing contracts. Exact-wide reviews require
same-contract CPU and direct-HIP baselines; the vector-ALU bounded baseline is
not applicable.
finite-u8 captures use prefix-zero finite storage with
`semantics: "finite_ring_u8"` or `"finite_field_u8"`, an explicit
`finite_modulus`, `bound_kind: "none"`, `bound: 0`, and
`epilogue_type: "canonical_u8_export"`.

Schema version 4 is the only accepted tracked capture schema. Current captures
must carry an explicit integer `"schema_version": 4`; missing version fields are
rejected instead of inferred. Schema v4 requires `backend_metadata` to mirror
the top-level `selected_kernel`, so accelerator readiness and selected-kernel
claims are tied to the public plan API instead of free-form benchmark text.
Schema v4 also includes a measured `scheduling` phase for the public
schedule-info query. The timing contract is:

```json
"raw_timings_us": {
  "planning": [123],
  "scheduling": [4],
  "matrix_alloc": [456],
  "pack": [10, 11],
  "rns_gemm": [20, 21],
  "crt_export": [30, 31],
  "end_to_end": [60, 63]
},
"timing_summary_us": {
  "planning": {"avg": 123, "median": 123, "p95": 123},
  "scheduling": {"avg": 4, "median": 4, "p95": 4},
  "pack": {"avg": 10.5, "median": 11, "p95": 11}
}
```

Schema v4 includes `timing_metadata.phase_availability`, per-tile adaptive
bounded capture metadata:
`bound_mode`, `tile_bounds_u64`, non-null `selected_kernel`, strict adaptive
schedule consistency, configured HIP toolchain metadata, and exact direct-HIP
event timing source/scope validation. The
current RNS bounded paths report `reduction.timed=false` with
`scope: "fused_into_rns_gemm"` because centered residue reduction happens inside
the `rns_gemm` phase. Strict wrap64 byte-limb captures report
`scope: "not_applicable_wrap64_byte_limb"`. Do not synthesize a reduction timing
from GEMM time.

Use `tools\benchmark_schema.py` to validate benchmark captures before using
them as comparison evidence. The validator enforces schema v4 required fields,
raw timing array lengths against `repeats`, average/median/p95 consistency,
phase-availability metadata, per-tile adaptive metadata, GPU event timing
nullability or completeness, `gpu_event_phase_order: null` when events are
unavailable, explicit event phase order for event-enabled captures, exact
matching of event timing keys to that phase order, and the strict wrap64
`prefix: 0` / `packed_layout_version: "byte_limb_v1"` metadata contract. It
also checks schedule metadata and the optional repeated packed-input contract:
`reuse_packed_inputs=true` requires `pack_mode: "prepacked_reuse"`,
`prepack_setup_us`, zero-valued repeated `pack` timing arrays, and zero-valued
pack HIP-event arrays when event timing is present. The CTest suite runs the
schema self-test, all
tracked current schema fixtures, and a same-contract `result_compare.py` check
so retired schemas and stale event labels are not only rejected manually.

Current benchmark inputs are inspectable planning contracts. Global bounded
captures remain fixed-prefix contracts. With `--bound-mode per-tile`, the
benchmark computes exact per-output-tile bounds from the seeded A/B inputs
before plan creation, passes those bounds through `rns8_gemm_desc.tile_bounds`,
requires actual prefix grouping or prefix skipping, and emits
`adaptive_execution_applied=true` only for the direct-HIP tiled bounded path.
Strict wrap64 captures report prefix zero and no RNS prefix groups.

Current direct-HIP benchmark timings use host `std::chrono::steady_clock`.
They include the current correctness backend's synchronization, first-use
matrix-owned upload/export buffer allocation when it occurs, host/device copies,
kernel launches, fused residue reduction, and GPU bounded or exact-wide export.
Exact-wide HIP event captures report `exact_wide_export_status_memset`,
`exact_wide_export_kernel`, `exact_wide_export_status_d2h`, and
`exact_wide_export_d2h` operation groups, plus the aggregate `crt_export`
phase used by the stable benchmark timing schema.

With `--reuse-packed-inputs`, the benchmark packs A/B once before warmups and
then times repeated GEMM/export calls against those persistent matrices. Such
captures report `pack_mode: "prepacked_reuse"`, `prepack_setup_us`,
`avg_prepack_setup_us`, and zero-valued per-repeat `pack` host/GPU-event
timings. `end_to_end` excludes the one-time setup. This mode is for pack
amortization evidence and does not imply a production prepack cache exists.
Release review marks `prepacked_reuse` captures ineligible for normal AUTO
autotune-cache promotion.

## Direct-HIP event timing status

The benchmark enables direct-HIP event timing through internal backend hooks for
measured repeats. Events are recorded inside the backend around operation groups
that the public benchmark phase cannot otherwise see.

When the selected backend has no event hook, or when a complete expected event
set is not available, event fields remain nullable:

```json
"timing_metadata": {
  "gpu_event_timing": false,
  "gpu_event_timing_reason": "backend_not_hip_direct"
},
"gpu_event_timings_us": null,
"gpu_event_timing_summary_us": null
```

For bounded direct-HIP captures with complete event data, `gpu_event_timing` is
`true`, `gpu_event_timings_us` contains raw per-repeat arrays, and
`gpu_event_timing_summary_us` contains average, median, and p95 summaries for:

- `pack_h2d`
- `pack_kernel`
- `pack`
- `rns_gemm_kernel_group`
- `rns_gemm`
- `crt_export_status_memset`
- `crt_export_kernel`
- `crt_export_status_d2h`
- `crt_export_d2h`
- `crt_export`

For strict wrap64 direct-HIP captures, event timing uses wrap64-specific labels
plus current aggregate phase labels:

- `pack_h2d`
- `pack_kernel`
- `pack`
- `wrap64_byte_gemm36_tiled_2d_kernel`
- `rns_gemm`
- `wrap64_export_kernel`
- `wrap64_export_d2h`
- `crt_export`

The wrap64 direct-HIP event source scope is
`direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups`. It
describes the tiled byte-limb correctness path, not an optimized matrix-engine
byte-GEMM backend.

For finite-u8 captures, pack/export event labels are finite-specific:
`finite_pack_h2d`, `finite_pack_kernel`, `finite_export_kernel`, and
`finite_export_d2h`. Direct-HIP finite captures use
`finite_resident_gemm_kernel` for the resident finite GEMM. hipBLASLt finite
captures combine those finite pack/export labels with the hipBLASLt operation
labels. CK and rocWMMA finite captures combine the finite pack/export labels
with the shared `rns_gemm_kernel_group` accelerator operation-group label.

For non-finite explicit CK and rocWMMA captures, event timing uses the same
direct-HIP pack/export labels plus a backend-owned `rns_gemm_kernel_group`
label around the accelerator GEMM device call. This is operation-group evidence
only. It does not expose CK-internal or rocWMMA-internal per-kernel, per-prefix,
or per-tile timing.

Host timings and HIP event timings answer different questions. Host
`std::chrono::steady_clock` timings include API dispatch, CPU scheduling,
allocations, and synchronous host-side overhead. HIP event timings record
default-stream backend operation groups only. Do not compare event timings to
host timings as replacements, and do not replace nullable event fields with
host wall-clock timings or estimates.

## Production-grade follow-up sweeps

Debug and one-repeat release captures are useful for correctness and early
shape triage, but final performance promotion requires production-grade release
sweeps. Promotable Windows `gfx1100` captures use fixed seeds, HIP event timing
where backend hooks exist, at least three warmups, at least nine measured
repeats, schema v4 validation, and same-contract baseline groups.

Bounded i64/u64 promotion requires CPU reference, direct-HIP correctness, and
`hip-vector-alu-int64` baselines for the same semantic contract, shape, layout,
target id, HIP SDK and accelerator library versions, seed, warmups, repeats,
prefix schedule, K-block, tile size, epilogue, and selected input
distribution. finite-u8 promotion requires CPU and direct-HIP finite baselines
for the same explicit modulus. Exact-wide signed/unsigned promotion requires
CPU and direct-HIP exact-wide baselines with the same fixed-width limb export
contract. Strict wrap64 promotion requires CPU byte-limb and direct-HIP
`direct_hip_wrap64_byte_gemm36_tiled_2d_v3` baselines.

Current Windows release sweep status:

- bounded i64/u64 square 64, 128, 512, and 1024 have local release-reviewed
  reports with complete baselines;
- adaptive bounded 65x65x64 and 1024x1024x1024 have local release-reviewed
  reports with complete baselines;
- finite-u8 ring moduli 251 and 255 plus finite-u8 field modulus 251 have a
  local release-reviewed matrix with 12 temp reviewed cache entries keyed by
  explicit modulus;
- exact-wide signed/unsigned 64, 128, 512, and 1024 have a local
  release-reviewed matrix with complete CPU/direct-HIP baselines and four temp
  reviewed CK cache entries: signed 1024 plus unsigned 128, 512, and 1024;
- strict wrap64 has local release-reviewed CPU/direct-HIP baselines for 64, 128,
  512, and 1024; the matrix-engine accelerator candidate remains open;
- 2048, 4096, and 8192 remain exploratory until complete baselines finish
  within the run cap.

Review reports must include per-phase medians, speedups versus direct-HIP and
vector-ALU baselines where applicable, promotion blockers, selected kernel,
target id, HIP SDK and accelerator library versions, event source, epilogue,
workspace bytes, winner rationale, and cache-write status. Durable docs may
summarize reviewed release results only; raw captures and temp cache files stay
under ignored `temp/`.

`tools/benchmark_sweep.py` review reports currently use `schema_version: 3`.
The schema is additive over raw capture schema v4 and records group-level
target/toolchain/library metadata plus candidate-level source metadata,
promotion rationale, and `eligible_after_review`, `not_requested`, `pending`,
`written`, or `not_eligible` cache-write states. This report schema is the
promotion artifact; it does not make raw captures performance claims by itself.

## Packed low-bit benchmark matrix

The near-ideal packed low-bit pipeline is measured as a separate set of
experiments before it becomes production layout policy. Required candidates:

- `rns_i8_modulus_major_v2` for current centered RNS INT8 planes;
- `rns_i8_tile_swizzled_b_v1` for repeated-B rocWMMA/CK panels;
- `finite_u8_centered_plane_v2` for canonical finite-u8 inputs converted to
  centered signed matrix-engine operands;
- `wrap64_byte_limb_gemm36_v2` for strict low-64 byte-limb GEMM36 candidates;
- research-only `rns_i4_packed_v0` for IU4/INT4 experiments.

Each layout experiment records pack A, pack B, raw GEMM, fused epilogue or
reduction, CRT/export, workspace allocation, and end-to-end phases. It must
separate one-shot, repeated-A, repeated-B, and repeated-A/B workloads so pack
amortization is visible. A layout cannot be promoted unless source-version
invalidation, operand-role mismatch rejection, tile tails, K-block splits,
adaptive prefix schedules, finite modulus metadata, and exact CPU/direct-HIP
differentials are all covered.

INT4/IU4, AMDGPU builtins, FP8/Ozaki, and wrap64 matrix-engine paths are
retired per semantic/target if they fail to beat the tuned INT8 or current
direct-HIP path after layout, epilogue, and ISA-confirmed matrix-instruction
tuning. No theoretical TOPS claim is accepted without reviewed same-contract
captures.

Future benchmark work must add deeper scheduler internals, reviewed raw sweeps,
comparison baselines, and performance gates before any speedup claims are made.

`tools/result_compare.py` validates both captures before comparing host timing
phases for schema v4 captures. Its same-contract check covers semantic contract,
bound mode, bounds, tile-bound source/order/min/max/hash, shape, prefix, seed,
input distribution, epilogue, packed layout, repeated packed-input mode, and
schedule metadata. Backend and selected-kernel differences are reported as
evidence, not as contract failures.
GPU compatibility is a separate gate: GPU-vs-GPU comparisons require matching
compiler, configured target, HIP toolchain, device target, runtime, and driver
fields, while CPU/reference and wrap64 byte-limb baselines can compare against a
GPU capture without fabricating a GPU target. It also compares
`gpu_event_timing_summary_us` phases only when both captures set
`timing_metadata.gpu_event_timing=true` and report the same event timing source,
source scope, and GPU event phase order. Per-modulus timing rows are flagged as
not applicable when a capture says `per_modulus_gemm_estimate_applicable:
false`; one-time `prepack_setup` timing is compared only when both captures
provide `avg_prepack_setup_us`.
