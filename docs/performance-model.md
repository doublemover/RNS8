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
  prefix/skip flags. Bounded input-range plans also report the public
  `bound_kind`, derived `effective_bound`, `lhs_bound`, `rhs_bound`, and
  `bound_contract` string used to produce the schedule,
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
- explicit GPU event timing availability metadata and per-backend HIP event
  timing arrays when hooks collect a complete repeat,
- one-time planning and matrix allocation time,
- one-time schedule metadata query time,
- one-time per-tile bound scan time for adaptive bounded captures,
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
present, layout, prefix schedule hash, accumulator contract, K-block, tile
size, epilogue, selected kernel, workspace bytes, reviewed median timings, and
validation status.
finite-u8 plan keys include the explicit finite modulus, so reviewed finite
cache entries are shape-and-modulus scoped. Unreviewed raw captures are not
performance validation claims, and the default smoke review mode is never a
production promotion path.
`tools/install_autotune_cache.py` is the deterministic install surface for
reviewed cache files: it validates schema-v1 reviewed-release entries, checks
identity fields against the autotune key including finite modulus, merges by
key, and writes either an explicit destination or the default cache path. The
installer accepts `hip-vector-alu-int64` only for bounded i64/u64 reviewed
entries with native final/export epilogues; finite, exact-wide, and wrap64
entries still require their explicit residue or byte-limb backend contracts.

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

Release-smoke wrap64 baseline captures originally kept
`direct_hip_wrap64_byte_gemm36_tiled_2d_v3` as the measured GPU path for strict
`mod 2^64`. The June 4, 2026 v4 validation superseded that direct-HIP kernel
locally with `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4`. The
production-threshold release baseline is recorded below. The
internal rocWMMA wrap64 byte-GEMM36 candidate has expanded Windows `gfx1100`
unit-level correctness differentials and ISA smoke evidence, but no wrap64
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
and two reviewed cache candidates. The durable summary is
[reviewed-local-evidence.md](reviewed-local-evidence.md); raw captures and
candidate cache files remain temp-only.

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

Current CK and rocWMMA RNS plans now report common-modulus reducer v2
identities after the shared epilogues gained explicit 256/255/251 reduction
paths: `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2`,
`ck_wmma_cshuffle_tiled_i8_i32_mod251_255_256_centered_epilogue_v2`,
`rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2`, and
`rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2`. The v1 timings in
this section remain historical reviewed evidence only; they should not be
installed or transferred into current CK/rocWMMA bounded or exact-wide cache
entries.

A current-v2 follow-up bounded-i64 validation on June 4, 2026 used seed
`20260604` and reran the 512 and 1024 groups with CPU reference, direct HIP,
`hip-vector-alu-int64`, hipBLASLt, CK, and rocWMMA release captures. It produced
12 captures, no missing required baselines, no incompatible metadata, no
duplicate backend records, and one reviewed cache candidate. All ten GPU
captures passed `tools/gpu_event_report.py --fail-on-unavailable`.

At 512x512x512, direct HIP
`direct_hip_tiled_active_prefix_rns_gemm_v2` stayed fastest at 1851 us median
end-to-end, followed by rocWMMA v2 at 2591 us, vector-ALU at 6147 us, CK v2 at
7172 us, hipBLASLt v2 at 10101 us, and CPU reference at 565180 us. No accelerator
cache entry was written for 512. At 1024x1024x1024, hipBLASLt
`hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` was fastest at
4174 us, followed by direct HIP at 4535 us, rocWMMA v2 at 12996 us, CK v2 at
15546 us, vector-ALU at 33945 us, and CPU reference at 4915270 us. The generated
reviewed temp cache contains that single 1024 hipBLASLt v2 entry. Installing it
with `tools/install_autotune_cache.py --replace-existing` replaced a stale local
default cache that failed reviewed-cache validation with a target-id/key
mismatch. A hipBLASLt release `rns8-bench --backend auto` smoke for the same
1024 bounded-i64 key selected hipBLASLt and reported
`backend_metadata.performance_validated: true`.

## Windows `gfx1100` release-reviewed bounded-u64 matrix

A release-mode review on June 3, 2026 covered bounded u64 square shapes 64,
128, 512, and 1024 with CPU reference, direct HIP, `hip-vector-alu-int64`,
hipBLASLt, CK, and rocWMMA captures from release builds. Every capture used
three warmups, nine measured repeats, and seed `20260602`. The report produced
24 captures, four same-contract review groups, no missing required baselines,
and zero promotable cache entries. The cache-write status was
`no_promotable_entries`.

Every reviewed shape was blocked by the vector-ALU baseline. Median end-to-end
leaders were `hip-vector-alu-int64` at 361 us for 64x64x64, 452 us for
128x128x128, 1653 us for 512x512x512, and 5649 us for 1024x1024x1024. The
closest accelerator candidates were rocWMMA at 1160 us for 64, rocWMMA at 1228 us
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
no missing required baselines, and one reviewed cache candidate.

The promoted entry is bounded i64 1024x1024x1024 with adaptive skip active:
rocWMMA `rocwmma_i8_i32_signed_tiled_hot_residue_v1` measured 5095 us median
end-to-end, followed by direct HIP at 6469 us, CK at 6854 us, vector-ALU at
13310 us, and CPU reference at 3774230 us. The cache entry records workspace
262144 bytes, runtime version `repo-local release/rocm-rel-7.1`, and validation
status `reviewed_release_same_contract_fastest_windows_gfx1100`. At the time,
`rns8-inspect` reported an exact validated hit for this key, and a matching
adaptive `rns8-bench --backend auto` smoke emitted
`backend_selected: "rocwmma"`, `backend_metadata.performance_validated: true`,
selected kernel `rocwmma_i8_i32_signed_tiled_hot_residue_v1`, and
schema-valid
`comparison_baseline.status: "reviewed_release_same_contract_baseline"`.
That reviewed cache identity is historical after the current tiled v2
selected-kernel rename and must not be installed.

A current-v2 adaptive-bands release review on June 4, 2026 used seed
`20260604`, three warmups, nine measured repeats, CPU reference, Direct HIP,
runtime `hip-vector-alu-int64`, CK, and rocWMMA. The corrected review grouped
the 15 captures into three same-contract groups with no missing required
baselines, no duplicate backend records, compatible target/toolchain metadata,
schema-valid captures, and required GPU events for GPU records. Direct HIP was
fastest in every group: 1848 us for bounded i64 256x256x512, 4937 us for
bounded i64 1024x1024x1024, and 4224 us for bounded u64 512x1024x512. CK and
rocWMMA current-v2 tiled paths lost to Direct HIP at all reviewed adaptive-bands
shapes, so no adaptive accelerator cache entry is promotable from the current
review.

## Windows `gfx1100` release-reviewed finite-u8 matrix

A release-mode review on June 3, 2026 covered finite-u8 ring moduli 251 and
255 plus finite-u8 field modulus 251 for square shapes 64, 128, 512, and 1024.
The matrix used CPU reference, direct HIP, hipBLASLt, CK, and rocWMMA captures
from release builds, three warmups, nine measured repeats, and seed `20260602`.
It produced 60 captures, 12 same-contract review groups, no missing required
baselines, and 12 reviewed cache candidates keyed by explicit modulus.

The promoted entries are scoped by explicit `finite_modulus` in the plan
autotune key. In the historical June 3 review, rocWMMA
`rocwmma_i8_i32_signed_finite_u8_hot_residue_v1` won the 64, 128, and 512
groups for field-251, ring-251, and ring-255. Median end-to-end timings were
835/862/1087 us for field-251, 900/863/1049 us for ring-251, and
852/851/1083 us for ring-255 at 64/128/512 respectively. CK
`ck_wmma_cshuffle_finite_u8_centered_epilogue_v1` won the 1024 ring groups at
1428 us for modulus 251 and 1354 us for modulus 255. hipBLASLt
`hipblaslt_int8_i32_scratch_reduce_baseline_v1` won the 1024 field-251 group
at 2327 us. Those reviewed timings remain attached to their recorded kernel
identities; they are not automatically transferred to later fixed-modulus v2
accelerator identities. `rns8-inspect` reports exact validated hits for
representative hipBLASLt, CK, and rocWMMA keys on runtime target `gfx1100`,
with runtime versions `hipBLASLt 100100` and
`repo-local release/rocm-rel-7.1`. Schema-valid AUTO smokes select
`backend_selected=hipblaslt`, `backend_selected=ck`, and
`backend_selected=rocwmma` for those representative keys, with
`backend_metadata.performance_validated: true` and reviewed-release comparison
metadata.

A focused follow-up on June 4, 2026 tested CK/rocWMMA fixed-modulus accelerator
reducer identities for finite ring modulus 256 at 512x512x512 with release
builds, three warmups, nine repeats, seed `20260602`, and required GPU events.
The v2 kernels were correct and event-visible, but they did not promote:
direct-HIP was 1382 us end-to-end, CK v2 was 1533 us, and rocWMMA v2 was
1486 us. Current CK/rocWMMA finite-u8 planning also reports explicit
fixed-modulus v2 identities for 251 and 255:
`ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2`,
`ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2`,
`rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2`, and
`rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2`. These identifiers
make schema/cache evidence precise, but their speedups remain unreviewed until
new release captures compare them against direct HIP and the historical
same-modulus accelerator baselines end-to-end.

A current-v2 finite-u8 release review on June 4, 2026 used seed `20260604` and
covered 512 and 1024 for ring moduli 251, 255, and 256 plus field modulus 251.
Each group included CPU reference, Direct HIP, hipBLASLt, CK, and rocWMMA
release captures, and all groups had complete required baselines and compatible
runtime/toolchain metadata. The review produced five event-valid cache entries:
rocWMMA ring-251 at 1024 measured 1709 us versus Direct HIP at 4682 us, CK
ring-255 at 1024 measured 1938 us versus Direct HIP at 5814 us, rocWMMA
ring-256 at 512 measured 1365 us versus Direct HIP at 5569 us, hipBLASLt
ring-256 at 1024 measured 1792 us versus Direct HIP at 12633 us, and CK
field-251 at 1024 measured 1860 us versus Direct HIP at 10564 us. GPU event
reports with `--fail-on-unavailable` passed for all five promoted source
captures.

The same review intentionally did not promote field-251 at 512 after tightening
the release cache gate: hipBLASLt measured 1471 us versus Direct HIP at 1476 us,
but the hipBLASLt capture had `gpu_event_timing_status:
unavailable_missing_expected_events`. `benchmark_sweep.py` now adds
`missing_required_gpu_events` to accelerator promotion blockers, so a raw timing
near-tie without required backend events cannot become a `performance_validated`
cache entry. Installing the four finite temp caches with
`tools/install_autotune_cache.py` merged five finite entries into the existing
default local runtime cache for six entries total, preserving the bounded-i64
1024 hipBLASLt entry.

A small-shape current-v2 finite-u8 follow-up on June 4, 2026 reran 64 and 128
for the same ring/field contracts with seed `20260604`, release builds, three
warmups, nine repeats, CPU and Direct-HIP baselines, and required GPU events.
Two 128x128x128 rocWMMA entries promoted and were installed in the local default
cache: ring-251 measured 1136 us versus Direct HIP at 1261 us and CPU at
1370 us, while ring-256 measured 1132 us versus Direct HIP at 1149 us and CPU at
1730 us. Ring-255 64 is deliberately not promoted even though rocWMMA measured
1257 us versus Direct HIP at 3388 us, because the CPU reference measured 167 us.
`benchmark_sweep.py` now adds `not_faster_than_cpu_reference` to accelerator
promotion blockers when a cache candidate loses to the required CPU baseline.
Installing the two small-shape finite temp caches increased the default local
runtime cache to 11 entries total.

## Windows `gfx1100` release-reviewed exact-wide matrix

Current exact-wide v2 release reviews now cover signed and unsigned 64, 128,
512, 1024, and 2048 with CPU reference, Direct HIP, hipBLASLt, CK, and rocWMMA
same-contract release captures. The 512/1024 pass used seed `20260604`; the
64/128 refresh and large 2048 pass used seed `20260605`. Promoted entries used
release builds, three warmups, nine measured repeats, schema-v4 validation,
same-contract CPU/direct baselines, compatible target/toolchain/commit metadata,
and required GPU events for the selected accelerator.

The installed exact-wide cache entries are: unsigned 64 hipBLASLt at 4611 us,
signed 512 rocWMMA at 7162 us, signed 1024 hipBLASLt at 17092 us, unsigned 1024
CK at 20481 us, signed 2048 hipBLASLt at 59074 us, and unsigned 2048 hipBLASLt
at 40985 us. Signed 64, signed 128, unsigned 128, and unsigned 512 remain on
Direct HIP in the current matrix.

The large exact-wide 2048 release-validation pass under
`temp/perf-work-queue/large-release-validation-2048-exact-wide-current/`
produced two clean review groups and two installed cache entries. For signed
2048, CPU measured 19040900 us, Direct HIP measured 131794 us, and hipBLASLt
won at 59074 us, 2.23x faster than Direct HIP. For unsigned 2048, CPU measured
15742000 us, Direct HIP measured 124570 us, and hipBLASLt won at 40985 us,
3.04x faster than Direct HIP. Required GPU events show the 2048 winners are
export-bound after GEMM acceleration: signed 2048 reported `crt_export` at
18940.650 us and `exact_wide_export_kernel` at 18773.540 us, while unsigned
2048 reported `crt_export` at 14407.720 us and `exact_wide_export_kernel` at
14182.880 us. That makes fixed-width CRT/export specialization and lazy
residue-current output the next exact-wide performance targets.

A follow-up exact-wide export specialization now treats signed three-limb
prefix-20 output as full-width for status-elision purposes. The default
prefix-20 product is 155 bits and the centered signed magnitude is 154 bits, so
three 64-bit limbs cover every signed exact-wide value produced by the current
device reconstruction range. Runtime export, benchmark metadata, and schema-v4
validation now agree that signed and unsigned limb counts 3..32 report
`exact_wide_export_status_check: "elided_full_width_device_reconstruction"` and
zero-valued status memset/D2H event phases. A focused Direct-HIP 2048 A/B under
`temp/exact-wide-signed-2048-limbs3-direct.json` and
`temp/exact-wide-signed-2048-limbs4-direct.json` measured 190940 us median
end-to-end for signed three-limb output versus 194115 us for signed four-limb
output. That is output-contract-specific export evidence, not a cache-promotion
claim.

## Windows `gfx1100` release-reviewed wrap64 baseline

A release-mode review on June 3, 2026 covered strict wrap64 64x64x64,
128x128x128, 512x512x512, and 1024x1024x1024 with CPU byte-limb reference and
direct HIP captures from release builds. Every capture used three warmups, nine
measured repeats, and seed `20260602`. The report produced eight captures, four
same-contract review groups, no missing required baselines, and no cache entries
because no public wrap64 accelerator backend exists.

Direct HIP `direct_hip_wrap64_byte_gemm36_tiled_2d_v3` was the June 3 measured
production GPU correctness path at 1828 us for 64, 2090 us for 128, 7757 us for
512, and 39359 us for 1024 median end-to-end. The CPU
`cpu_wrap64_byte_limb_reference_v1` measured 710 us, 5845 us, 576082 us, and
4729230 us at those shapes while consuming persistent byte-limb storage and
using exact unsigned `uint64_t` wraparound arithmetic for the low-64 product.
On June 4, 2026, paired release captures under
`temp/perf-work-queue/wrap64-v4/` updated the direct-HIP path to
`direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4`. The v4 path uses direct
unsigned byte products, a safe uint32 low-diagonal accumulator for `K <= 4096`,
and scalar small-shape pack/export fallbacks. Against same-seed v3 captures,
median end-to-end speedups were 1.07x, 1.17x, 1.02x, and 5.60x for default
64/128/512/1024 captures, and 1.22x, 4.67x, 1.07x, and 6.74x for
reuse-packed-input captures. Final v4 median end-to-end times were 1137 us,
1245 us, 7812 us, and 6496 us for the default 64/128/512/1024 captures.
Any future wrap64 matrix-engine candidate must beat the direct-HIP v4 release
baseline with exact byte-limb differentials and ISA evidence before it can
displace the current path.

A June 5, 2026 large-shape release-validation follow-up covered strict wrap64
2048x2048x2048 with seed `20260605`, release builds, three warmups, and nine
measured repeats. The same-contract review had no missing required baselines,
duplicate backends, target/toolchain incompatibilities, or commit mismatches.
The optimized CPU byte-limb reference measured 13423400 us median end-to-end,
while Direct HIP v4 measured 58331 us median end-to-end. Required Direct-HIP GPU
events were present; the event report attributed the median GPU stream time
primarily to `wrap64_byte_gemm36_tiled_2d_kernel` at 43597.1 us. No cache entry
is written because strict wrap64 Direct HIP is a correctness backend, not an
AUTO-promoted accelerator entry.

The internal rocWMMA wrap64 byte-GEMM36 candidate can now be captured with
`rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64` or added to
wrap64 sweeps with `--include-rocwmma-wrap64-candidate`. Candidate captures use a
fixed 16x16 WMMA schedule, report `backend_selected: "rocwmma"` and
`selected_kernel: "rocwmma_wrap64_byte_gemm36_candidate_v0"`, expose the
`wrap64_rocwmma_candidate_gemm36_kernel_group` HIP event phase, and remain
`performance_validated: false`. Sweep promotion keeps an explicit
`internal_candidate_not_public_backend` blocker until this path becomes a real
public backend with reviewed release evidence. Current unit-level correctness
coverage checks single-cell K tails, exact 16x16x16 tiles, padded carry-heavy
tails, ragged two-tile output, and the `k=32768` accepted / `k=32769` rejected
candidate boundary against direct HIP and the CPU byte-pair oracle. It also
checks release-shaped 64x64x64 and 128x128x128 full-output differentials
against direct HIP and the CPU byte-pair oracle, plus 512x512x512 and
1024x1024x1024 full candidate-vs-direct-HIP output checks with sampled
CPU-oracle cells. The benchmark smoke additionally validates same-seed
64x64x64 CPU byte-limb, direct-HIP, and rocWMMA-candidate captures through
matching `checksum_u64` values; this is release-shape smoke evidence, not
reviewed release promotion or performance evidence.

A follow-up candidate-inclusive release review on June 3, 2026 used three
warmups, nine repeats, and seed `20260603` for 64, 128, 512, and 1024 square
wrap64 shapes. The CPU byte-limb, direct-HIP, and rocWMMA-candidate captures
produced matching `checksum_u64` values within each shape, but the candidate
lost to direct HIP at every release shape:

| shape | historical direct-HIP v3 median us | rocWMMA candidate median us | candidate blocker |
|---|---:|---:|---|
| 64x64x64 | 3653 | 4825 | `internal_candidate_not_public_backend`, `not_faster_than_direct_hip` |
| 128x128x128 | 1852 | 5202 | `internal_candidate_not_public_backend`, `not_faster_than_direct_hip` |
| 512x512x512 | 9430 | 37481 | `internal_candidate_not_public_backend`, `not_faster_than_direct_hip` |
| 1024x1024x1024 | 41237 | 264657 | `internal_candidate_not_public_backend`, `not_faster_than_direct_hip` |

The review produced zero promotable entries, so the candidate should not be
integrated as a public wrap64 backend without a materially different kernel.

Bounded i64/u64 captures use persistent RNS matrices, a nonzero requested RNS
prefix budget, and `epilogue_type: "crt_export"`. In schema v4, `prefix`
remains the requested max prefix for compatibility, while optional additive
fields `selected_prefix`, `requested_max_prefix`, `contract_prefix_policy`,
`residue_planes_requested`, `residue_planes_selected`,
`residue_planes_skipped`, and `residue_plane_skip_fraction` make plane deletion
explicit. The default RNS policy is `minimum_proven`: global bounded and
exact-wide plans execute the minimum prefix proven by the contract. Use
`--prefix-policy fixed-requested` for controlled full-prefix comparison
captures, and `--max-prefix N` to change the requested ceiling. Strict wrap
captures use byte-limb storage with either the CPU byte-limb reference backend
or the direct-HIP tiled byte-limb correctness path: `semantics:
"wrap_u64_mod_2_64"`, `bound_kind: "none"`, `bound: 0`, `prefix: 0`,
`packed_layout_version: "byte_limb_v1"`, and `epilogue_type:
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
Schema v4 also requires `backend_metadata.accumulator_safety`, and autotune
keys must include the accumulator type, signedness, modulus policy, K-block
size, and K-block cap before evidence can be accepted.
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

Schema v4 includes `timing_metadata.phase_availability`, optional prefix-policy
metadata, per-tile adaptive bounded capture metadata:
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
matching of event timing keys to that phase order, scope-aware deep
accelerator/vector-ALU event labels, and the strict wrap64 `prefix: 0` /
`packed_layout_version: "byte_limb_v1"` metadata contract. It also checks
schedule metadata and the optional repeated packed-input contract:
`reuse_packed_inputs=true` requires one of `pack_mode: "prepacked_reuse"`,
`"prepacked_reuse_a"`, or `"prepacked_reuse_b"`, matching
`prepack_reuse_operands`, and `prepack_setup_us`. Full A/B reuse requires
zero-valued repeated `pack` timing arrays and zero-valued pack HIP-event arrays
when event timing is present; A-only and B-only reuse keep per-repeat `pack`
timings for the non-reused operand. The CTest suite runs the schema self-test, all
tracked current schema fixtures, and a same-contract `result_compare.py` check
so retired schemas and stale event labels are not only rejected manually.

Current benchmark inputs are inspectable planning contracts. Global RNS
captures default to minimum-proven uniform selected prefixes; fixed-prefix
comparison captures opt in with `--prefix-policy fixed-requested`. With
`--bound-mode per-tile`, the benchmark computes exact per-output-tile bounds
from the seeded A/B inputs before plan creation, passes those bounds through
`rns8_gemm_desc.tile_bounds`, requires actual prefix grouping or prefix
skipping unless fixed-requested policy is selected, and emits
`adaptive_execution_applied=true` only for backends that execute a real tiled
adaptive path. Strict wrap64 captures report prefix zero and no RNS prefix
groups.

Current direct-HIP benchmark timings use host `std::chrono::steady_clock`.
They include the current correctness backend's synchronization, first-use
matrix-owned upload/export buffer allocation when it occurs, host/device copies,
kernel launches, fused residue reduction, and GPU bounded or exact-wide export.
Exact-wide HIP event captures report `exact_wide_export_status_memset`,
`exact_wide_export_kernel`, `exact_wide_export_status_d2h`, and
`exact_wide_export_d2h` operation groups, plus the aggregate `crt_export`
phase used by the stable benchmark timing schema.

With `--reuse-packed-inputs`, the benchmark packs A/B once before warmups and
then times repeated GEMM/export calls against those persistent matrices. With
`--reuse-packed-a` or `--reuse-packed-b`, it pre-packs only the selected operand
and includes per-repeat packing of the other operand in `pack` and
`end_to_end`. Such captures report `prepack_reuse_operands`,
`prepack_reuse_strategy`, `prepack_setup_us`, and `avg_prepack_setup_us`.
Eligible rocWMMA non-tiled RNS `--reuse-packed-b` captures use
`rns8_create_prepack_cache` plus `rns8_gemm_rns_prepacked_b` and stamp
`prepack_reuse_strategy: "rocwmma_reusable_b_cache"`; other current reuse
captures stamp `persistent_matrix_residency`. This mode family is for pack
amortization evidence and does not imply a production prepack cache exists.
Release review marks all prepacked-reuse captures ineligible for normal AUTO
autotune-cache promotion.

## GPU event timing status

The benchmark enables HIP event timing through internal hooks for measured
repeats. Events are recorded around default-stream device operation groups that
the public host phase cannot otherwise see. Event collection is deferred until a
timing snapshot, reset, or disable call, so deep per-kernel labels do not force a
host synchronization after every suboperation.

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
- `direct_hip_zero_output_tile_memset`, only for adaptive per-tile bounded
  direct-HIP schedules with proven zero-output tiles
- `rns_gemm`
- `crt_export_status_memset`
- `crt_export_kernel`
- `crt_export_status_d2h`
- `crt_export_d2h`
- `crt_export`

For all-zero adaptive per-tile Direct-HIP schedules, input packing and export
status traffic are both elided. The trusted tile-bound schedule proves every
output tile zero, so the measured repeat reports zero-valued `pack_h2d`,
`pack_kernel`, and `pack` phases, materializes resident zero RNS output without
reading A/B, zero-fills the compact native export buffer, and avoids export
tile schedule/bounds uploads. Captures keep the same phase order, but
`crt_export_status_memset` and `crt_export_status_d2h` are also reported as
zero-valued arrays.

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
with `rns_gemm_kernel_group` under the older operation-group accelerator scope;
they do not emit selected-prefix deep labels because finite-u8 GEMM uses one
explicit benchmark modulus rather than the default RNS prefix ladder.

Explicit bounded/exact CK and rocWMMA captures use
`accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export`.
Older valid captures with
`accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export`
remain readable only when they contain the old operation-group labels. New deep
captures add aggregate accelerator labels and selected-prefix labels:

- CK aggregate labels: `ck_pack_a_kernel`, `ck_pack_b_kernel`,
  `ck_wmma_cshuffle_matmul`, `ck_copy_centered_kernel`, and
  `ck_add_centered_kernel`.
- CK prefix labels: `ck_prefix_XX_pack_a`, `ck_prefix_XX_pack_b`,
  `ck_prefix_XX_matmul`, `ck_prefix_XX_copy_centered`, and
  `ck_prefix_XX_add_centered`.
- rocWMMA aggregate labels: `rocwmma_pack_a_kernel`,
  `rocwmma_pack_b_kernel`, and `rocwmma_matmul_kernel`.
- rocWMMA prepacked-B labels: `rocwmma_pack_a_prepacked_b_kernel`,
  `rocwmma_matmul_prepacked_b_kernel`,
  `rocwmma_prefix_XX_pack_a_prepacked_b`, and
  `rocwmma_prefix_XX_matmul_prepacked_b`.
- rocWMMA prefix labels without prepacked B: `rocwmma_prefix_XX_pack_a`,
  `rocwmma_prefix_XX_pack_b`, and `rocwmma_prefix_XX_matmul`.

The `XX` prefix index is zero-based. Fixed-requested captures emit labels
through the requested benchmark prefix. Minimum-proven global and adaptive
captures emit labels through `schedule_metadata.max_selected_prefix`, also
reported as `selected_prefix` when prefix-policy metadata is present. Values
aggregate all tiles and K-blocks for that prefix within one repeat. Optional
copy/add labels are zero-filled
when the corresponding operation does not launch; missing required pack/matmul
labels make event timing unavailable.

Native vector-ALU captures use
`vector_alu_default_stream_native_int64_operation_groups` and report
`vector_alu_pack_a_h2d`, `vector_alu_pack_b_h2d`, `pack`,
`vector_alu_status_memset`, `vector_alu_i64_kernel` or
`vector_alu_u64_kernel`, `rns_gemm`, `vector_alu_status_d2h`,
`vector_alu_output_d2h`, and `crt_export`. The same labels are used by the
benchmark-only baseline and the runtime `RNS8_BACKEND_HIP_VECTOR_ALU_INT64`
path. Tiny status memset/D2H labels can be zero when the HIP SDK does not
surface a measurable default-stream event for the 4-byte status operation; the
kernel, pack, and output-copy labels remain the required attribution points.

Use `tools\gpu_event_report.py <capture.json>` after schema validation to rank
event phases by median and share. Use `tools\gpu_isa_report.py --target gfx1100
--object <hip-object>` or `--build-tree <build-dir>` for LLVM objdump-based ISA
summaries. Reports default to `temp\isa-reports\` and must not be committed.
The ISA report records symbol names, WMMA/MFMA counts, global store counts, LDS
mentions, waits, and VGPR/SGPR/occupancy fields when the disassembler exposes
them. RGA CLI use is optional.

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
repeats, schema v4 validation, concrete HIP `device.gcn_arch` target identity,
and same-contract baseline groups. Review groups with a missing HIP GPU target
are blocked as `missing_gpu_target_id`; groups that mix GPU targets are blocked
as `gpu_target_mismatch`. Review groups with missing or mixed HIP SDK/ROCm
version metadata for HIP-resident captures are blocked as
`missing_hip_toolchain_version` or `hip_toolchain_version_mismatch`.
Configured AMDGPU target metadata plus HIP runtime and driver versions are
also complete-and-compatible promotion gates. Compiler identity and source
checkout identity must be complete and consistent across the same-contract
group before a cache entry can be written. Warmup and repeat counts must be
present, meet the release minimum, and match across the reviewed group. Review
groups with duplicate captures for the same backend are not promotable.

Bounded i64/u64 promotion requires CPU reference, direct-HIP correctness, and
`hip-vector-alu-int64` baselines for the same semantic contract, shape, layout,
target id, HIP SDK and accelerator library versions, seed, warmups, repeats,
prefix schedule, K-block, tile size, epilogue, and selected input
distribution. finite-u8 promotion requires CPU and direct-HIP finite baselines
for the same explicit modulus, and an accelerator cache candidate must beat both
required baselines. Exact-wide signed/unsigned promotion requires CPU and
direct-HIP exact-wide baselines with the same fixed-width limb export contract,
and an accelerator cache candidate must beat both required baselines. Strict
wrap64 promotion requires CPU byte-limb and direct-HIP
`direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` baselines for local
`K <= 4096` shapes, or the corresponding v4 u64-accumulator fallback for larger
K shapes.

Current Windows release sweep status:

- bounded i64/u64 square 64, 128, 512, and 1024 have local release-reviewed
  reports with complete baselines;
- adaptive bounded 65x65x64 and 1024x1024x1024 have local release-reviewed
  reports with complete baselines;
- finite-u8 ring moduli 251, 255, and 256 plus finite-u8 field modulus 251 have
  current local v2 release-reviewed matrices at 64/128/512/1024 plus
  hot-modulus 2048; 11 event-valid entries are installed in the local default
  cache, and accelerator cache promotion now requires beating CPU as well as
  Direct HIP;
- exact-wide signed/unsigned 64, 128, 512, 1024, and 2048 have current local v2
  release-reviewed matrices with complete CPU/direct-HIP baselines; six
  event-valid exact-wide entries are installed: unsigned 64 hipBLASLt, signed
  512 rocWMMA, signed 1024 hipBLASLt, unsigned 1024 CK, and signed/unsigned
  2048 hipBLASLt;
- strict wrap64 has local release-reviewed CPU/direct-HIP baselines for 64, 128,
  512, 1024, and 2048; the matrix-engine accelerator candidate remains open;
- 4096 and 8192 remain exploratory until complete baselines finish within the
  run cap.

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
Created plans expose the current non-benchmark packing contract through
`rns8_get_plan_packing_info`. The report names persistent resident layouts,
backend-specific transient A/B pack layouts, accumulator or library workspace
bytes, selected input/output domains, host/device output currentness,
next-operation flags, and cache availability. hipBLASLt and CK plans report transient
per-dispatch matrix-engine pack workspaces; rocWMMA plans report the same
transient A workspace and, for eligible non-tiled RNS B operands with
`K <= 65536`, a reusable `rns_i8_tile_swizzled_b_v1` B prepack cache path. The
cache is a measured-runtime surface, not a performance claim by itself: it must
still be benchmarked as one-time B setup plus repeated GEMM, and
`production_prepack_cache_available` remains false. CPU, direct-HIP, and wrap64
reference plans report resident layouts without transient pack workspaces.
For autotune exact-hit inspection, `rns8-inspect` also reports an internal
`plan_lowering` object derived from backend, packing, and schedule metadata so
review can distinguish final export, residue-chain continuation, native-chain
continuation, conversion, transient packing, and prepack reuse decisions.
Matrix handles expose the companion source version, finite modulus, host/device
currentness flags, byte counts, and persistent layout version through
`rns8_get_matrix_storage_info`; reusable cache tooling must include that
matrix-side state in cache keys and mismatch rejection.
`rns8_get_prepack_cache_key_info` is the current validator for plan/operand
cache-key material. Its serialized `prepack-v2` key names the backend, target
id, selected kernel, B prepack kernel variant, semantic, prefix-schedule hash,
tile shape, K-block, operand role, source version, finite modulus, device id,
matrix layout, and operand layout, and it rejects incompatible role, shape,
backend, semantic, layout, device id, currentness, source-version, and
finite-modulus inputs before returning a key. `rns8_get_prepack_cache_info`
reports the created runtime cache's matching key/hash material, device id, and
allocation byte contract. No current backend reports a reusable production
prepack cache.

INT4/IU4, AMDGPU builtins, FP8/Ozaki, and wrap64 matrix-engine paths are
retired per semantic/target if they fail to beat the tuned INT8 or current
direct-HIP path after layout, epilogue, and ISA-confirmed matrix-instruction
tuning. No theoretical TOPS claim is accepted without reviewed same-contract
captures.

Future benchmark work must add deeper scheduler internals, reviewed raw sweeps,
comparison baselines, and performance gates before any speedup claims are made.

`tools/result_compare.py` validates both captures before comparing host timing
phases for schema v4 captures. Its same-contract check covers semantic contract,
bound mode, bounds, tile-bound source/order/min/max/hash, shape, requested and
selected prefix metadata, seed, input distribution, epilogue, packed layout,
repeated packed-input mode, and schedule metadata. Backend and selected-kernel
differences are reported as evidence, not as contract failures.
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

## Helper-Lane Evidence Metadata

Current schema-v4 benchmark captures emit additional optimizer-facing metadata
without changing the public C/C++ ABI or AUTO promotion policy:

- `plan_packing` mirrors `rns8_get_plan_packing_info` and names the selected
  input/output domains, resident/transient layout use, prepack-cache
  availability, next-operation flags, and transient workspace byte counts.
- `plan_lowering` is a private benchmark/inspect explanation derived from
  backend, packing, and schedule metadata. It distinguishes final export,
  RNS-continuation, native-continuation, native-to-RNS, transient-pack, and
  prepack-reuse lowering paths.
- `requested_next_op` records the benchmark-only hint
  `final-export|rns-gemm|native-gemm|native-to-rns|reuse-b`; residue-current
  chain captures must resolve to `rns-gemm`.
- `output_policy` records contiguous versus padded destination layout, logical
  leading dimension, zero per-repeat export for residue-current chains, final
  checksum export after measured repeats, and status handling as `required`,
  `structurally_elided`, or `not_applicable`. When HIP events are available,
  schema validation checks the status memset/D2H phase labels against this
  policy.
- `target_variant` normalizes concrete GPU identity into review namespaces:
  `gfx1100`, future `gfx11xx`, future `gfx12xx`, `gfx9xx_gfx94x`, `cpu`, or
  `unknown`. New HIP helper captures must include a concrete target id,
  namespace, and review grouping key.
- `auto_selector` explains AUTO cache load state, runtime identity, selected
  key, validated-hit status, fallback reason, and fixed-vocabulary rejected
  candidates. It is diagnostic only; exact-cache-only promotion is unchanged.
- `device_allocation` snapshots HIP allocation counters before warmup, after
  warmup, and after measured repeats so persistent-plan captures can prove
  whether repeats allocate after warmup.
- `timing_metadata.pack_layout`, `fusion_mode`, `residue_group_width`,
  `residue_group_layout`, and `generated_reducer_identity` become
  same-contract comparison inputs when they change the measured work.

Direct-HIP generated/fixed reducer captures use declared identities such as
`direct_hip_fixed_prefix_1_generated_reducer_v1` through
`direct_hip_fixed_prefix_9_generated_reducer_v1` and
`direct_hip_fixed_prefix_20_generated_reducer_v1`; stale generic reducer names
are rejected for generated captures. The corresponding ISA gate is explanatory:
generated reducers should avoid integer divide instructions and expose the
expected prefix-specific symbols before any kernel is considered for a reviewed
speedup claim.

`tools/gpu_isa_report.py --capture <capture.json>` validates and cross-links a
capture before writing temp-only ISA summaries under `temp/isa-reports/`.
`tools/gpu_counter_report.py` validates captures, optionally ingests JSON/CSV
profiler counter exports and ISA summaries, and writes JSON/Markdown reports
under `temp/gpu-counter-reports/`. Counter and ISA reports explain bottlenecks
and next experiments only; they do not replace exact correctness checks, host
timings, HIP event timings, or release baseline gates.
