# Performance Wins

This document is the durable short list of RNS8 performance improvements that
currently have local evidence. It is intentionally narrower than the research
roadmap and work queue: it records what is winning now, what it beat, and what
still blocks promotion.

Scope:

- Platform: Windows HIP SDK on Radeon RX 7900 XTX / `gfx1100`.
- Semantics: bounded i64/u64 and finite-u8 square GEMM for the latest post-fix
  validation passes, plus same-backend strict wrap64 Direct-HIP implementation
  comparisons.
- Evidence standard: release builds, fixed seeds, three warmups, nine measured
  repeats, schema-valid captures, CPU reference checks, and required GPU event
  timing for GPU captures.
- Boundary: Windows `gfx1100` evidence does not imply Linux ROCm, Linux Radeon,
  Instinct, multi-GPU, or production profiling readiness.
- Durable evidence: [reviewed-local-evidence.md](reviewed-local-evidence.md)
  records the curated platform, command family, seed, shape, backend, result,
  review status, caveat, and reproduction command families. Raw captures stay
  ignored and are not durable docs.

## Current One-Shot Winners

The current bounded-i64 validation pass covered 512 and 1024 after the vector
event-capture fixes, hipBLASLt full A+B event-contract fixes, and CK/rocWMMA
common-modulus reducer identity update. It used seed `20260604`, release builds,
three warmups, nine measured repeats, and required GPU events. The durable
summary lives in [reviewed-local-evidence.md](reviewed-local-evidence.md).
The sweep wrote one reviewed temp cache entry, and
`tools/install_autotune_cache.py --replace-existing` installed that current
1024 hipBLASLt v2 entry into the local default runtime cache after the existing
local cache failed reviewed-cache validation with a stale target-id/key mismatch.

| Shape | Current winner | Winner median end-to-end | Direct HIP median | Vector ALU median | Speedup | Decision |
|---:|---|---:|---:|---:|---:|---|
| 512 | Direct HIP `direct_hip_tiled_active_prefix_rns_gemm_v2` | 1851 us | 1851 us | 6147 us | No accelerator win | Keep direct HIP; no cache entry |
| 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 4174 us | 4535 us | 33945 us | 1.09x vs direct HIP, 8.13x vs vector ALU | Current reviewed v2 cache entry installed locally |

Both groups had no missing required baselines, incompatible metadata, or
duplicate backend records, and both satisfied release-review requirements. The
ten GPU captures from the sweep passed
`tools/gpu_event_report.py --fail-on-unavailable`. At 512, direct HIP beat
rocWMMA v2 at 2591 us, vector ALU at 6147 us, CK v2 at 7172 us, and hipBLASLt v2
at 10101 us. At 1024, hipBLASLt v2 beat direct HIP at 4535 us, rocWMMA v2 at
12996 us, CK v2 at 15546 us, and vector ALU at 33945 us.

This supersedes the earlier June 3, 2026 bounded-i64 one-shot snapshots for
current cache decisions. The seed `20260602` four-shape matrix and seed
`20260603` post-event-fix matrix remain useful historical release-reviewed
evidence, but their old CK/rocWMMA v1 identities must not be mixed into current
v2 autotune cache evidence.

The large-shape validation pass separately covered bounded i64/u64
2048x2048x2048 with CPU, Direct HIP, runtime vector ALU, hipBLASLt, CK, and
rocWMMA comparators. It used release builds, three warmups, nine repeats,
schema-valid captures, and required GPU events for the promoted accelerators.
The two non-reuse winners were installed into the local reviewed cache on
June 5, 2026.

| Contract | Shape | Current winner | Winner median end-to-end | Direct HIP median | Speedup vs Direct HIP | Decision |
|---|---:|---|---:|---:|---:|---|
| bounded i64 | 2048 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 14220 us | 22331 us | 1.57x | Current reviewed v2 cache entry installed locally |
| bounded u64 | 2048 | rocWMMA `rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2` | 15128 us | 18524 us | 1.22x | Current reviewed v2 cache entry installed locally |

The same bounded 2048 pass also produced repeated-B captures. Those are
retained as workload-contract evidence rather than AUTO cache entries because
`prepacked_reuse` intentionally changes the pack/reuse contract.

A later bounded 4096 exploratory pass used seed `20260605`, release binaries,
three warmups, nine repeats, and required GPU events, but intentionally did not
include CPU or runtime vector baselines. It is therefore throughput
classification only, not cache or public promotion evidence. The same-commit
matrix reran 2048 beside 4096 so scaling could be read from the same build and
driver state.

| Contract | Shape | Mode | GPU-only winner | Winner median end-to-end | Direct HIP median | Speedup vs Direct HIP | Best-path scale vs same-commit 2048 |
|---|---:|---|---|---:|---:|---:|---:|
| bounded i64 | 4096 | non-reuse | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 52259 us | 140393 us | 2.69x | 4.11x |
| bounded i64 | 4096 | repeated-B | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 40108 us | 132674 us | 3.31x | 4.22x |
| bounded u64 | 4096 | non-reuse | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 47467 us | 191947 us | 4.04x | 3.76x |
| bounded u64 | 4096 | repeated-B | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 45843 us | 133831 us | 2.92x | 3.43x |

The exploratory 4096 results indicate that the bounded large-shape matrix is no
longer launch-bound in the same way as 512/1024. Pack and export are still large
shares of hipBLASLt time, while Direct HIP is dominated by RNS GEMM at 4096.
Do not use the GPU-only rows above as AUTO cache entries.

A later budgeted 4096 gate under
`temp/perf-work-queue/large-4096-budgeted-release-current-v2/` reran bounded i64
and bounded u64 with CPU, Direct HIP, runtime vector ALU, hipBLASLt, CK, and
rocWMMA. Both groups have required baselines and GPU events. The promotion
ledger closeout under
`temp/perf-work-queue/large-4096-cache-closeout-current/` verified the reviewed
cache entries against the installed local cache with zero blockers.

| Contract | Shape | Budgeted-gate winner | Winner median end-to-end | Direct HIP median | Runtime vector median | CPU reference median | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| bounded i64 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 35303 us | 128995 us | 853232 us | 21244300 us | Current reviewed 4096 cache entry installed locally |
| bounded u64 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 37543 us | 118674 us | 416960 us | 16592600 us | Current reviewed 4096 cache entry installed locally |

A follow-up non-bounded 4096 exploratory pass used the same seed and release
settings for exact-wide, finite-u8, and strict wrap64. It is also GPU-only
classification, not promotion evidence: CPU and runtime vector baselines were
intentionally omitted. The original pass had four non-promoted hipBLASLt finite
captures missing the residue-reduce event label; a later timing-wrapper fix
reran all 17 stale finite hipBLASLt reduce-label misses under
`temp/perf-work-queue/finite-hipblaslt-event-reruns-all/`, and every focused
rerun passed `tools\gpu_event_report.py --require-events`. The event-valid 4096
winners were:

| Contract | Shape | GPU-only event-valid winner | Winner median end-to-end | Direct HIP median | Speedup vs Direct HIP | Best-path scale vs same-commit 2048 |
|---|---:|---|---:|---:|---:|---:|
| exact-wide signed | 4096 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 279231 us | 834962 us | 2.99x | 4.19x |
| exact-wide unsigned | 4096 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 223816 us | 821968 us | 3.67x | 3.77x |
| finite field u8 mod 251 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 7970 us | 46818 us | 5.87x | 1.58x |
| finite ring u8 mod 251 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 9101 us | 82054 us | 9.02x | 1.33x |
| finite ring u8 mod 255 | 4096 | CK `ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2` | 10131 us | 41924 us | 4.14x | 2.01x |
| finite ring u8 mod 256 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 13632 us | 37808 us | 2.77x | 3.26x |
| strict wrap64 u64 | 4096 | Direct HIP `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 352449 us | 352449 us | same backend | 7.35x |

These rows are useful for deciding whether the next large-shape work should
target GEMM throughput or export specialization. They are not reviewed cache
entries and should not appear in public snapshot tables until the missing
CPU/reference and vector baselines are run.

The budgeted 4096 gate then reran the finite hot-modulus rows with CPU and
Direct-HIP baselines. These rows supersede the finite GPU-only scout for local
review evidence, and the promotion ledger closeout installed the eligible
non-reuse 4096 finite rows in the local reviewed cache.

| Contract | Shape | Budgeted-gate winner | Winner median end-to-end | Direct HIP median | CPU reference median | Decision |
|---|---:|---|---:|---:|---:|---|
| finite field u8 mod 251 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 6396 us | 33587 us | 4782790 us | Current reviewed 4096 cache entry installed locally |
| finite ring u8 mod 251 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 7284 us | 32508 us | 4906360 us | Current reviewed 4096 cache entry installed locally |
| finite ring u8 mod 255 | 4096 | CK `ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2` | 8786 us | 33643 us | 4714350 us | Current reviewed 4096 cache entry installed locally |
| finite ring u8 mod 256 | 4096 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 6881 us | 32520 us | 4685440 us | Current reviewed 4096 cache entry installed locally |

The signed exact-wide 4096 row was rerun on the current branch after the first
CPU reference and GPU rows were found to be mixed-commit evidence. The current
same-commit group under
`temp/perf-work-queue/large-4096-signed-cache-closeout-current/` and
`temp/perf-work-queue/large-4096-signed-cpu-current/` has CPU, Direct-HIP,
hipBLASLt, CK, and rocWMMA rows with three warmups, nine measured repeats,
matching checksum `5508849193854467465`, and required GPU events. hipBLASLt is
the reviewed winner at 176943 us median end-to-end versus Direct HIP at
639360 us, CK at 205438 us, rocWMMA at 253023 us, and CPU reference at
113085000 us. That is 3.61x faster than Direct HIP and 639.1x faster than CPU,
and the reviewed 4096 cache entry is installed locally.

A follow-up budgeted exact-wide unsigned 4096 group under
`temp/perf-work-queue/large-4096-unsigned-budgeted-release-current/` completed
the same CPU, Direct-HIP, hipBLASLt, CK, and rocWMMA comparator set with
release settings, schema-valid captures, matching checksum
`9643325300233475427`, and required GPU events for every GPU capture. The
review winner changed from the earlier GPU-only exploratory CK row to
hipBLASLt at 162382 us median end-to-end. Direct HIP measured 614116 us, CK
180252 us, rocWMMA 241248 us, and CPU reference 105462000 us. That makes the
budgeted unsigned hipBLASLt row 3.78x faster than Direct HIP and 649.5x faster
than CPU. The reviewed 4096 cache entry is installed locally.

The strict wrap64 4096 budget row also captured Direct HIP at 295657 us median
end-to-end with required wrap64 GPU events. Its required byte-limb CPU reference
exceeded the initial 300-second per-capture timeout, but the release-reference
rerun completed with three warmups and nine measured repeats. The
`wrap64-byte-limb` median was 102905000 us with checksum
`13518998852724169131`, matching the Direct-HIP row. This makes Direct HIP
348.06x faster than the byte-limb reference at 4096. It remains correctness-path
evidence, not an AUTO cache entry.

## Finite-u8 Accelerator Wins

The current finite-u8 v2 release review covered 64, 128, 512, 1024, the
generic field-127 512/2048 follow-up, generic ring 127/253 2048 follow-up, the
field-251 512 refresh, and the large 2048 hot-modulus validation slice for ring
moduli 251, 255, and 256 plus field modulus 251. The 512/1024 and small-shape
passes used seed `20260604`; the generic refreshes and large 2048 pass used
seed `20260605`. All promoted entries used release builds, three warmups, nine
repeats, CPU and Direct-HIP baselines, schema-valid
captures, and required GPU events for promoted accelerators.
`tools/benchmark_sweep.py` now blocks reviewed cache promotion when an
accelerator capture lacks required GPU event timing or loses to the CPU
reference; the ring-255 64 rocWMMA result was therefore not installed. The
hipBLASLt finite timing fallback cleared the stale reducer-event label blocker,
and the post-fix finite 2048 rerun refreshed the local cache entries for the
hot 2048 contracts.

| Contract | Shape | Current winner | Winner median end-to-end | Direct HIP median | Speedup vs Direct HIP | Decision |
|---|---:|---|---:|---:|---:|---|
| finite ring u8 mod 251 | 128 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2` | 1136 us | 1261 us | 1.11x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 251 | 1024 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2` | 1709 us | 4682 us | 2.74x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 251 | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 3244 us | 11259 us | 3.47x | Current reviewed v2 cache entry refreshed locally |
| finite ring u8 mod 127 | 2048 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_hot_residue_v1` | 3427 us | 5445 us | 1.59x | Current reviewed generic-modulus cache entry installed locally |
| finite ring u8 mod 253 | 2048 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_hot_residue_v1` | 4856 us | 5460 us | 1.12x | Current reviewed generic-modulus cache entry installed locally |
| finite ring u8 mod 255 | 1024 | CK `ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2` | 1938 us | 5814 us | 3.00x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 255 | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 2425 us | 7395 us | 3.05x | Current reviewed v2 cache entry refreshed locally |
| finite ring u8 mod 256 | 128 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2` | 1132 us | 1149 us | 1.02x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 256 | 512 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2` | 1365 us | 5569 us | 4.08x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 256 | 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 1792 us | 12633 us | 7.05x | Current reviewed v2 cache entry installed locally |
| finite ring u8 mod 256 | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 3017 us | 5778 us | 1.92x | Current reviewed v2 cache entry refreshed locally |
| finite field u8 mod 127 | 512 | CK `ck_wmma_cshuffle_finite_u8_centered_epilogue_v1` | 1289 us | 1421 us | 1.10x | Current reviewed generic-modulus cache entry installed locally |
| finite field u8 mod 127 | 2048 | CK `ck_wmma_cshuffle_finite_u8_centered_epilogue_v1` | 3424 us | 5388 us | 1.57x | Current reviewed generic-modulus cache entry installed locally |
| finite field u8 mod 251 | 512 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2` | 1241 us | 1303 us | 1.05x | Current reviewed v2 cache entry installed locally |
| finite field u8 mod 251 | 1024 | CK `ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2` | 1860 us | 10564 us | 5.68x | Current reviewed v2 cache entry installed locally |
| finite field u8 mod 251 | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 3079 us | 9154 us | 2.97x | Current reviewed v2 cache entry refreshed locally |

Non-promoted finite groups are still useful tuning signals. Ring-251 512 stayed
on Direct HIP at 1521 us and ring-255 512 stayed on Direct HIP at 1381 us.
Ring-255 64 had a rocWMMA accelerator result at 1257 us versus Direct HIP at
3388 us, but CPU reference was 167 us; the CPU gate correctly kept it out of
the reviewed runtime cache. The earlier same-day finite 2048 run is superseded
for hot 2048 cache decisions: the post-fix rerun makes hipBLASLt the reviewed
winner for ring 251, ring 255, ring 256, and field 251 with required GPU events
and CPU-backed release review.

The field-127 generic-modulus refresh also reran hipBLASLt with complete
`hipblaslt_pack_transpose_centered`, `hipblaslt_int8_i32_matmul`, and
`hipblaslt_i32_to_residue_reduce` event timing. It was not promoted because it
lost to Direct HIP and CK at 512, not because event data was missing.

The generic ring 127/253 2048 refresh closed the previous GPU-only evidence gap
with CPU-backed release review. rocWMMA won both reviewed contracts and was
installed locally. The stale hipBLASLt ring-127 2048 capture was
event-incomplete and lost to Direct HIP; the focused timing-fallback rerun is
event-complete but remains diagnostic until the full same-contract release group
is rerun. Ring-253 hipBLASLt had required events but also lost to Direct HIP and
rocWMMA.

The field refreshes added CK for field-127 2048 and rocWMMA for field-251 512.
The field-251 512 hipBLASLt capture now has required GPU events, including
pack/transpose, matmul, and i32-to-residue reduction, but it is slower than
Direct HIP and rocWMMA in the current release review.

## Exact-Wide Accelerator Wins

The current exact-wide v2 release review covered signed and unsigned 64, 128,
512, 1024, and the large 2048 validation slice with CPU reference, Direct HIP,
hipBLASLt, CK, and rocWMMA. The 512/1024 pass used seed `20260604`; the 64/128
refresh and 2048 validation pass used seed `20260605`. All promoted entries used
release builds, three warmups, nine repeats, CPU and Direct-HIP baselines,
schema-valid captures, and required GPU events for promoted accelerators. The
reviewed cache entries were merged into the local default runtime cache without
replacing existing bounded or finite-u8 entries.

| Contract | Shape | Current winner | Winner median end-to-end | Direct HIP median | Speedup vs Direct HIP | Decision |
|---|---:|---|---:|---:|---:|---|
| exact-wide unsigned | 64 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 4611 us | 7714 us | 1.67x | Current reviewed v2 cache entry installed locally |
| exact-wide signed | 512 | rocWMMA `rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2` | 7162 us | 7297 us | 1.02x | Current reviewed v2 cache entry installed locally |
| exact-wide signed | 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 17092 us | 22543 us | 1.32x | Current reviewed v2 cache entry installed locally |
| exact-wide unsigned | 1024 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 20481 us | 25029 us | 1.22x | Current reviewed v2 cache entry installed locally |
| exact-wide signed | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 59074 us | 131794 us | 2.23x | Current reviewed v2 cache entry installed locally |
| exact-wide unsigned | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 40985 us | 124570 us | 3.04x | Current reviewed v2 cache entry installed locally |

The June 6, 2026 `gfx1100` closeout reran the benchmark-only exact-wide fixed
limb export selector under
`temp/gfx1100-pending-validation-20260606/`. These rows are local
export-selector evidence only: they are release-reviewed, schema-valid, and
event-valid, but the selector variant is still marked experimental and is not a
README headline claim, installed cache entry, default route, or Linux/CDNA
claim.

| Contract | Shape | Export selector winner | Winner median end-to-end | Direct HIP median | CPU reference median | Speedup vs Direct HIP | Status |
|---|---:|---|---:|---:|---:|---:|---|
| exact-wide signed, 4 limbs | 1024 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 14659 us | 17634 us | 3261790 us | 1.20x | Local export-selector candidate only |
| exact-wide unsigned, 4 limbs | 1024 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 15337 us | 16137 us | 2512390 us | 1.05x | Local export-selector candidate only |
| exact-wide signed, 4 limbs | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 49161 us | 103762 us | 19073800 us | 2.11x | Local export-selector candidate only |
| exact-wide unsigned, 4 limbs | 2048 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 44964 us | 101018 us | 15714000 us | 2.25x | Local export-selector candidate only |

Exact-wide signed 64, signed 128, unsigned 128, and unsigned 512 remain on
Direct HIP in the current v2 matrix. The signed 512 win is narrow and should be
watched in future reruns, but it is release-reviewed, event-valid, and beats the
same-contract Direct-HIP baseline. At 2048, hipBLASLt removes most GEMM time and
the promoted captures are export-bound, making fixed-width export and lazy
residue-current workflows the next exact-wide tuning target.

## Direct-HIP Implementation Wins

These rows compare one Direct-HIP implementation against the previous
Direct-HIP implementation for the same public API and shape. They are not
cross-backend AUTO winners.

| Surface | Shape | New selected kernel | Average end-to-end speedup | Median end-to-end speedup | Event phase speedup | Status |
|---|---:|---|---:|---:|---:|---|
| Public bounded-i64 one-shot | 512 | `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2` | 2.72x | 3.07x | 1.04x median | Routed only for Direct-HIP bounded-i64 `m/n/k >= 512`; persistent resident Direct-HIP remains faster for non-one-shot workflows |
| Public bounded-u64 one-shot | 512 | `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2` | 1.09x | 1.21x | 1.06x | Routed only for Direct-HIP bounded-u64 `m/n/k >= 512`; smaller u64 remains on v1 |
| Public exact-wide signed 4-limb export | 1024 | `hip_direct_export_exact_wide_signed_fixed_prefix18_fixed_limbs_device` | 1.12x | 1.13x | 1.64x export kernel | Local Windows `gfx1100` same-backend A/B; schema/event-valid, not README/cache/CDNA claim material |
| Public exact-wide unsigned 4-limb export | 1024 | `hip_direct_export_exact_wide_unsigned_fixed_prefix18_fixed_limbs_device` | 1.22x | 1.24x | 6.42x export kernel | Local Windows `gfx1100` same-backend A/B; schema/event-valid, not README/cache/CDNA claim material |
| Public exact-wide signed 4-limb export | 2048 | `hip_direct_export_exact_wide_signed_fixed_prefix18_fixed_limbs_device` | 1.09x | 1.09x | 1.61x export kernel | Local Windows `gfx1100` same-backend A/B; schema/event-valid, not README/cache/CDNA claim material |
| Public exact-wide unsigned 4-limb export | 2048 | `hip_direct_export_exact_wide_unsigned_fixed_prefix18_fixed_limbs_device` | 1.21x | 1.22x | 5.92x export kernel | Local Windows `gfx1100` same-backend A/B; schema/event-valid, not README/cache/CDNA claim material |
| Public large exact-wide signed 4-limb tree/CRT export route | 1024 | `hip_direct_export_exact_wide_signed_tree_crt_limbs_device` | 1.07x | 1.05x | 2.04x export phase | Routed only for Direct-HIP signed prefix18, four requested limbs, and large outputs; local Windows `gfx1100`, not README/cache/CDNA claim material |
| Public large exact-wide signed 4-limb tree/CRT export route | 2048 | `hip_direct_export_exact_wide_signed_tree_crt_limbs_device` | 1.07x | 1.06x | 2.16x export phase | Routed only for Direct-HIP signed prefix18, four requested limbs, and large outputs; local Windows `gfx1100`, not README/cache/CDNA claim material |

The colpair one-shot kernel is now routed for bounded i64 and bounded u64 when
`m/n/k >= 512`. Smaller bounded one-shot shapes keep the prior v1 native-input
grouped kernel because 64/128 evidence was noisy or not favorable on Windows
`gfx1100`. These are public one-shot implementation wins only; they are not
evidence that one-shot beats resident matrix reuse for repeated calls.

The exact-wide export rows compare the new Direct-HIP fixed-prefix18 fixed-limb
export launch path against the immediately previous Direct-HIP export path with
the same public shape, selected prefix, limb count, signedness, and target. They
also include the removal of a redundant host-side device synchronization before
the already-synchronizing export copy/status phase.

The tree/CRT route is deliberately narrower than the benchmark selector
surface: it is used only for signed prefix18, four requested limbs, and large
outputs. The same June 6 follow-up showed that unsigned tree/CRT improved some
GPU export events but did not clear the end-to-end route gate, that 128/512
signed shapes should stay on fixed-prefix Garner export, and that the prefix20
fixed-export selector was too marginal to call a durable win because its
export-kernel event did not improve. Those dispositions are recorded in
[reviewed-local-evidence.md](reviewed-local-evidence.md) and stay out of cache,
README, and Linux/CDNA claims.

## Planner And Prepass Wins

These rows reduce benchmark/planner setup cost for adaptive bounded captures.
They do not change math semantics, selected GPU kernels, or AUTO backend
selection.

| Surface | Shape | Change | Before | After | Speedup | Status |
|---|---:|---|---:|---:|---:|---|
| Exact per-tile bound discovery, bounded-u64 adaptive bands | 512 | Nonzero A-row/B-column summaries skip exact scans for proven-zero tile, row, and column products | 557635 us `tile_bound_scan` | 414379 us `tile_bound_scan` | 1.35x | Schema-valid/event-valid; tile-bound hash, selected prefix, prefix groups, zero-output tile count, and selected kernel unchanged |

The same scanner change also passed a bounded-i64 512 adaptive-band release
capture with schema v4 and required Direct-HIP GPU events. Raw captures live
under `temp/perf-work-queue/tile-bound-zero-shortcut/`. This is a setup-path
gain; measured per-repeat GPU phases still need separate backend/kernel
optimization.

A later setup-inclusive release gate deliberately did not promote
bound-discovery routing. The June 5, 2026 `bound-discovery` scenario matrix used
seed `20260605`, three warmups, nine repeats, and 51 schema-valid captures over
bounded-i64 256/1024 adaptive-band workloads and a bounded-u64 512x1024
adaptive-band workload. The generic release review had nine groups with no
missing required baselines, duplicate backends, git/target mismatches, or
missing GPU events for non-CPU captures. `tools/bound_discovery_report.py`
compared 18 global input-scan captures and 15 proof-mask per-tile captures
against same-backend and fastest static global-bound baselines with scan setup
cost included. All 33 candidates were deprioritized. hipBLASLt global
input-scan improved over its own 256 bounded-i64 static baseline, and CK global
input-scan improved over its own rectangular bounded-u64 static baseline, but
both lost to the fastest static workload baseline after setup cost. Per-tile
proof masks were event-visible and correct, but exact tile-bound scans dominated
setup-inclusive timing.

## Shape-Specialized Runtime Wins

These rows compare a new shape-gated runtime kernel against the previous kernel
inside the same backend and semantic contract.

| Backend | Shape | Semantics | New selected kernel | Average end-to-end speedup | Median end-to-end speedup | Median kernel speedup | Status |
|---|---:|---|---|---:|---:|---:|---|
| Vector ALU | 1x1x65536 | bounded u64 | `hip_vector_alu_u64_gemv_n1_exact_192b_v1` | 2.22x | 3.39x | 20.30x | Measured at 1x1x65536; active route now covers long-K N=1 captures: `n == 1`, `k >= 4096` |
| Vector ALU | 1x1x65536 | bounded i64 | `hip_vector_alu_i64_gemv_n1_exact_192b_v1` | 4.44x | 7.41x | 35.87x | Measured at 1x1x65536; active route now covers long-K N=1 captures: `n == 1`, `k >= 4096` |

The vector long-K dot captures used release binaries, three warmups, nine
measured repeats, seed `20260604`, and required GPU events. The pre-change
baseline was built from a temporary detached `96781eb` worktree; retained raw
before/after captures live under `temp/perf-work-queue/vector-gemv-n1/`.
Current schema intentionally rejects those old `n == 1` captures if they claim
the stale generic vector kernel. A broader 1024x1x1024 smoke stayed on
`hip_vector_alu_*_exact_192b_v1` after gating because that shape was
pack-dominated and did not produce a setup-inclusive GEMV win.

The current `skinny-gemv` release review used seed `20260605`, three warmups,
nine repeats, CPU and Direct-HIP baselines, all optional accelerator backends,
schema-valid captures, and required GPU events. Direct HIP won every reviewed
N=1 scenario shape: bounded-i64 512x1x512 at 1689 us, bounded-i64
256x1x4096 at 2712 us, and bounded-u64 1024x1x1024 at 2307 us. The vector
GEMV kernels remain useful explicit-backend microkernel evidence, but they do
not currently justify AUTO/cache promotion for these setup-inclusive scenario
contracts.

The current `many-small` release review used seed `20260605`, three warmups,
nine repeats, and 61 same-commit schema-valid captures: 41 independent-call
baselines and 20 host API batch captures. The mixed review has no missing
required baselines, no duplicate backend records, compatible git/target
metadata, and no cache entries promoted. Independent-call winners are CPU for
bounded-i64 32, bounded-u64 64, and finite-u8 64; runtime vector ALU for
bounded-i64 128; and Direct HIP for bounded-u64 128x1x1024 and exact-wide
signed 64. `tools/host_api_batch_report.py` adds the setup-inclusive per-task
batch comparison against same-backend and fastest independent-call baselines:
only Direct-HIP exact-wide signed 64 hostbatch32 wins, at 1903 us per task
versus the 3880 us independent Direct-HIP baseline, or 2.04x faster. The other
19 host-batch candidates are deprioritized. This is benchmark-only workload
evidence, not an AUTO cache entry or public batching API promotion. The small
Direct-HIP one-shot resident-fallback diagnostic and hipBLASLt finite ring-251
diagnostic remain focused event-cleanup evidence only.

A follow-up grouped-dispatch capture on the same exact-wide signed 64
contract now beats both the independent Direct-HIP baseline and the earlier
hostbatch32 candidate. The first branch-local `rns8-bench --grouped-dispatch
32` capture under `temp/perf-work-queue/many-small-grouped-dispatch-current/`
reported 991.94 us per task. The current async exact-wide export-slab capture
under `temp/perf-work-queue/many-small-grouped-dispatch-slab-current/` uses
three warmups, nine repeats, seed `20260605`, schema v4, required Direct-HIP
GPU events, and checksum parity with the earlier grouped capture. It reports
792.66 us per task, or 1.25x faster than the earlier grouped-dispatch path,
4.89x faster than the 3880 us independent Direct-HIP baseline, and 2.40x
faster than the 1902.97 us Direct-HIP hostbatch32 row. The export-slab path
queues all grouped exact-wide export kernels without per-task synchronization,
copies one compact device output slab back to host, and materializes per-task
checksum buffers outside the measured repeat for contiguous output. The later
one-kernel grouped export follow-up under
`temp/perf-work-queue/many-small-grouped-dispatch-kernel-current/` keeps the
same checksum and required events, changes the strategy metadata to
`device_grouped_exact_wide_export_kernel_batched_d2h`, and cuts the signed
aggregate export phase from 5670 us average in the async slab capture to
1212 us average. Its signed end-to-end median is 795.19 us per task, effectively
flat versus the 792.66 us slab median but still 4.88x faster than independent
Direct HIP and 2.39x faster than hostbatch32. The current grouped pack+export
follow-up under `temp/perf-work-queue/many-small-grouped-pack-current/` keeps
the same exact-wide signed 64 group32 contract and changes the strategy metadata
to `device_grouped_pack_and_exact_wide_export_kernels_batched_d2h`. It copies
compact A and B slabs once per measured repeat, launches one grouped pack kernel
per operand, keeps GEMM in the existing per-task resident loop, and uses the
one-kernel grouped export path. The signed median falls to 228.06 us per task,
or 17.01x faster than independent Direct HIP, 8.34x faster than hostbatch32,
and 3.49x faster than the previous one-kernel grouped-export capture; host
aggregate pack average falls from 16873 us to 778 us, and GPU event pack average
falls from 12815 us to 461 us. The unsigned exact-wide 64 group32 twin is
schema/event-valid at 249.63 us per task, but the current many-small corpus
lacked the matching independent unsigned baseline at the time, so that capture
stayed smoke-only. The grouped pack+GEMM+export follow-up under
`temp/perf-work-queue/many-small-grouped-gemm-current/` changes the strategy
metadata to
`device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h`. It keeps
the grouped pack and one-kernel export path, and replaces the host-side GEMM
task loop with one same-shape task-prefix Direct-HIP grouped GEMM kernel group.
The signed median falls again to 66.47 us per task, or 58.37x faster than
independent Direct HIP, 28.63x faster than hostbatch32, and 3.43x faster than
the grouped pack+export capture. Event median `rns_gemm` drops from
4701.94 us to 168.70 us; export is now the largest remaining event-visible
phase in the signed capture. The unsigned exact-wide 64 group32 twin is
schema/event-valid at 63.56 us per task but was still smoke-only in that
capture set.

The focused unsigned closeout under
`temp/perf-work-queue/many-small-grouped-unsigned-current/` adds the missing
exact-wide unsigned 64 CPU and Direct-HIP independent baselines plus matching
CPU and Direct-HIP hostbatch32 baselines. The stricter
`tools/many_small_grouped_report.py` gate now requires same-task-count
same-backend hostbatch checksum parity before a grouped row can become a
candidate. With that gate, the Direct-HIP exact-wide unsigned 64 group32
pack+GEMM+export capture is schema/event-valid and classified as a candidate
at 79.09 us per task, versus 1479 us for independent Direct HIP and 1066 us per
task for Direct-HIP hostbatch32. That is 18.70x faster than the same-backend
independent baseline and 13.48x faster than the same-backend hostbatch32
baseline, with matching combined checksum `666386315613239916`. This remains
benchmark-owned persistent-task evidence, not a public grouped API, generic
descriptor queue, AUTO cache entry, or Linux/Instinct claim.

The bounded grouped export closeout under
`temp/perf-queue-grouped-bounded-export-release/` removes the per-task bounded
CRT export loop from the same bounded-i64/u64 64 group32 benchmark contract.
The strategy is now
`device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h`: grouped
pack kernels build resident RNS inputs for the task batch, the same-shape
Direct-HIP grouped GEMM covers all tasks, grouped bounded CRT export kernels
write one compact device output slab, and a single compact D2H copy returns the
native output. The release controls use seed `20260606`, three warmups, nine
repeats, CPU and Direct-HIP independent baselines, Direct-HIP hostbatch32
baselines, schema v4, required Direct-HIP GPU events, and same-task-count
checksum parity. Bounded-i64 64 group32 is now 53.625 us per task, 19.97x
faster than the 1071 us best independent CPU baseline and 23.32x faster than
Direct-HIP hostbatch32 at 1250.59 us per task. Bounded-u64 64 group32 is now
53.25 us per task, 10.95x faster than the 583 us best independent CPU baseline
and 22.95x faster than Direct-HIP hostbatch32 at 1222.19 us per task. This is
still benchmark-owned same-shape grouped evidence, not a public grouped API,
generic descriptor queue, AUTO cache entry, or Linux/Instinct claim.

The broader grouped matrix closeout under
`temp/perf-queue-grouped-broader-release/` reruns the current grouped-dispatch
family with scenario-owned CPU/Direct-HIP independent controls and matching
same-task-count Direct-HIP hostbatch controls. It uses seed `20260606`, three
warmups, nine repeats, schema v4, required Direct-HIP GPU events, and
`tools/many_small_grouped_report.py` checksum parity. All five grouped rows are
candidate wins with no missing baselines: bounded-i64 64 group32 is
57.41 us per task, 20.02x faster than the best independent baseline and
19.98x faster than hostbatch32; bounded-i64 128 group64 is 55.88 us per task,
26.59x faster than best independent and 23.09x faster than hostbatch64;
bounded-u64 64 group32 is 59.38 us per task, 16.62x faster than best
independent and 15.87x faster than hostbatch32; bounded-u64 128x1x1024
group128 is 100.76 us per task, 16.14x faster than best independent and
14.62x faster than hostbatch128; finite-ring u8 mod251 64 group32 is
33.53 us per task, 2.33x faster than the best independent CPU baseline and
26.41x faster than Direct-HIP hostbatch32. The finite row is the first
release-reviewed grouped finite pack+GEMM+export path: it uses grouped finite
pack, grouped finite GEMM, grouped finite canonical export, and one compact
output D2H. These rows supersede narrower grouped smokes for the same contracts
but remain benchmark-owned same-shape workload evidence, not public grouped
ABI, AUTO routing, cache entries, or Linux/Instinct proof.

The exact-wide 128 grouped-control closeout under
`temp/perf-queue-grouped-exact128-release/` keeps the same report gate but adds
signed and unsigned 128-square independent, hostbatch32, and grouped-dispatch
controls. The 20-capture release set has schema v4 throughout, required GPU
events for all 12 GPU captures, and four candidate wins with no missing
baselines. Exact-wide signed 64 group32 is 68.12 us per task, 18.47x faster
than independent Direct HIP and 16.78x faster than hostbatch32; exact-wide
signed 128 group32 is 184.97 us per task, 7.93x faster than independent
Direct HIP and 7.59x faster than hostbatch32; exact-wide unsigned 64 group32
is 75.53 us per task, 16.16x faster than independent Direct HIP and 13.96x
faster than hostbatch32; exact-wide unsigned 128 group32 is 155.88 us per
task, 11.66x faster than independent Direct HIP and 8.94x faster than
hostbatch32. The 128 rows are still benchmark-owned same-shape grouped
evidence; they do not create public grouped descriptors, cache entries, AUTO
routing, or Linux/Instinct claims.

The strict wrap64 Direct-HIP v4 kernel supersedes the previous v3 scalar path
for local `K <= 4096` shapes. It uses direct unsigned byte products, uint32
low-diagonal accumulation where safe, uint64 carry propagation, vectorized
compact byte-limb load/store where shape evidence permits, and scalar
pack/export fallbacks for 64-like shapes. These rows compare v4 against v3 with
release binaries, three warmups, nine measured repeats, seed `20260604`,
schema-valid/event-valid v4 captures, and checksum-matched before/after output.

| Surface | Shape | New selected kernel | Median end-to-end speedup | Event GEMM median speedup | Status |
|---|---:|---|---:|---:|---|
| Strict wrap64 Direct-HIP default pack/GEMM/export | 64 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.08x | 1.38x | Local implementation win; CPU byte-limb still faster at 64 |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 128 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.17x | 2.17x | Local implementation win |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 512 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.02x | 2.04x | Positive but pack/export-noisy |
| Strict wrap64 Direct-HIP default pack/GEMM/export | 1024 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 5.60x | 8.94x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 64 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.22x | 1.45x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 128 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 4.67x | 3.16x | Local implementation win |
| Strict wrap64 Direct-HIP reuse-packed inputs | 512 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 1.07x | 1.99x | Positive but export-noisy |
| Strict wrap64 Direct-HIP reuse-packed inputs | 1024 | `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` | 6.74x | 7.83x | Local implementation win |

The large-shape release-validation follow-up on June 5, 2026 covered strict
wrap64 2048x2048x2048 with the same-contract byte-limb CPU reference and Direct
HIP v4. The review had no missing required baselines, duplicate backends, target
or toolchain mismatches, or commit mismatches. Direct HIP measured 58331 us
median end-to-end versus 13423400 us for the CPU byte-limb reference, a 230.1x
same-contract speedup, with required wrap64 GPU events. This is durable evidence
for the current Direct-HIP strict wrap64 path, not an AUTO cache entry.

## RNS Chain Evidence

Final-output RNS-chain captures now have a dedicated same-contract report.
`tools/rns_chain_report.py` validates `residue_chain_final_host_export` and
`residue_chain_independent_final_host_export` captures, requires CPU baselines
and GPU events, and folds reusable-input setup cost into per-repeat medians
before classifying rows. The focused Windows `gfx1100` set under
`temp/perf-work-queue/rns-chain-final-output-current/` covers bounded-i64 128
and exact-wide signed 128 with CPU, Direct HIP, and Direct-HIP reusable-B rows.
Direct HIP wins versus CPU for the final-output chain contract: bounded-i64
128 is 1614 us versus 15745 us, and exact-wide signed 128 is 1358 us versus
41656 us. Reusable-B does not survive setup cost in this contract: bounded-i64
reusable-B is 3002 us setup-inclusive versus 1614 us non-reuse, and exact-wide
signed reusable-B is 3161 us versus 1358 us non-reuse.

Bounded independent export/repack controls are now measured separately under
`temp/perf-work-queue/rns-chain-independent-final-output-current/`. The
schema/event-valid report classifies two Direct-HIP resident final-output chain
candidate wins against same-backend independent controls: bounded-i64 128
chain3 is 1805 us versus 3324 us, 1.84x faster, and bounded-u64 256 chain3 is
2240 us versus 4387 us, 1.96x faster. This is useful bounded chain evidence,
not an AUTO/cache entry.

Exact-wide independent export/repack controls now use an explicit benchmark
limb-import/repack path and are measured under
`temp/perf-work-queue/exact-wide-rns-chain-independent-final-output-current/`.
The schema/event-valid report classifies two Direct-HIP resident final-output
chain wins against same-backend independent controls: exact-wide signed 128
chain3 is 1770 us versus 17344 us, 9.80x faster, and exact-wide unsigned 256
chain3 is 2842 us versus 30687 us, 10.80x faster. This closes the focused
exact-wide same-output control gap, but it is still benchmark evidence only:
no AUTO/cache entry, no public lazy-output API, and no Linux/Instinct claim.

The broader rank-9 release matrix under
`temp/rank9-rns-chain-broader-release/` extends the same final-output contract
to 512 and 1024 for bounded-i64, bounded-u64, exact-wide signed, and
exact-wide unsigned. Each group has CPU final-output baseline, Direct-HIP
resident-chain capture, and same-backend Direct-HIP independent export/repack
control. `tools/rns_chain_report.py` reports eight candidate wins, zero missing
baselines, zero experimental rows, and required GPU events for every Direct-HIP
row:

| Contract | Shape | Resident Chain Median | Independent Export/Repack Median | Speedup Vs Independent |
|---|---:|---:|---:|---:|
| bounded-i64 chain3 final output | 512 | 3696 us | 5630 us | 1.52x |
| bounded-i64 chain3 final output | 1024 | 14610 us | 19097 us | 1.31x |
| bounded-u64 chain3 final output | 512 | 3497 us | 6022 us | 1.72x |
| bounded-u64 chain3 final output | 1024 | 14090 us | 18435 us | 1.31x |
| exact-wide signed chain3 final output | 512 | 6475 us | 228670 us | 35.32x |
| exact-wide signed chain3 final output | 1024 | 32339 us | 1116390 us | 34.52x |
| exact-wide unsigned chain3 final output | 512 | 5817 us | 113142 us | 19.45x |
| exact-wide unsigned chain3 final output | 1024 | 30178 us | 472985 us | 15.67x |

These rows close the current benchmark-owned RNS-chain final-output matrix.
They do not install cache entries, change AUTO routing, expose a public lazy
output API, or imply Linux/CDNA behavior.

Exact-wide residue-current chain pairs now have a dedicated report in
`tools/exact_wide_chain_report.py`. The rank-46 release sweep under
`temp/rank46-exact-wide-output-chain-release/`, paired with the rank-9
final-output chain3 captures, reports eight ready residue-current/final-output
pairs with no missing baselines. The ratio below is final-output median divided
by residue-current median for the same Direct-HIP exact-wide chain contract:

| Contract | Shape | Chain | Residue-Current Median | Final-Output Median | Final/Residue Ratio |
|---|---:|---:|---:|---:|---:|
| exact-wide signed | 512 | 3 | 4513 us | 6475 us | 1.43x |
| exact-wide signed | 512 | 4 | 5799 us | 7727 us | 1.33x |
| exact-wide signed | 1024 | 3 | 26612 us | 32339 us | 1.22x |
| exact-wide signed | 1024 | 4 | 34372 us | 40388 us | 1.18x |
| exact-wide unsigned | 512 | 3 | 4473 us | 5817 us | 1.30x |
| exact-wide unsigned | 512 | 4 | 5852 us | 7218 us | 1.23x |
| exact-wide unsigned | 1024 | 3 | 25974 us | 30178 us | 1.16x |
| exact-wide unsigned | 1024 | 4 | 33509 us | 37377 us | 1.12x |

These rows prove that the benchmark-owned residue-current chain contract is
event-complete and paired with exact final-output checks. They remain API
design evidence until public resident-output handles and lifetime rejection
rules exist.

## Reuse And Prepack Wins

The latest reuse validation compares each reuse path against the same backend
with normal per-repeat packing. These are implementation wins, not default AUTO
promotion claims, because the comparison intentionally changes `pack_mode` and
reuse metadata. The headline speedup includes one-time setup over the measured
nine-repeat validation capture:

```text
setup_inclusive_speedup =
    (9 * non_reuse_avg_end_to_end_us) /
    (prepack_setup_us + 9 * reuse_avg_end_to_end_us)
```

Steady-state speedup is the per-repeat limit after setup has already been paid,
computed as `1 / end_to_end_ratio` from `tools\result_compare.py`.

Current reuse promotion work should use `tools\reuse_contract_report.py`, not
one-off spreadsheet math. The report normalizes reuse captures against their
non-reuse workload contract, adds setup cost back into the per-repeat median,
computes break-even repeats, requires GPU events for GPU captures, and records
whether source-version/setup-scope metadata is strong enough for a workload
claim. The latest large-shape reports and bounded reuse-contract matrix produced
these explicit workload candidate rows:

| Capture family | Candidate | Setup-inclusive per-repeat | Same-backend speedup | Fastest non-reuse baseline | Workload speedup | Break-even repeats | Decision |
|---|---|---:|---:|---|---:|---:|---|
| CPU-backed 2048 large-release | hipBLASLt bounded-i64 repeated-B | 11048 us | 1.32x | CK | 1.29x | 3 | explicit workload candidate |
| CPU-backed 2048 large-release | CK bounded-u64 repeated-B | 12930 us | 1.82x | rocWMMA | 1.17x | 2 | explicit workload candidate |
| CPU-backed 2048 large-release | hipBLASLt bounded-u64 repeated-B | 10683 us | 2.28x | rocWMMA | 1.42x | 2 | explicit workload candidate |
| CPU-backed 2048 large-release | rocWMMA bounded-u64 repeated-B | 13106 us | 1.15x | rocWMMA | 1.15x | 5 | explicit workload candidate |
| bounded 4096 exploratory | CK bounded-i64 2048 repeated-B | 12357 us | 4.27x | hipBLASLt | 1.03x | 1 | exploratory workload candidate |
| bounded 4096 exploratory | hipBLASLt bounded-i64 2048 repeated-B | 11096 us | 1.15x | hipBLASLt | 1.15x | 5 | exploratory workload candidate |
| bounded 4096 exploratory | hipBLASLt bounded-i64 4096 repeated-B | 43842 us | 1.19x | hipBLASLt | 1.19x | 3 | exploratory workload candidate |
| reuse-contract release matrix | hipBLASLt bounded-i64 2048 repeated-A | 13404 us | 1.82x | CK 14223 us | 1.06x | 6 | explicit workload candidate |
| reuse-contract release matrix | hipBLASLt bounded-i64 2048 repeated-A+B | 13651 us | 1.78x | CK 14223 us | 1.04x | 8 | explicit workload candidate |
| reuse-contract release matrix | hipBLASLt bounded-u64 2048 repeated-A | 11532 us | 3.18x | same-run rocWMMA 21609 us | 1.87x | 2 | explicit workload candidate |
| reuse-contract release matrix | hipBLASLt bounded-u64 2048 repeated-A+B | 9198 us | 3.99x | same-run rocWMMA 21609 us | 2.35x | 2 | strongest explicit hipBLASLt reuse candidate |
| reuse-contract release matrix | hipBLASLt bounded-u64 2048 repeated-B | 17034 us | 2.15x | same-run rocWMMA 21609 us | 1.27x | 3 | explicit candidate with conservative caveat |
| reuse-contract release matrix | hipBLASLt bounded-u64 1024 repeated-B | 8171 us | 1.21x | Direct HIP 8477 us | 1.04x | 8 | narrow explicit workload candidate |

The same reports also deprioritize most reuse candidates after setup or against
the fastest non-reuse backend. Direct HIP bounded repeated-B does not currently
clear the large-shape fastest-baseline gate, runtime vector reuse-B is
downgraded where source-identity metadata is incomplete, and hipBLASLt 1024
stable-A/full-reuse loses the workload gate. The bounded-u64 2048 hipBLASLt
repeated-B row wins against the same-run reuse-contract baseline, but an older
installed rocWMMA 2048 non-reuse cache row measured 15128 us, so it should stay
explicit-contract evidence until a same-seed rerun confirms the faster baseline
does not erase the win. None of these rows are AUTO cache entries.

The older mechanism table below remains useful for event-contract validation,
especially the corrected hipBLASLt A+B event shape. The 2026-06-05
reuse-contract matrix supersedes it for 1024 workload-promotion decisions.

| Backend | Shape | Reuse mode | Setup-inclusive speedup over 9 repeats | Steady-state per-repeat speedup | Saved per repeat | Setup | Break-even repeats | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| hipBLASLt | 512 | A | 1.47x | 1.55x | 8029.9 us | 7104 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | B | 4.72x | 5.05x | 18116.7 us | 2811 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 512 | A+B | 5.84x | 7.68x | 19649.0 us | 8323 us | 1 | Event-valid experimental reuse win |
| hipBLASLt | 1024 | A | 3.36x | 3.57x | 20712.4 us | 4345 us | 1 | Historical same-backend mechanism win; current reuse-contract matrix deprioritizes workload promotion |
| hipBLASLt | 1024 | B | 1.74x | 1.80x | 12810.9 us | 4744 us | 1 | Historical same-backend mechanism win; current matrix has only a narrow workload candidate |
| hipBLASLt | 1024 | A+B | 4.32x | 4.81x | 22794.2 us | 5992 us | 1 | Historical same-backend mechanism win; current reuse-contract matrix deprioritizes workload promotion |
| Direct HIP | 1024 | B, uniform-small i8 A/B colpair v2, bounded i64 | 1.19x | 1.20x | 1379.7 us | 768 us | 1 | Event-valid explicit reuse-path win |
| Vector ALU | 512 | A | 1.54x | 1.56x | 3460.6 us | 883 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | B | 1.53x | 1.55x | 3412.6 us | 943 us | 1 | Event-valid local reuse win |
| Vector ALU | 512 | A+B | 1.01x | 1.10x | 838.8 us | 6357 us | 8 | Barely positive at 9 repeats |
| Vector ALU | 1024 | B | 1.21x | 1.23x | 4683.0 us | 3187 us | 1 | Event-valid local reuse win |
| Direct HIP | 512 | B, uniform-small i8 A/B colpair v2, bounded u64 | 1.34x | 1.39x | 981.0 us | 813 us | 1 | Event-valid explicit reuse-path win |
| Direct HIP | 1024 | B, uniform-small i8 A/B colpair v2, bounded u64 | 1.17x | 1.18x | 1282.9 us | 695 us | 1 | Conservative rerun result; export timing volatile |
| rocWMMA | 512 | B | 1.14x | 1.22x | 1210.3 us | 3428 us | 3 | Experimental; same-strategy review baselines still incomplete |

Non-winners from the same validation pass:

| Backend | Shape | Reuse mode | Setup-inclusive result over 9 repeats | Steady-state result |
|---|---:|---|---:|---:|
| Vector ALU | 1024 | A | 0.94x | 0.96x |
| Vector ALU | 1024 | A+B | 0.69x | 0.71x |
| Direct HIP | 512 | B, uniform-small i8 A/B colpair v2, bounded i64 | 1.00x | 1.02x |
| hipBLASLt | 1024 | A | 0.28x vs fastest non-reuse, 0.45x same-backend | 0.45x same-backend |
| hipBLASLt | 1024 | A+B | 0.87x vs fastest non-reuse, 1.40x same-backend | same-backend win does not clear fastest-baseline gate |
| hipBLASLt | 2048 | B, bounded i64 | 0.94x vs fastest non-reuse, 1.61x same-backend | same-backend win does not clear CK gate |
| rocWMMA | 1024 | B | 0.64x | 0.68x |

The hipBLASLt full A+B reuse captures intentionally omit
`hipblaslt_pack_transpose_centered` from per-repeat event timing because the
operands are already packed. That absence is the corrected event contract, not
missing telemetry.

The Direct-HIP colpair v2 entries supersede the earlier single-column
uniform-small i8 A/B reuse kernel. In the first before/after release matrix,
v2 improved per-repeat end-to-end time against that v1 implementation by 2.10x
for bounded i64 512, 2.54x for bounded i64 1024, and 1.43x for bounded u64 512.
The same matrix showed bounded u64 1024 export volatility, so the table uses the
most conservative setup-inclusive speedup from three focused reruns against the
same-backend non-reuse baseline.

A separate Direct-HIP bounded-u64 adaptive-band repeated-B colpair route is
positive only at higher repeat counts so far. The June 4, 2026 Windows
`gfx1100` release smoke in
`temp/perf-work-queue/direct-hip-u64-reuse-b-colpair/` selected
`direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2`, was
schema-valid and event-valid, and at 512 with 33 repeats measured 3218.94 us
for same-build non-reuse direct HIP versus 2842.46 us setup-inclusive per repeat
for the reuse-B colpair route, a 1.13x setup-amortized win. The corresponding
5-repeat smoke still lost after setup cost, so this is a many-repeat explicit
reuse-path result rather than an AUTO/default-routing claim.

Direct-HIP reuse-A fixed-prefix reruns used release binaries, 3 warmups, and 33
measured repeats. The pre-change baseline came from a clean `a75b0a2` worktree
and the candidate used the `transient_uniform_small_i8_b_resident_i8_a_reuse`
benchmark path. All rows were schema-valid, event-valid, and checksum-matched.

| Backend | Shape | Semantics | Reuse mode | Setup-inclusive speedup over 33 repeats | Steady-state per-repeat speedup | Saved per repeat | Setup | Break-even repeats | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Direct HIP | 512 | bounded i64 | A, uniform-small i8 A/B colpair v1 | 3.04x | 2.93x | 6296 us | 618 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 1024 | bounded i64 | A, uniform-small i8 A/B colpair v1 | 1.32x | 1.26x | 1211 us | 580 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 512 | bounded u64 | A, uniform-small i8 A/B colpair v1 | 1.33x | 1.18x | 271 us | 536 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 1024 | bounded u64 | A, uniform-small i8 A/B colpair v1 | 1.30x | 1.24x | 1102 us | 544 us | 1 | Event-valid explicit fixed-prefix reuse-path win |
| Direct HIP | 512 | bounded u64 | A, adaptive native-B colpair v1 | 1.40x | 1.48x | 2100 us | 8514 us | 5 | Event-valid explicit reuse-path win; GEMM phase is slower |
| Direct HIP | 1024 | bounded u64 | A, adaptive native-B colpair v1 | 1.07x | 1.11x | 1014 us | 9679 us | 10 | Event-valid explicit reuse-path win; GEMM phase is slower |

The adaptive native-B rows are same-build Windows `gfx1100` release captures
from June 4, 2026 under
`temp/perf-work-queue/direct-hip-u64-reuse-a-colpair/`. They selected
`direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1` and used
3 warmups with 33 measured repeats. The event traces show the kernel itself is
slower than the normal Direct-HIP grouped GEMM, but the path still wins
setup-inclusively because A packing is removed from the measured repeats and
CRT export timing was lower in these captures.

## Promotion Boundaries

- Promote now: the current local default runtime cache includes the reviewed
  bounded-i64 1024 hipBLASLt v2 entry, the installed 2048 bounded entries,
  current finite-u8 v2 entries, four generic finite-u8 entries, post-fix
  finite-u8 hot 2048 entries, six current exact-wide v2 entries, and eight
  eligible non-reuse 4096 entries for bounded, finite hot-modulus, and
  exact-wide signed/unsigned contracts. The installed reviewed cache contains
  39 validated entries overall after the June 5, 2026 4096 promotion-ledger
  closeout.
  There is no bounded-i64 512 accelerator entry; Direct HIP remains the current
  512 bounded-i64 winner.
- Keep experimental for AUTO selection: Direct-HIP, hipBLASLt, vector ALU, CK,
  and rocWMMA reuse/prepack wins. They are correct and event-visible, and the
  current setup-inclusive matrix identifies explicit reusable-input workload
  candidates, but they compare different reuse contracts and need public
  lifetime/source policy before AUTO selection. The mechanism split is
  summarized in [reviewed-local-evidence.md](reviewed-local-evidence.md).
- Deprioritize for now: vector 1024 repeated-A/full-reuse, hipBLASLt 1024
  repeated-A/full-reuse, and rocWMMA 1024 repeated-B, which lose the latest
  setup-inclusive workload gate.
