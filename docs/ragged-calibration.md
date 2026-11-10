# Public ragged ownership calibration

`rns8-ragged-calibrate` measures the explicit CPU/HIP
[ragged materialization owner](ragged-residues.md#public-device-side-preparation-authored-execution-pending).
The driver, collector, manifests and fixtures are authored, with compilation and
execution pending. No measured compression benefit, traffic saving or automatic
dispatch decision is asserted.

## Matched results and explicit controls

`uniform_export` retains the ordinary uniform source and exports its complete
wide value. `ragged_export` prepares the compact owner once, reconstructs a
uniform output, then performs the identical wide export. Both provide the same
host result after each repetition. `ragged_resident` only reconstructs a resident
uniform output inside timing; its host audit follows outside. It has no public
uniform-copy comparator because the public API does not provide that operation.
Its timing must not be divided by the uniform host-export timing as a speedup.

```powershell
build/issues-all/rns8-ragged-calibrate.exe cpu-reference -1 uniform_export 17 19 mixed signed 3 5 2 9 3 uniform-example
build/issues-all/rns8-ragged-calibrate.exe cpu-reference -1 ragged_export 17 19 mixed signed 3 5 2 9 3 ragged-example
build/issues-all/rns8-ragged-calibrate.exe hip-direct 0 ragged_resident 17 19 mixed unsigned 3 5 2 9 3 resident-example
```

Arguments are backend, device, mode, rows, columns, pattern, signedness, tile rows,
tile columns, reuse, repeats, warmups and run ID. CPU requires device -1; HIP
requires an explicit 0..63 device, without fallback. Unsupported context creation
exits 77. Rows/columns are 1..512, tile extents 1..128, reuse 1..256 and
repeats/warmups 1..1000. `rows*cols*reuse<=16777216`. The wide source construction
also requires `rows*cols*16384<=134217728`. IDs are 1..128 ASCII alphanumeric
characters, underscores or hyphens. Invalid controls fail before context creation.

The baseline carries the same tile controls solely to identify its matched
compression experiment; they do not change ordinary uniform export execution.
Each batch repeats the selected materialization `reuse` times and overwrites the
same output. Host modes export each repetition. Resident mode exports only in
the subsequent audit, so its reported export count inside timing is zero.

## Reproducible values and full independent checks

Recipe `ragged_tile_bands_seed5152_or_constant_dot16384_v1` has three patterns.
`fullwidth` packs all SplitMix64 bits from seed 5152; every 257th cell uses
INT64_MIN or UINT64_MAX and the next uses INT64_MAX or zero. `mixed` starts from
the same words, then uses four tile-column bands: zero, magnitude below 127,
`2^32 + word%65536`, and full-width. Signed small bands alternate sign by row and
column. This creates exact neighboring low/high prefixes and partial tile tails.

`wide140` builds a source with an untimed public exact-wide GEMM at K=16384.
Each factor is constant along K, with row/column tile bands 0, 1, 2^32 or the
64-bit word 2^63. The last word means INT64_MIN under signed semantics. Two high
bands therefore produce 2^140, beyond native 128-bit range. Every source cell has
an independent `cpp_int` constant-dot reference; complete scalar K loops audit
both edge columns of every row. Construction factors, matrices and workspace
are released before lifecycle timings. Source construction time is reported
inside setup, separately from preparation and reconstruction.

Native recipes retain the explicit host-pack source version 1. Generated sources
report the positive version returned by the public storage query as an opaque
value. The collector does not predict its numeric value or confuse it with the
caller's host-pack version.

All sources and outputs use explicit exact-wide signed/unsigned prefix 20.
Independent CPU tile maxima determine expected prefix counts from the recorded
default modulus ladder. No absent plane is treated as zero. Each public export
uses three little-endian uint64 limbs, with 192-bit two's complement for signed
values. Every value and three padding columns per row are checked outside timing,
including the unchanged source. Full FNV-1a hashes cover all logical limbs;
native recipes also report complete native-word hashes. Python independently
replays all values, prefixes, payload sizes and hashes.

## Lifecycle and capacities

Setup creates the context, source, optional output, export plan and source value,
then the retained owner. First preparation has a separate timer. The first batch
follows setup before any full export audit, so initial output staging growth
remains charged to its first use. Warmups precede repeated wall samples and the
sustained interval: at least 16 batches, 250ms target and 4096-batch cap. No
reference, export beyond the selected host-result scope, checksum or formatting
occurs inside that interval. A complete audit follows it.

After production timing, fresh owners are prepared against the same source for
separate preparation samples. Their destruction is measured separately. These
owners are not substituted into the retained production owner. Resource queries
must agree with first preparation; the retained owner's capacities and generation
binding must remain unchanged from first use to the final audit. The suite does
not infer allocator call counts from stable capacities.

Reports include queried payload bytes, host directory bytes, retained device
directory/status bytes and materialization staging capacity. The ragged API
retains the original uniform source, and this experiment accounts for both the
source and compact owner. Public matrix allocation queries
record payload and cached staging at setup, first use and final audit. They are
boundary capacities, not measured peak memory or RSS.

HIP preparation models one directory D2H transfer, one four-byte proof status
D2H transfer and a directory H2D transfer for compaction. There is no full uniform
snapshot D2H or compact-payload H2D transfer in this preparation implementation.
Materialization separately models its directory upload and status download.
These byte counts describe explicit source-level copies; `observed_hbm_bytes`,
`observed_transfer_bytes` and `observed_overlap` remain null. They do not measure
cache traffic, actual bus transactions or kernel bandwidth.

The 512MiB logical envelope charges `512*cells + 64*construction_operand_cells`
plus, for ragged routes, two conservative padded payload/directory allocations
and one full uniform staging capacity. This covers fresh preparation while the
retained owner remains live, cpp_int values/limbs, host/device uniform storage,
packing and export staging. Tile directory entry capacity is queried, with the
model admitting 32..128 bytes per entry; its exact current size is retained in
receipts. Construction workspace is added separately to the envelope and checked
before its allocation. Allocator bookkeeping, spare capacity, context/runtime
and fixed metadata are excluded explicitly.

## Collection and acceptance

```powershell
python tools/collect_ragged.py --exe build/issues-all/rns8-ragged-calibrate.exe --cases benchmarks/ragged/cpu-smoke.json --output temp/ragged-cpu-smoke --runs 1 --repeats 2 --warmups 1
python tools/collect_ragged.py --exe build/issues-all/rns8-ragged-calibrate.exe --cases benchmarks/ragged/hip-smoke.json --output temp/ragged-hip-smoke --runs 1 --repeats 2 --warmups 1
python tools/collect_ragged.py --exe build/issues-all/rns8-ragged-calibrate.exe --cases benchmarks/ragged/hip-large.json --output temp/ragged-large --runs 5 --repeats 9 --warmups 3
```

CPU/HIP case sets are separate explicit cohorts. Each smoke set covers all modes,
patterns and signednesses. Larger sets include mixed/full64 128x129, 256x257 and
512x512 matrices, plus 140-bit 17x19 and 65x67 sources, with reuse 1/8. Both small
and large cases retain exact complete checks outside production intervals.

One process is one independent observation; internal repetitions are correlated.
The collector shuffles process order, hashes executable/cases/collector/helper
and interpreter identities, rejects mutations and metadata/checksum mismatches,
and preserves failed stdout/stderr receipts. Diagnostic dirty, unavailable or
Garner-table provenance remains collectible but ineligible for promotion. Summary
groups disclose whether the matched host-result pair was observed and retain
reconstruction-only results as a separate scope. No unmeasured break-even,
performance win, hardware traffic qualification or planner promotion follows.
