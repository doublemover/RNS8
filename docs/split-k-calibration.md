# Public split-K ownership calibration

`rns8-split-k-calibrate` compares ordinary public GEMM with the explicit
[full-scratch and bounded-window owners](native-split-k.md#public-reusable-residue-split-k-owner).
All three modes receive identical full-width data, persistent RNS inputs,
exact-wide signed or unsigned semantics and fixed prefix 20. These owners use
residue producers; the separate private native-WMMA split-K path is outside this
experiment. No AUTO selection changes. The driver, collector and fixtures are
authored, with compilation, execution and measurements pending.

## Explicit controls

```powershell
build/issues-all/rns8-split-k-calibrate.exe 0 unsplit 2 3 17 signed 1 0 2 9 3 unsplit-example
build/issues-all/rns8-split-k-calibrate.exe 0 full 2 3 17 signed 3 3 2 9 3 full-example
build/issues-all/rns8-split-k-calibrate.exe 0 window 2 3 17 signed 3 2 2 9 3 window-example
```

Arguments are device, mode, M, N, K, signedness, fragments, window, reuse,
repeats, warmups and run ID. HIP device 0..63 is explicit, without fallback.
Unsupported context creation exits 77. Invalid controls fail before context
creation. M,N are 1..512, K 1..65537, reuse 1..256 and repeats/warmups 1..1000.
`M*N*K*reuse` is at most 134217728. IDs contain 1..128 ASCII alphanumeric
characters, underscores or hyphens.

`unsplit` requires fragments 1/window 0; `full` requires window equal to
fragments; `window` requires `1<=window<fragments`. Split fragments are 1..4096,
cannot exceed K and keep each fragment at most 65536 products. The ordinary
control can process K=65537 through its existing accumulation slabs. Every batch
recomputes the same complete product `reuse` times, overwriting C. GEMM rates
count all recomputations; independent verification checks the final result.

## Full reference and generation changes

Recipe `splitmix64_fullwidth_seed5117_xor_A0_generation2_v1` generates row-major A
then B with seed 5117. Every 257th word is INT64_MIN for signed or UINT64_MAX for
unsigned; the next word is INT64_MAX or zero. Remaining words retain all random
bits. Generation 2 flips A[0] by hex `0123456789abcdef`; B is unchanged. Both
inputs are explicitly repacked with source version 2. Source versions are user
metadata; internal generations remain opaque and are enforced by the public API.

Boost `int256_t` computes every output from preconverted A and transposed B.
K<=65537 and native products below 2^128 bound each complete sum below 2^145.
`cpp_int` independently audits both edge columns of every row. The full-expression
CRT range is checked against prefix 20 before GPU setup. Temporary reference
operands are released before timing. The collector separately computes every
output with Python integers for both generations and caches only final hashes.

Public export uses three little-endian uint64 limbs per cell, with 192-bit two's
complement for signed outputs. Every cell and three padding columns per row are
checked outside timing. FNV-1a hashes include complete canonical little-endian
input words and all three output limbs; padding is checked separately.

Generation 1 setup includes context, plan, workspace, matrices, pack and optional
owner creation. Preparation and the first complete batch are separately timed.
Warmups precede repeated call samples and an uninterrupted sustained interval:
at least 16 batches, 250ms target, 4096-batch cap. The interval contains no export,
oracle, checksums or formatting, and has a final complete audit.

Generation 2 oracle work is outside timing. Packing has its own timer. The stale
split owner must reject with `RNS8_INVALID_ARGUMENT` and preserve the previous C;
that probe and full preservation audit are outside all measured intervals.
Checked destroy/recreate has a separate timer. Changed setup is the sum of pack
and recreation, excluding the diagnostic probe; it is explicitly additive, not
one contiguous wall interval. Unsplit has the same repack and zero owner cost.
First-use, warmup, repeated and sustained phases then repeat on the changed data.

`setup_plus_first_batch_ns` adds the reported setup and separately observed first
batch, excluding oracle/verification. Public synchronous wall cost includes
launches, merges, carries, publication and synchronization. Kernel-event timing,
observed overlap and hardware-resource measurements remain unavailable. The
driver does not infer hardware cycles or overlap.

## Memory and reproducible receipts

Pure public queries report partial/merge/running bytes, window count and levels.
Each owner receives exactly the queried scratch budget. A/B/C allocation queries
at setup and after each final audit record payload and cached upload/export
staging. Workspace and split scratch remain separate. These boundary capacities
are not measured peaks or process RSS.

The 512MiB logical payload cap uses the larger of oracle peak
`8*(MK+KN)+sizeof(int256_t)*(MK+KN)+24*MN+2048` and resident bound
`256*(MK+KN)+256*MN+split_scratch+workspace+16384`. This conservatively covers
host inputs, temporary signed packing, persistent host/device residues, cached
staging, expected/observed wide outputs and old/new expected output overlap at
generation change. The resident bound also covers the temporary fixed Boost
operands while already allocated GPU state remains live for generation 2; each
fixed operand object is statically limited to 128 bytes. Queried capacities plus
explicit scratch/workspace must fit. Allocator bookkeeping, vector spare capacity, context/compiler runtime
and fixed metadata are excluded.

```powershell
python tools/collect_split_k.py --exe build/issues-all/rns8-split-k-calibrate.exe --cases benchmarks/split-k/smoke.json --output temp/split-k-smoke --runs 1 --repeats 2 --warmups 1
python tools/collect_split_k.py --exe build/issues-all/rns8-split-k-calibrate.exe --cases benchmarks/split-k/merge-tails.json --output temp/split-k-tails --runs 5 --repeats 9 --warmups 3
python tools/collect_split_k.py --exe build/issues-all/rns8-split-k-calibrate.exe --cases benchmarks/split-k/large-reuse.json --output temp/split-k-reuse --runs 5 --repeats 9 --warmups 3
```

Smoke covers signed/unsigned and all routes. Merge tails cover K=65535/65536/65537,
127/128/129/4096 fragments, windows 1/2/128 and short final windows. Large cases
include 512³ where the work/payload bounds permit each explicit route/reuse count.
Full Python replay may be expensive, but is outside child timing.

One process is one independent observation; samples inside it are correlated.
The collector shuffles cases, binds executable/case/tool/interpreter hashes and
retains stdout/stderr/failures. Metadata, generation actions, complete output
hashes, resource formulas and exact timing boundaries are replayed independently.
Dirty, unavailable and Garner-table diagnostic provenance remains structurally
collectible with `provenance_complete=false`. Cohorts match device, shape,
signedness and reuse, retaining explicit fragment/window choices and disclosing
whether all three modes were observed. Summaries retain actual process values;
no untested reuse point, break-even or speedup is extrapolated. Automatic planner
promotion remains disabled.
