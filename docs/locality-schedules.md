# Logical locality schedules

`tools/locality_schedule.py` provides an offline #36 experiment descriptor and
simulator. Each work item owns an output tile and a contiguous logical modulus
group for one K slab. Prefixes may vary from zero (caller-proven zero) to twenty;
partial groups and matrix tails remain explicit. Sorting never changes basis
indices. Forward and inverse tables map launch IDs to exact logical owners.

Every K slab is a separate phase with a full completion barrier. Arbitrarily
shuffling work within a phase cannot race another writer to the same output
plane/tile. A consumer must preserve these barriers; the descriptor does not
authorize concurrent accumulation from different slabs.

The three orders are output-neighbor (row, group, column), modulus (group, row,
column) and same-output (row, column, group). Each first partitions work by
matching tile extents and active plane count. This makes incomplete groups and
tails separate cost buckets. The offline object itself does not install a HIP launch policy or establish
physical cache-domain affinity. The explicit runtime consumer below installs the
corresponding ordering for one-plane resident-residue work.

Traffic distinguishes native 64-bit input blocks reused across groups from
already materialized byte planes, whose different plane indices do not share
source bytes. It reports requested source/read/write bytes, globally unique
source/block bytes, per-domain work counts, scheduling waves and unused slots.
Output reads for later slabs and every output write are included. Modulus-zero
tiles issue no work and require caller handling of their output storage.

The cache estimate is a fully associative LRU over whole logical tile blocks,
with write allocation and round-robin hypothetical domain assignment. Oversized
blocks bypass the cache. Reuse distances count distinct intervening block bytes
in each domain trace. Cold starts empty; warm replay first runs the complete trace,
so an undersized cache does not become magically resident. This diagnostic
implementation can take quadratic time for reuse-distance analysis and stores
the complete mapping; it is not a production scheduling algorithm.

No cache line size, associativity, occupancy, writeback traffic, real launch
overhead, physical XCD placement or HBM counter is inferred. Target and topology
version are labels supplied by the experimenter. `actual_hbm_bytes` and measured
hit rates remain null and promotion remains false. gfx1100 is a single-device
control, not a substitute for gfx942/gfx950 measurements. Actual counter-based comparisons and planner promotion remain open; no Instinct
qualification has been performed.

Authored synthetic tests enumerate awkward mappings and inverse coverage, compare
modular GEMM under shuffled phase execution, and check native-versus-packed
traffic and cold/warm capacity. Execution is deferred to combined validation.


## Explicit direct-HIP runtime consumer

`rns8_create_locality_workspace` (also a C++ `Workspace` constructor) lowers an
already validated adaptive plan's active residue directory into one of these
orders. It first buckets compatible tail extents, then sorts immutable
(tile-row, tile-column, actual-plane) owners. The existing grouped-active HIP
kernel consumes this permuted directory directly; there is no extra device
lookup table or per-submit host sort. Every K slab replays it in stream order,
retaining the original accumulator cap and inter-slab dependencies. Proven-zero
handling, the original mathematical plan seal, canonical equivalent-plan reuse,
and output publication are unchanged. Prefix offsets remain count certificates,
not assumptions about physical directory ordering.

The descriptor binds topology version 1 to actual `gfx942` or `gfx950`; `gfx1100`
is an explicit single-device control. An unknown target or requested/actual
mismatch falls back to the ordinary directory and reports `applied=0`. This
backend is direct-HIP residue arithmetic, not a claim of a new CDNA matrix-engine
kernel. A sorted launch ID is not XCD pinning. `cache_domains`, concurrent groups
and cache capacity are hypothetical simulator inputs, never discovered hardware
facts. No CU masks or system configuration changes are performed.

`rns8_get_workspace_locality_info` reports the applied policy/target, immutable
permutation fingerprint, directory bytes, ordered K phases, requested and unique
packed-plane source bytes and output read/write bytes. Unique source accounting
keeps different residue planes distinct. Traffic covers active arithmetic only,
excluding packing, separate zero-fill and export. These are logical work counts; physical
traffic, cache hit rates and affinity remain unknown. Existing metadata capacity
is reused; sorting/preparation is charged once before repeated submissions.

`rns8-bench --locality-order` uses this real public workspace and records its
identity. The tracked `public-calibration/locality.json` corpus compares all
orders on tail, occupancy-capable, thin and multi-slab K cases with repacked and
retained inputs. Repacking is a lifecycle control, **not** a guarantee of a cold
physical cache. Complete export/padding verification, setup, production host
latency, sustained reuse and separate event diagnostics are inherited from the
public harness. Reports never pool different permutations, and both review and
promotion-ledger paths block automatic routing from these explicit experiments.

Authored host permutation/inverse/overflow-target tests and public HIP repeated
GEMM/export tests await execution. Raw-native multi-modulus/split-fragment locality
composition is a separate next integration; this consumer's group size is exactly
one plane. GPU compilation, target counters, cache-cold control and measured
end-to-end benefit are unverified.
