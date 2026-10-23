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
tails separate cost buckets. These are offline permutations, not installed HIP
launch policies or physical cache-domain affinity claims.

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
control, not a substitute for gfx942/gfx950 measurements. Runtime remapping,
calibration/planner integration and actual counter-based comparisons remain open;
the unavailable Instinct qualification host is deferred by the user.

Authored synthetic tests enumerate awkward mappings and inverse coverage, compare
modular GEMM under shuffled phase execution, and check native-versus-packed
traffic and cold/warm capacity. Execution is deferred to combined validation.
