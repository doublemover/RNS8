# Public block-sparse cost and reuse controls

`rns8-block-sparse-calibrate` compares complete public direct-HIP calls on the
same signed exact-wide inputs and output shape. The target, collector, case sets
and fixtures are authored; they have not run. No speedup, hardware qualification
or default dispatch change is claimed. The [block-sparse owner](block-sparse.md)
owns the arithmetic, generation binding and WMMA/hybrid contracts.

## Matched boundaries

The caller explicitly chooses a HIP device, prefix20, a deterministic block-mask
recipe, the method, reuse count and hybrid threshold. All modes pack identical
retained-native A/B matrices with import source_version=1. There is no fallback
to another backend, device or sparse method. Unsupported selection exits 77;
the collector retains it as a diagnostic failure, never a passing observation.

| Mode | Work inside one measured batch |
|---|---|
| `dense` | R ordinary public `rns8_gemm_rns` calls on the dense resident inputs |
| `sparse_scalar`, `sparse_wmma`, `sparse_hybrid` | R public sparse calls using the two owners prepared once during setup |
| Corresponding `_cold` mode | Prepare both owners, execute R calls, then explicitly destroy both owners with checked status |

The cold observation is measured, not constructed by adding setup and warm
medians. Every cold sample also reports its measured preparation and destruction
subintervals. Warm preparation is a separate setup duration, and total setup
includes context/plan/workspace/matrix creation and input packing. Native/RNS
setup before warm preparation has its own duration. Each sparse public call
includes proof admission, temporary metadata allocation/upload, capability probe
when applicable, execution, synchronization and publication. Those costs remain
inside the measurement. Dense RNS computation is the ordinary public control;
no optimized CPU comparator or matrix-engine dense route is implied by its name.

R is explicit in 1..256. One batch leaves one repeatedly overwritten output,
which receives a complete Boost and padding audit outside the interval. It does
not claim that every overwritten intermediate was independently exported. The
sustained interval has at least 16 batches and ends at 250ms or 4096 batches after
that minimum. No independent audit runs inside the sustained interval; its final
output is checked afterward. Counts distinguish batches from R*batch GEMMs and
report both means. Per-sample medians average the central values for even counts.

## Exact reference and dataset

`splitmix64_signed64_block_masks_seed3135_v1` generates A then B in row-major
order using SplitMix64 starting at 3135. The 64-bit word is interpreted as a
signed two's-complement value. Operand indices divisible by 257 become INT64_MIN;
indices congruent to 1 become INT64_MAX. An active cell with generated zero
becomes one. Inactive block cells become exact integer zero. The generator still
consumes a word for inactive cells. Block size is 16x64 for A and 64x16 for B.

| Pattern | Present K blocks |
|---|---|
| `dense` | All blocks in both operands |
| `disjoint` | A retains even K-block indices; B retains odd indices |
| `mixed` | Output row/column tile index modulo 3 selects all blocks, even A/odd B blocks, or only block0 |

Mixed cases exercise zero, low and higher intersection counts. Hybrid uses the
explicit threshold; it never prunes small nonzero values. Non-hybrid modes reject
a nonzero threshold. The collector independently reconstructs the exact native
recipe, computes its byte SHA256 and verifies the complete input FNV checksums.
All checksums use little-endian bytes; the output checksum covers all three
192-bit two's-complement limbs of every logical result.

The full independent product uses preconverted/transposed Boost `int256_t`
operands. `|sum| <= K*2^126 < 2^143` under K<=65537 proves every accumulation
exact in that fixed Boost backend; it is not a native128 comparison. Separate
`cpp_int` calculations audit both edge columns of every row. Every exported
192-bit value and padding word is compared to this full reference. Reference
operand buffers are released before GPU setup and timing. The benchmark also
checks that the public modulus product at prefix20 admits the complete signed
full-expression bound, independently of runtime admission.

Source content generations are private to the owner. The record exposes real
public matrix instance IDs and import source versions, leaves content generation
numbers null, and states that the public owner API checks them. It does not
mislabel import source_version as an engine generation. Inputs are not repacked
after preparation. Cached generation invalidation and rollback are exercised by
the separate owner fixtures, not inferred from this timing experiment.

## Memory and provenance

M,N are 1..512, K is 1..65537 and M*N*K<=134217728. The logical live-payload
cap is 256MiB and may further restrict the shape. Let AB=MK+KN and P=MN. Before
allocating, the reference bound is `8*AB + 24*P + sizeof(int256_t)*AB + 2048`.
The separate execution bound is `80*AB + 112*P + 2*(packed+directory) +
8*max(MK,KN) + directory + 4 + plan_workspace`. The greater bound is charged.
It conservatively covers retained native inputs and RNS matrices, reference
outputs, full padded export, both cached operands, overlapping preparation
snapshot/packed buffers, per-call metadata and public workspace. Reference
operand buffers and GPU state are not charged as if simultaneously live.

Expected packed bytes/directories and intersection counts come from the exact
recipe and documented block format. They are explicitly modeled, not queried
device counters or observed cache allocation. Actual sparse-cache allocation
remains null because its public owner exposes no allocation query. The public
matrix allocation API separately reports A/B/C requested payload capacities and
cached upload/export/status/schedule staging at setup and after the final audit.
These snapshots exclude sparse owners and transient peaks. Vector spare capacity,
allocator overhead, context runtime and fixed metadata are outside the logical
cap; it is not a process RSS/free-device-memory guarantee. The actual queried
matrix capacities are also checked against the conservative execution envelope.

The executable receives the same immutable build manifest as other calibrators.
The collector binds executable, case file, native dataset, interpreter, collector
and both validator-module SHA256 values, retains raw stdout/stderr on failure,
checks metadata/resource/timing/audit identities, and rejects build/runtime or
matched-output changes. Dirty/unavailable/experimental-Garner diagnostics remain
ineligible for provenance completion where applicable. All records unconditionally
retain `promotion_eligible=false`; no production-schema or verified ISA claim is
manufactured by this thin adapter.

## Commands and observed comparisons

Build the existing project with `RNS8_BUILD_BENCHMARKS=ON`. For an explicitly
selected device0, a bounded example is:

```powershell
build/issues-all/rns8-block-sparse-calibrate.exe 0 sparse_hybrid 17 19 257 mixed 8 1 9 3 warm-hybrid-1
build/issues-all/rns8-block-sparse-calibrate.exe 0 sparse_hybrid_cold 17 19 257 mixed 8 1 9 3 cold-hybrid-1
build/issues-all/rns8-block-sparse-calibrate.exe 0 dense 17 19 257 mixed 8 0 9 3 dense-1
python tools/collect_block_sparse.py --exe build/issues-all/rns8-block-sparse-calibrate.exe --cases temp/benchmark-cases/current/block-sparse/tail-reuse.json --output temp/block-sparse-tail --runs 5 --repeats 9 --warmups 3
```

Tracked [smoke cases](../benchmarks/cases/block-sparse.json) cover all seven modes.
The [tail/reuse set](../benchmarks/cases/block-sparse.json) uses all three
patterns at R=1/8/32; the [512-cubed set](../benchmarks/cases/block-sparse.json)
uses dense/mixed patterns at those same reuse counts. Copy and edit a manifest
under ignored `temp/` to select another actual device. Do not silently filter
unsupported modes or replace their measurements with scalar timing.

Each case/run is a fresh process. Only sustained batch means from independent
processes enter the reuse summary. A point requires matched dense/warm/cold
process-index sets; missing controls stay incomplete. The summary reports
medians, minima/maxima and ratios at tested R values. A first observed cold win
is reported only at a tested R with at least five processes and the entire
observed cold range below the dense range. This is a descriptive range test,
not a confidence interval, interpolation, calibrated universal crossover or
production promotion. Larger shapes, other targets, resource/ISA reports and
measured end-to-end break-even remain qualification work.

## Campaign-order control

The collector no longer executes every repeat of dense before any sparse case.
`--order-seed` (default 5151) generates bounded matched blocks keyed by device,
shape, dataset and reuse count. Each block contains one process per requested
mode/threshold at the same independent-repeat index. Its initial order is
seed-shuffled, rotates by one position each repeat, and matched-block order is
also shuffled. Thus controls do not occupy one fixed early/late position and
all requested case/repeat pairs occur exactly once. This is counterbalancing,
not proof that thermal or frequency drift was absent.

Before launching any process, an immutable campaign schedule records every
ordinal, original case index, independent repeat, matched block and position.
Its SHA256 and seed accompany every raw receipt and the summary; each receipt
also records its actual attempt start time. Schedule/case/executable changes
reject the campaign. Missing/cohort-mismatched data still cannot become a reuse
win. Scheduling/receipt regressions are authored but unexecuted.

Structured Sparse-A event families are separately centralized in
`metadata/event_phases.yaml`. The generated C++ helper is consumed by actual
backend hooks and benchmark JSON construction; the Python validator consumes
the same generated rows. Both CDNA3 16x16x64 and 32x32x32 SMFMAC, RDNA4 sparse,
and dense controls therefore bind their finite versus bounded/exact-wide phase
names to one source. This fixes larger-family collection metadata, not sparse
arithmetic, issued ISA or the >=1.5x measured structured-sparse gate.
