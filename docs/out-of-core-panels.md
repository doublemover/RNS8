# Bounded out-of-core panel planning

`RNS8_STREAM_GEMM_OVERLAPPED` explicitly selects a second bounded schedule.
It uses two native/RNS input slots and two output/status slots with three owned
nonblocking streams and eight reusable events. The query charges the complete
device arena, two aligned pinned input/export rings, cache tags and a 256-byte
control reservation. It reports pinned and ordinary host bytes independently.
Driver-internal handle/allocator overhead remains opaque. Serial scheduling is
the zero-valued default and retains its smaller unpinned storage contract.

The host refills an input slot only after its upload event; upload waits for
the previous compute consumer before replacing device bytes. Conversion and
GEMM share the compute stream, including retained-B cache replacement. Export
staging cannot be overwritten until its download/status completes. Caller
outputs receive only checked completed tiles; whole-operation validity still
requires every tile and terminal stream drain. Failed drains quarantine the
entire owner and poison its context. Destruction retires handles/buffers in a
retryable order and prevents further execution after partial retirement.

The v3 calibration manifests match in-core, serial and overlapped modes with
the same arithmetic/output boundary. Receipts include logical H2D/D2H submitted
bytes and transfer counts; the collector independently replays tail geometry
and direct-mapped cache conflicts. These quantities are not hardware bus
traffic. `observed_overlap` remains unknown until a real trace demonstrates it.
The new schedule and fixtures are source implementations awaiting the deferred
validation and benchmark batch.

`rns8_plan_stream_gemm` evaluates at most 64 explicit M/N/K, schedule and cache
choices through the same runtime range/resource query. Returned descriptors can
be passed directly to creation; `choice_index` preserves the original identity.
Admitted choices with complete known costs sort by upper amortized estimate,
then required bytes; rejected and unknown-cost choices remain visible.

The cost profile reuses the pipeline planner's transfer, encoded-byte,
multiply-add, general-CRT and output-transfer rates. It additionally requires
input-scan, host-copy and boundary rates, and an explicit calibrated duplex
penalty for overlap. Target, build, calibration label, bounded signedness,
prefix and schedule mask bind the profile. Each endpoint simulates actual
direct-mapped cache evictions and ring dependencies, GPU conversion/GEMM sharing,
upload/compute/download serialization, CPU fill/scatter, pipeline fill/drain
and cold-versus-warm reuse. It never uses an ideal maximum of three isolated
rates. The duplex penalty is conservatively applied to the overlapped transfer
lane; it is a supplied calibration range, not inferred hardware contention.

Prediction work has an explicit cap of at most 65536 panels per modeled call.
Oversized predictions, missing stage rates or stale identities keep resource
admission intact and costs unknown. Synthetic profile fixtures validate model
ordering without becoming evidence. A shortlist never changes AUTO routing,
installs a reviewed record or establishes measured benefit.

`StreamPanelProblem` is an internal versioned contract for one host-backed
bounded i64/u64 product, with explicit M/N/K panels and one uniform prefix.
`plan_stream_panels` checks the complete padded host spans before admitting
locally qualified GPU panel dimensions. It does not reinterpret finite, wrap,
or exact-wide contracts as bounded arithmetic.

The serial resource plan contains eight nonoverlapping 256-byte-aligned regions:
A/B residue panels, two ping-pong C accumulators, native A/B upload panels,
native output staging and a status word. A single device arena owns all regions.
Host payload allowance covers compact A/B and export-tile staging; caller input/output
spans are reported separately and never copied into full-sized library shadows.
An optional retained B residue cache follows those regions in the same arena.
Its tags consume a fixed 32 bytes per slot within the host budget. There are no
pinned buffers or event slots in this serial schedule. Bounded export explicitly
disables nested pinned staging, including when `RNS8_HIP_PINNED_EXPORT_STAGING=1`;
the owned compact output tile is its only library host payload. Driver/allocator
overhead and pageable runtime staging remain
outside requested payload capacity; no observed peak claim is made.

The query returns required and exact 1x1x1 minimum capacities even when the
caller budget is insufficient. It does not silently shrink an explicitly
requested panel or allocate past the budget. Panel extents are clipped to the
logical problem, so oversized user panels do not inflate resource admission.
Every arithmetic size and global/panel address has a checked bound.

`StreamPanelCoverage` enumerates disjoint output macrotiles and visits their
complete K ranges in serial order. Every step covers all admitted modulus
planes. Acknowledgment requires the exact step index, execution epoch and both
input instance/generation snapshots. Missing, duplicate, reordered or stale
acknowledgments cannot complete a product. Only a last-K panel makes a tile
eligible for export; only full coverage makes whole-operation success eligible.
Sources must remain immutable throughout execution. The planned streaming host
destination is uncommitted until whole-operation success and may contain
completed tiles after an issued failure; rollback would require separately
budgeted full staging.

The experimental public C interface comprises `rns8_query_stream_gemm`,
`rns8_create_stream_gemm`, `rns8_execute_stream_gemm`, and
`rns8_destroy_stream_gemm`. Query works without a GPU. Creation requires direct
HIP and allocates all staging plus the single device arena before execution.
The immutable plan retains its context; context destruction is BUSY until plan
release. Execution is synchronous and serialized by a try-lock.

The descriptor declares signed/unsigned semantics, global and panel dimensions,
strides, prefix, maximum input magnitudes and separate hard payload budgets.
Plan admission proves K*amax*bmax fits native output and the uniform CRT basis.
Execution independently scans both complete native inputs to verify those
magnitudes before any output mutation. It does not assume final cancellation
bounds intermediate native products. Each panel converts to residues, computes
A*B+the previous residue accumulator with fused modular reduction, then swaps
two disjoint accumulators. No partial K sum is reconstructed into native64.
Only after complete K coverage does the existing checked CRT export reconstruct
the tile into owned host staging and copy its logical cells to caller output.
Padded destination cells remain untouched.

Source instances/generations bind the execution epoch but never bypass the
input scan. Host inputs are borrowed and must
remain immutable throughout the synchronous call. The result reports the epoch,
completed tiles/panels and whole-output validity. Static rejection preserves
caller output and result. After execution begins, failure publishes invalid
result status; earlier completed tiles may already be present. On any execution
error, an explicit drain precedes cleanup. A failed drain poisons the context
and quarantines the plan's arena and owned host staging; destroy preserves the
plan until a drain and free succeed. Caller output is never a DMA target, so
pending DMA cannot retain it after return. Poisoned contexts stay poisoned.

`retained_b_budget_bytes` enables a bounded direct-mapped residue cache. Zero
preserves the uncached path. The allowance is part of the total device budget;
tag storage is part of the host budget. Admission chooses only complete aligned
slots that fit both hard budgets and the allowance. If every slot would conflict
on every sequential B scan, admission disables caching instead of reserving
payload that cannot remove a reload. It never reduces the requested panel size.

A cache key contains the native B address, instance, generation and panel ordinal;
the immutable plan fixes shape, strides, prefix and semantics. Callers must advance
the generation when B values change. Hits skip host panel packing, upload and
native-to-RNS conversion, including across calls that change only A. Slot eviction
invalidates its old tag before conversion; conversion publishes the new tag only
on success. Any issued failure clears all tags before draining and preserves the
existing whole-output invalidation contract.

The resource query reports selected entries/device payload and cold/warm B load
counts for the actual direct mapping. These counts include conflict reloads.
Execution separately reports observed loads and hits for that call. With Q B
panels and S admitted slots, only singleton slots can hit on a repeated ordered
scan; a full cache uploads each B panel once on a cold call and none on a warm
call. Counts describe removed transfers/conversions, not measured time savings.

CPU fixtures exhaust small panel combinations, count every output/plane/K cell
exactly once, verify slot nonoverlap, reject budget-minus-one, stale/duplicate
completion and overflowing global spans. Public HIP fixtures force many M/N/K
panels under a budget smaller than complete resident operands, compare full
signed/unsigned output to Boost, preserve padding and reject violated bounds.
All execution remains pending. Async overlap, exact-wide/finite/wrap extensions,
calibrated panel selection and reviewed capacity/performance receipts remain
development/qualification work; no throughput claim follows from the serial
implementation.

An installed public-header consumer is available as
`examples/downstream-cmake/stream_bounded_i64.cpp`. Its default mode checks
budget queries without a device; explicit `--hip` performs two full products,
checks padding and generation/coverage receipts, and verifies BUSY ownership
and preflight rejection. Installed CTest GPU execution is opt-in via
`RNS8_DOWNSTREAM_GPU_TESTS`. This authored example is awaiting the final batch.

## Public serial panel calibration

`rns8-stream-panel-calibrate` compares the public serial owner with ordinary
in-core direct-HIP GEMM under the same native host input to checked native host
output boundary. In-core calls pack both inputs with source version zero on
every call, compute the product and export it; the panel call includes its full
input bound scan, panel copies, conversion, K accumulation and checked export.
The in-core recipe's complete input bound is independently checked before timing.
This is an application wall-cost control, not a kernel-only comparison.

Both modes use explicit signed-i64 or unsigned-u64 semantics, a selected fixed
RNS prefix and the conservative complete-expression bound
`K * (2^BITS - 1)^2`. Cases that exceed the native output or selected CRT range
are rejected before context creation. The versioned SplitMix64 recipe uses seed
6811, records all three checksums, starts A[0] and B[0] at the declared maximum,
then changes A[0] to zero in generation two. All input/output row strides include
three padding cells. Every output and padding cell is compared outside timing
with a full Boost int256 product, independently audited by cpp_int at both edge
columns of every row. Fixed Boost operand buffers are released before timing.

The first phase reports query, context/owner or plan/workspace/matrix setup, and
the first native host-to-host batch separately. The same owner then runs warmups,
sampled batches and an uninterrupted sustained interval, followed by changed
input execution with the same plan. Sustained timing has no independent oracle
inside it, targets 250 ms and stops after 16 to 4096 batches; a final full audit
follows. Stream mode also measures checked destroy/create plus first execution,
and probes a violated source bound to verify output and epoch preservation.
Generation labels do not imply cache hits or a skipped input scan; observed cache
receipts report the selected policy's actual loads and hits for each audited call.

Stream receipts report the public query's required device arena, host staging,
minimum capacities, caller spans, output tiles and K panels per tile. The
collector independently replays the eight 256-byte-aligned arena regions and
checks observed completed-panel counts, cache loads/hits and epochs at every
sampled audit. Receipts and manifests use schema v2 with an explicit retained-B
allowance. The CLI takes `RETAINED_B_BYTES` after `RUN_ID`. Device-budget-minus-one
and host-budget-minus-one probes disable optional caching and must preserve the
baseline query receipt and return `RNS8_WORKSPACE_TOO_SMALL`. Panels may be clipped to the
logical shape, but insufficient budgets never shrink the requested panel.
In-core receipts query matrix payload/staging capacities after audits and record
plan workspace separately. The aggregate 512 MiB logical-payload admission
charges all retained caller and reference output buffers, a conservative
512-byte-per-logical-cell allowance, queried panel capacities and workspace.
The allowance covers fixed Boost operands and public matrix/packing payloads;
queried retained capacities are checked against it. Runtime/context internals,
allocator bookkeeping and spare capacity remain outside that model. These are
required or queried capacities, not measured process peak, HBM traffic or proof
that an allocation exceeded device memory. The serial path asserts no overlap.

After building with `RNS8_BUILD_BENCHMARKS=ON`, collect independent processes:

```powershell
python tools/collect_stream_panels.py --exe build/issues-all/rns8-stream-panel-calibrate.exe --cases benchmarks/stream-panels/smoke.json --output temp/stream-panels-smoke --runs 5 --repeats 9 --warmups 3
python tools/collect_stream_panels.py --exe build/issues-all/rns8-stream-panel-calibrate.exe --cases benchmarks/stream-panels/tails.json --output temp/stream-panels-tails --runs 5 --repeats 9 --warmups 3
python tools/collect_stream_panels.py --exe build/issues-all/rns8-stream-panel-calibrate.exe --cases benchmarks/stream-panels/large-bounded.json --output temp/stream-panels-large --runs 5 --repeats 9 --warmups 3
```

Use the same executable names without `.exe` on Linux. The manifests explicitly
select direct-HIP device 0; edit and retain a separate case manifest when another
device is intended. `--help` lists the complete positional driver contract.
`smoke.json` covers both signednesses and matched controls. `tails.json` adds
clipped panels, uneven tiles and K=65535/65536/65537. `large-bounded.json` includes
128, 256 and 512 square cases with fixed panel budgets; these remain within the
bounded host reference domain, rather than claiming an out-of-VRAM workload.
`retained-b.json` adds matched uncached, undersized, conflict-eviction and fully
retained B controls for both signednesses. Its future runs include the public
input scans and checked exports in every timed host-to-host call.

The collector rejects malformed or duplicate cases, metadata/binding/coverage
mismatches and artifact changes; it retains raw failed receipts. It hashes the
executable, collector and validator dependencies, interpreter and manifest around
each child process, validates existing build provenance, and keeps dirty,
unknown-dependency or diagnostic CRT builds ineligible for promotion. Summaries
compare observed phase-matched independent processes, with panel sizes and
budgets still visible; they do not extrapolate crossover or capacity claims.
The [October 25 integration](local-validation-20261025.md#allocation-and-global-control-integration-retry)
passes the applicable local small correctness, policy and receipt fixtures.
Actual beyond-device-capacity, independent crossover and trace qualification
remain separate gates.

## Explicit physical capacity fixture

The authored `rns8-stream-capacity-qualify` gate addresses the small harness's
512 MiB ceiling separately. It uses signed M=65537, N=3, K=65537 with prefix 9,
256×4×4096 panels, padded native inputs/output and no retained B cache. The
complete native A exceeds 32 GiB; its complete nine-plane RNS representation
exceeds 36 GiB. The gate requires both to exceed the queried actual device
memory, before calling the case a beyond-device-capacity demonstration.
Inputs are affine periodic values near 2^20 with negative B columns; the
complete signed bound requires at least eight default planes. Even that minimum
A representation exceeds 32 GiB and must exceed actual VRAM. The capacity
claim therefore does not depend on charging unnecessary prefix-9 planes.

It requires explicit caller, device and library-host budgets and available
host RAM for all caller payload, the library-host budget and an additional
8 GiB reserve. No full-sized reference/RNS shadow is allocated on the host.
A small period table fills the real complete native allocation; an independent
Boost period-sum oracle checks every output plus input/output padding. Cold,
unchanged-generation warm and changed-A-generation calls reuse one stream owner.
Public observations check device/pinned requested peaks against the budgets and
verify retirement to the prior live baseline. Ordinary host capacities and
allocator/driver exclusions remain explicit; process RSS is not measured.

```powershell
.\build\rns8-stream-capacity-qualify.exe --describe
.\build\rns8-stream-capacity-qualify.exe 0 serial 42949672960 67108864 67108864 capacity-serial-001
.\build\rns8-stream-capacity-qualify.exe 0 overlap 42949672960 67108864 67108864 capacity-overlap-001
python tools/collect_stream_capacity.py --binary build/rns8-stream-capacity-qualify.exe --output temp/physical-stream-capacity --caller-budget-bytes 42949672960 --device-budget-bytes 67108864 --host-budget-bytes 67108864 --runs 2
```

`--describe` performs only resource queries. The regular suite tests that mode
and rejects an insufficient caller budget without querying a GPU or allocating
the dataset. Physical execution is always explicit; unavailable RAM/device
capacity returns 77. This three-call gate establishes capacity/correctness when
executed, and does not substitute for independent crossover or overlap traces.
The fixture, CPU oracle regressions and CLI contracts are authored; their build,
runtime and actual capacity qualification remain pending.

The collector pins the executable, assigns separate process/run identities,
counterbalances schedule order and retains timeouts/skips/failures without a
retry or resume. It independently checks the exact generation-dependent weighted
checksums, complete tail-panel coverage and submitted payload/transfer totals,
as well as queried/observed memory bounds. Its output directory must be new.
