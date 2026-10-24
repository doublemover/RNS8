# Native conversion and scheduling calibration

The [larger RX 7900 XTX collection](7900xtx-large-throughput-2026-10-23.md)
covers native-output and resident-residue comparisons at 512³/1024³, with five
independent processes, paired instrumented/uninstrumented calls and sustained
bursts. The [initial 256³ diagnostic](7900xtx-throughput-2026-10-23.md) is retained
as historical small-workload evidence. Other qualification gaps remain open.

The `rns8-native-calibrate` research executable exposes explicit product/packing and epilogue microcases:

| Candidate | Timed work | Input boundary |
| --- | --- | --- |
| tiled_register, tiled_lds | Four-wave 32x32 native WMMA register control or one/two raw-native LDS buffers | Resident full-width native 64-bit |
| `pack_pair` | Pack both native operands | Resident native 64-bit |
| `packed_gemm` | Direct residue GEMM | Already packed residue planes |
| `pack_gemm` | Pack both operands, then direct GEMM | Resident native 64-bit |
| `wmma` | Fused native conversion/grouped WMMA | Resident native 64-bit |
| `encoded_b_cold` | Allocate, probe, encode B, execute R cached GEMMs, release cache | Same resident native A/B; explicit R |
| `encoded_b_warm` | R cached GEMMs; one measured allocation/preparation reported separately | Resident native A and immutable prepared B |
| `adaptive_grid`, `adaptive_compact` | Fresh axis-L1 discovery, adaptive native WMMA, exact suffix and staged commit | Resident full-width native 64-bit to complete prefix20 residues |
| `queue` | Descriptor setup and persistent WMMA queue | Resident native 64-bit |
| `queue_prepared` | Reuse admitted device descriptors; reset cursor and execute | Resident native 64-bit and prepared descriptors |
| `grid_prepared` | Reuse the same preparation; launch ordinary grids without a cursor | Same resident inputs and descriptor allocation |
| `complex_3m`, `complex_4m` | Three/four real products with staged paired recombination | Four resident input components to two residue outputs |
| `split_k` | Three fragments (or K if smaller), queue, canonicalize, merge/store | Resident native 64-bit |

The product cases in this table produce the same default-basis residue planes for matched
shape/prefix/signedness. This output boundary does not assert that a small prefix
uniquely represents the full integer product. Packing-only and already-packed
cases have different boundaries and cannot be compared as complete native-input
pipelines without accounting for missing stages. Group must be one for the
packing/direct baseline cases; WMMA candidates accept one, two or four.

Arguments are candidate, M, N, K, prefix, group, unsigned flag (0/1), repeat count,
warmup count, and a unique run ID. `queue`, `queue_prepared` and `split_k` accept an optional final
worker count in 1..1024; it defaults to 32 to preserve existing invocations.
`adaptive_compact` has its own required worker/method suffix described below.
`wmma` interprets its optional suffix as a reuse count, not a worker count;
the encoded-B candidates require that suffix, as described below. Other
candidates reject a worker argument, including an explicit zero. `--help`
lists every candidate and the argument domains. For example:

```powershell
# Existing invocation retains the requested 32 workers.
.\build\windows-msvc-hip-debug\rns8-native-calibrate.exe queue 17 19 257 3 2 0 10 3 queue-default
# Explicit requests are appended after the unique run ID.
.\build\windows-msvc-hip-debug\rns8-native-calibrate.exe split_k 17 19 257 3 2 0 10 3 split-eight 8
```

M/N are at most 1024, K at most 65537, and
M*N*K at most 1073741824 to bound the independent CPU reference (up to 1024³). A fixed seed and
full-width signed/unsigned extremes are included. Boost computes expected
residues before timing. Full-width product cases use
preconverted/transposed Boost `int256_t` arithmetic: K<=65537 bounds the full
product sum below 2^145, within that exact integer type. Independent `cpp_int`
audits cover both edge columns of every row, and epilogue coefficient arithmetic
remains `cpp_int`. Packing references use `cpp_int`. The emitted `reference_method`
names the oracle method. Older compiled artifacts retain their original oracle
identity; collection does not rewrite their evidence. Fixed-width Boost operand
storage is reported as `reference_operand_storage_bytes` and released before
timing. For every case, outputs are poisoned before each call, and every warmup
and repeat checks output bytes outside the timed region. Product outputs also
carry tail canaries; packing-only cases compare the complete operand planes.

The report preserves fractional host nanoseconds and separate HIP event labels,
and pairs each instrumented call with an additional call with timing hooks
disabled (`uninstrumented_host_call_ns`). Both calls undergo full output and
canary verification outside timing. The uninstrumented column retains production
admission/status and synchronization; it excludes event instrumentation. The
instrumented call always precedes its paired uninstrumented call, so comparisons
between these columns are diagnostic rather than randomized overhead estimates.
The report also preserves
compiled provenance, device/runtime identity, complete case parameters, and
logical buffer sizes. Host calls include candidate-specific allocation,
capability probing and synchronization. Initial uploads, reference construction,
poisoning and verification copies are excluded. Native and packed control buffers
are retained across all cases; source buffers are reused and destination buffers
are initialized between calls. Logical storage bytes are not actual HBM traffic.

After the individually verified pairs, each process runs a sustained burst of
the same synchronous checked operation with timing hooks disabled. It runs at
least 16 calls and targets 250 ms, capped at 4096 calls. The burst has no output
poisoning or reference readbacks between calls; the final full output and canaries
are checked afterward. `sustained` reports the actual call count, duration and
mean latency. It measures repeated resident execution, including required
admission/status work, rather than transfer-inclusive one-shot latency. Per-call
verification and burst verification are reported as different evidence scopes.

`queue_prepared` separately reports `queue_prepare_ns`: host admission,
allocation, capability probing and descriptor upload before warmups. Its timed
calls retain cursor reset, checked launch and synchronization but reuse that
prepared storage. Report setup-inclusive cost for a declared reuse count R as
`queue_prepare_ns + R * sustained.mean_call_ns`; compare against R matched
one-shot calls before claiming amortization. `queue_descriptor_storage_bytes`
records descriptor/cursor storage, not input or output allocations. The tracked
[prepared-queue comparison](../benchmarks/native-calibration/prepared-queue.json)
includes 32³ and 512³ workloads, ordinary native WMMA and both packed/direct
controls. Already-packed input is a distinct boundary, so charge packing when
comparing native-input workloads. The prepared mode remains a private
synchronous experiment, not public prepared-group integration.

`grid_prepared` uses that same owner and preparation scope, but launches one
ordinary native WMMA grid per task with one final synchronization. It accepts
groups 1/2/4 and no worker override. Queue scheduling fields are zero/not-applicable;
the retained descriptor allocation and preparation time remain reported. This
intentionally matched control separates scheduling from allocation/capability
overhead. It does not establish the minimum setup cost of an ordinary-only
implementation. The tracked comparison now includes this control at both sizes;
the earlier ten-case collection predates it.

`adaptive_grid` and `adaptive_compact` are matched private raw-device controls.
Both require prefix20, group1/2/4 and explicit `axis_l1`; both use the same seeded
full-width signed/unsigned native inputs and complete residue-output oracle as
ordinary `wmma`. Every timed call includes capability/admission, allocation,
fresh bounds/compaction, grouped native WMMA, exact suffix completion, staged
commit and synchronization. Initial native upload, common packed controls,
poisoning and full reference/canary copies remain outside timing. These modes
do not use the separate public retained-native matrix API or cache bound metadata.

The CLI suffix is intentionally explicit:

```powershell
# Grid control: the bound method is required after run identity.
.\build\issues-all\rns8-native-calibrate.exe adaptive_grid 17 19 257 20 2 0 9 3 adaptive-grid axis_l1
# Compact control: requested workers1..65535, then the required method.
.\build\issues-all\rns8-native-calibrate.exe adaptive_compact 17 19 257 20 2 0 9 3 adaptive-compact 32 axis_l1
```

Collector cases require `"bound_method":"axis_l1"` for both adaptive candidates.
Compact cases additionally require an explicit integer `workers` in1..65535;
there is no default. Grid cases reject `workers`. Prefixes other than20,
unsupported methods, booleans/floats, missing/irrelevant fields and out-of-range
workers are rejected before launching a child. Existing queue defaults/bounds
remain unchanged. The [adaptive case set](../benchmarks/native-calibration/adaptive-native.json)
covers32³/512³, signedness, groups1/2/4, compact32/512 and matched ordinary
native/pack-direct controls. It is authored input, not completed measurement.

`adaptive_dense_group_upper_bound` is
`ceil(M/16)*ceil(N/16)*ceil(20/group)`. It describes the capacity-sized grid,
not the actual compact work. Device offsets determine the latter; this call
does not read them back, so `adaptive_actual_compact_groups` is null and
`adaptive_compact_count_source` is `device_offsets_not_read_back`. Compact
`requested_workers` and `effective_workers` are equal: exactly that bounded
number of worker blocks is launched, including potentially idle workers.
There is no clamp to a host-inferred descriptor count. Queue descriptor fields
remain zero/not-applicable; `adaptive_launch`, `adaptive_worker_policy`, method
and kernel-selection identity distinguish these schedules. These fields do not
measure occupancy, executed workgroups or hardware utilization. The output
scope states that exact suffix completion makes every capacity20 plane valid.
Compiled differentials, no-GPU CLI/collector fixtures and measurements for this
new calibration slice await the consolidated validation run. No adaptive
speedup or default dispatch promotion is claimed.

`complex_3m` and `complex_4m` compare the private split-complex compositions with
identical full-width signed input recipes, prefix 20 and group 1. Both components
are compared in full against four independent integer products; an exact
256-bit reference covers every output and separate unbounded Boost arithmetic
audits both boundary columns. The declared maximum K keeps each component
below 2^145. Eight tail bytes per output detect out-of-range stores.

The resident timing includes admission, scratch allocation/free, sum transforms,
three or four real GEMMs, recombination, dual commit and synchronization.
Native upload/packing is measured separately as `input_prepare_ns`; neither
candidate performs final integer export. No transformed panel cache is used,
including during the sustained reused-input interval. The report records both
output storage and the actual admitted scratch peak: 3M needs two input-sum
panels plus five output-sized planes; 4M needs six output-sized planes. Thus
fewer products does not imply less memory. The
[complex comparison cases](../benchmarks/native-calibration/complex-residues.json)
include 32³, a 1×512×512 skinny case and 512³. These modes await compiled
validation and measurement; no complex speedup is established yet.

`collect_native_calibration.py` accepts an explicit JSON list of case objects
with `candidate,m,n,k,prefix,group,unsigned_inputs`. Queue and split-K cases may
also specify `"workers": 8`; omitting it normalizes to `"workers": 32` in the
saved case and executable command. Worker values must be JSON integers in
1..1024: booleans, floats, strings, zero and out-of-range counts are invalid.
The distinct adaptive compact contract above requires workers1..65535.
Supplying `workers` for other candidates is invalid. Unknown or missing case
fields, duplicate JSON keys and non-finite JSON numbers are rejected before any
child starts. Duplicate cases are rejected after normalization, including an
omitted queue worker count alongside an otherwise identical explicit 32; use
`--runs` for independent processes. The collector randomizes process order,
defaults to five independent processes per case, checks executable identity
before and after each process, and retains unique raw stdout/stderr records,
commands, UUIDs and failure records. The authored collector test mocks child
processes and does not claim GPU execution.

The current executable schema is `rns8_native_microcase_v2`, and the collector
writes `rns8_native_collection_v2`. Historical v1 captures remain historical
records; the v2 collector rejects them rather than guessing their scheduling
controls. All v2 candidates emit `requested_workers`, `effective_workers`,
`queue_descriptors`, `queue_kernel_selection` and `splits`. Queue launch size is
`min(requested_workers, queue_descriptors)`, with

```text
queue_descriptors = ceil(M/16) * ceil(N/16) * ceil(prefix/group) * splits
splits = min(K, 3) for split_k, otherwise 1
```

For example, M=17, N=19, prefix=3 and group=2 yield eight queue descriptors.
A request for 32 workers launches eight workers; the same split-K case at K>=3
has 24 descriptors and launches 24. This is a clamp to available descriptors,
not an occupancy or hardware-resource measurement. Every calibration queue is
homogeneous in group and signedness, so `queue_kernel_selection` reports
`homogeneous_group_<1|2|4>_<signed|unsigned>` for all queue candidates. Other
candidates report zero for the three worker/descriptor fields,
`not_applicable` for kernel selection and one split. The former misleading
`workers=32` field is removed.

The collector validates each output boundary, storage size, fixed seed, split
count, requested and effective workers, descriptor count and selected queue
variant against the command. It also checks the integer/bool distinction,
every paired timing/reference result, HIP event metadata and sustained final
verification. A sample with invalid host timing is rejected; an unavailable
HIP event duration may be null and remains unavailable. Each successful record
includes a `comparison_identity` containing the executable SHA-256, complete
compiled manifest, device, target, runtime, canonical case, scheduling controls,
input/output contracts, seed, repeat/warmup counts and timing/cache scope.
Requests with different worker counts remain distinct even when both clamp to
the same launch size. Build/device/runtime identity must remain constant within
a campaign. The executable digest is checked immediately before and after each
process.

Compiled provenance must have a valid v2 manifest structure and declared external
header identities. Diagnostic dirty-source builds, experimental Garner-table
configurations and
manifests with `dependency_digest="unavailable"` are retained, with
`provenance_complete=false`. The original compiled manifest remains intact.
Each collection record also preserves `collector_sha256` and
`collector_sources_sha256` for the collector, build-manifest validator and
modulus-ladder dependency. These identify collection-tool changes independently
of the executable artifact/source manifest.
This flag reports the existing release-provenance checks; neither it nor
`validated=true` qualifies performance or enables promotion. Every record keeps
`promotion_eligible=false`. Invalid captures, child failures, launch errors and
timeouts retain raw stdout/stderr, commands and a failure reason; partial timeout
output is preserved too.

The tracked [queue worker sweep](../benchmarks/native-calibration/queue-workers.json)
contains 14 explicit 512³/prefix-9 cases spanning ordinary WMMA groups, queue
worker counts, split-K and direct pack+GEMM. To collect it after building the
current executable:

```powershell
python tools/collect_native_calibration.py --exe build/windows-msvc-hip-debug/rns8-native-calibrate.exe --cases benchmarks/native-calibration/queue-workers.json --output temp/native-queue-workers --runs 5 --repeats 9 --warmups 3
```

Each child also performs the sustained burst described above. This fixture is a
measurement plan; it asserts no result or speedup.

Run the focused authored tests with
`python tools/test_collect_native_calibration.py`. To additionally check compiled
CLI help and pre-device rejection paths, set `RNS8_NATIVE_CALIBRATION_EXE` to the
built executable and run the same test. Those optional checks launch no valid GPU
case. Actual GPU correctness and timing collection remain separate work.

The linked VS2026/gfx1100 collection now covers all listed product cases and
groups one, two and four on the recorded dense shape. Calibrated clock resolution,
per-capture ISA and counter evidence remain absent; raw records cannot promote a
candidate. Instruction-dependency/throughput roofs,
dedicated HBM/LDS probes, measured reuse break-even and held-out planner regret
remain separate #31 acceptance work.

## Four-wave register and LDS controls

The tiled_register and tiled_lds candidates use the
[private native tiled family](native-tiled-wmma.md). Both append two arguments
after the run ID: k_depth and lds_buffers. The register control requires 16 and
0. Staged cases require depth 16 or 32 and one or two buffers. Controls are
required; irrelevant worker or bound-method fields reject.

    .\build\windows-msvc-hip-debug\rns8-native-calibrate.exe tiled_register 17 19 257 20 2 0 9 3 tiled-register 16 0
    .\build\windows-msvc-hip-debug\rns8-native-calibrate.exe tiled_lds 17 19 257 20 2 0 9 3 tiled-two-buffer 32 2

Every case uses four waves and a 32x32 output tile, with groups 1/2/4 and explicit
signedness. Static raw LDS reservation is
buffers*(32*(depth+1)+depth*33)*8 bytes, within an explicit 65536-byte per-block
budget. Register, depth16/buffer1, depth16/buffer2, depth32/buffer1 and
depth32/buffer2 reserve 0, 8576, 17152, 16896 and 33792 bytes respectively.
This is source resource admission, not measured compiled LDS, register usage,
occupancy or device qualification.

Reports bind exact depth/buffers, raw LDS, output tile/wave count, storage,
prefetch policy and kernel-selection identity. Two buffers mean the next tile
loads into disjoint storage before current arithmetic; asynchronous DMA overlap
is not claimed. Full-width inputs, the int256/cpp_int oracle, complete output
canaries, paired calls and sustained intervals use the same existing scopes.
Checked capability/status/synchronization remains inside host call timing.

The tracked [tail campaign](../benchmarks/native-calibration/tiled-native-tail.json)
uses 17x19x257; the [512 campaign](../benchmarks/native-calibration/tiled-native-512.json)
uses 512 cubed. Each has 38 cases: all five four-wave storage variants plus the
single-wave wmma control for every signedness/group, and a pack-plus-direct
control per signedness. Storage variants compare against the identical four-wave
register control; the single-wave case separately measures the output-tile/wave
change. Do not attribute that change to LDS staging.

    python tools/collect_native_calibration.py --exe build/windows-msvc-hip-debug/rns8-native-calibrate.exe --cases benchmarks/native-calibration/tiled-native-tail.json --output temp/native-tiled-tail --runs 5 --repeats 9 --warmups 3
    python tools/collect_native_calibration.py --exe build/windows-msvc-hip-debug/rns8-native-calibrate.exe --cases benchmarks/native-calibration/tiled-native-512.json --output temp/native-tiled-512 --runs 5 --repeats 9 --warmups 3

These are reproducible candidate commands, not executed results. Both manifests,
collector metadata/identity fixtures and no-GPU malformed CLI checks are authored
for the final consolidated campaign. No performance or resource improvement is
claimed.

## Matched native integer comparison

`narrow_wmma`, `narrow_scalar`, `narrow_vector` and `narrow_rns` use the same seeded byte-domain
values stored in native 64-bit arrays and produce bounded exact native 64-bit
integers. Signed data cover [-128,127]; unsigned data cover [0,255]. Both operands
include the maximum magnitude explicitly. These cases require prefix=group=1
(the prefix is an unused CLI placeholder) and K<=32768, which proves all INT32
absolute partial sums fit even for unsigned bytes. The scalar implementation is
signed-only; requesting unsigned scalar fails explicitly. WMMA and vector ALU
support both interpretations.

Host timing includes each complete checked call: WMMA/scalar scan actual device
inputs and prove partial sums; vector ALU resets and reads its overflow status.
All cases retain identical native A/B/C and status buffer allocations. RNS also
retains its operand and product residue planes, with their storage reported.
Candidate-local
scratch allocation, capability probing and synchronization remain timed. HIP
event labels retain component timing separately; output poisoning and exact CPU
comparison of every integer and eight tail canaries are outside timing. The byte
alphabet and K limit bound every absolute partial sum below INT32_MAX, so the
narrow CPU oracle uses exact int64 arithmetic with contiguous B-row traversal.
Boost independently audits both boundary columns of every row. Full-width
residue cases continue to use Boost for every output.

The RNS case scans current device inputs, packs both operands, runs direct residue
GEMM and reconstructs to the device native output with overflow status readback.
It uses the smallest default prefix whose product strictly exceeds twice the
absolute bound for signed data, or the bound for unsigned data. The dataset
determines capacity before timing, but every call independently scans and checks
the actual bound before packing. Admission, packing, GEMM and export have separate
event labels. This is the direct HIP RNS baseline; it does not claim to select the
fastest available RNS kernel for each shape.

The collector validates the dataset, native output contract, partial-sum bound,
minimal prefix and buffer sizes. These cases cannot be compared directly with
the residue-output cases above. No collection, ISA qualification or performance
result beyond the linked local collections is claimed.

## Matched resident epilogue comparison

`epilogue_direct`, `epilogue_rocwmma` and `epilogue_unfused` compare the same
integer-coefficient `alpha*AB + beta*D` expression from resident A/B/D residue
planes. Signed cases use alpha=INT64_MIN; unsigned cases use alpha=INT64_MAX;
beta is INT64_MAX in both. D follows A/B in the fixed seeded native dataset.
Boost evaluates the original integer expression before reducing each channel.
These cases require group=1 and report a distinct expression/output contract;
they are not comparable with bare GEMM cases.

The direct and rocWMMA routes fuse combination into the product store. The
unfused route runs direct GEMM into an intermediate, then a standalone modular
combine kernel into the distinct final output. All three reserve and poison the
same intermediate allocation outside timing to match allocation footprint. Only
the unfused path accesses it during timed work; the report records reserved and
materialized intermediate bytes separately. Actual HBM/counter traffic remains
unavailable. The collector checks coefficients, D dataset, residency and output
contracts in addition to shape and run identity. Optional rocWMMA unavailability
is a retained failed case, never a fallback silently reported as rocWMMA.

The standalone combine has signed-byte modular semantics and rejects output
aliases. It is a correctness/measurement baseline, not integer range admission.
Authored full-width-coefficient CPU/Boost differentials compare fused and unfused
paths with padded tails and K=65537. The extended calibration executable and tests compiled with VS2026/HIP 7.1,
optional rocWMMA and gfx1100. Local differentials and the linked larger collection
have passed; broader performance qualification remains pending.

## Encoded-B preparation and observed reuse break-even

The [encoded-B owner](encoded-b-native.md) has public proof/generation checks.
This calibrator measures its private `hip_prepare_encoded_b` /
`hip_gemm_encoded_b` raw-residue boundary. Public matrix creation, full-integer
range admission, generation ownership and export are not included. Captures
record `encoded_b_public_ownership_measured=false`; they cannot qualify the
complete public call or automatic planner selection.

`encoded_b_cold` and `encoded_b_warm` require an explicit `reuse_count` in 1..256.
`wmma` accepts the same count and defaults to one for existing invocations. Each
timed execute is one batch of R synchronous GEMMs at the same shape, prefix,
group and signedness. Both input matrices stay unchanged during the process;
every call writes the same full output. Cold includes allocation, capability
probe, B conversion, R consumers and checked cache release in its host interval.
Warm measures one allocation/probe/conversion as `encoded_b_prepare_ns`, then
times batches of R consumers against that retained cache. Warm cache release
occurs outside timing. The native grouped control performs R ordinary calls,
including their existing per-call admission/capability policy.

```powershell
.\build\windows-msvc-hip-debug\rns8-native-calibrate.exe encoded_b_cold 17 19 257 9 2 0 9 3 cold-eight 8
.\build\windows-msvc-hip-debug\rns8-native-calibrate.exe encoded_b_warm 17 19 257 9 2 0 9 3 warm-eight 8
.\build\windows-msvc-hip-debug\rns8-native-calibrate.exe wmma 17 19 257 9 2 0 9 3 direct-eight 8
python tools/collect_native_calibration.py --exe build/issues-all/rns8-native-calibrate.exe --cases benchmarks/native-calibration/encoded-b-tail.json --output temp/encoded-b-tail --runs 5 --repeats 9 --warmups 3
```

The two tracked sets, `encoded-b-tail.json` (17x19x257) and
`encoded-b-512.json` (512 cubed), each contain 54 matched cases: three candidates,
groups 1/2/4, both signednesses, and R=1/8/32. These commands and new candidates
are authored only; no measurements from them have been collected.

Cache metadata records the actual K-word padding, retained capability word,
logical cache bytes, resident versus transient bytes, selected group/signedness,
dataset and preparation scope. The full-width reference and tail canaries are
the same as the native grouped control. Verification runs after sampled batches
and after the sustained burst, outside timing. It checks the final output of
each batch, rather than reading it back between identical GEMMs. Sustained
`calls` and `mean_call_ns` count batches; `gemm_calls`, `mean_gemm_ns` and
`batch_gemm_calls` make the logical call count explicit.

When encoded-B cases are present, the collector writes
`encoded-b-reuse-comparison-<campaign-id>.json`. It uses one sustained batch
mean per independent process, then reports the median and process min/max for
each matched case. Warm preparation has its own median. A matched point needs
all three candidates and the same independent-repeat set, artifact, build,
device and runtime. Missing controls/repeats remain incomplete. The reported
break-even is only the first tested R whose cold batch median is lower than the
matched native grouped batch median. There is no interpolation, extrapolated
threshold, confidence claim or promotion. Warm timing alone omits its setup
cost and is not a setup-inclusive win.

Fresh WMMA captures now carry this explicit batch contract even at R=1. Older
records retain their original executable and collector identities; they are
not rewritten or silently interpreted as new batch captures.


### Encode-once candidates and prepared conversion owners

The private `NativeTiledPrepared` owner admits one borrowed native A/B/C binding
and a typed `NativeTiledConfig`. Preparation checks actual-device LDS capacity
and the loaded RDNA3 capability kernel once; execution rereads native values,
launches one grid and synchronizes, without allocating or re-probing. Failed
re-preparation preserves the old binding; execution failure poisons the owner.
Borrowed allocations and the device/code-module lifetime must outlive it. This
is not public matrix generation tracking or a cross-context capability cache.
The old tiled entrypoint remains an honest one-shot prepare/execute boundary.

Two new source-level conversion candidates retain the same signed/unsigned
reciprocal map, independent modular accumulators and 65536 live-K slab:

- `NativeConversion::cooperative`: lanes 0–15 own A and lanes 16–31 own B,
  run a single common conversion stream, then exchange four packed words via
  wave32 XOR16 shuffle. Both replicated operand halves are populated, including
  zero-filled tails. One-wave and four-wave register controls share this body.
- `NativeConversion::encoded_lds`: disjoint producers load and encode each raw
  scalar once per channel into padded packed-word A-row/K and B-column/K LDS.
  Four consumer waves read their replicated fragments. A producer/consumer
  barrier and an end-of-consumption barrier protect every single-buffer reuse.
  There is no simultaneous raw LDS copy; depths 16/32 and groups 1/2/4 are explicit.

The checked resource query reports raw and encoded LDS separately, simultaneous
fragment/accumulator source storage and conversion ownership counts. These are
not measured VGPR allocation, HBM traffic or speedups. The independent host
fragment interpreter uses exact native remainder and checks coordinate-tagged
operands, replication and tails; authored mutations cover wrong partners/roles,
missing tail initialization, wrong shared stride and premature reuse. Device
regressions cover all prefixes, extrema, padded outputs and K-slab boundaries,
including changed-input reuse and zero warm-call allocations/transfers.

The layout assumptions follow AMD's [RDNA3 fragment
specification](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html)
and [lane-operation
reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/wavefront-builtins.html).
They do not qualify other RDNA/CDNA layouts. HIP compilation, emitted shuffle/
conversion/barrier/resource inspection, differential execution and matched
pipeline measurements are deferred; these candidates do not change default AUTO.
