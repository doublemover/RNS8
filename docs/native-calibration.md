# Native conversion and scheduling calibration

The [larger RX 7900 XTX collection](7900xtx-large-throughput-2026-10-23.md)
covers native-output and resident-residue comparisons at 512³/1024³, with five
independent processes, paired instrumented/uninstrumented calls and sustained
bursts. The [initial 256³ diagnostic](7900xtx-throughput-2026-10-23.md) is retained
as historical small-workload evidence. Other qualification gaps remain open.

The `rns8-native-calibrate` research executable exposes explicit product/packing and epilogue microcases:

| Candidate | Timed work | Input boundary |
| --- | --- | --- |
| `pack_pair` | Pack both native operands | Resident native 64-bit |
| `packed_gemm` | Direct residue GEMM | Already packed residue planes |
| `pack_gemm` | Pack both operands, then direct GEMM | Resident native 64-bit |
| `wmma` | Fused native conversion/grouped WMMA | Resident native 64-bit |
| `queue` | Descriptor setup and persistent WMMA queue | Resident native 64-bit |
| `queue_prepared` | Reuse admitted device descriptors; reset cursor and execute | Resident native 64-bit and prepared descriptors |
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
Other candidates reject this argument, including an explicit zero. `--help`
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

`collect_native_calibration.py` accepts an explicit JSON list of case objects
with `candidate,m,n,k,prefix,group,unsigned_inputs`. Queue and split-K cases may
also specify `"workers": 8`; omitting it normalizes to `"workers": 32` in the
saved case and executable command. Worker values must be JSON integers in
1..1024: booleans, floats, strings, zero and out-of-range counts are invalid.
Supplying `workers` for another candidate is invalid. Unknown or missing case
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
