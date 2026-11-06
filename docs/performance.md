# Performance evidence and promotion

Inspector and benchmark integer controls require complete decimal tokens with
an optional minus sign: no whitespace, trailing text, base prefixes, plus signs,
empty strings or overflow. `--device` accepts only `-1` or an ordinal representable
by a nonnegative `int`; explicit values are checked before backend/context access.
Omitting the option retains the existing inspector default (`-1`) and benchmark
backend-dependent default (`-1` for CPU, `0` for GPU routes). A valid numeric
ordinal still needs actual device admission. Parser-only CLI checks use a final
`--help` to avoid probing a device or starting a benchmark.

## Current evidence boundary

The [SDK10.1 local follow-up](../metadata/sdk101-followup-20261026.json) records
clean compiled source `72c55c2` on RX7900XTX/gfx1100, Core SDK10.1.0 /
HIP7.16.26385 / AMDClang24, with unchanged Windows driver 32.0.31041.5005.
The original 350-process cohort and the separate 350-process post-restart idle
cohort retain their own timings, ranges and frozen prices. The local Debug,
OpenMP-OFF and supported CPU-ASan checks are recorded in the
[Windows guide](platform-windows.md#october-26-configuration-follow-up).

The [October 23 larger-workload report](7900xtx-large-throughput-2026-10-23.md)
and [256³ diagnostic](7900xtx-throughput-2026-10-23.md) remain historical,
source/toolchain-bound observations. Current [backend availability](backend-notes.md),
correctness and measured routing admission remain separate records.

## October 26 SDK10.1 bounded measurements

Each cohort contains 140 public-backend processes, 50 resident-output, 55 CRT,
35 system-roof, 20 instruction-roof, 10 stage-training and 40 held-out graph
processes. All collectors complete successfully. Production outputs pass exact
checks; the 70 public instrumented captures per cohort additionally cross-match
contracts and output checksums to their validated production observations.
The new reference-cache path below adds full element comparisons to those
instrumented native runs as well. Five independent
processes per case use three warmups and nine latency samples. Production
collectors retain three sustained windows, each at least 250 ms and 16 calls unless
the hard 4096-call cap is reached. Oracles stay outside timed intervals; required
proofs, status, completion and publication stay inside. Instrumented phase
observations come from separate processes and are never subtracted from
production calls.

The fixed gain gate requires disjoint observed process ranges and a
(max−min)/median spread of at most 10% in both baseline/candidate groups. These are the beneficial
post-restart comparisons that passed that timing gate within the original harness:

The November4 review of [issue88](https://github.com/doublemover/RNS8/issues/88)
traced discarded timing JSON inside the sustained narrow and graph callbacks.
The graph manual controls also rebuilt prefix-report JSON inside each timed call,
favoring compiled/manual comparisons relative to a reporting-free baseline.
The table preserves those measured envelopes; its narrow/graph ratios, rates and
payback projections await matched remeasurement with the repaired callbacks.
The overhead magnitude and corrected ratios are unknown. Direct CRT uses a
different driver and is unaffected by this trace; exact-output checks remain valid.

| Addition / measured boundary | Baseline → candidate | Speedup | Candidate rate |
|---|---:|---:|---:|
| Fused native WMMA: signed 512³, complete P20 resident RNS output | 0.606801 → **0.530806 ms** | **1.143×** | **0.50571 useful integer TOPS** |
| Direct CRT: 1,048,576 values, P20, resident three-limb output | 3.664167 → **0.204691 ms** | **17.901×** | **5.123 Gvalues/s** |
| Scalar-first proofs vs eager proofs: signed 128³, three-product chain, P20 host output | 11.539241 → **1.162741 ms** | **9.924×** | **0.010822 useful integer TOPS** |
| Costed recipe vs scalar-first control: signed 128³, three-product chain, P20 host output | 1.162741 → **1.144116 ms** | **1.016×** | **0.010998 useful integer TOPS** |
| Scalar-first proofs vs eager proofs: signed 256³, three-product chain, P20 host output | 41.772023 → **1.646298 ms** | **25.373×** | **0.061145 useful integer TOPS** |
| Costed recipe vs scalar-first control: signed 256³, three-product chain, P20 host output | 1.646298 → **1.625372 ms** | **1.013×** | **0.061932 useful integer TOPS** |

The three-product graph fixtures use signed all-one inputs and one final checked
P20 host export. The costed 128³ resident-recipe model still fails the fixed 25% held-out
interval gate; the 256³ resident model passes in both original and post-restart
cohorts. The priced publication stage is a resident GPU copy; host CRT export is
timed separately and remains included in the historical complete-call comparisons.
[Boundary review](../metadata/resident-model-boundary-review-20261027.json)
reproduces the archived resident-model outcomes without new GPU observations or
a changed tolerance. Held-out means
independent processes on the same graph/dataset/build, not unseen workloads.
In the idle cohort the training charges are 2.693 s and 4.070 s; median-based
setup/training break-even projections are 144,672 and 194,549 replays. These are
projections, not observed amortization runs. The earlier 256³ cohort's 1.114×
steady gain and 24,612-call projection remain separate; they are superseded for
current idle-host expectations by the smaller repeated effect. The planner uses
chain DP and bounded materialization search; it does not establish global
optimality or authorize default routing/cache promotion.

Public complete-call 1024³ bounded-i64/P20 medians are 16.589 ms for CK,
16.781ms for rocWMMA, 16.865 ms for hipBLASLt, 17.604 ms for native vector,
20.277 ms for builtins and 27.823 ms for direct HIP. CK's useful integer rate
is 0.12945 TOPS, but no public backend passes the strongest-other-backend gain
gate. Its spread is 10.054%, above the fixed 10% threshold. These calls include
repacking and checked host export. The separate CK diagnostic medians are
9.479 ms packing, 7.457 ms GEMM and 5.078 ms export; independently aggregated phase
medians need not sum to the complete diagnostic median 21.811 ms. Packing/export
and resident reuse deserve the next optimization pass; these observations do not
prove a particular optimization's benefit.

At 17×19×65,537, retained RNS product execution takes 4.158 ms versus 14.156 ms
for scalar native conversion/product. Input preparation and retained storage
have different costs and remain separate; this is a reuse-boundary observation.
The balanced raw CRT tree takes 83.549 ms, a substantial loss. Checked quarter
export and complete-GEMM comparisons retain their inconclusive/negative timing
gates in the metadata, rather than inheriting the raw direct-CRT gain.

The eight-independent-chain instruction probe records 43.513 source-operation
normalized INT8 TOPS (one chain 23.723); adding reduce251 records 13.942 and
6.891 TOPS. Global-copy source-byte normalization records 525.674 GB/s for a
256 MiB source plus destination. Actual HBM transactions, issued-instruction
counters, cycles, occupancy and sustainable clock policy remain unmeasured.
The original matching-artifact ISA audit identifies the emitted
`v_wmma_i32_16x16x16_iu8` kernels; binary hashes are unchanged for the repeat.

Full commands, cases, raw observations, ranges, stage prices, failed gates and
ISA text are in the [original archive](../metadata/sdk101-followup-archive-20261026.json)
and [post-restart archive](../metadata/sdk101-idle-archive-20261026.json).
Within-SDK comparisons do not establish a matched old/new-SDK regression.
Linux, other targets, physical counters/resources, broader held-out workloads
and observed profitable setup amortization remain release gates.

## Measurement contract

A comparable capture must identify:

- Exact commit and dirty-tree state; compiler, HIP/ROCm and accelerator versions
- GPU architecture/device identity, OS, relevant visibility variables, and clocks
- Semantic contract, dimensions and strides, bounds, selected/requested prefix,
  tile schedule, source/input profile, seed, and output representation
- Selected backend and actual kernel, warmups/repeats, timing source, allocations,
  setup/reuse lifetime and workspace requirements
- Pack/H2D, GEMM/reduction, CRT/export/D2H, scheduling and end-to-end costs
- Exact CPU/reference result comparison and required HIP event coverage

Schema v4 validates capture structure and consistency; it is not a proof that a
GPU instruction was executed or that a declaration is true. Keep raw JSON,
stdout/stderr, build logs, exact commands, and disassembly together.

### Reusable exact reference fixtures

The public collector restores the compressed fixtures in
`tests/fixtures/native-reference/` into ignored `temp/benchmark-references/` and
passes that directory to ordinary single-product bounded, exact-wide, finite
ring/field and strict wrap64 runs. Every
logical output element is compared outside production timing. A cache miss
computes the complete Boost.Multiprecision reference; a malformed existing
fixture fails the run. Results beyond the 128 MiB cache payload budget use full
streaming Boost comparison without allocating or writing a cache payload; explicit
population beyond that budget rejects. `--no-reference-cache` requests fresh computation in the
collector. Composed and batched modes keep their existing oracles.

Fixture identity hashes the arithmetic contract, M/N/K and every actual input
word in canonical little-endian order. Extended keys also bind finite modulus,
input width and output limb count; ring, field, exact-wide and strict wrap
contracts have distinct identities. The file also binds the complete expected
output by SHA256. Backend, prefix, timing hooks and device identity do not change
the exact mathematical answer. Input-generator or C++ distribution differences
produce a different input digest and a cache miss. No correctness claim rests
on the seed or output checksum alone. Captures report the method, content key,
payload digest, checked-cell count and time spent outside production intervals.

The paired diagnostic collector uses the same read-only cache by default.
Benchmark sweeps accept an explicit `--reference-cache-dir` and restore the
selected fixture index before execution. The existing three checked-in fixtures
cover bounded signed cases; other keys initially miss until independently
generated and explicitly packaged. The unchanged RNS8REF1 header counts uint64
words, so wide payloads contain cells times limbs words. Every logical limb is
compared, and exact-wide padding and magnitude receipts remain checked.

Restore the reviewed fixtures for direct benchmark invocation:

```sh
python tools/benchmark_references.py
build/rns8-bench --backend hip-direct --semantics bounded-i64 --m 1024 --n 1024 --k 1024 --seed 3135 --reference-cache-dir temp/benchmark-references
```

On Windows use the corresponding build path and `.exe` suffix. Fixture
population is a separate explicit operation: point `--reference-cache-dir` at a
new ignored directory and pass `--populate-reference-cache`. The benchmark
computes the independent full Boost result, checks its output and atomically
publishes the reference. Normal collection never populates a cache. Package
reviewed generation receipts and raw references with
`tools/benchmark_references.py --pack DIR --provenance RECEIPT --output tests/fixtures/native-reference`;
the indexed gzip files are deterministic, and expanded data stays ignored.

Uncached postflight receipts also bind the actual input words. Composed streaming
checks combine input keys in task order and account for every logical output.
Final verification metadata reports `exact_cpu_reference_compared` only for a
complete input-bound receipt; a recorded checksum, including zero, remains pending
without that coverage. Ordinary bounded smoke commands request full verification
explicitly. Existing uncached exact-wide comparisons record the work they already
perform. All oracle, input-hash and cache work remains outside production timers.
These producer changes are source-only pending the deferred validation batch.

The [cache validation](../metadata/native-reference-cache-validation-20261027.json)
records three reviewed fixtures, 14 GPU cache-hit cases, a non-populating miss,
clean corrupt-file rejection, two C++ cases, five Python cases and four automatic
collector observations. The fixtures occupy 3.13 MiB compressed. For the 1024³
case, one full Boost reference takes 24.29 seconds; six cached full-element
checks take 139.6–142.3 ms each. These are observations of reference work outside
production intervals. The library's arithmetic and routing are unchanged, and
the helper remains C++17-compatible.

The vector/native-to-RNS chain now has a separate complete `(A × B) × D`
postflight oracle for requested native verification or reference caching. Its
cache key binds all three actual inputs, shape and explicit signed/unsigned
bounded semantics. A miss computes the producer independently with Boost,
checks its native-width boundary, and compares every final logical output using
one row of scratch; a hit compares the bound final payload. Verification runs
after timed iterations and GPU-owner cleanup. The plain GEMM cache format and
keys are preserved through a shared output/cache comparison helper. New chain
padding, three-input invalidation, signed/unsigned boundaries, intermediate
overflow, cache-hit and streaming-budget controls are authored but unexecuted;
CPU and HIP-enabled benchmark frontend compilation passes.

Vector/native-to-RNS chain recommendations now use qualified process medians
and matching complete work/source/device/timing domains. Both approaches need
the embedded complete CPU oracle and matching mathematical input keys/checksums;
any supplied matching CPU-anchor mismatch still blocks them. Standalone CPU
anchor requirements remain the default for other shared report comparisons.
Use `vector_to_rns_chain_report.py --diagnostic-binding` with separate production
and diagnostic captures. Recommendations must clear the policy threshold and
observed noise; legacy/singleton/unstable evidence keeps descriptive phase values
with unavailable ratios. Submitted implementation views and selected process
bindings remain visible. An identical filename alias cannot erase an intact
binding or add another independent process. These report controls are authored
but unexecuted and carry no new performance credit.

Native benchmark startup now records an application-scoped OS-instance identity
before workload timing. Windows reads the 64-bit native registry view; Linux
reads a bounded machine-id source. Output contains a derived HMAC-SHA256 value,
source/status and OS scope. Raw installation identifiers stay out of captures.
This records an observed OS instance, without claiming hardware attestation.
The source follows the [Windows registry API](https://learn.microsoft.com/windows/win32/api/winreg/nf-winreg-reggetvaluew)
and [machine-id application-identity guidance](https://www.freedesktop.org/software/systemd/man/latest/machine-id.html).
RFC 4231 digest vectors and identity normalization/output controls are authored;
unit and HIP-enabled benchmark frontend compilation passes. Native acquisition
and link/runtime checks remain pending.

Shared variance/cohort, timing-pair and production-companion readers now require
the observed OS identity and stable native runtime-device witness. CPU anchors
join only the same observed OS instance; GPU comparisons also retain the native
UUID/PCI observation and runtime properties. Missing, changed or malformed
identities leave captures descriptive and unqualified. OS scope does not prove
a physical-board count. Protocol controls distinguish host changes, identical
GPU names with different identifiers, CPU anchors, failed witnesses and legacy
captures; these controls are unexecuted.

Vector handoff and host-export/repack chain modes now admit
`--verify-native-output` and `--reference-cache-dir` through the real command-line
parser, including optional cache population. Unsupported compound/output modes
retain explicit rejection until their own complete oracle integration exists.
The argument-parser controls cover the seven ordinary integer semantics and
both bounded vector chain approaches without creating GPU contexts. CPU and
HIP-enabled parser/support/unit frontend checks pass; parser execution, linking
and end-to-end chain validation remain deferred.

Bounded and exact-wide residue chains now also admit complete references and
read-only caching through the native parser and collectors. The independent
Boost oracle computes `A × B^L` from the original inputs, using two rows of
scratch on a cache miss. Keys bind both actual input arrays, shape, chain length
and the explicit semantic/limb contract; single-product keys remain unchanged.
Every final logical cell and limb is compared outside repeated timings. Bounded
chains reject native-width intermediate overflow. Wide resident intermediates
remain mathematical integers, with the declared limb width checked at final
output. These full-output receipts do not masquerade as single-GEMM magnitude
admission certificates. Cache hits compare the entire output without recomputing
the chain. New cancellation, boundary, input/key, corruption, padded-output and
real-parser/collector controls are authored; execution and timing remain pending.

The exact-wide chain report now aggregates independent production processes and
revalidates every GPU process's diagnostic companion and complete CPU oracle.
It retains all implementation views and partitions source, observed host and
runtime device identities before choosing a qualified comparison. Copied files,
singletons and missing companions cannot create ratios. Final-export versus
resident-chain time is explicitly a comparison of different completion
boundaries, with no host-output speedup or promotion implication. CPU/GPU
host-output ratios use a separate explicit policy requiring the same observed
OS instance, source/build, host clock and threading policy. GPU-only device
fields differ in that policy; it cannot compare two GPU devices or two CPU
implementations. Complete final mathematical references can cross continuation
boundaries, while timing contracts retain those boundaries. Source controls are
authored and await the consolidated validation batch.

Shared report comparisons can also request setup-inclusive reuse measurements.
Each independent process contributes its own steady median plus observed
prepack setup divided by its own completed repeat count. The report takes the
median of those process totals, retains their individual receipts and rejects
missing setup/counts or excessive setup-inclusive process spread. This describes
amortization over the recorded run length; it does not establish an application
lifetime break-even. The default policy continues to compare steady complete
calls. Remaining consumer migration and runtime controls remain pending.

Native setup is a one-time scalar, not a repeated timing array. The cohort reader
requires matching `prepack_setup_us` / `avg_prepack_setup_us` values and an explicit
timed `one_time_before_warmups` scope. It carries that actual process observation
into setup amortization without creating or changing raw timing samples. Missing
scope or inconsistent aliases leave setup-inclusive comparisons unqualified.

The final-output RNS-chain report now uses this policy for resident, independent
export/repack and prepacked-reuse implementations. It retains all process views,
selects baselines within the matching work and hardware scope, and requires
complete references and production companions before emitting ratios. Reuse
recommendations must clear the observed noise margin against the same-backend
non-reuse, strongest non-reuse, independent materialization and same-host CPU
controls. Missing source-version/setup custody blocks reuse ratios. Break-even
counts are labeled projections from observed cohort medians. The shared work
contract now explicitly includes the native exact-wide limb count, including
legacy records without the nested export descriptor. Exact-wide boundary
diagnostics also account for the native emitter's D2H/status selector overlays.
Protocol and loader controls are authored; runtime qualification remains pending.

Chain lifetime and reuse source-version checks inspect every actual unique
process record, with individual content identities retained in the report.
The representative capture cannot supply missing custody for a different run.
Incomplete lifetime/source metadata suppresses the affected ratio even when
timing, full-output and companion evidence otherwise qualify. These controls
are authored and remain unexecuted.

Reuse-contract and delegated Direct-HIP reuse reports now use the same cohort,
original-input reference, production-companion and per-process setup readers.
The strongest non-reuse control is selected within matching GPU scope or an
explicit same-host CPU comparison. Every run contributes source/stale/cache
metadata; a representative cannot supply another run's missing owner or default
promotion policy. Explicit workload-selector eligibility retains its separate
source-version, runtime-cache and no-AUTO guards. Phase costs remain descriptive,
and break-even counts remain projections. Native-format eight-process controls
and delegated loader/binding controls are authored; runtime execution is pending.

## Commands

A bounded smoke is prepared by [gpu_qualification.py](../tools/gpu_qualification.py).
It performs no promotion. For a later authorized release campaign, start with
reviewable scenario families instead of an unbounded "all" run:

```sh
python tools/benchmark_sweep.py --list-scenarios
python tools/benchmark_sweep.py --scenario release-candidates \
  --bench build/qualification/gfx942/hip-direct/rns8-bench \
  --bench-for amdgpu-builtins=build/qualification/gfx942/amdgpu-builtins/rns8-bench \
  --bench-for hipblaslt=build/qualification/gfx942/hipblaslt/rns8-bench \
  --bench-for ck=build/qualification/gfx942/ck/rns8-bench \
  --bench-for rocwmma=build/qualification/gfx942/rocwmma/rns8-bench \
  --review-mode release --warmups 3 --repeats 9 --seed 20261006 \
  --capture-timeout-seconds 600 --progress --dry-run \
  --out-root temp/mi300x-release-plan
```

Inspect the resulting plan, resource requirements, selected scenarios and CPU
reference costs before removing `--dry-run`. On Windows, use `gfx1100` build
paths, `.exe` suffixes, and PowerShell continuations. Do not use smoke captures
as release timing evidence. At least three warmups and nine repeats are required
by the existing release-review policy.

Extract ISA from the actual build, not from an unrelated compiler probe:

```sh
python tools/gpu_isa_report.py --build-tree build/qualification/gfx942/amdgpu-builtins \
  --backend amdgpu-builtins --target gfx942 --out-dir temp/mi300x-isa
```

Use `--backend direct-hip`, `ck`, `rocwmma`, `wrap64`, or `vector-alu` as
appropriate. Preserve the tool output and compiler identity. Matrix-instruction
and no-divide rules must be interpreted for the specific kernel family, rather
than applied to every helper indiscriminately.

After collecting matching independent runs, use `perf_variance_report.py` with
`--require-multi-run`, `target_validation_report.py` with real target-status
records, and the relevant workload report. Review their blockers. Reuse, graph,
grouped, sparse, result-cache, and chain workloads require matching lifetime and
output contracts; their measurements cannot be substituted for ordinary one-shot
latency.

Cache installation requires a current `rns8_cache_promotion_receipt_v1` in each
ready ledger row. The ledger derives the exact runtime payload from the reviewed
capture and binds every payload field except publication-only `updated_utc`.
It retains content-addressed candidate, independent-observation, baseline,
review, variance and supplied diagnostic/target/shape/counter inputs. Installation
rechecks their bytes and the payload digest before changing the destination.
Same-key conflicting source payloads reject; timestamp-only regeneration is
idempotent. A legacy ready ledger remains diagnostic input and cannot authorize
a new write. The new publication and real producer-chain controls are authored;
their runtime execution remains deferred.

The generic sweep writer now uses the same admission and atomic installer as
the standalone command. `--write-autotune-cache` requires `--promotion-ledger`;
a ready single process cannot satisfy the two-run minimum, and required baseline
captures need their own matching independent variance evidence. Conflicting
review/variance rows reject. Before authoring a ledger, export the reviewed
payload with `--emit-candidate-cache temp/my-review/candidates.json` and matching
`--out-root temp/my-review`. This artifact has schema
`rns8_autotune_candidates_v1` and no runtime schema version, so AUTO rejects it.
Use it as the ledger's `--cache` and as an explicit installer source; only
successful shared admission publishes the runtime format. Candidate export
never merges into an installed cache. Runtime/generic-writer/CLI and C++ AUTO
rejection controls are authored and remain unexecuted.

The strict wrap64 tuning report now uses shared independent-process cohorts,
full-output CPU-reference verdicts and production/diagnostic binding for every
unique process. Supply its `--diagnostic-binding` receipts alongside production
captures. It compares process medians only within matching input/output and
device/build/runtime contracts, with an observed noise margin. Physical tiling
and K-slab variants may differ; reuse and graph output cadences are retained as
distinct work. Missing or incompatible evidence leaves descriptive phase values
and blockers, without a performance ratio. Order, filename, copied-ID,
high-spread and companion controls are authored but unexecuted. No historical
capture or archived classification has been remeasured or reclassified.

Host API batching is an explicit per-task throughput comparison, rather than
equal-work latency. Its shared normalization admits the registered complete
batch, task count, checksum and per-task host-output boundary; it preserves
arithmetic/input/reuse and device/build/runtime identity. Each batch and
independent baseline needs its own qualified process cohort and CPU reference
covering its complete task set. GPU timings use separate production companions
through `host_api_batch_report.py --diagnostic-binding`; CPU correctness anchors
do not serve as GPU performance baselines. Unknown/legacy normalization and
incompatible evidence retain blockers and omit ratios. The new controls have
not run during this source pass.

CPU small-shape selector recommendations likewise require actual release
sampling, independent production processes, complete CPU-reference coverage and
separate GPU diagnostic companions. They use process-cohort medians rather than
the fastest file, partition full work/source/device/runtime domains, and retain
those identities in non-routing threshold recommendations. The promotion margin
must clear both the selector policy and observed noise. Policy annotations alone
cannot mark a group reviewed; missing or unqualified evidence yields no ratio
or recommendation. Supply production captures and `--diagnostic-binding`
receipts. The new warmup/order/copied-ID/device/companion controls are unexecuted.

The many-small grouped report uses the same production and numerical gates.
Its uniform-task adapter preserves input recipe, arithmetic, output completion
and device/build/runtime identity while normalizing each native aggregate by
its declared task count exactly once. A grouped candidate also needs the same
task count and combined checksum as its host-batch control. Complete CPU
host-batch references cover the identical ordered grouped input sequence;
physical pointer-table placement does not change numerical work. Heterogeneous
task descriptors and compact-slab completion remain visible with blockers and
cannot borrow scalar or uniform host-output timings. Supply production captures
and `many_small_grouped_report.py --diagnostic-binding` receipts. The new loader,
order, copied-ID, filename, noise, warmup and companion controls are authored but
unexecuted; these source changes establish no new measured gain.

Result-cache recommendations now use the shared report-facing comparison
reader. Named cache-contract groups are partitioned by complete work, source
and physical execution environment. Each public candidate needs independent
production cohorts, all-process companions, complete numerical coverage and a
matched full-recompute implementation. Its gain must reach 1.10× and exceed
the observed noise margin. Excluded optional comparisons remain visible without
changing a qualified pair; missing or incompatible required evidence yields no
ratio. Dirty-region composition and cache policy remain distinct implementation
identities. Supply `incremental_result_cache_report.py --diagnostic-binding`
receipts. New parser controls remain unexecuted.

Pending-validation summaries now consume versioned generic-review receipts
bound to the complete capture, work/environment identity, required baseline
files and production companions. They reconstruct independent cohorts and full
numerical coverage before a local recommendation. Path reuse, changed input or
review content, conflicting sidecars and unsupported receipts cannot authorize
a refreshed observation. Unchanged content can be located under a new filename;
copied files do not add independent runs. Historical reviewed flags and declared
review ratios remain descriptive. This local disposition does not install an
AUTO cache entry. The generic review producer's existing duplicate-backend
policy remains separate from receipt qualification; reconciling that producer
with the complete report migration is still implementation work. New producer
round-trip and stale-evidence controls are authored but unexecuted.

## Promotion boundaries

- CPU success and static/host arithmetic tests do not qualify GPU code.
- Header discovery and a compiled object do not imply runtime correctness.
- A reported kernel name does not prove the selected machine instruction.
- Correctness success is not a throughput/latency claim.
- Windows Radeon results do not qualify Linux Instinct.
- A measured helper improvement does not prove an end-to-end improvement.
- Exact-shape cache entries do not authorize nearby-shape routing.
- Historical entries omit this revision's implementation key and cannot hit it.

`benchmark_sweep.py` and the cache install/review tools retain their explicit
promotion workflow. The qualification runner never passes
`--write-autotune-cache`. Keep broad tuning, paid hardware, publishing, and
installed-cache changes outside a source-preparation run.

### Synchronous transfer payload accounting

The benchmark allocation snapshots also record `h2d_calls`, `h2d_bytes`,
`d2h_calls`, `d2h_bytes`, `d2d_calls` and `d2d_bytes`. Their scope is successful
synchronous copy APIs inside the direct HIP implementation, including its
packing, descriptor, status and export helpers. A 2D copy counts only width
multiplied by height, excluding pitch gaps. Setup and measured-repeat deltas
use the same boundaries as allocation counters. The counters are process-wide;
measure a single workload without concurrent counter reset or other activity.

These are observed API payload counts, not profiler memory-transaction counts,
physical PCIe traffic, host RSS or transient device peak. Asynchronous copies,
graph replay, copies internal to vendor libraries and other backend translation
units are outside this scope; zero here cannot establish that such a route
performed no transfer. Existing retained-native signed/unsigned differential
fixtures check one compact logical upload for both compact and padded sources.
The new counters and fixtures await the final validation campaign.

## Capture resume and retired reports

A resumed sweep satisfies a requested capture only when its versioned request
receipt matches the complete arguments (including seed and sampling), relevant
runtime environment and compiled executable digest. Matching captures are not
relabelled or rewritten. Historical captures without this receipt remain valid
review inputs, but are not a substitute for a new requested observation. Stale
observations are retained beside a recollected capture; the sweep reports matched,
stale, attempted and deferred counts separately. RNS8 linkage is static, as
specified by the compiled build manifest; no checkout revision replaces this
artifact identity.

The golden suite records retired reports separately from attempted/tested cases.
Its three retired legacy reports do not execute and do not contribute passes.
Current sweep admission, scheduler evidence and autotune installation keep their
own executable coverage.

### Selected static ISA and proof-work identity

ISA inventory is no longer joined to a timing capture by backend name alone.
Qualified static resource fields require the concrete target, unchanged captured
executable/library hash, build-manifest hash, code-object hash and selected
kernel/symbol mapping. `gpu_isa_report.py --capture ... --artifact ...
--selected-symbol ...` checks the artifact hash and extracts the target code
object to prove byte-identical membership. Repeat `--selected-symbol` for an
explicit static symbol set. Unbound reports remain separately available under
`isa_inventory`; missing target/build/symbol associations return unknown, never
another target's VGPR/occupancy numbers. Reports from different artifacts keep
distinct hash-suffixed filenames, and duplicate copies of one symbol are not
summed. A static mapping is not a dispatch trace or an executed-instruction count.

Strict comparison, sweep grouping, repeatability cohorts and timing eligibility
now share normalized proof-policy and host-batch work identities. Historical
omission of the proof policy means the documented eager-moments behavior.
`result_compare.py --allow-proof-policy-diff` is an explicitly labeled A/B
review, not repeated measurements of one proof workload. Batch task counts and
inner repetitions describe work per observation, not the number of observations.
The host-batch scaling report intentionally compares separately labeled aggregate
and per-task values; ordinary strict comparisons do not call batch4 and batch8
the same work. Regression fixtures are authored; their execution is deferred.

## Independent observations and profiler partitions

The [AMD profiler guide](amd-profiler.md) owns feature-level tool preflight,
native dispatch/database adapters and their remaining acquisition work. Flat
counter exports remain inventory unless a native receipt binds their evidence.

Adaptive, finite-modulus, finite-distribution, layout, bound-discovery and FHE
proxy reports share `observation_cohorts.py`. They aggregate **process medians**
within an exact arithmetic/work, implementation, artifact and execution cohort
before selecting an implementation. Copied captures do not add weight; reused
execution IDs with conflicting records block qualification. Reports retain all
submitted hashes/paths, distinct-run counts and process spread. Raw capture
arrays are not rewritten. Legacy observations remain descriptive, with no
performance qualification or promotion implication.

Environment cohorts use `rns8_relevant_environment_v1`, derived from the actual
versioned collector request's `environment` object and explicit-override digest.
The request seal and executable binding must validate; a claimed standalone
`environment_sha256` never replaces the recorded values. Relevant OMP, HIP,
ROCR, HSA, KMP, RNS8 and loader controls separate cohorts. An explicitly recorded
empty object is valid. Missing/malformed environment evidence, unsupported request
versions and tampered seals remain readable but cannot supply independent runs.
Command arguments and report/output paths remain outside this environment key,
so different destinations do not split otherwise matching process observations.

The collector, resume validator, variance report and observation aggregator share
the same canonical identity routine. Report entries expose its version/digest and
specific blockers. Comparison environments, including CPU anchors, preserve the
same recorded controls. Ledger consumers recompute the cohort and reject stale
variance bindings or a forged ready entry with an unavailable cohort. No timing
arrays, statistical thresholds or installed-cache policy are changed by this
identity repair. Its real-emitter integration fixtures use synthetic timings;
they establish admission behavior, not hardware performance.

At **15ce333**, all **6 real-emitter integration tests**, **5 diagnostic-companion
tests** and **111 configured Python self-tests** pass. The new suites are also
registered in CTest. The [source-bound validation receipt](../metadata/environment-cohort-qualification-20261026.json)
records the inventory preflight, actual commands/log hashes and unchanged C++
source boundary; no performance campaign or logical-proof rerun was required.

FHE workload reports keep the backend-specific export selector in each
implementation receipt. It is excluded from the cross-backend work key: CPU and
GPU selectors can implement the same public output. Semantic, signedness, layout,
completion, shape, proof-policy, source and timing identities remain checked.

The default admission requires two independent processes, three warmups, nine
within-process observations, measured uncensored end-to-end samples, at most
15% within-process p95/median spread and at most 10% process-median range. Those
checks are necessary, not sufficient, for scientific qualification. Same-source
CPU arithmetic controls are matched by complete work contract; GPU performance
controls additionally match device/runtime/configuration/timing policy. Explicit
layout or bound-discovery A/B reports name their permitted contract differences.
Bound-discovery setup-inclusive comparisons sum setup and execution **within
each process before taking the median**. Host batch tasks are work per sample,
not independent observations. FHE recommendations carry shape/work/environment
identity and retain their proxy-only status.

Profiler imports retain per-metric totals, present counts, ranges and units,
partitioned by kernel, phase, device and execution identity. Uneven export-file
partitions no longer give small files extra statistical weight. Mixed-kernel
means are diagnostic inventories, not a kernel's register requirement; occupancy
observation means are not duration-weighted occupancy. Selected scalar resources
require execution/symbol association and an unambiguous phase/unit group. Raw
ISA inventories likewise cannot supply selected-kernel resources without the
artifact/target/symbol association described above.

## CI smoke is not a regression gate

The retained `perf-regression.yml` entrypoint is named **Non-promoting Benchmark
Smoke**. It builds Release and collects six bounded 512³ CPU/direct-HIP captures
(bounded signed, bounded unsigned, finite field 251), with per-capture and job
budgets and serialized device use. `ci_benchmark_smoke.py` generates both the
plan and execution through the supported sweep parser. Missing or invalid
captures fail the collection result; partial collection cannot pass by omission.
Its one warmup/three repeats are diagnostic only. There is deliberately no
invented baseline artifact, numeric regression threshold, cache write or speed
claim. A later true regression gate needs a separately versioned, same-contract
qualified baseline and independent-process threshold review. The CPU workflow
owns a no-executable parser/plan integration regression for this entrypoint.

## Paired residue-dot scheduler controls

`rns8-residue-dot-calibrate` and `tools/collect_residue_dot.py` isolate two axes:
scalar versus packed signed-byte arithmetic **inside the same production
scheduler**, then independent versus grouped/scheduled execution **with the same
arithmetic build**. Region cases explicitly recompute a nonzero `(1,1)` rectangle
and check the cache's recomputation receipt; their full-output export cost is
reported; the reduced compute domain is excluded from measured batching gains.

Build the same revision/dependencies/compiler twice. One build enables
`RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL`; the other enables
`RNS8_EXPERIMENTAL_RESIDUE_DOT4`. They are mutually exclusive. OFF/OFF preserves
legacy arithmetic and is **not** the scalar control: the legacy uniform finite
256/255/251 kernels already use packed dot. The scalar switch overrides those
explicit packed calls inside the shared accumulation helper without changing
kernel dispatch, tile geometry, reduction or slab carries. The collector rejects
legacy/mislabeled builds and only compares manifests identical after removing
these two explicit switches. Different artifact hashes remain recorded.

The bounded shortlist includes every default ladder plane, prefix 1/3/9/18/20,
other finite rings, group size one/several, sparse nonzero/zero/centered/full-width
native sources, padded host rows, K tails through 65,535/65,536/65,537, and larger
sustained grids. Every output and every residue plane is checked with an
independent Boost product outside timing, including export padding. Setup,
first call, complete compute+export p50/p95 and sustained intervals remain
separate. Work/payload admission bounds reference cost before allocation.

Optional diagnostic companions run in separate processes against the same
immutable artifacts. Their final hook-enabled invocation reports actual HIP
**event group spans**, not inferred launch counts; production comparison samples
remain hook-disabled. `gpu_isa_report.py` accepts these captures, binds explicitly
selected symbols to the captured target code object and reports static dot4,
scalar multiply/add, byte-extract/permute, LDS load/store, VGPR/SGPR, LDS and spill
metadata. Static instruction inventory does not prove dynamic execution, sign
interpretation or measured acceleration. Actual launch counts remain unknown without a dynamic
profiler receipt. The existing arithmetic slab certificate is unchanged:
`65536*128*128+128 < INT32_MAX`; numerical-zero tails never read inactive bytes.

All authored regressions and experiments remain unexecuted; no target is newly
qualified and no default is promoted by these source-level update counts.

## Unavailable diagnostic timing

HIP event creation/recording, synchronization and elapsed-time failures now have
separate diagnostic records containing label, phase and backend status. Invalid
durations remain censored; they are never replaced with zero. Recording a
failure uses bounded thread-local storage and cannot make an otherwise successful
arithmetic call fail. Observer overflow/truncation is explicitly incomplete.
Reset and disable retire the previous diagnostic epoch.

The main benchmark marks GPU-event evidence unavailable if a timing failure or
incomplete observer occurs, including on optional phases. Synchronized host
fallbacks from CK, rocWMMA and hipBLASLt use distinct
`host_synchronized_fallback:` labels and mark the requested GPU-event label
unavailable. A host duration therefore cannot masquerade as an event duration.
The per-tile GPU differential still checks every produced/untouched residue
against CPU output; its separate event assertion accepts an available duration
or an explicit timing-failure record. New telemetry regressions await execution.


Benchmark schema v4 retains censored event captures as valid observations when
structured failure diagnostics reconcile with the unavailable reasons and all
GPU-event arrays/source fields are null. The bounded diagnostic list records
unique label/phase/backend-status triples; observer overflow is explicit. Such a
capture still fails event-timing and promotion admission. AUTO-cache and packing
correctness smokes accept this explicit loss while retaining their numerical,
selection and reuse assertions; silent loss and fabricated zero timings fail.


## Residue-dot calibration ownership and timing body

The residue-dot microcase v3 records the created plan's global bound, per-tile
bound count, physical schedule-entry count, required/selected-prefix ranges and
effective dispatch. Scheduled requests use
only per-tile bounds (`global_bound=0`); fixed-prefix selection preserves the same
residue-channel workload across ordinary, grouped, scheduled and region cases.
A uniform no-skip request can legitimately fold to ordinary execution when every
required prefix equals the selected prefix. Such observations are labelled
`scheduled:folded_uniform` and never reported as scheduled-kernel batching gains.
Capture validation checks this distinction and rejects inconsistent ownership.

Individual samples return fixed-size timing values; JSON serialization happens
after the measured phase. Sustained bodies execute compute plus full host export,
with Boost, every-plane residue checks and padding audits outside the interval.
The outer wall timer still includes its loop/termination bookkeeping. The v1/v2
contracts remain historical evidence and are not admitted as v3. Scalar
controls and dot4 candidates must differ only in their recorded arithmetic
options; CRT-table, compiler, dependency and other build settings must match.

## Research evidence and benchmark inputs

Keep reviewed conclusions, matched gain/loss tables, semantic boundaries and
compact provenance summaries in Git. Full per-process captures, sample arrays,
diagnostic logs and expanded experiment inputs belong in a durable artifact,
with its source commit, byte count and SHA256 recorded in the repository.
Do not remove failed measurements or weaken qualification gates when archiving.
Future collections should follow this split instead of embedding entire captures
in dated metadata JSON.

The [PR15 archive manifest](../metadata/research-evidence-20261026.json) pins the
[published evidence archive](https://github.com/doublemover/RNS8/releases/tag/research-evidence-pr15-20261026)
at source `25f7746`. It preserves all 2,075 benchmark JSON inputs and 35 metadata
records as exact Git bytes. Each archived file has a SHA256 in the internal
`archive-index.json`; this archive contains data, not a software release.
The compact dated summaries retain original schemas as `original_schema` and
replace large payloads with JSON pointers into their full archived records.
Restore and verify the complete records without modifying source files:

```sh
python tools/research_evidence.py --output temp/restored-pr15-evidence
```

`--archive /path/to/rns8-pr15-evidence-25f7746.zip` supports offline restoration.
The helper verifies the pinned archive and every member before publishing files,
rejects unsafe paths and restores into ignored scratch storage. Archived Git
bytes use LF; the original records and historical result identities are unchanged.

Benchmark definitions remain editable and reproducible from 25 compact JSON
catalogs under `benchmarks/cases/`, plus 38 retained standalone definitions.
Catalogs factor shared defaults and ordered record layouts into explicit rows;
they retain case names, field ordering and an exact SHA256 for each expanded
document. Generate all ordinary case files before running a collector:

```sh
python tools/benchmark_cases.py --output temp/benchmark-cases/current --json
```

Collectors continue accepting their ordinary JSON inputs, now under
`temp/benchmark-cases/current/<family>/`. The materializer verifies canonical
Git bytes, then uses native line endings so Windows case hashes match the
earlier checkout. CMake generates its own verified inputs under the build tree
automatically; Python corpus tests use an isolated content-addressed scratch
directory. No benchmark is executed by materialization or restoration.

The [October27 reference extension](../metadata/replay-extension-validation-20261027.json)
passes 28 GPU/CPU captures across all seven integer contracts, seven deliberate
corrupt-file rejections, two cached exact-wide diagnostic pairs, and negative-input
retained-recipe/stage captures. These are correctness and integration checks,
not a new elapsed-performance campaign.


Fresh first-call collection consumes the benchmark's versioned
first_call_setup_inventory. For an ordinary persistent product, the complete
caller-host boundary after context creation and caller-buffer preparation sums
planning/workspace, matrix allocation, optional one-time prepacking and one
completed call. Prepack A/B/cache subphases are children of the prepack total and
are not added again. The narrower planning/allocation/prepared-input invocation
sum remains available under its own name. Context creation, caller-buffer
allocation, input generation and the independent oracle are excluded explicitly.

The collector rejects graph, streaming and specialized-owner modes whose setup
inventory is incomplete. Zero warmups do not make prior preparation free. A
missing, unsupported or legacy inventory cannot produce a current complete
first-call total. Review tools use the same accounting helper; first-call,
steady and sustained observations remain separate. This source change does not
replace historical measurements or claim that the October25 ordinary first-call
table used prepacked inputs.


Full pack/GEMM/export graph lanes now stage caller inputs and deliver caller
output inside every completed replay, including warmups and zero-warmup first
use. Their caller_transaction receipt records per-call payloads and first/steady
observations, additional pinned/private preparation, stream creation and
teardown. Capture and instantiate remain separate children. Preparation excludes
those separately measured children and common matrix allocation exactly once.

The setup-inclusive comparison deliberately excludes common planning/workspace
and matrix allocation on both operands; it includes prepack and graph-specific
additional preparation, stream creation, capture and instantiate. Teardown is
reported separately. This is not a whole-owner-lifetime total. Production uses
only the outer wall-clock pair. Separate diagnostic graph runs can report host
staging/replay/delivery phases; their companion kind is graph_caller_host_phases,
which does not assert HIP-event timings or executed ISA. Ordinary GPU controls
use their separately bound HIP-event diagnostics. The shared timing/provenance
gates reject instrumented-versus-production and incomplete/legacy boundary
comparisons before classifying a win or break-even result.
The graph report now retains independent process cohorts and validates every
member's production companion. It aggregates setup-inclusive observations per
process and applies the shared spread/run-count gates; duplicated records cannot
supply independent runs. Unqualified rows retain observed ratios for diagnosis
while classified speedups and break-even conclusions remain unavailable.


Tile and K-block reports use complete independent process cohorts, including
hooks-disabled production timings and a validated diagnostic companion for each
GPU process. Default-tile comparisons preserve input seed/profile, exact range
and proof contracts, complete CPU output references, source/build identity and
observed native device identity. Only physical tile/grid and derived selector
fields are excluded from the work comparison. Per-tile bound proofs remain part
of that contract. Setup is charged from each process's own measured setup and
repeat count; speedups require the shared observed-noise margin.

`tools/tile_shape_report.py --diagnostic-binding receipt.json` accepts repeatable
companion receipts. Its `--resource-manifest` accepts ISA paths or content-bound
`{"path": "selected-isa.json", "sha256": "<64 lowercase hex characters>"}`
references in capture entries. Capture filenames are inventory labels; they do
not establish a resource association. The report rechecks artifact, code object,
build, target, selected symbols and physical tile/K-block configuration for every
process. It requires both register counts, LDS allocation bytes (zero is valid)
and positive static occupancy metadata for local promotion. ISA instruction
mentions, flat counter labels and inline resource summaries remain inventory.
Static occupancy metadata is not measured wave occupancy or dynamic launch proof.
Resource inputs are held as one parsed snapshot and rehashed before publication;
changed or ambiguous input files cannot produce a current report.


Prefix-fusion review follows the same independent process/reference/companion
policy as tile review. It never selects a baseline from a `before` filename,
compares a different requested prefix or output contract implicitly, or awards a
speedup to a legacy-invalid record. One-shot and resident timing boundaries stay
separate. Legacy records and external derived export-selector rows are inventory;
the report recomputes schema validity and shared qualification from raw captures.
Comparisons between source builds still require a separately bound paired
experiment, rather than ignoring source identity. No such campaign is restarted.


Resident-redesign review likewise selects a qualified strongest default cohort,
checks every process's declared research checklist, and revalidates original
selected ISA sidecars from counter reports. Inline resource explanations and
flat counter labels are inventory. The checklist describes declared research
coverage; it is not proof of a measured layout or scheduling change. Complete
comparisons use the observed noise margin (at least 1.03), while host component
medians stay descriptive and cannot independently establish phase speedups.
Legacy-invalid records cannot authorize routing; missing process companions,
selected resource metadata or checklist entries prevent a route candidate.
