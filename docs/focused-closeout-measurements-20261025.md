# Focused closeout measurements — 2026-10-25

The latest [stage-price and proof-policy experiment](#subsequent-stage-price-and-proof-policy-experiment)
adds 160 exact captures and a stronger baseline. It supersedes the earlier
compiled-versus-default comparison as the current baseline assessment.

The first 400 captures passed numerical/schema/provenance validation: 280 public
magnitude captures and 120 materialization captures. These are new bounded
comparisons; the stopped 5,382/16,513-run campaign was not resumed. The user
explicitly authorized collection with NuLARRY still running. Results describe
that shared Windows/RX 7900 XTX/gfx1100 host, HIP 7.1 and compiled code `6ce489b`;
the checkout was `e17c6c7`. Both executables identify their actual clean build.

The [reviewed receipt](../metadata/focused-closeout-observations-20261025.json)
retains every capture's identity/hash, per-process results, cases, phases,
footprints and comparisons. Raw captures remain in ignored
`temp/closeout-remaining-20261025/focused-public/` and `focused-materialization/`.
The [correctness checkpoint](local-validation-20261025.md#subsequent-materialization-integration)
records the four passing configurations and exact build arguments.

## Method

Each case has five independent processes. Public cases additionally have
separate diagnostic and production processes in seeded AB/BA order, with case
order reversed on alternate runs. Materialization cases rotate/reverse order
using seed 4242. Every case has three warmups and nine latency samples.

Production sustained observations execute three bounded bursts: at least 16
completed calls and 250 ms each, with a 4096-call cap. Each process contributes
`sum(duration)/sum(completed calls)`; reported medians and ranges are across
five processes. This differs from the raw recipe collector's descriptive median
of three burst means, so both original observations and this aggregation are
preserved. Independent full-integer audits occur outside repeated and sustained
timing; required proof/status work and completion remain inside the calls.

Times use C++ `std::chrono::steady_clock`. These observations do not export
planner rates, infer causal instrumentation overhead or promote an AUTO route.

## Earlier compiled resident graphs: default public proof policy

These `6ce489b` comparisons use the default eager-axis public proof policy.
The subsequent `31aa691` experiment below adds an explicit scalar-first public
control. Compiled execution does not beat that stronger control in the larger
cohort medians; the earlier speedups are not speedups over the best public path.

Manual baseline and compiled full-prefix owner execute the same complete DAG
and final full-prefix three-limb padded host export. Both use all 20 planes and
unchanged packed A/B inputs. The baseline invokes the public resident GEMM for
each node; the compiled owner retains the admitted graph bounds and arena.
It still validates input generations and completes all required writes/status.
No explicit result-cache owner is created by either route. Signed all-one inputs
produce 22-bit and 25-bit final values at 128³ and 256³. This comparison holds the
20-plane compute prefix fixed; graph admission/metadata/publication work differs.

| Addition / case | Public resident sequence | Compiled full-prefix DAG | Observed steady speedup | Setup charged over 100 replays, projected speedup |
| --- | ---: | ---: | ---: | ---: |
| Three-product 128³ signed chain | 13.461 ms | 1.370 ms | **9.83×** | **3.67×** |
| Three-product 256³ signed chain | 56.205 ms | 1.980 ms | **28.39×** | **9.28×** |

Steady process ranges are 13.360–14.966 ms versus 1.314–1.435 ms at 128³, and
51.818–62.962 ms versus 1.957–2.342 ms at 256³. The 100-replay column computes
`setup/100 + measured steady mean` within each process, then takes its median;
it is not a directly timed 100-call batch. Its observed process ranges are also
separated for these two cases. Projected per-call totals are 18.682→5.090 ms
and 61.906→6.669 ms respectively.

Measured harness setup medians are 371.651→373.584 ms at 128³ and
405.880→466.044 ms at 256³. Setup includes context/plans, matrices/input packing
and recipe construction, including common control objects. It is not an isolated
planner-preparation measurement or a promise about minimal application setup.
The median-difference break-even projections are one and two replays; cold
setup has substantial variation, retained in the receipt.

Useful graph throughput counts `2MNK` for each of the three actual products,
without multiplying by residue planes. It is 0.00919 TOPS at 128³ and
0.05084 TOPS at 256³ for the compiled route, versus 0.000935 and 0.001791 TOPS
for the public sequences. These are complete-graph/host-output rates, not raw
matrix-instruction throughput.

The 17³ chain and 32³ unsigned diamond also favor full-prefix compilation in
steady medians. Their setup-inclusive 100-replay comparisons are not clear wins,
so they are omitted from the gains table.

## Packed storage and the analytical model

No packed-B candidate has a lower median than the plain full-prefix compiled
owner in any of the four tested cohorts. At 256³, packed-full takes 2.494 ms and
the analytical layout model 2.940 ms, versus 1.980 ms for plain compiled full.
The minimum-prefix owner takes 2.613 ms. Packing and additional exact lifts must
be charged; fewer compute planes are not automatically a faster complete call.

At 256³ the full-prefix owner retains 2,621,440 arena bytes and no lift scratch.
Packed-full retains 3,932,160 arena bytes and packs B once per replay. The
layout-model choice computes prefixes 2/3/4, retains 2,031,616 arena bytes plus
1,048,580 lift bytes and performs three incremental pack calls. Every recipe
mode reports zero HIP allocations/frees during its nine measured replays.
These capacities exclude external matrices/host metadata and do not establish
process RSS or allocator/device peak.

The model prices are explicitly analytical and unqualified. Their route choices
lose to full-prefix compilation here. Independent measured stage prices,
held-out selection, actual peak/capacity evidence and a native per-operation
import/export control remain work under #42/#30/#31. These findings justify no
packed default promotion; the best compiled control stays visible.

## Genuine wide-value public fixtures

The public subset uses 8×7×512, prefix 20, three exported limbs, output pitch
padding 3 and seed 7373. It covers signed/unsigned small, constant two/three-limb,
and nonconstant mixed profiles, CPU/reference and direct-HIP, with retained and
selected repack controls. All 280 captures check 56 outputs and 168 limbs plus
padding. Distribution/checksum/magnitude identities agree across backends,
residency, instrumentation and all five process runs.

| Profile | Actual result magnitude |
| --- | --- |
| Constant `2^40` operands | Exactly `2^89`: 90 bits, two magnitude limbs |
| Constant `2^62` operands | Exactly `2^133`: 134 bits, three magnitude limbs |
| Signed two-limb mixed | Output magnitudes 52–86 bits; maximum uses two limbs |
| Signed three-limb mixed | Output magnitudes 74–130 bits; maximum uses three limbs |
| Unsigned mixed controls | 90-bit and 134-bit maxima respectively |

Three-limb storage alone is not treated as evidence of a wide value. All high
limbs are checked against Boost.Multiprecision full integer products.

Retained-input signed constant observations are 0.455 ms CPU versus 1.677 ms HIP
for two limbs, and 0.485 ms versus 1.858 ms for three limbs. Unsigned values are
0.419/1.781 ms and 0.458/1.822 ms. These tiny fixtures favor the CPU; higher value
magnitude does not establish a GPU speedup. Their purpose is the missing genuine
wide-result control under #73, not a peak-throughput claim.

## Closeout

#73 is completed by the implemented profiles, numerical admission/schema checks
and this paired independent-process collection. With #19/#29 and the preceding
six closures, nine issues are closed and 42 remain open. Their individual
implementation/hardware/performance gates remain in the
[ledger](../metadata/remaining-implementation-20261024.json).


## Subsequent stage-price and proof-policy experiment

**160/160 captures passed exact reference checks:** 20 independent stage-training
processes and 140 complete-DAG processes. All executables were compiled from clean
`31aa691`; their hashes were pinned before collection. Subsequent checkout edits
are not attributed to these artifacts. The
[reviewed receipt](../metadata/product-stage-observations-20261025.json) preserves
all cases, commands, process identities, raw samples, sustained bursts, copy and
allocation-call counters, exact checks, frozen price bundles and negative results.
This brings the new focused collection to 560 captures. The old campaign remains
stopped. The user authorized this shared-host collection with NuLARRY active.

Each of four graphs has five independent training processes and five processes
for each of seven complete-route controls. Stage training uses three warmups and
31 samples per shortlisted cell. Complete routes use three warmups, nine latency
samples and three sustained bursts, each at least 16 calls and 250 ms. Burst means
are weighted by completed calls within each process; the table reports the median
of the five process means. CPU integer oracles and padding checks run outside
these intervals. Required admission, synchronization and final three-limb padded
host publication remain inside. Event hooks are disabled.

All fixtures use all-one inputs, a full 20-plane destination and the same graph
and final observable result. `manual-full` uses default eager-axis proof discovery;
`manual-scalar-first` explicitly asks the existing deterministic scalar-first
proof policy to admit the same fixed-prefix public operations. It does not skip
proofs, cache a result or change exactness. Native roundtrip imports native A/B at
each product and exports/reimports intermediate native integers. Intermediate
values in these fixtures fit that native contract; the final output uses the
common three-limb export. Measured recipes consume frozen, identity-bound stage
prices from the separate training processes.

### Beneficial proof-policy comparisons

Only the two positive comparisons with separated observed process ranges appear
in this gains table. These are gains from selecting the explicit proof policy;
they are not attributed to the compiled recipe or packed storage.

| Graph / addition | Default public eager-axis | Explicit scalar-first | Steady speedup | Useful integer TOPS, before → after |
| --- | ---: | ---: | ---: | ---: |
| 128³ signed chain, three products | 38.330 ms | **5.022 ms** | **7.63×** | 0.000328 → **0.002505** |
| 256³ signed chain, three products | 113.176 ms | **8.064 ms** | **14.04×** | 0.000889 → **0.012483** |

Process ranges are 31.270–47.579 versus 1.559–6.010 ms at 128³, and
100.100–122.067 versus 2.621–10.393 ms at 256³. Useful operations count `2MNK`
once for each of the three graph products. These are complete host-output graph
rates. The tiny 17³ and 32³ comparisons have overlapping process ranges and do
not enter the gains table. No automatic policy change is made from these fixtures.

### All complete-route controls

This comparison table retains the slower controls; it is not a gains table.
Values are steady weighted process medians in milliseconds. The 32³ unsigned
case is a four-product diamond; the others are three-product signed chains.

| Side | Public eager-axis | Public scalar-first | Native roundtrip | Compiled full | Compiled minimum | Analytical layout | Measured layout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 5.714 | 5.235 | 14.490 | 5.018 | 8.529 | 5.778 | 5.222 |
| 32 | 7.450 | 5.258 | 29.704 | 5.307 | 16.784 | 14.383 | 6.854 |
| 128 | 38.330 | 5.022 | 120.012 | 7.753 | 8.484 | 9.884 | 7.533 |
| 256 | 113.176 | 8.064 | 372.881 | 10.671 | 13.561 | 17.246 | 9.041 |

Compiled recipes do not beat scalar-first public calls in the larger cohort
medians. Candidate-versus-scalar process ranges overlap in all four cohorts.
The stronger baseline therefore removes the earlier interpretation that graph
compilation itself is an established best-baseline gain. The earlier `6ce489b`
tables remain historical observations against their named eager-axis control;
source identities and host conditions differ and are not pooled.

The receipt includes setup, first-call, repeated-call and sustained observations,
owner-only `setup/100 + steady` projections, and separate projections charging
all measured stage training to 100 calls. Training cost includes backend setup
and measured calibration work; process startup and report serialization are
outside that recorded boundary. A conditional pretrained projection is not a
cold-start gain. No new setup-inclusive compiled gain qualifies against the
scalar-first control. Replay allocation-call counters are zero for the compiled
owners; these older captures predate the live/peak observation API and do not
establish an allocator peak.

### Frozen-price held-out results

All four models fail the predeclared 25% held-out prediction tolerance. Training
and held-out execution IDs are disjoint, but use the same graph, all-one dataset
and build: this does not establish generalization across shapes or distributions.
The model underpredicts the complete measured graph intervals. Confidence bounds
use the existing conservative run-mean method and its stated stationarity
assumptions; shared-host variance is substantial.

| Side | Predicted graph interval | Held-out process mean | Held-out 95% interval | Pass |
| --- | ---: | ---: | ---: | --- |
| 17 | 0.165–0.241 ms | 1.529 ms | 0.027–3.031 ms | No |
| 32 | 0.229–0.313 ms | 3.338 ms | -0.199–6.875 ms | No |
| 128 | 0.294–0.370 ms | 2.097 ms | 0.260–3.934 ms | No |
| 256 | 0.590–0.727 ms | 5.904 ms | 2.093–9.716 ms | No |

The negative lower confidence bound at 32³ is uncertainty from the interval
construction, not a negative measured duration. Unstable training cells whose
lower interval is not positive are omitted as unknown, never replaced with free
work. Tolerances are not widened after seeing these results. Price-file ingestion
and complete-route controls are implemented; prediction qualification, hardware
traffic/instruction roofs, representation-choice integration and physical
capacity qualification remain open under #30/#31/#42. No AUTO route is promoted.
