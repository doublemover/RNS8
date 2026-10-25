# Focused closeout measurements — 2026-10-25

All 400 captures passed numerical/schema/provenance validation: 280 public
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

## Beneficial compiled resident graphs

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
