# Materialized replay lifetimes

The explicit materialization API in `rns8/experimental_product.h` uses cost
schema3. Callers provide `expected_replays` in 1..UINT64_MAX-1 and initialize an
unknown baseline to UINT64_MAX. Earlier outer profile sizes and schema versions
are rejected before node tables are read. Physical per-primitive stage price
files retain schema2: their capacity/source/target axes have not changed.

The planner reports first and steady replay costs, one-time preparation plus
calibration cost, total lifetime cost and rounded-up amortized cost. Every replay
includes explicit orchestration and publication costs. Selection minimizes the
total cost over the declared lifetime, subject to the same exact range and
physical memory admission as execution. UINT64_MAX means unknown or overflow;
it cannot become a free stage. `estimated_ns` is the first complete replay cost.

An optional baseline supplies complete replay and preparation costs. The report
distinguishes an unknown baseline, projected profit and projected nonprofit, and
gives the first replay count with strict projected improvement. Setting
`RNS8_PRODUCT_REQUIRE_PROJECTED_AMORTIZATION` admits only a projected profitable
candidate at the declared lifetime. This is a model guard; it does not establish
observed amortization or authorize AUTO promotion.

`RNS8_PRODUCT_RETAIN_EXTERNAL_PACKED_B` retains packed external B planes across
replays. Only immutable input instance/generation bindings qualify. Intermediates
are repacked each replay. Retained regions occupy the arena for the owner's full
lifetime and cannot alias temporary values, even before their first reader.
Packing growth is charged once for retained inputs and each replay otherwise.
The report exposes retained bytes, completed replay count, last replay's actual
packing calls and external reads that reused existing packing.

Input mutation rejects execution. Successful explicit rebinding invalidates only
changed input caches; failed rebinding preserves the prior binding and cache.
Execution failure invalidates cached packing before attempting its device drain.
Experimental-basis lift packets are uploaded during preparation, including for
the first replay; replay does not introduce a hidden packet transfer.
An uncertain drain retains the existing owner and reservations.

Product calibration cases optionally accept `input_values: [A, B]`, two constant
native integers with explicit signedness. Omission preserves all-one fixtures.
Each graph's constant values have independent exact closed-form references;
shape, signed constants and dataset identity must match frozen prices. Recipe
cases may also supply `replay_policy` using the public cost terms. Measured
training and publication charges cannot disappear behind smaller overrides.
Held-out checks compare execution plus final export against the same predicted
boundary, with declared common overhead identified as modeled. Independent
process checks remain scoped to the same graph, dataset and build; neighboring
shapes or datasets require their own calibration and held-out observations.

These additions do not change the source/toolchain boundary of earlier archived
measurements. The [combined CPU/HIP and collector validation](../metadata/replay-extension-validation-20261027.json)
passes the new runtime and collector fixtures. No new elapsed performance gain is
asserted from implementation alone.
