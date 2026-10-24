# Retained Gram computation and reconstruction cohorts

The public workload `integer-cross-product --gram` computes `XᵀX`, with K
observations and M=N integer features. Its explicit batch controls separate
repeated computation on retained inputs from repeated output materialization.
The new source, collector checks and CPU/HIP fixtures are authored only. They
await the consolidated validation batch; no timings or speedup are claimed.

## Matched application contracts

`--gram-reuse R` accepts1..256 complete recomputations per application call.
R is a computation count, not a result-cache hit count: each product overwrites
the same output with the same mathematical result. `--gram-export each` exports
after every product; `--gram-export once` exports after the final product only.
The default export policy for an explicit batch is `once`. Without
`--gram-reuse`, the existing single-product workload keeps its existing scope.
Both controls require `--gram`; other workload families reject them.

| Candidate | Retained preparation | Work inside each batch |
|---|---|---|
| `rns8_prepared_dense_gram_control` | Public `rns8_prepare_gemm` for packed `Xᵀ` and X | R synchronous `rns8_execute_prepared` calls, then the selected export policy |
| `rns8_single_source_gram_scalar` | Packed X and output storage | R public `rns8_gram_residues` calls with scalar triangular computation and mirrored output |
| `rns8_single_source_gram_wmma` | Packed X and output storage | R public Gram calls with explicit WMMA method, when supported |
| Native and Boost CPU controls | Native inputs and comparator buffers | R full products, each serialized into the common output word format |

This cohort uses the ordinary synchronous Gram API. Its triangular job-list
allocation remains inside each call, and the WMMA capability probe remains
charged there. Metadata states these costs explicitly. The dense control does
use a prepared owner; it does not rebuild that owner inside unchanged-input
samples. CPU native serialization happens R times under both export policies;
it is recorded separately from RNS reconstruction/downloads. These boundaries
must be considered when interpreting application costs.

The workload explicitly requests CPU reference or direct HIP. It rejects
other backends for this batch mode and never falls back. The WMMA route is
offered only for its implemented RDNA3 target family, with runtime admission
still authoritative. No prepared-group queue, asynchronous overlap, hardware
qualification or dispatch promotion follows from running this experiment.

## Exactness, generations and measured work

Admission uses `K * max_abs(X including changed X[0])²`, with no cancellation
assumption. Bounded mode requires this full-expression bound to fit signed64;
`--wide` requires the selected odd-modulus range to represent it uniquely and
exports three u64 limbs representing a signed192-bit value per cell. The original versioned recipe and
full output checksum remain unchanged by R or export policy.

Setup packs each input once at generation1 and prepares the dense owner.
Unchanged cold-call, warmup, steady and sustained samples reuse those packed
matrices. A real change to X[0] creates generation2, repacks both dense inputs
or the single Gram input, refreshes the prepared dense owner's input bindings,
and recomputes a complete independent Boost reference. The changed-input call
charges repacking/refresh in its own named phase. It is not included in the
unchanged-input steady median.

Every instrumented application call verifies the complete final native output
after timing. Intermediate identical products are not individually audited.
Sustained intervals disable diagnostic phase clocks and perform complete final
verification after the interval. They run until the requested elapsed time or
4096 application batches, recording whether the duration was reached. Each
batch still performs R computations and its declared RNS export count.

`calls_per_second` counts complete application batches. A separate
`gram_products_per_second` divides by the R products actually performed;
`median_gram_product_ns` is the batch median divided by R, not an isolated
kernel latency. Mathematical work fields count R times the dense-equivalent
`M*N*K` products and, for triangular candidates, R times
`N*(N+1)/2*K` independent products. Residue planes are implementation work.

The existing aggregate memory envelope retains all comparator buffers,
references, matrices and workspace simultaneously, plus a bounded preparation
metadata allowance. Reuse adds no per-product retained output arrays. The
additional batch work bound is `M*N*K*R <= 134217728`; the full independent
reference remains a real CPU cost. The optional timed naive-comparator limit
also counts all R products; blocked comparators remain present. See the
[workload resource limits](../examples/workloads/README.md#range-comparators-and-resource-limits).

## Reproducible commands

Use an already configured build with workload targets enabled. These commands
describe pending runs; they are not recorded evidence. On Linux omit `.exe`.
The installed workload source uses the same public headers and exported target.

```powershell
build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --reference-only --size tiny --gram --gram-reuse 3 --gram-export once --repeats 2 --warmups 1 --json
python tools/collect_workload_examples.py --exe build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --output temp/workloads/gram-each.json --runs 5 --timeout 600 -- --backend hip-direct --device 0 --size large --m 128 --n 128 --k 128 --bits 8 --seed 3135 --gram --gram-reuse 8 --gram-export each --repeats 9 --warmups 3 --sustained-ms 250
python tools/collect_workload_examples.py --exe build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --output temp/workloads/gram-once.json --runs 5 --timeout 600 -- --backend hip-direct --device 0 --size large --m 128 --n 128 --k 128 --bits 8 --seed 3135 --gram --gram-reuse 8 --gram-export once --repeats 9 --warmups 3 --sustained-ms 250
```

The collector binds requested R/export policy to every capture, keeps them in
comparison identity, and retains mismatches as failures. It independently
replays all original and changed integer outputs, including serialization and
range admission; checks preparation, work/rate and export/verification counts;
and requires the prepared dense, triangular scalar, blocked Boost and admitted
blocked native64 controls. Existing executable, build/source, command, process,
clock and device provenance remains attached. Raw captures remain ineligible
for performance promotion. The fixtures and CTest smoke registrations cover
both export policies; their authored presence is not execution evidence.
