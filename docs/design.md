# Implemented design

The [research specification](RNS8_RESEARCH_SPEC.md) defines intended architecture;
this page describes the source structure and API boundaries.

## Arithmetic layers

- `src/core/moduli.cpp`: descriptor validation, exact prefix products, bounds,
  and supported semantic contracts
- `src/cpu/`: residue packing and blocked ring multiplication; Boost.Multiprecision
  provides independent full-integer cells and CRT reference behavior
- `src/reconstruct/crt.cpp`: incremental Garner reconstruction and checked limb
  conversion on the CPU
- `src/backend_common/`: host/device finite-modulus reduction and the blocked
  scalar residue dot product used by the small persistent HIP path
- `src/backend_hip_direct/`: packing, RNS/finite GEMM, grouped and scheduled work,
  checked 192-bit reconstruction, and host/device transfers
- `src/backend_wrap64/`: separate strict-wrap storage, CPU oracle, and HIP kernels
- `src/backend_{hipblaslt,ck,rocwmma,amdgpu_builtins,vector_alu}/`: optional
  accelerators or native exact comparators

The default ladder has 28 moduli; execution supports prefixes through 20.
Direct-HIP wide reconstruction uses three unsigned 64-bit limbs, sufficient for
the supported prefix range. Export uses that implementation for every bounded
prefix. A short-prefix weighted sum in uint64 is not safe merely because the
modulus product fits uint64: each weighted term may still overflow.

## Handles and ownership

`rns8_context` binds the device/backend. A `rns8_plan` owns a validated contract,
selected prefix/tile schedule, lowering metadata, and backend selection. A
`rns8_workspace` is bound to the plan's full contract, not just its shape.
`rns8_matrix` owns its storage; source versions and currentness distinguish native
values, RNS residues, and wrap64 limbs on host and device.

A successful pack establishes current input storage. Reusing the same source
version is the caller's assertion that the input has not changed. A successful
GEMM establishes the backend-specific output domain. Export requires matching
semantics, shape, layout, schedule, and currentness. Stale or incompatible
handles must be rejected rather than silently reinterpreted.

For bounded/exact-wide work, device residues use modulus-major compact row-major
planes. Finite storage has one explicit modulus. Wrap storage uses eight bytes
per element. Packed host inputs may have padded leading dimensions.

## Explicit CPU exact-wide lifting

`src/core/api_exact_lift.cpp` implements `rns8_lift_exact_wide_cpu`; the C++
`Matrix::lift_exact_wide_cpu` wrapper forwards the same contract. Matrix capacity
and initialized planes remain distinct. The operation stages only the missing
modulus-major suffix within existing capacity, with an explicit byte budget.
It reconstructs one cell at a time from a proven source prefix and commits only
after all cells pass. Signed values use the centered full-integer representative,
not a reinterpretation of unsigned CRT output. Range and axis proofs describe
the same integer after lifting and remain intact.

This additive symbol changes no enum, descriptor layout, ABI version, existing
call behavior, plan/workspace ownership, or backend routing.

`rns8_gemm_exact_wide_cpu_auto` shares the validated lift preparation/staging and
pure range-bound builder. It operates within the original plan/workspace contract,
selects a sufficient prefix above that plan's selected floor, and computes each
plane from a view of initialized input planes plus staged suffixes. It stages
complete output as well, so no first-input commit or partial output can survive
a second-input failure. Shared inputs are deduplicated. The CPU route serializes
a private blocked ring primitive internally, reusing one `4 * N` byte row
accumulator across all selected modulus planes; ordinary CPU calls retain their
parallel policy. Caller serialization of the involved handles remains required.
The returned prefix supports an explicit fixed-prefix export plan without
rewriting the original plan/workspace. The public synchronous `rns8_lift_exact_wide_hip` route also exists for compatible
direct-HIP device-resident values; [correctness](correctness.md#exact-wide-output)
and [resident graphs](resident-graphs.md) describe its admission/publication rules.

## Reuse and grouping

Persistent matrices avoid repeated conversion when values are unchanged.
Prepack caches add backend-specific operand identity and lifetime constraints.
Result caches track source versions and explicitly dirty output regions; they
are not general automatic incremental computation.

The ordinary public grouped GEMM calls accept compatible current resident tasks.
[Prepared execution](prepared-execution.md) retains resources and generation
bindings. Explicit fixed-prefix direct-HIP submission and owned capture/replay
use caller streams, completion tokens and read/write reservations. Same-device
prepared groups permit independent submissions; [resident DAGs](resident-graphs.md)
retain intermediate values and validate whole-expression proofs before work.
The synchronous and asynchronous subsets have distinct failure-publication rules.

Benchmark-owned pack/export, graph, streaming and workload combinations remain
separate experiments. Their flags do not enlarge the public API envelope. See
[exact epilogues](exact-epilogues.md) for public complete-expression composition
and [immutable plans](plan-validation.md) for reusable structural validation.

## Experimental public composition and storage

The [experimental complex composition](complex-residue-composition.md) connects
paired exact-wide signed values to CPU/direct-HIP execution and synchronous
resident graphs. Each component retains its own bound and logical generation.
The [retained-native adaptive operation](adaptive-native-product.md#explicit-matrix-owned-native-inputs)
uses an explicit matrix flag to own native device inputs alongside residues;
other value-producing writes retire that native materialization. These public
bindings are authored for the current batch and are distinct from the earlier
private-kernel validation.

The opt-in [public Strassen owner](strassen-device.md#public-owner-authored-validation-pending)
uses ordinary exact-wide signed/unsigned plans on direct HIP. It validates the
complete product and publishes through existing matrix ownership before export.
Rank-23 straight-line execution and default routing remain separate work.

The [distributed M0 plan/mock](distributed-mock.md) models exact tile/plane
membership, transport phases and whole-epoch publication on the CPU. It supplies
neither a public distributed matrix nor actual GPU transport.

## Files and generated metadata

Several C++ translation units include `.inc` fragments to keep related APIs or
kernel wrappers in one compilation context. The fragments are not independent
libraries. New helpers should be small, testable, and shared only when their
contracts genuinely match.

`metadata/*.yaml` are JSON-compatible registry files. Generate Python/C++
constants with `python tools/metadata_registry.py --write-generated` and check
with `--check`. Registration is metadata validation, not an execution proof.

hipBLASLt workspace operand packs bind the stable matrix instance ID and its
engine-owned content generation, together with shape, prefix and device. Caller
source labels and generation equality across distinct handles never authorize
reuse. Zero or exhausted generations force repacking while retaining allocated
scratch capacity. Exact-wide chains and same-generation operand swaps cover this
contract in the optional-backend differentials.
