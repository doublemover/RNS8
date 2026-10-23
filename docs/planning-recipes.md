# Offline typed recipe admission

`tools/planning_recipe.py` adds a typed provider boundary around the existing
arithmetic verifier and resource-constrained predictor. It does not launch
kernels, alter public planning, create reviewed cache entries or qualify a
target. Providers enumerate concrete stage descriptions; the central gate
admits recipes before consulting costs.

The records distinguish problem shape/strides, full expression and output
boundary; generation-bound input range/materialization facts; supported
kernel/certificate/layout combinations; calibrated rates; complete tile/plane/
K-slab coverage; and aligned temporary storage. Initial expressions are `A*B`
under finite-ring/field or bounded signed/unsigned semantics. Other expressions
reject explicitly until their composed proof and recipe are supplied. An
epilogue cannot reuse an ordinary GEMM identity.

Every basis plane needs its ordered algebraic certificate. The actual current
operand envelope must fit each bounded-integer certificate, and the chosen K
slab must respect every instruction bound. Kernel capability records bind
target/compiler/artifact, certificate, opcode, layout and instruction-evidence
digest. These records consume evidence identities; a supplied digest is not
itself verified hardware evidence. Cost environments must match the target
artifact, and every stage symbol and work term needs calibration. Use
`profile_from_calibrations` to apply the existing independent-run, ISA and
reference admission checks when importing measurements. CPU mock fixtures
construct explicitly fabricated profiles instead.

Coverage is checked as disjoint half-open ranges over the full flattened
output-tile/plane/K-slab domain. Duplicate arrivals cannot hide missing work.
Address admission conservatively checks 64-bit element spans. Providers must
account for conversions, imports/exports and the contract's declared reuse in
their work/stage descriptions; the boundary does not invent amortization or
infer omitted kernels.

Workspace queries return success, unsupported, overflow or unknown. Failure
never becomes zero scratch. Successful queries assign aligned nonoverlapping
arena regions with checked 64-bit arithmetic; padding counts against the
budget. This arena is conservative and does not reuse storage across lifetimes.
Each region names its producer and direct consumers, while prediction retains
the producer allocation through all dependent consumers. A retained region
survives to the end of the graph. The model's lifetime peak and the arena's
actual required bytes are reported separately.

Shortlists are deterministic for the same typed inputs, with ties resolved by
workspace then recipe name. Cost changes can reorder mathematically admissible
recipes but cannot waive failed proofs. Binding rechecks instance, generation,
range and materializations before a caller could submit. This is an offline
integration boundary, not a C++ runtime provider or asynchronous submitter;
those adapters and qualified calibration cohorts remain pending under #30.

Authored mock-provider tests cover deterministic serialization, cost-only rank
changes, missing instructions/calibration, stale bindings, insufficient CRT
range, incomplete/duplicate coverage, workspace fallback, alignment overflow,
unknown query results and malformed numerical stage fields. Execution remains
pending during frontloaded development.
