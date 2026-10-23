# Native conversion and scheduling calibration

The `rns8-native-calibrate` research executable exposes explicit product/packing and epilogue microcases:

| Candidate | Timed work | Input boundary |
| --- | --- | --- |
| `pack_pair` | Pack both native operands | Resident native 64-bit |
| `packed_gemm` | Direct residue GEMM | Already packed residue planes |
| `pack_gemm` | Pack both operands, then direct GEMM | Resident native 64-bit |
| `wmma` | Fused native conversion/grouped WMMA | Resident native 64-bit |
| `queue` | Descriptor setup and persistent WMMA queue | Resident native 64-bit |
| `split_k` | Three fragments (or K if smaller), queue, canonicalize, merge/store | Resident native 64-bit |

The product cases in this table produce the same default-basis residue planes for matched
shape/prefix/signedness. This output boundary does not assert that a small prefix
uniquely represents the full integer product. Packing-only and already-packed
cases have different boundaries and cannot be compared as complete native-input
pipelines without accounting for missing stages. Group must be one for the
packing/direct baseline cases; WMMA candidates accept one, two or four.

Arguments are candidate, M, N, K, prefix, group, unsigned flag (0/1), repeat count,
warmup count, and a unique run ID. M/N are at most 1024, K at most 65537, and
M*N*K at most 16777216 to bound the independent CPU reference. A fixed seed and
full-width signed/unsigned extremes are included. Boost computes expected
residues before timing; outputs are poisoned before each call, and every warmup
and repeat checks output bytes and a tail canary outside the timed region.

The report preserves fractional host nanoseconds and separate HIP event labels,
compiled provenance, device/runtime identity, complete case parameters, and
logical buffer sizes. Host calls include candidate-specific allocation,
capability probing and synchronization. Initial uploads, reference construction,
poisoning and verification copies are excluded. Native and packed control buffers
are retained across all cases; source buffers are reused and destination buffers
are initialized between calls. Logical storage bytes are not actual HBM traffic.

`collect_native_calibration.py` accepts an explicit JSON list of case objects
with `candidate,m,n,k,prefix,group,unsigned_inputs`. It randomizes process order,
defaults to five independent processes per case, checks executable identity
before and after each process, and retains unique raw stdout/stderr records,
commands, UUIDs and failure records. The authored collector test mocks child
processes and does not claim GPU execution.

The VS2026/gfx1100 executable compiles. No collection has been run yet. Clock
resolution, ISA, counter evidence and qualification are explicitly absent; raw
records cannot promote a candidate. Instruction-dependency/throughput roofs,
dedicated HBM/LDS probes, measured reuse break-even and held-out planner regret
remain separate #31 acceptance work.

## Matched native integer comparison

`narrow_wmma`, `narrow_scalar` and `narrow_vector` use the same seeded byte-domain
values stored in native 64-bit arrays and produce bounded exact native 64-bit
integers. Signed data cover [-128,127]; unsigned data cover [0,255]. Both operands
include the maximum magnitude explicitly. These cases require prefix=group=1
(the prefix is an unused CLI placeholder) and K<=32768, which proves all INT32
absolute partial sums fit even for unsigned bytes. The scalar implementation is
signed-only; requesting unsigned scalar fails explicitly. WMMA and vector ALU
support both interpretations.

Host timing includes each complete checked call: WMMA/scalar scan actual device
inputs and prove partial sums; vector ALU resets and reads its overflow status.
All cases retain identical A/B/C and status buffer allocations. Candidate-local
scratch allocation, capability probing and synchronization remain timed. HIP
event labels retain component timing separately; output poisoning and Boost
comparison of every integer and eight tail canaries are outside timing.

The collector validates the dataset, native output contract, partial-sum bound
and buffer sizes. These cases cannot be compared directly with the residue-output
cases above. They do not yet include an RNS pipeline with integer reconstruction,
and no collection, ISA qualification or performance result is claimed.

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
optional rocWMMA and gfx1100. Collection and runtime qualification remain pending.
