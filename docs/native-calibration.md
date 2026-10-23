# Native conversion and scheduling calibration

The `rns8-native-calibrate` research executable exposes six explicit microcases:

| Candidate | Timed work | Input boundary |
| --- | --- | --- |
| `pack_pair` | Pack both native operands | Resident native 64-bit |
| `packed_gemm` | Direct residue GEMM | Already packed residue planes |
| `pack_gemm` | Pack both operands, then direct GEMM | Resident native 64-bit |
| `wmma` | Fused native conversion/grouped WMMA | Resident native 64-bit |
| `queue` | Descriptor setup and persistent WMMA queue | Resident native 64-bit |
| `split_k` | Three fragments (or K if smaller), queue, canonicalize, merge/store | Resident native 64-bit |

All product cases produce the same default-basis residue planes for matched
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
