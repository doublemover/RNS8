# Backend Notes

RNS8 backends are selected by explicit semantic contracts. Unsupported
backends return unsupported status; they do not route through compatibility
shims or hidden fallback semantics.

## Implemented Families

- CPU reference: deterministic correctness backend for bounded i64/u64,
  exact-wide RNS output, and finite u8.
- Direct HIP: Windows `gfx1100` correctness backend for persistent RNS
  matrices, bounded i64/u64, exact-wide limb export, finite u8, and strict
  wrap64 byte-limb storage.
- Native vector ALU: Windows HIP bounded i64/u64 backend for explicit bounded
  contracts. It is not an exact-wide, finite, or wrap64 backend.
- Wrap64 byte-limb CPU: strict `mod 2^64` reference backend. It is separate
  from odd-modulus CRT.
- hipBLASLt: opt-in Windows HIP baseline for fixed-prefix bounded, exact-wide
  RNS output, and finite u8. It uses INT8-to-INT32 scratch plus a separate
  residue-reduction kernel. Fixed-prefix single-K-block RNS work can reuse
  workspace-local repeated-A and repeated-B transposed operands when identity
  matches, but this is not a public production prepack cache.
- CK: opt-in fused matrix-engine backend for bounded, adaptive bounded,
  exact-wide RNS output, and finite u8. It selects CK WMMA on RDNA targets and
  CK XDL on CDNA targets.
- rocWMMA: opt-in fused matrix-engine backend for bounded,
  adaptive bounded, exact-wide RNS output, and finite u8. It also contains an
  internal wrap64 candidate harness that is not public or AUTO-selected.
- AMDGPU builtins: opt-in target-specific backend for MFMA, WMMA, SMFMAC, and
  SWMMAC kernels. The enum, capability surface, schemas, reviews, autotune
  metadata, runtime contexts, and GEMM dispatch name the path honestly; release
  promotion still requires exact CPU parity, timing, and ISA evidence gates.

## Sparse-A v1

Sparse support is an explicit A-side 4:2 structured contract, not automatic
dense pruning and not a general sparse GEMM API. A valid sparse-A v1 descriptor
requires dense B, expanded K divisible by four, exactly two nonzero A values in
each four-lane K group, ascending canonical two-bit indices, explicit signed or
unsigned byte interpretation, and semantic identity for either RNS prefix
planes or one finite-u8 modulus plane.

Sparse handles are separate from dense `rns8_matrix` handles. Dense GEMM APIs
cannot consume sparse A, sparse GEMM APIs reject dense A, and accelerator
sparse paths require an explicit sparse-A handle on CDNA3 SMFMAC or RDNA4
SWMMAC targets. The CPU sparse path expands resident sparse A into the existing
exact dense CPU reference before GEMM; this remains the correctness anchor for
CDNA3 SMFMAC and RDNA4 SWMMAC claims.

## Selection Policy

`RNS8_BACKEND_AUTO` is a context-default selector. In HIP-capable contexts it
starts from direct-HIP correctness and may select a compiled HIP backend only
when a reviewed release autotune-cache entry exactly matches the plan key,
target id, and runtime library version. hipBLASLt, CK, and rocWMMA entries
remain residue-current accelerator selections; `hip-vector-alu-int64` entries
are accepted only for bounded i64/u64 final/native-output plans. Without that
exact hit, AUTO remains on the configured correctness path.

Dependency discovery, header probes, CMake probes, and tiny compile probes are
candidate evidence only. They do not enable CK, rocWMMA, hipBLASLt, AMDGPU
builtins, or sparse matrix-core paths by themselves.

## Public Metadata

The ABI exposes backend capability through `rns8_get_backend_capability_info`
and plan-selected backend metadata through `rns8_get_plan_backend_info`.
Current metadata reports selected kernel, accelerator library/version,
capability status, epilogue mode, workspace mode, workspace byte requirement,
ISA evidence, and autotune key.

`rns8_get_plan_packing_info` reports resident layout names and transient
accelerator pack workspace bytes. It also reports the selected input/output
domain, whether a successful GEMM leaves host or device output current, and
next-operation flags for final export, RNS continuation, native continuation,
native-to-RNS conversion eligibility, and reusable B prepack availability.
rocWMMA reports production B-side prepack-cache availability only for the
narrow validated non-tiled RNS path; other accelerator backends still report
transient per-dispatch packing or resident direct input layouts.

`rns8_get_grouped_dispatch_contract_info` reports the current Direct-HIP
same-shape grouped descriptor contract for a created plan: task count,
shape/stride policy, unique matrix and workspace ownership, source-version
repack policy, device-current output, host-output export requirement,
status/checksum policy, descriptor reuse, and device descriptor layout. This is
contract metadata for `rns8_gemm_rns_grouped` and
`rns8_gemm_finite_u8_grouped`, the narrow public resident grouped GEMM entry
points. Those calls require already-current resident task inputs and do not
perform grouped host packing, native-to-RNS conversion, final host export, or
AUTO routing.

`rns8-inspect` derives an internal plan-lowering description for autotune
exact-hit plans from the same backend, packing, and schedule metadata. The
diagnostic explains the `MatMul` semantic contract, backend family, selected
domain transition, schedule, packing, reuse, conversion strategy, and final
lowering path without adding a new public algebra API.

`rns8-bench` adds evidence-only metadata around that public inspection surface:
`requested_next_op`, `output_policy`, `target_variant`, `auto_selector`,
`device_allocation`, `reuse_contract`, `exact_output_contract`,
`export_variant`, `reconstruction_variant`, `modulus_set`,
`residue_count_policy`, `tile_shape_variant`, `grouped_dispatch`,
`adaptive_grouped_scheduler`, `hip_graph_replay`, `resident_lifetime`,
`workspace_arena`, `streaming_overlap`, `release_gate`,
`verification_amortization`, and `workload_proxy`. These objects explain
benchmark intent, padded versus contiguous export layout, status-event
expectations, target namespace, AUTO cache fallback/rejection reasons, HIP
allocation counters before warmup and after repeats, experimental
modulus/tile/export/CRT metadata, graph/grouped/adaptive descriptor identity,
resident currentness and arena identity, stream-overlap contracts, release
gate blockers, and FHE/lattice-inspired proxy workload shape. They do not
change backend selection or public API semantics.

`rns8-inspect --selector-shadow` reports a non-routing family recommendation
and fixed blocker vocabulary beside the exact-cache selector state. The
recommendation is advisory only: AUTO still requires an exact reviewed cache key
for the concrete semantic contract, shape, target, workspace, and selected
kernel identity.

## Benchmark Promotion

Raw `rns8-bench` captures cannot write production autotune entries.
`tools/benchmark_sweep.py --review-mode release --write-autotune-cache` is the
promotion boundary: it validates schema, groups same-contract captures, checks
required baselines, requires release repeat counts, and writes only fastest
reviewed accelerator winners. Review output separately reports the fastest
production route, which may be Direct HIP, and the fastest accelerator route,
which excludes correctness baselines. This prevents a correct Direct-HIP win
from being misread as "no wins."

`tools/gpu_event_report.py`, `tools/gpu_isa_report.py`, and
`tools/gpu_counter_report.py` are attribution tools for optimizer work. Their
outputs stay under ignored `temp/`; ISA and counter conclusions explain likely
bottlenecks but cannot replace exact correctness, host timing, HIP event timing,
or release baseline requirements.

See [performance-model.md](performance-model.md) for benchmark schema and
release-evidence policy.
