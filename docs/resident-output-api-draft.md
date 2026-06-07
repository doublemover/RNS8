# Resident Output API Draft

This draft closes the current performance-queue design requirement for
resident matrix lifetime and residue-current output handles without changing
the public ABI in this PR.

## Goals

- Make persistent RNS matrix storage explicit: creation/import, source version,
  device-current state, workspace binding, output domain, reset, and release.
- Allow a GEMM output that is still RNS-current on device to feed another RNS
  consumer without host CRT export unless the caller requests native/final
  output.
- Keep semantics explicit. Signed bounded, unsigned bounded, exact-wide,
  finite-u8, and wrap64 byte-limb states must not be inferred from C++ element
  type alone.

## Draft Handle Model

The future C API should keep `rns8_context` and `rns8_plan` as owners of
runtime and lowering policy, then add opaque resident matrix handles with
explicit descriptors:

- semantic contract: bounded signed, bounded unsigned, exact-wide signed,
  exact-wide unsigned, finite ring/field, or wrap64 byte-limb;
- input/output domain: native host, resident RNS, resident finite, resident
  wrap-byte, or exact-wide limb host;
- shape and leading dimensions for logical host view and device storage view;
- prefix/modulus schedule identity and target/backend identity;
- source version and currentness state;
- workspace binding and stream-safety policy.

Resident matrix operations should fail deterministically when a source version,
descriptor, semantic contract, prefix/modulus schedule, target id, backend
version, output policy, or workspace fingerprint no longer matches the plan
that wants to consume it.

## Benchmark Evidence Boundary

Current `rns8-bench` captures use `resident_lifetime`, `reuse_contract`,
`workspace_arena`, `exact_output_contract`, and `requested_next_op` metadata to
model this API without exposing new public handles. Promotion requires:

- exact CPU final-output comparison for the requested output domain;
- stale-source and stale-workspace rejection tests;
- allocation-free measured-repeat proof for arena-backed resident workflows;
- setup-inclusive and steady-state comparisons against same-contract
  independent calls;
- target-specific reviewed evidence before any AUTO routing.

Until those gates are satisfied, residue-current output and resident reuse are
optimizer evidence, not default public routing.

## Rank 46 Chain Contract

The current exact-wide chain evidence models residue-current output as a
benchmark-owned lifetime, not a public handle. A residue-current chain output
is consumable by a later RNS GEMM only while all of these remain true:

- the context, physical device, stream owner, and workspace fingerprint match;
- the semantic contract, shape, leading dimensions, prefix schedule, modulus
  set, limb contract, output policy, and selected backend metadata match;
- every source matrix version is unchanged since the residue-current output was
  produced;
- the output is device-current and host-not-current until an explicit final
  export or checksum export occurs;
- no caller assumes host-visible exact limbs, native integers, or finite values
  from a residue-current output;
- the final host export is the only boundary that can be compared to the CPU
  exact reference.

`tools/exact_wide_chain_report.py` enforces this benchmark contract for the
current evidence by requiring a residue-current capture, a same-backend
final-output capture, a CPU final-output baseline, release settings, required
GPU events, and explicit output-currentness metadata for each exact-wide
chain/shape/limb contract.
