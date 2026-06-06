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
