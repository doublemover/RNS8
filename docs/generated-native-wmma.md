# Bounded generated native WMMA family

`RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA=ON` compiles six explicit research
variants: signed/unsigned native 64-bit inputs crossed with groups of one, two,
or four default-basis channels. The default is OFF and no public dispatcher
selects these symbols. Exactly one gfx1100/gfx1101/gfx1102 target is required.

The supported envelope is a 16x16x16 matrix instruction, one wave per CTA,
masked tails, native row-major inputs, planar centered-byte outputs, independent
channel accumulators, and K=65536 reduction slabs. Vector width, pipeline depth,
split-K, layout, output semantics and group specifications outside that envelope
are rejected before emitting files. The generator emits standalone kernels that
share the inspected native WMMA tile implementation. Target/builtin mismatches
are compile errors; a scalar fallback cannot acquire a generated variant ID.

Stable IDs bind the exact specification, ordered basis, per-modulus arithmetic
certificates, generator and shared source/proof dependency hashes. Both generated
source and registry headers have separate hashes. Each variant has its own
external kernel and launch symbol; launch wrappers check full qualified address
ranges and output overlap before dispatch. A/B sharing remains legal. The
compiler's actual configuration is recorded by the existing build manifest,
including the opt-in flag. Generator metadata alone never asserts compilation.

Each ID also binds a per-modulus composed proof of native reduction, byte
fragment values, the INT32 slab plus centered carry, and the final centered byte
store. Carry/output closure establishes the same range for subsequent slabs.
The existing proof-DAG verifier checks physical widths and modular domains;
the existing planner resource query checks the empty global temporary arena.
The compatibility record names form, group, slab, layout and output boundary.
It does not certify lane mapping, register use, spills or instruction execution.

The existing GPU differential fixture is reused for every generated variant,
including signed/unsigned full-width values, prefix/group tails, padding and
K=65535/65536/65537. Generator fixtures cover reproducibility, duplicate IDs,
unsupported specifications and source/header drift. The gfx1100 variants and
their test executable compile with VS2026/HIP; runtime tests remain pending.
Paired fixtures additionally compare each generated route with separate native
packing and direct residue GEMM, then reverse the contraction axis of both
operands to move values across tiles and slab boundaries while preserving AB.
Both routes check the same Boost result, padded bytes and unused suffix plane.
These cross-feature fixtures are authored; runtime qualification is pending.

`check_generated_native_wmma.py` regenerates expected artifacts before inspecting
the exact manifested symbols with the existing ISA extraction and resource
parsers. Its report binds host/device object hashes, code size, disassembly
hashes, matrix opcodes, resources, spill metadata and remainder diagnostics.
Missing matrix instructions, missing/incompatible wave/LDS metadata and global
INT32 stores fail inspection. This inspection is authored but not yet executed;
it is not a substitute for the differential suite or independent timing runs.

This is a bounded first matrix-family implementation for #55. Broader generated
layouts, split-K combinations, measured Pareto pruning, held-out performance,
upstream regression qualification and production promotion remain open. No
register-use, occupancy, throughput or compile-time improvement is claimed.
