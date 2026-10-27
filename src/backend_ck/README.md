# Composable Kernel Backend

Public availability and target evidence are summarized in the
[backend reference](../../docs/backend-notes.md); this page owns implementation
layout and admission details. The [dated local validation](../../docs/local-all-features-20261023.md)
records the specific Windows/gfx1100 configuration exercised.


Opt-in Composable Kernel accelerator backend for the active HIP/ROCm target.

The backend is compiled only with `RNS8_ENABLE_CK=ON`. It uses the selected
installed SDK or explicit CK source headers plus RNS8-owned HIP pack/output
helper kernels to run INT8-to-INT32 matrix GEMM for fixed-prefix bounded plans,
adaptive per-tile bounded plans, exact-wide RNS output, and finite u8. CK
CShuffle keeps the INT32 accumulation internal to the matrix-engine kernel and
uses an RNS8 centered-residue epilogue to store `int8_t` output. RNS8 helper
kernels only copy or add centered byte tiles when padding or K-split accumulation
requires temporary output storage. The path is optional and is not required for
CPU or direct-HIP correctness.

Operand interpretation is explicit. Core SDK10.1's CK1.3 WMMA defaults to signed
operands, so packing retains centered `int8_t` bytes. The retained CK1.1 source
defaults to unsigned WMMA operands and requires canonical unsigned-byte packing.
CMake reads the actual selected header defaults, rejects mixed/unknown contracts,
and records `ck_wmma_signed_inputs` in the compiled manifest. XDL retains signed
centered inputs. The interpretation is never inferred from the C++ storage type
or an SDK version label. CPU algebra covers all byte moduli; GPU regressions
exercise negative residues, poisoned padding and K=64/65536/65537. Accumulation
remains fused with centered-byte reduction; no INT32 global scratch is added.

Dependency discovery and compile probes remain evidence only. CK reports an
enabled correctness backend only in the explicit CK preset after the compiled
kernels, exact CPU/direct-HIP differential tests, benchmark schema fixtures,
and ISA gate are present. The current ISA gate is target-aware: RDNA targets
must produce CK WMMA matrix instructions, while CDNA targets must produce CK XDL
MFMA matrix instructions. It rejects scalar divide/remainder mnemonics and
reports INT32 global stores in matched CK GEMM symbols; the CDNA XDL gate allows
CK's internal block-map `v_rcp_iflag_f32` because it is launch geometry
indexing, not modular arithmetic.
Current CK captures keep
`performance_validated=false` until reviewed target-shape captures prove it is
the fastest accepted backend.

The CK preset generates RNS8's WMMA no-divide block-map include overlay from
the pinned repo-local CK header during configure. The patch is exact-match
guarded, no-ops when the source already has the patched form, fails fast if the
expected `MakeDefaultBlock2CTileMap` block has drifted, puts the overlay before
CK's include directory for the CK HIP compile, and registers the generated
patched header as a dependency of the compiled CK HIP object. CDNA builds select
CK XDL and do not use the WMMA overlay at runtime.
