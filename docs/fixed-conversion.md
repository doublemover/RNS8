# Fixed packed-byte conversion transforms

The private `hip_direct_fixed_conversion_device` entry exposes three research
modes: signed native import, unsigned native import, and canonical three-limb
CRT export. The first implementation uses RDNA3 unsigned packed-dot4 instructions
and retains existing public dispatch. It is not a native arbitrary-width dtype
or an approximate/floating reconstruction.

Immutable constants derive from the existing default modulus ladder and exact
CRT provider. Import weights are 256^j mod m for j=0..7, packed into two words.
Two unsigned dot4 instructions sum the eight byte products (at most 520200).
Signed import subtracts 2^64 mod m when the native sign bit is set; independent
signed-byte reinterpretation would be incorrect. Reduction produces the ordinary
centered residue representation. Each source cell is an aligned native64 value;
output remains modulus-major bytes.

CRT computes v_i=(r_i*(P/m_i)^-1) mod m_i, then multiplies these canonical bytes
by fixed byte columns of P/m_i. Five packed words cover up to20 planes; padding
coefficients and weights are zero. Each column sum is at most1300500 before
carry. Exact base256 carry propagation reconstructs the same unreduced integer
S as the direct CRT path. Since S<pP<2^165, 24 bytes suffice, and the existing
192-bit comparison/subtraction gives canonical S mod P. No floating quotient,
probabilistic correction, or signed matrix bias shortcut is used.

The [Clang builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
specifies `__builtin_amdgcn_udot4` and its dot7 target feature. The current
candidate admits only gfx1100/1101/1102 with that builtin and probes a compiled
device capability before writes. It checks complete spans, native alignment,
nonoverlap, prefix1..20 and a four-byte probe budget. After launch, failure marks
writes as potentially started; this private interface grants no public output
publication by itself.

CPU tests independently compare imported residues and unreduced CRT sums to
Boost over every prefix, full-width signs and byte products. GPU fixtures cover
cell tails, every prefix, signed/unsigned boundaries, aliases, budget rejection
and padding. Compilation, ISA inspection, matched packing/CRT timing and
performance promotion are pending. Scalar direct/Garner/balanced CRT remain
controls. WMMA transform variants, signed-bias matrix layouts, and broader-target
qualification are follow-on development, not implied by this packed-dot path.
