# Fixed packed-byte conversion transforms

The private `hip_direct_fixed_conversion_device` entry exposes six research
modes: signed native import, unsigned native import, and canonical three-limb
CRT export, each with packed-dot4 and unsigned-byte WMMA candidates.
The first implementation uses RDNA3 unsigned packed-dot4 instructions
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
controls. Broader-target qualification remains deferred. Neither candidate changes
public dispatch or establishes a measured speedup.

The [conversion calibration driver and collector](conversion-calibration.md)
provide matched scalar import, Garner/direct/balanced CRT and fixed-dot4 cases
for every prefix in resident and cold scopes. Full Boost output/canary audits,
paired production timings and sustained large-case manifests are authored;
execution, ISA inspection and performance results remain pending.

## Unsigned-byte WMMA transforms

Modes3/4/5 select signed import, unsigned import and canonical CRT through the
RDNA3 wave32 `wmma_i32_16x16x16_iu8` builtin with both operand signedness flags
false. Signed native import still corrects the whole 64-bit sign after the
unsigned-byte transform. No signed-byte bias is assumed or omitted.

Import maps16 native cells to rows and16 modulus planes to columns. The eight
native digits and fixed powers are padded with eight zero digits; output fragment
row `2*register + lane/16` and column `lane%16` map directly to modulus-major
residues. Tail cells and planes are predicated after the wave-wide instruction.
Every accumulator is at most520200. No intermediate global allocation is needed.

CRT maps16 cells to rows and24 output bytes to columns, with two column tiles and
up to two K=16 slabs covering all20 planes. Canonical inverse-scaled coefficients
and fixed CRT byte weights are unsigned; inactive planes and columns are zero.
The wave stores at most1300500 per column in1536 bytes of LDS. After a block
barrier,16 lanes independently propagate exact carries across24 bytes and apply
the same 192-bit canonical correction as packed-dot CRT. LDS is per-block kernel
storage, not an unreported host/device scratch allocation. Probe budget stays
four bytes. Capability gating requires the compiled RDNA3 WMMA builtin.

Every-prefix Boost differential cases now include15/16/17 and31/32/33 cell tails,
full-width signed inputs, all-byte extrema and canaries for both transforms.
Matched resident/cold manifests include `fixed_wmma`, `import_wmma_signed` and
`import_wmma_unsigned`; compilation, ISA and measurements remain pending.

Capability scratch has a preallocated deferred-cleanup record. A failed drain or
status copy retains that allocation instead of freeing a possibly active device
pointer. Successful device synchronization reclaims only records detached before
that drain; records from later concurrent calls remain queued. Context teardown
also attempts collection. A failed free preserves the record for another cleanup
attempt. No allocation-failure campaign has been run.
